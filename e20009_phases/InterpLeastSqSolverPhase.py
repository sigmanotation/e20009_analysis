from spyral.core.phase import PhaseLike, PhaseResult
from spyral.core.status_message import StatusMessage
from spyral.core.cluster import Cluster
from spyral.core.spy_log import spyral_warn, spyral_error, spyral_info
from spyral.core.run_stacks import form_run_string
from spyral.core.track_generator import (
    generate_track_mesh,
    MeshParameters,
    check_mesh_needs_generation,
)
from spyral.core.constants import QBRHO_2_P, BIG_PAD_HEIGHT
from spyral.core.estimator import Direction
from spyral.interpolate.track_interpolator import (
    create_interpolator_from_array,
    TrackInterpolator,
)
from spyral.solvers.guess import Guess
from spyral.solvers.solver_interp_leastsq import (
    solve_physics_interp,
    create_params,
    objective_function,
)
from spyral.phases.schema import ESTIMATE_SCHEMA, INTERP_SOLVER_SCHEMA

# Import e20009 specific data classes
from e20009_phases.config import SolverParameters, DetectorParameters

from spyral_utils.nuclear.target import load_target, GasTarget
from spyral_utils.nuclear.particle_id import deserialize_particle_id, ParticleID
from spyral_utils.nuclear.nuclear_map import NuclearDataMap
from spyral_utils.plot import Cut2D
from spyral_utils.nuclear import NucleusData
import h5py as h5
import polars as pl
from pathlib import Path
from lmfit import Parameters, minimize, fit_report
from lmfit.minimizer import MinimizerResult
from multiprocessing import SimpleQueue
from numpy.random import Generator
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager

"""
Changes from attpc_spyral package base code (circa July 30, 2024):
    - InterpLeastSqSolverPhase run method pulls gain-match factor for the run being analyzed from the specified file 
      and applies it. The estimates_gated dataframe now has additional gates to only select events with the 
      correct IC and IC SCA information. StatusMessage now takes self.name as first argument instead of "Interp. Solver".
      Run method also pulls in window and micromegas time buckets and feeds them to solve_physics_interp() for
      errors to the fit.
    - solve_physics_interp() function from solver_interp_leastsq.py edited to take in window and micromegas timebuckets 
      directly and their errors. The x and y errors are split up as the triangular pads are circumscribed by a 
      rectangle not a square. The z error is fully propagated with errors from time bucket edges. Removed the 
      float when taking difference of two edges in z error.
"""

DEFAULT_PID_XAXIS = "dEdx"
DEFAULT_PID_YAXIS = "brho"


class InterpLeastSqSolverError(Exception):
    pass


def form_physics_file_name(run_number: int, particle: ParticleID) -> str:
    """Form a physics file string

    Physics files are run number + solved for isotope

    Parameters
    ----------
    run_number: int
        The run number
    particle: ParticleID
        The particle ID used by the solver.

    Returns
    -------
    str
        The run file string

    """
    return f"{form_run_string(run_number)}_{particle.nucleus.isotopic_symbol}.parquet"


class InterpLeastSqSolverPhase(PhaseLike):
    """The default Spyral solver phase, inheriting from PhaseLike

    The goal of the solver phase is to get exact (or as exact as possible) values
    for the physical observables of a trajectory. InterpSolverPhase uses a pre-calculated
    mesh of ODE solutions to interpolate a model particle trajectory for a given set of
    kinematics (energy, angles, vertex) and fit the best model trajectory to the data. InterpSolverPhase
    is expected to be run after the EstimationPhase.

    Parameters
    ----------
    solver_params: SolverParameters
        Parameters controlling the interpolation mesh and fitting
    det_params: DetectorParameters
        Parameters describing the detector

    Attributes
    ----------
    solver_params: SolverParameters
        Parameters controlling the interpolation mesh and fitting
    det_params: DetectorParameters
        Parameters describing the detector
    nuclear_map: spyral_utils.nuclear.NuclearDataMap
        A map containing isotopic information
    track_path: pathlib.Path
        Path to the ODE solution mesh
    """

    def __init__(self, solver_params: SolverParameters, det_params: DetectorParameters):
        super().__init__(
            "InterpLeastSqSolver",
            incoming_schema=None,
            outgoing_schema=None,
        )
        self.solver_params = solver_params
        self.det_params = det_params
        self.nuclear_map = NuclearDataMap()
        self.shared_mesh_shape: tuple | None = None
        self.shared_mesh_name: str | None = None
        self.shared_mesh_type: np.dtype | None = None

    def create_assets(self, workspace_path: Path) -> bool:
        global manager
        target = load_target(Path(self.solver_params.gas_data_path), self.nuclear_map)
        pid = deserialize_particle_id(
            Path(self.solver_params.particle_id_filename), self.nuclear_map
        )
        if pid is None:
            raise InterpLeastSqSolverError(
                "Could not create trajectory mesh, particle ID is not formatted correctly!"
            )
        if not isinstance(target, GasTarget):
            raise InterpLeastSqSolverError(
                "Could not create trajectory mesh, target is not a GasTarget!"
            )
        mesh_params = MeshParameters(
            target,
            pid.nucleus,
            self.det_params.magnetic_field,
            self.det_params.electric_field,
            self.solver_params.n_time_steps,
            self.solver_params.interp_ke_min,
            self.solver_params.interp_ke_max,
            self.solver_params.interp_ke_bins,
            self.solver_params.interp_polar_min,
            self.solver_params.interp_polar_max,
            self.solver_params.interp_polar_bins,
        )
        self.track_path = (
            self.get_asset_storage_path(workspace_path)
            / mesh_params.get_track_file_name()
        )
        meta_path = (
            self.get_asset_storage_path(workspace_path)
            / mesh_params.get_track_meta_file_name()
        )
        do_gen = check_mesh_needs_generation(self.track_path, mesh_params)
        if do_gen:
            generate_track_mesh(mesh_params, self.track_path, meta_path)

        return True

    def create_shared_data(
        self, workspace_path: Path, manager: SharedMemoryManager
    ) -> None:
        # Create a block of shared memory of the same total size as the mesh
        # Note that we don't have a lock on the shared memory as the mesh is
        # used read-only
        mesh_data: np.ndarray = np.load(self.track_path)
        self.memory = manager.SharedMemory(
            mesh_data.nbytes
        )  # Stored as class member for windows reasons, simply a keep-alive
        spyral_info(
            __name__,
            f"Allocated {mesh_data.nbytes * 1.0e-9:.2} GB of memory for shared mesh.",
        )
        memory_array = np.ndarray(
            mesh_data.shape, dtype=mesh_data.dtype, buffer=self.memory.buf
        )
        memory_array[:, :, :, :] = mesh_data[:, :, :, :]
        # The name allows us to access later, shape and dtype are for casting to numpy
        self.shared_mesh_name = self.memory.name
        self.shared_mesh_shape = memory_array.shape
        self.shared_mesh_type = memory_array.dtype

    def construct_artifact(
        self, payload: PhaseResult, workspace_path: Path
    ) -> PhaseResult:
        if not self.solver_params.particle_id_filename.exists():
            spyral_warn(
                __name__,
                f"Particle ID {self.solver_params.particle_id_filename} does not exist, Solver will not run!",
            )
            return PhaseResult.invalid_result(payload.run_number)

        pid: ParticleID | None = deserialize_particle_id(
            Path(self.solver_params.particle_id_filename), self.nuclear_map
        )
        if pid is None:
            spyral_warn(
                __name__,
                f"Particle ID {self.solver_params.particle_id_filename} is not valid, Solver will not run!",
            )
            return PhaseResult.invalid_result(payload.run_number)

        result = PhaseResult(
            artifact_path=self.get_artifact_path(workspace_path)
            / form_physics_file_name(payload.run_number, pid),
            successful=True,
            run_number=payload.run_number,
        )
        return result

    def run(
        self,
        payload: PhaseResult,
        workspace_path: Path,
        msg_queue: SimpleQueue,
        rng: Generator,
    ) -> PhaseResult:
        # Need particle ID the correct data subset
        pid: ParticleID | None = deserialize_particle_id(
            Path(self.solver_params.particle_id_filename), self.nuclear_map
        )
        if pid is None:
            msg_queue.put(StatusMessage("Waiting", 0, 0, payload.run_number))
            spyral_warn(
                __name__,
                f"Particle ID {self.solver_params.particle_id_filename} does not exist, Solver will not run!",
            )
            return PhaseResult.invalid_result(payload.run_number)

        # Check the cluster phase and estimate phase data
        cluster_path: Path = payload.metadata["cluster_path"]
        estimate_path = payload.artifact_path
        if not cluster_path.exists() or not estimate_path.exists():
            msg_queue.put(StatusMessage("Waiting", 0, 0, payload.run_number))
            spyral_warn(
                __name__,
                f"Either clusters or esitmates do not exist for run {payload.run_number} at phase 4. Skipping.",
            )
            return PhaseResult.invalid_result(payload.run_number)

        # Setup files
        result = self.construct_artifact(payload, workspace_path)
        cluster_file = h5.File(cluster_path, "r")
        estimate_df = pl.scan_parquet(estimate_path)

        cluster_group: h5.Group = cluster_file["cluster"]  # type: ignore
        if not isinstance(cluster_group, h5.Group):
            spyral_error(
                __name__,
                f"Cluster group does not eixst for run {payload.run_number} at phase 4!",
            )
            return PhaseResult.invalid_result(payload.run_number)

        # Extract cut axis names if present. Otherwise default to dEdx, brho
        xaxis = DEFAULT_PID_XAXIS
        yaxis = DEFAULT_PID_YAXIS
        if not pid.cut.is_default_x_axis() and not pid.cut.is_default_y_axis():
            xaxis = pid.cut.get_x_axis()
            yaxis = pid.cut.get_y_axis()

        # Retrieve gain-matching factor
        gm_lf: pl.LazyFrame = pl.scan_csv(self.solver_params.gain_match_factors_path)
        gm_df: pl.DataFrame = gm_lf.filter(
            pl.col("run") == payload.run_number
        ).collect()
        if gm_df.shape[0] > 1:
            spyral_error(
                __name__,
                f"Multiple gain match factors found for run {payload.run_number}, solving phase cannot be run!",
            )
            return PhaseResult.invalid_result(payload.run_number)
        elif gm_df.shape[0] == 0:
            spyral_error(
                __name__,
                f"No gain match factor found for run {payload.run_number}, solving phase cannot be run!",
            )
            return PhaseResult.invalid_result(payload.run_number)
        gain_factor: float = gm_df.get_column("gain_factor")[0]

        # Apply gain-matching factor to PID
        pid_vertices: list[tuple[float, float]] = list(pid.cut.get_vertices())
        pid_vertices_matched: list[tuple[float, float]] = [
            (point[0] / gain_factor, point[1]) for point in pid_vertices
        ]
        pid.cut = Cut2D(
            pid.cut.name,
            pid_vertices_matched,
            pid.cut.get_x_axis(),
            pid.cut.get_y_axis(),
        )

        # Load drift velocity information
        dv_lf: pl.LazyFrame = pl.scan_csv(self.det_params.drift_velocity_path)
        dv_df: pl.DataFrame = dv_lf.filter(
            pl.col("run") == payload.run_number
        ).collect()
        if dv_df.shape[0] > 1:
            spyral_error(
                __name__,
                f"Multiple drift velocities found for run {payload.run_number}, interpolation solver cannot be run!",
            )
            return PhaseResult.invalid_result(payload.run_number)
        elif dv_df.shape[0] == 0:
            spyral_error(
                __name__,
                f"No drift velocity found for run {payload.run_number}, interpolation solver cannot be run!",
            )
            return PhaseResult.invalid_result(payload.run_number)
        mm_tb: float = dv_df.get_column("average_micromegas_tb")[0]
        w_tb: float = dv_df.get_column("average_window_tb")[0]
        mm_err: float = dv_df.get_column("average_micromegas_tb_error")[0]
        w_err: float = dv_df.get_column("average_window_tb_error")[0]

        # mm_tb: float = 62
        # w_tb: float = 396
        # mm_err: float = 0
        # w_err: float = 0

        # Select the particle group data, beam region of ic, convert to dictionary for row-wise operations
        estimates_gated = estimate_df.filter(
            pl.struct([xaxis, yaxis]).map_batches(pid.cut.is_cols_inside)
            & (pl.col("ic_multiplicity") == 1)
            & (pl.col("ic_sca_multiplicity") == 1)
        )
        estimates_gated = (
            estimates_gated.filter(
                (pl.col("ic_amplitude") >= self.solver_params.ic_min_val)
                & (pl.col("ic_amplitude") < self.solver_params.ic_max_val)
                & (abs(pl.col("ic_centroid") - pl.col("ic_sca_centroid")) <= 10)
            )
            .collect()
            .to_dict()
        )

        # Check that data actually exists for given PID
        if len(estimates_gated["event"]) == 0:
            msg_queue.put(StatusMessage("Waiting", 0, 0, payload.run_number))
            spyral_warn(__name__, f"No events within PID for run {payload.run_number}!")
            return PhaseResult.invalid_result(payload.run_number)

        nevents = len(estimates_gated["event"])
        total: int
        flush_val: int
        if nevents < 100:
            total = nevents
            flush_val = 0
        else:
            flush_percent = 0.01
            flush_val = int(flush_percent * (nevents))
            total = 100

        count = 0

        msg = StatusMessage(
            self.name, 1, total, payload.run_number
        )  # We always increment by 1

        # Result storage
        phys_results: dict[str, list] = {
            "event": [],
            "cluster_index": [],
            "cluster_label": [],
            "vertex_x": [],
            "sigma_vx": [],
            "vertex_y": [],
            "sigma_vy": [],
            "vertex_z": [],
            "sigma_vz": [],
            "brho": [],
            "sigma_brho": [],
            "ke": [],
            "sigma_ke": [],
            "polar": [],
            "sigma_polar": [],
            "azimuthal": [],
            "sigma_azimuthal": [],
            "redchisq": [],
        }

        # load the ODE solution interpolator
        if self.shared_mesh_shape is None or self.shared_mesh_name is None:
            spyral_warn(
                __name__,
                f"Could not run the interpolation scheme as there is no shared memory!",
            )
            return PhaseResult.invalid_result(payload.run_number)
        # Ask for the shared memory by name and cast it to a numpy array of the correct shape
        mesh_buffer = SharedMemory(self.shared_mesh_name)
        mesh_handle = np.ndarray(
            self.shared_mesh_shape, dtype=self.shared_mesh_type, buffer=mesh_buffer.buf
        )
        mesh_handle.setflags(write=False)
        # Create our interpolator
        interpolator = create_interpolator_from_array(self.track_path, mesh_handle)

        # Process the data
        for row, event in enumerate(estimates_gated["event"]):
            count += 1
            if count > flush_val:
                count = 0
                msg_queue.put(msg)

            event_group = cluster_group[f"event_{event}"]
            cidx = estimates_gated["cluster_index"][row]
            local_cluster: h5.Dataset = event_group[f"cluster_{cidx}"]  # type: ignore
            cluster = Cluster(
                event, local_cluster.attrs["label"], local_cluster["cloud"][:].copy()  # type: ignore
            )

            # Do the solver
            guess = Guess(
                estimates_gated["polar"][row],
                estimates_gated["azimuthal"][row],
                estimates_gated["brho"][row],
                estimates_gated["vertex_x"][row],
                estimates_gated["vertex_y"][row],
                estimates_gated["vertex_z"][row],
                Direction.NONE,  # type: ignore
            )
            solve_physics_interp(
                payload.run_number,
                event,
                cidx,
                cluster,
                guess,
                pid.nucleus,
                interpolator,
                self.det_params,
                w_tb,
                mm_tb,
                w_err,
                mm_err,
                phys_results,
            )

        # Write out the results
        physics_df = pl.DataFrame(phys_results)
        physics_df.write_parquet(result.artifact_path)
        spyral_info(__name__, "Phase 4 complete.")
        return result


def solve_physics_interp(
    run: int,
    event: int,
    cluster_index: int,
    cluster: Cluster,
    guess: Guess,
    ejectile: NucleusData,
    interpolator: TrackInterpolator,
    det_params: DetectorParameters,
    w_tb: float,
    mm_tb: float,
    w_err: float,
    mm_err: float,
    results: dict[str, list],
):
    """High level function to be called from the application.

    Takes the Cluster and fits a trajectory to it using the initial Guess. It then writes the results to the dictionary.

    Parameters
    ----------
    cluster_index: int
        Index of the cluster in the hdf5 scheme. Used only for debugging
    cluster: Cluster
        the data to be fit
    guess: Guess
        the initial values of the parameters
    ejectile: spyral_utils.nuclear.NucleusData
        the data for the particle being tracked
    interpolator: TrackInterpolator
        the interpolation scheme to be used
    det_params: DetectorParameters
        Configuration parameters for detector characteristics
    w_tb: float
        Window time bucket
    mm_tb: float
        Micromegas time bucket
    w_err: float
        Window time bucket error
    mm_err: float
        Micromegas time bucket error
    results: dict[str, list]
        storage for results from the fitting, which will later be written as a dataframe.
    """
    traj_data = cluster.data[:, :3] * 0.001
    momentum = QBRHO_2_P * (guess.brho * float(ejectile.Z))
    kinetic_energy = np.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    if not interpolator.check_values_in_range(kinetic_energy, guess.polar):
        return

    # Uncertainty due to z-position reconstruction
    dv = det_params.detector_length / (w_tb - mm_tb) * 0.001
    tb_err = 0.5
    z_error = np.sqrt(
        (dv * tb_err) ** 2.0
        + (w_err * (-traj_data[:, 2] + dv * (w_tb - mm_tb)) / (w_tb - mm_tb)) ** 2.0
        + (mm_err * traj_data[:, 2] / (w_tb - mm_tb)) ** 2.0
    )
    # uncertainty due to pad size, treat as rectangle
    x_error = cluster.data[:, 4] * BIG_PAD_HEIGHT * 0.5
    y_error = cluster.data[:, 4] * BIG_PAD_HEIGHT / np.sqrt(3.0)
    # total positional variance per point
    total_error = np.sqrt(x_error**2.0 + y_error**2.0 + z_error**2.0)
    weights = 1.0 / total_error

    fit_params = create_params(guess, ejectile, interpolator, det_params)

    try:
        best_fit: MinimizerResult = minimize(
            objective_function,
            fit_params,
            args=(traj_data, weights, interpolator, ejectile),
            method="leastsq",
        )
    except ValueError:
        spyral_warn(
            "spyral." + __name__,
            f"Run {run} event {event} generated NaN's while fitting!",
        )
        return

    scale_factor = QBRHO_2_P * float(ejectile.Z)
    brho: float = best_fit.params["brho"].value  # type: ignore
    p = brho * scale_factor  # type: ignore
    ke = np.sqrt(p**2.0 + ejectile.mass**2.0) - ejectile.mass

    results["event"].append(cluster.event)
    results["cluster_index"].append(cluster_index)
    results["cluster_label"].append(cluster.label)
    # Best fit values and uncertainties
    results["vertex_x"].append(best_fit.params["vertex_x"].value)  # type: ignore
    results["vertex_y"].append(best_fit.params["vertex_y"].value)  # type: ignore
    results["vertex_z"].append(best_fit.params["vertex_z"].value)  # type: ignore
    results["brho"].append(best_fit.params["brho"].value)  # type: ignore
    results["ke"].append(ke)
    results["polar"].append(best_fit.params["polar"].value)  # type: ignore
    results["azimuthal"].append(best_fit.params["azimuthal"].value)  # type: ignore
    results["redchisq"].append(best_fit.redchi)

    if hasattr(best_fit, "uvars"):
        results["sigma_vx"].append(best_fit.uvars["vertex_x"].std_dev)  # type: ignore
        results["sigma_vy"].append(best_fit.uvars["vertex_y"].std_dev)  # type: ignore
        results["sigma_vz"].append(best_fit.uvars["vertex_z"].std_dev)  # type: ignore
        results["sigma_brho"].append(best_fit.uvars["brho"].std_dev)  # type: ignore

        # sigma_f = sqrt((df/dx)^2*sigma_x^2 + ...)
        ke_std_dev = np.fabs(
            scale_factor**2.0
            * brho
            / np.sqrt((brho * scale_factor) ** 2.0 + ejectile.mass**2.0)
            * best_fit.uvars["brho"].std_dev  # type: ignore
        )
        results["sigma_ke"].append(ke_std_dev)

        results["sigma_polar"].append(best_fit.uvars["polar"].std_dev)  # type: ignore
        results["sigma_azimuthal"].append(best_fit.uvars["azimuthal"].std_dev)  # type: ignore
    else:
        results["sigma_vx"].append(1.0e6)
        results["sigma_vy"].append(1.0e6)
        results["sigma_vz"].append(1.0e6)
        results["sigma_brho"].append(1.0e6)
        results["sigma_ke"].append(1.0e6)
        results["sigma_polar"].append(1.0e6)
        results["sigma_azimuthal"].append(1.0e6)


# For testing, not for use in production
def fit_model_interp(
    cluster: Cluster,
    guess: Guess,
    ejectile: NucleusData,
    interpolator: TrackInterpolator,
    det_params: DetectorParameters,
    w_tb: float,
    mm_tb: float,
    w_err: float,
    mm_err: float,
) -> Parameters | None:
    """Used for jupyter notebooks examining the good-ness of the model

    Parameters
    ----------
    cluster: Cluster
        the data to be fit
    guess: Guess
        the initial values of the parameters
    ejectile: spyral_utils.nuclear.NucleusData
        the data for the particle being tracked
    interpolator: TrackInterpolator
        the interpolation scheme to be used
    det_params: DetectorParameters
        Configuration parameters for detector characteristics
    w_tb: float
        Window time bucket
    mm_tb: float
        Micromegas time bucket
    w_err: float
        Window time bucket error
    mm_err: float
        Micromegas time bucket error

    Returns
    -------
    lmfit.Parameters | None
        Returns the best fit Parameters upon success, or None upon failure
    """
    traj_data = cluster.data[:, :3] * 0.001
    momentum = QBRHO_2_P * (guess.brho * float(ejectile.Z))
    kinetic_energy = np.sqrt(momentum**2.0 + ejectile.mass**2.0) - ejectile.mass
    if not interpolator.check_values_in_range(kinetic_energy, guess.polar):
        return None

    # Uncertainty due to z-position reconstruction
    dv = det_params.detector_length / (w_tb - mm_tb) * 0.001
    tb_err = 0.5
    z_error = np.sqrt(
        (dv * tb_err) ** 2.0
        + (w_err * (-traj_data[:, 2] + dv * (w_tb - mm_tb)) / (w_tb - mm_tb)) ** 2.0
        + (mm_err * traj_data[:, 2] / (w_tb - mm_tb)) ** 2.0
    )
    # uncertainty due to pad size, treat as rectangle
    x_error = cluster.data[:, 4] * BIG_PAD_HEIGHT * 0.5
    y_error = cluster.data[:, 4] * BIG_PAD_HEIGHT / np.sqrt(3.0)
    # total positional variance per point
    total_error = np.sqrt(x_error**2.0 + y_error**2.0 + z_error**2.0)
    weights = 1.0 / total_error

    fit_params = create_params(guess, ejectile, interpolator, det_params)

    result: MinimizerResult = minimize(
        objective_function,
        fit_params,
        args=(traj_data, weights, interpolator, ejectile),
        method="leastsq",
    )
    print(fit_report(result))

    return result.params  # type: ignore
