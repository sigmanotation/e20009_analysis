from spyral.core.phase import PhaseLike, PhaseResult
from spyral.core.status_message import StatusMessage
from spyral.core.cluster import Cluster
from spyral.core.spy_log import spyral_warn, spyral_error, spyral_info
from spyral.core.track_generator import (
    generate_track_mesh,
    MeshParameters,
    check_mesh_needs_generation,
)
from spyral.core.estimator import Direction
from spyral.interpolate.track_interpolator import create_interpolator
from spyral.solvers.guess import Guess
from spyral.solvers.solver_interp import solve_physics_interp
from spyral.phases.schema import ESTIMATE_SCHEMA, INTERP_SOLVER_SCHEMA
from spyral.phases.interp_solver_phase import form_physics_file_name

# Import e20009 specific data classes
from e20009_phases.config import SolverParameters, DetectorParameters

from spyral_utils.nuclear.target import load_target, GasTarget
from spyral_utils.nuclear.particle_id import deserialize_particle_id, ParticleID
from spyral_utils.nuclear.nuclear_map import NuclearDataMap
from spyral_utils.plot import Cut2D
import h5py as h5
import polars as pl
from pathlib import Path
from multiprocessing import SimpleQueue
from numpy.random import Generator

"""
Changes from attpc_spyral package base code (circa June 1, 2024):
    - InterpSolverPhase run method pulls gain-match factor for the run being analyzed from the specified file 
      and applies it. The estimates_gated dataframe now has additinoal gates to only select events with the 
      correct IC and IC SCA information. StatusMessage now takes self.name as first argument instead of "Interp. Solver".
"""


class InterpSolverPhase(PhaseLike):
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
            "InterpSolver",
            incoming_schema=ESTIMATE_SCHEMA,
            outgoing_schema=INTERP_SOLVER_SCHEMA,
        )
        self.solver_params = solver_params
        self.det_params = det_params
        self.nuclear_map = NuclearDataMap()
        self.track_path = Path("Invalid")

    def create_assets(self, workspace_path: Path) -> bool:
        target = load_target(Path(self.solver_params.gas_data_path), self.nuclear_map)
        pid = deserialize_particle_id(
            Path(self.solver_params.particle_id_filename), self.nuclear_map
        )
        if pid is None:
            print(
                "Could not create trajectory mesh, particle ID does not have the correct format!"
            )
            print("Particle ID is required for running the solver stage (phase 4).")
            raise Exception
        if not isinstance(target, GasTarget):
            print(
                "Could not create trajectory mesh, target data does not have the correct format for a GasTarget!"
            )
            print("Gas Target is required for running the solver stage (phase 4).")
            raise Exception
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
            print("Creating the trajectory mesh... This may take some time...")
            generate_track_mesh(mesh_params, self.track_path, meta_path)
            print("Done.")
        return True

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
        # Need particle ID and target to select the correct data subset/interpolation scheme
        pid: ParticleID | None = deserialize_particle_id(
            Path(self.solver_params.particle_id_filename), self.nuclear_map
        )
        if pid is None:
            msg_queue.put(StatusMessage("Waiting", 0, 0, payload.run_number))
            spyral_warn(
                __name__,
                f"Particle ID {self.solver_params.particle_id_filename} does not exist, Solver will not run!",
            )
            return PhaseResult(Path("null"), False, payload.run_number)
        target = load_target(Path(self.solver_params.gas_data_path), self.nuclear_map)
        if not isinstance(target, GasTarget):
            msg_queue.put(StatusMessage("Waiting", 0, 0, payload.run_number))
            spyral_warn(
                __name__,
                f"Target {self.solver_params.gas_data_path} is not of the correct format, Solver will not run!",
            )
            return PhaseResult(Path("null"), False, payload.run_number)

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
        pid.cut = Cut2D(pid.cut.name, pid_vertices_matched)

        # Check the cluster phase and estimate phase data
        cluster_path: Path = payload.metadata["cluster_path"]
        estimate_path = payload.artifact_path
        if not cluster_path.exists() or not estimate_path.exists():
            msg_queue.put(StatusMessage("Waiting", 0, 0, payload.run_number))
            spyral_warn(
                __name__,
                f"Either clusters or esitmates do not exist for run {payload.run_number} at phase 4. Skipping.",
            )
            return PhaseResult(Path("null"), False, payload.run_number)

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
            return PhaseResult(Path("null"), False, payload.run_number)

        # Select the particle group data, beam region of ic, convert to dictionary for row-wise operations
        estimates_gated = (
            estimate_df.filter(
                pl.struct(["dEdx", "brho"]).map_batches(pid.cut.is_cols_inside)
                & (pl.col("ic_amplitude") > self.solver_params.ic_min_val)
                & (pl.col("ic_amplitude") < self.solver_params.ic_max_val)
                & (pl.col("ic_multiplicity") == 1.0)
                & (pl.col("ic_sca_multiplicity") == 1.0)
                & (abs(pl.col("ic_sca_multiplicity") - pl.col("ic_multiplicity")) <= 10)
            )
            .collect()
            .to_dict()
        )

        # Check that data actually exists for given PID
        if len(estimates_gated["event"]) == 0:
            msg_queue.put(StatusMessage("Waiting", 0, 0, payload.run_number))
            spyral_warn(__name__, f"No events within PID for run {payload.run_number}!")
            return PhaseResult(Path("null"), False, payload.run_number)

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
            "polar": [],
            "sigma_polar": [],
            "azimuthal": [],
            "sigma_azimuthal": [],
            "redchisq": [],
        }

        # load the ODE solution interpolator
        interpolator = create_interpolator(self.track_path)

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
                Direction.NONE,
            )
            solve_physics_interp(
                cidx,
                cluster,
                guess,
                pid.nucleus,
                interpolator,
                self.det_params,
                phys_results,
            )

        # Write out the results
        physics_df = pl.DataFrame(phys_results)
        physics_df.write_parquet(result.artifact_path)
        spyral_info(__name__, "Phase 4 complete.")
        return result
