from spyral.core.phase import PhaseLike, PhaseResult
from spyral.core.config import EstimateParameters
from spyral.core.status_message import StatusMessage
from spyral.core.cluster import Cluster
from spyral.core.estimator import Direction, choose_direction
from spyral.core.spy_log import spyral_warn, spyral_error, spyral_info
from spyral.core.run_stacks import form_run_string
from spyral.phases.schema import CLUSTER_SCHEMA, ESTIMATE_SCHEMA

# Import e20009 specific data classes
from e20009_phases.config import DetectorParameters

import h5py as h5
import polars as pl
from pathlib import Path
from multiprocessing import SimpleQueue
from numpy.random import Generator


from spyral.geometry.circle import generate_circle_points, least_squares_circle

import numpy as np
import math
from scipy.stats import linregress

"""
Changes from attpc_spyral package base code (circa July 29, 2024):
    - Estimation results now includes IC SCA information written to the output parquet file. 
      Including the IC SCA information means that we have to now add these parameters as
      inputs to all the functions downstream of it.
    - estimatephysics function now takes IC SCA information.
    - estimate_physics_pass function now takes IC SCA information.
"""


class EstimationPhase(PhaseLike):
    """The default Spyral estimation phase, inheriting from PhaseLike

    The goal of the estimation phase is to get reasonable estimations of
    the physical properties of a particle trajectory (B&rho; , reaction angle, etc.)
    for use in the more complex solving phase to follow. EstimationPhase should come
    after ClusterPhase and before InterpSolverPhase in the Pipeline.

    Parameters
    ----------
    estimate_params: EstimateParameters
        Parameters controlling the estimation algorithm
    det_params: DetectorParameters
        Parameters describing the detector

    Attributes
    ----------
    estimate_params: EstimateParameters
        Parameters controlling the estimation algorithm
    det_params: DetectorParameters
        Parameters describing the detector

    """

    def __init__(
        self, estimate_params: EstimateParameters, det_params: DetectorParameters
    ):
        super().__init__(
            "Estimation",
            incoming_schema=None,
            outgoing_schema=None,
        )
        self.estimate_params = estimate_params
        self.det_params = det_params

    def create_assets(self, workspace_path: Path) -> bool:
        return True

    def construct_artifact(
        self, payload: PhaseResult, workspace_path: Path
    ) -> PhaseResult:
        result = PhaseResult(
            artifact_path=self.get_artifact_path(workspace_path)
            / f"{form_run_string(payload.run_number)}.parquet",
            successful=True,
            run_number=payload.run_number,
            metadata={"cluster_path": payload.artifact_path},
        )
        return result

    def run(
        self,
        payload: PhaseResult,
        workspace_path: Path,
        msg_queue: SimpleQueue,
        rng: Generator,
    ) -> PhaseResult:
        # Check that clusters exist
        cluster_path = payload.artifact_path
        if not cluster_path.exists() or not payload.successful:
            spyral_warn(
                __name__,
                f"Cluster file for run {payload.run_number} not present for phase 3. Skipping.",
            )
            return PhaseResult.invalid_result(payload.run_number)

        result = self.construct_artifact(payload, workspace_path)

        cluster_file = h5.File(cluster_path, "r")
        cluster_group: h5.Group = cluster_file["cluster"]  # type: ignore
        if not isinstance(cluster_group, h5.Group):
            spyral_error(
                __name__, f"Cluster group not present for run {payload.run_number}!"
            )
            return PhaseResult.invalid_result(payload.run_number)

        min_event: int = cluster_group.attrs["min_event"]  # type: ignore
        max_event: int = cluster_group.attrs["max_event"]  # type: ignore

        nevents = max_event - min_event + 1
        total: int
        flush_val: int
        if nevents < 100:
            total = nevents
            flush_val = 0
        else:
            flush_percent = 0.01
            flush_val = int(flush_percent * nevents)
            total = 100

        count = 0

        # estimation results
        data: dict[str, list] = {
            "event": [],
            "cluster_index": [],
            "cluster_label": [],
            "ic_amplitude": [],
            "ic_centroid": [],
            "ic_integral": [],
            "ic_multiplicity": [],
            "ic_sca_centroid": [],
            "ic_sca_multiplicity": [],
            "vertex_x": [],
            "vertex_y": [],
            "vertex_z": [],
            "center_x": [],
            "center_y": [],
            "center_z": [],
            "polar": [],
            "azimuthal": [],
            "brho": [],
            "dEdx": [],
            "sqrt_dEdx": [],
            "dE": [],
            "arclength": [],
            "direction": [],
        }

        msg = StatusMessage(
            self.name, 1, total, payload.run_number
        )  # We always increment by 1
        # Process data
        for idx in range(min_event, max_event + 1):
            count += 1
            if count > flush_val:
                count = 0
                msg_queue.put(msg)

            event: h5.Group | None = None
            event_name = f"event_{idx}"
            if event_name not in cluster_group:
                continue
            else:
                event = cluster_group[event_name]  # type: ignore

            nclusters: int = event.attrs["nclusters"]  # type: ignore
            ic_amp = float(event.attrs["ic_amplitude"])  # type: ignore
            ic_cent = float(event.attrs["ic_centroid"])  # type: ignore
            ic_int = float(event.attrs["ic_integral"])  # type: ignore
            ic_mult = float(event.attrs["ic_multiplicity"])  # type: ignore
            ic_sca_cent: float = float(event.attrs["ic_sca_centroid"])  # type: ignore
            ic_sca_mult: float = float(event.attrs["ic_sca_multiplicity"])  # type: ignore
            # Go through every cluster in each event
            for cidx in range(0, nclusters):
                local_cluster: h5.Group | None = None
                cluster_name = f"cluster_{cidx}"
                if cluster_name not in event:  # type: ignore
                    continue
                else:
                    local_cluster = event[cluster_name]  # type: ignore

                cluster = Cluster(
                    idx, local_cluster.attrs["label"], local_cluster["cloud"][:].copy()  # type: ignore
                )

                # Cluster is loaded do some analysis
                estimate_physics(
                    cidx,
                    cluster,
                    ic_amp,
                    ic_cent,
                    ic_int,
                    ic_mult,
                    ic_sca_cent,
                    ic_sca_mult,
                    self.estimate_params,
                    self.det_params,
                    data,
                )

        # Write the results to a DataFrame
        df = pl.DataFrame(data)
        df.write_parquet(result.artifact_path)
        spyral_info(__name__, "Phase 3 complete.")
        # Next step also needs to know where to find the clusters
        return result


def estimate_physics(
    cluster_index: int,
    cluster: Cluster,
    ic_amplitude: float,
    ic_centroid: float,
    ic_integral: float,
    ic_multiplicity: float,
    ic_sca_centroid: float,
    ic_sca_multiplicity: float,
    estimate_params: EstimateParameters,
    detector_params: DetectorParameters,
    results: dict[str, list],
):
    """Entry point for estimation

    This is the parent function for estimation. It handles checking that the data
    meets the conditions to be estimated, applying splines to data, and
    esuring that the estimation results pass a sanity check.

    Parameters
    ----------
    cluster_index: int
        The cluster index in the HDF5 file.
    cluster: Cluster
        The cluster to estimate
    ic_amplitude: float
        The ion chamber amplitude for this cluster
    ic_centroid: float
        The ion chamber centroid for this cluster
    ic_integral: float
        The ion chamber integral for this cluster
    ic_multiplicity: float
        Number of peaks in IC trace for this cluster
    ic_sca_centroid: float
        Centroid of peak in IC SCA trace for this cluster
    ic_sca_multiplicity: float
        Number of peaks in IC SCA trace for this cluster
    detector_params:
        Configuration parameters for the physical detector properties
    results: dict[str, int]
        Dictionary to store estimation results in

    """
    # Check if we have enough points to estimate
    if len(cluster.data) < estimate_params.min_total_trajectory_points:
        return
    # Generate smoothing splines, these will give us better distance measures
    try:
        cluster.apply_smoothing_splines(estimate_params.smoothing_factor)
    except Exception as e:
        # Spline creation can fail for two main reasons:
        # - Not enough points in the data (need at least 5)
        # - The data is inherentily multivalued in z (a spark event where the pad plane lights up at one instance in time)
        # We do not analyze un-splineable events. But we do report a warning in the log file that these events failed
        spyral_warn(
            __name__,
            f"Spline creation failed for event {cluster.event} with error: {e}",
        )
        return

    # Run estimation where we attempt to guess the right direction
    is_good, direction = estimate_physics_pass(
        cluster_index,
        cluster,
        ic_amplitude,
        ic_centroid,
        ic_integral,
        ic_multiplicity,
        ic_sca_centroid,
        ic_sca_multiplicity,
        detector_params,
        results,
    )

    # If estimation was consistent or didn't meet valid criteria we're done
    if is_good or (not is_good and direction == Direction.NONE):
        return
    # If we made a bad guess, try the other direction
    elif direction == Direction.FORWARD:
        estimate_physics_pass(
            cluster_index,
            cluster,
            ic_amplitude,
            ic_centroid,
            ic_integral,
            ic_multiplicity,
            ic_sca_centroid,
            ic_sca_multiplicity,
            detector_params,
            results,
            Direction.BACKWARD,
        )
    else:
        estimate_physics_pass(
            cluster_index,
            cluster,
            ic_amplitude,
            ic_centroid,
            ic_integral,
            ic_multiplicity,
            ic_sca_centroid,
            ic_sca_multiplicity,
            detector_params,
            results,
            Direction.FORWARD,
        )


def estimate_physics_pass(
    cluster_index: int,
    cluster: Cluster,
    ic_amplitude: float,
    ic_centroid: float,
    ic_integral: float,
    ic_multiplicity: float,
    ic_sca_centroid: float,
    ic_sca_multiplicity: float,
    detector_params: DetectorParameters,
    results: dict[str, list],
    chosen_direction: Direction = Direction.NONE,
) -> tuple[bool, Direction]:
    """Estimate the physics parameters for a cluster which could represent a particle trajectory

    Estimation is an imprecise process (by definition), and as such this algorithm requires a lot of
    manipulation of the data.

    Parameters
    ----------
    cluster_index: int
        The cluster index in the HDF5 file.
    cluster: Cluster
        The cluster to estimate
    ic_amplitude: float
        The ion chamber amplitude for this cluster
    ic_centroid: float
        The ion chamber centroid for this cluster
    ic_integral: float
        The ion chamber integral for this cluster
    ic_multiplicity: float
        Number of peaks in IC trace for this cluster
    ic_sca_centroid: float
        Centroid of peak in IC SCA trace for this cluster
    ic_sca_multiplicity: float
        Number of peaks in IC SCA trace for this cluster
    detector_params: float
        Configuration parameters for the physical detector properties
    results: dict[str, int]
        Dictionary to store estimation results in
    chosen_direction: Direction, default=Direction.NONE
        Optional direction for the trajectory. Default
        estimates the direction.

    """

    direction = chosen_direction
    vertex = np.array([0.0, 0.0, 0.0])  # reaction vertex
    center = np.array([0.0, 0.0, 0.0])  # spiral center
    # copy the data so we can modify it without worrying about side-effects
    cluster_data = cluster.data.copy()

    # If chosen direction is set to NONE, we want to have the algorithm
    # try to decide which direction the trajectory is going
    if direction == Direction.NONE:
        direction = choose_direction(cluster_data)

    if direction == Direction.BACKWARD:
        cluster_data = np.flip(cluster_data, axis=0)

    # Guess that the vertex is the first point; make sure to copy! not reference
    vertex[:] = cluster_data[0, :3]

    # Find the first point that is furthest from the vertex in rho (maximum) to get the first arc of the trajectory
    rho_to_vertex = np.linalg.norm(cluster_data[1:, :2] - vertex[:2], axis=1)
    maximum = np.argmax(rho_to_vertex)
    first_arc = cluster_data[: (maximum + 1)]

    # Fit a circle to the first arc and extract some physics
    center[0], center[1], radius, _ = least_squares_circle(
        first_arc[:, 0], first_arc[:, 1]
    )
    # radius = np.linalg.norm(cluster_data[0, :2] - center[:2])
    circle = generate_circle_points(center[0], center[1], radius)
    # Re-estimate vertex using the fitted circle. Extrapolate back to point closest to beam axis
    vertex_estimate_index = np.argsort(np.linalg.norm(circle, axis=1))[0]
    vertex[:2] = circle[vertex_estimate_index]
    # Re-calculate distance to vertex, maximum, first arc
    rho_to_vertex = np.linalg.norm((cluster_data[:, :2] - vertex[:2]), axis=1)
    maximum = np.argmax(rho_to_vertex)
    first_arc = cluster_data[: (maximum + 1)]

    # Do a linear fit to small segment of trajectory to extract rho vs. z and extrapolate vertex z
    test_index = max(10, int(maximum * 0.5))
    # test_index = 10
    fit = linregress(cluster_data[:test_index, 2], rho_to_vertex[:test_index])
    vertex_rho = np.linalg.norm(vertex[:2])
    # Since we fit to rho_to_vertex, just find intercept point
    if fit.slope == 0.0:
        return (False, Direction.NONE)
    vertex[2] = -1.0 * fit.intercept / fit.slope  # type: ignore
    center[2] = vertex[2]

    # Toss tracks whose verticies are not close to the origin in x,y
    if vertex_rho > detector_params.beam_region_radius:
        return (False, Direction.NONE)

    polar = math.atan(fit.slope)  # type: ignore
    # We have a self consistency case here. Polar should match chosen Direction
    if (polar > 0.0 and direction == Direction.BACKWARD) or (
        polar < 0.0 and direction == Direction.FORWARD
    ):
        return (
            False,
            direction,
        )  # Our direction guess was bad, we need to try again with the other direction
    elif direction is Direction.BACKWARD:
        polar += math.pi

    # From the trigonometry of the system to the center
    azimuthal = math.atan2(vertex[1] - center[1], vertex[0] - center[0])
    if azimuthal < 0:
        azimuthal += 2.0 * math.pi
    azimuthal += math.pi * 0.5
    if azimuthal > math.pi * 2.0:
        azimuthal -= 2.0 * math.pi

    brho = (
        detector_params.magnetic_field * radius * 0.001 / np.abs(math.sin(polar))
    )  # Sometimes our angle is in the wrong quadrant
    if np.isnan(brho):
        brho = 0.0

    # arclength = 0.0
    charge_deposited = first_arc[0, 3]
    small_pad_cutoff = -1  # Where do we cross from big pads to small pads
    for idx in range(len(first_arc) - 1):
        # Stop integrating if we leave the small pad region
        if np.linalg.norm(first_arc[idx + 1, :2]) > 152.0:
            small_pad_cutoff = idx + 1
            break
        # arclength += np.linalg.norm(first_arc[idx + 1, :3] - first_arc[idx, :3])
        charge_deposited += first_arc[idx + 1, 3]
    if charge_deposited == first_arc[0, 3]:
        return (False, Direction.NONE)

    # Use the splines to do a fine-grained line integral to calculate the distance
    points = np.empty((1000, 3))
    points[:, 2] = np.linspace(first_arc[0, 2], first_arc[small_pad_cutoff, 2], 1000)
    points[:, 0] = cluster.x_spline(points[:, 2])  # type: ignore
    points[:, 1] = cluster.y_spline(points[:, 2])  # type: ignore
    arclength = np.sqrt((np.diff(points, axis=0) ** 2.0).sum(axis=1)).sum()  # integrate

    dEdx = charge_deposited / arclength

    # fill in our map
    results["event"].append(cluster.event)
    results["cluster_index"].append(cluster_index)
    results["cluster_label"].append(cluster.label)
    results["ic_amplitude"].append(ic_amplitude)
    results["ic_centroid"].append(ic_centroid)
    results["ic_integral"].append(ic_integral)
    results["ic_multiplicity"].append(ic_multiplicity)
    results["ic_sca_centroid"].append(ic_sca_centroid)
    results["ic_sca_multiplicity"].append(ic_sca_multiplicity)
    results["vertex_x"].append(vertex[0])
    results["vertex_y"].append(vertex[1])
    results["vertex_z"].append(vertex[2])
    results["center_x"].append(center[0])
    results["center_y"].append(center[1])
    results["center_z"].append(center[2])
    results["polar"].append(polar)
    results["azimuthal"].append(azimuthal)
    results["brho"].append(brho)
    results["dEdx"].append(dEdx)
    results["sqrt_dEdx"].append(np.sqrt(np.fabs(dEdx)))
    results["dE"].append(charge_deposited)
    results["arclength"].append(arclength)
    results["direction"].append(direction.value)
    return (True, direction)
