import vector
import polars as pl
import h5py as h5
import numpy as np
from pathlib import Path
from attpc_engine import nuclear_map

# Configuration parameters
kine_path = Path("/Volumes/e20009_sim/1.78_mev_0-60cm/1.78_mev_kine.parquet")
sim_directory = Path("/Volumes/e20009_sim/1.78_mev_0-60cm/workspace/PointcloudLegacy")
write_path = Path("/Volumes/e20009_sim/1.78_mev_0-60cm")

run_min = 1
run_max = 20

# Schema is [Z, A]
target = [1, 2]
beam = [4, 10]
product = [4, 11]

vertex_z_min = 0.004  # Units of meters
vertex_z_max = 0.958  # Units of meters


def find_sim_cm(
    kine_path: Path,
    sim_directory: Path,
    run_min: int,
    run_max: int,
    target: list,
    beam: list,
    product: list,
    vertex_z_min: float,
    vertex_z_max: float,
):
    """
    Finds the center of mass scattering angle of the input product in all the simulated events
    that survived the detector effects.

    Parameters
    ----------
    kine_path: Path
        Path to the simulated kinematics parquet made with the convert-kinematics that
        comes with attpc_engine.
    sim_directory: Path
        Path to the folder containing the point clouds made by Spyral from analyzing
        the simulated data.
    run_min: int
        Minimum run number of simulated data.
    run_max: int
        Maximum run number of simulated data.
    target: list
        List of [Z, A] of target nucleus.
    beam: list
        List of [Z, A] of beam nucleus.
    product: list
        List of [Z, A] of product nucleus whose center of mass scattering angle will
        be found.
    vertex_z_min: float
        Minimum vertex z-coordinate of points that survived detector effects.
    vertex_z_max: float
        Maximum vertex z-coordinate of points that survived detector effects.
    """
    # Find events that survived applying detector effects
    events = np.empty(0, int)
    for run in range(run_min, run_max + 1):
        cloud_f = None
        path = sim_directory / f"run_{run:04d}.h5"
        if not path.exists():
            continue
        cloud_f = h5.File(path, "r")

        events_run = np.array(list(cloud_f["cloud"]))
        events_run = np.char.strip(events_run, chars="cloud_").astype(int)
        events = np.append(events, events_run)

    target_vector = vector.array(
        {
            "px": [0.0],
            "py": [0.0],
            "pz": [0.0],
            "E": [nuclear_map.get_data(target[0], target[1]).mass],
        }
    )

    # Find events in kinematics file that survived applying detector effects
    kine_f = pl.scan_parquet(kine_path)
    kine_f = kine_f.filter(
        pl.col("event").is_in(events)
        & (pl.col("vertex_z") >= vertex_z_min)
        & (pl.col("vertex_z") <= vertex_z_max)
    )

    beam_coords = (
        kine_f.filter((pl.col("Z") == beam[0]) & (pl.col("A") == beam[1]))
        .select("px", "py", "pz", "energy")
        .collect()
        .to_numpy()
    )
    beam_vectors = vector.array(
        {
            "px": beam_coords[:, 0],
            "py": beam_coords[:, 1],
            "pz": beam_coords[:, 2],
            "E": beam_coords[:, 3],
        }
    )

    product_coords = (
        kine_f.filter((pl.col("Z") == product[0]) & (pl.col("A") == product[1]))
        .select("px", "py", "pz", "energy")
        .collect()
        .to_numpy()
    )
    product_vectors = vector.array(
        {
            "px": product_coords[:, 0],
            "py": product_coords[:, 1],
            "pz": product_coords[:, 2],
            "E": product_coords[:, 3],
        }
    )

    cm_ang_det = product_vectors.boostCM_of(beam_vectors + target_vector).theta

    np.save(write_path / "cm_ang.npy", cm_ang_det)


if __name__ == "__main__":
    find_sim_cm(
        kine_path,
        sim_directory,
        run_min,
        run_max,
        target,
        beam,
        product,
        vertex_z_min,
        vertex_z_max,
    )
