import vector
import polars as pl
import h5py as h5
import numpy as np
from pathlib import Path
from attpc_engine import nuclear_map
from spyral_utils.nuclear import NucleusData

# Configuration parameters
kine_path = Path("E:\\elastic_0-90cm\\elastic_kine.parquet")
sim_directory = Path("E:\\elastic_0-90cm\\workspace\\PointcloudLegacy")
write_path = Path("E:\\elastic_0-90cm")

run_min = 1
run_max = 21

# Define reaction
target: NucleusData = nuclear_map.get_data(1, 2)
beam: NucleusData = nuclear_map.get_data(4, 10)
product: NucleusData = nuclear_map.get_data(4, 10)

# Specify analysis gates
vertex_z_min = 0.004  # Units of meters
vertex_z_max = 0.958  # Units of meters
product_mass_low = -1.0 + product.mass  # Units of MeV
product_mass_high = 1.0 + product.mass  # Units of MeV


def find_sim_cm(
    kine_path: Path,
    sim_directory: Path,
    run_min: int,
    run_max: int,
    target: NucleusData,
    beam: NucleusData,
    product: NucleusData,
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
            print(f"{run} does not exist in the input directory!")
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
            "E": [target.mass],
        }
    )

    # Find events in kinematics file that survived applying detector effects and vertex z-coordinate gate
    kine_f = pl.scan_parquet(kine_path)
    kine_f = kine_f.filter(
        pl.col("event").is_in(events)
        & (pl.col("vertex_z") >= vertex_z_min)
        & (pl.col("vertex_z") <= vertex_z_max)
    )

    # beam_coords = (
    #     kine_f.filter((pl.col("Z") == beam.Z) & (pl.col("A") == beam.A))
    #     .select("px", "py", "pz", "energy")
    #     .collect()
    #     .to_numpy()
    # )

    # For cases where the beam and product are the same
    beam_coords = (
        kine_f.gather_every(4, offset=1)
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

    # product_coords = (
    #     kine_f.filter((pl.col("Z") == product.Z) & (pl.col("A") == product.A))
    #     .select("px", "py", "pz", "energy")
    #     .collect()
    #     .to_numpy()
    # )

    # For cases where the beam and product are the same
    product_coords = (
        kine_f.gather_every(4, offset=3)
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

    # Apply analysis excitation energy gate
    mask: np.ndarray = (product_mass_low <= product_vectors.mass) & (
        product_vectors.mass < product_mass_high
    )

    cm_ang_det = (
        product_vectors[mask].boostCM_of(beam_vectors[mask] + target_vector).theta
    )

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
