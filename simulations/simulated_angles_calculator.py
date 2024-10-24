import vector
import polars as pl
import h5py as h5
import numpy as np
from pathlib import Path
from attpc_engine import nuclear_map
from spyral_utils.nuclear import NucleusData

# Configuration parameters
kine_path = Path(
    "/Volumes/e20009_sim/engine_v0.3.0/elastic_0-180cm/elastic_kine.parquet"
)
write_path = Path("/Volumes/e20009_sim/engine_v0.3.0/elastic_0-180cm")

# Define reaction
target: NucleusData = nuclear_map.get_data(1, 2)
beam: NucleusData = nuclear_map.get_data(4, 10)
product: NucleusData = nuclear_map.get_data(4, 10)

# Specify analysis gates
vertex_z_min = 0.004  # Units of meters
vertex_z_max = 0.700  # Units of meters


def find_sim_cm(kine_path: Path):
    """
    Finds the center of mass scattering angle of the input product in all the simulated events.

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
    """

    # Load in all simulated events
    kine_f = pl.scan_parquet(kine_path)
    # kine_f = kine_f.filter(
    #     (vertex_z_min <= pl.col("vertex_z")) & (pl.col("vertex_z") <= vertex_z_max)
    # )

    target_vector = vector.array(
        {
            "px": [0.0],
            "py": [0.0],
            "pz": [0.0],
            "E": [target.mass],
        }
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

    cm_ang_det = product_vectors.boostCM_of(beam_vectors + target_vector).theta

    np.save(write_path / "cm_ang.npy", cm_ang_det)


if __name__ == "__main__":
    find_sim_cm(
        kine_path,
    )
