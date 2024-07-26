import vector
import polars as pl
import h5py as h5
import numpy as np
from pathlib import Path
from attpc_engine import nuclear_map

# Configuration parameters
kine_path = Path("/Volumes/e20009_sim/3.4_breit_wig_0-60/3.4_mev_kine.parquet")
sim_directory = Path(
    "/Volumes/e20009_sim/3.4_breit_wig_0-60/workspace/PointcloudLegacy"
)
write_path = Path("/Users/attpc/Desktop/")

run_min = 1
run_max = 16

# Schema is [Z, A]
target = [1, 2]
beam = [4, 10]
product = [4, 11]


def find_sim_cm(kine_path, sim_directory, run_min, run_max):
    """ """
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
    kine_f = kine_f.filter(pl.col("event").is_in(events))

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

    ejectile_coords = (
        kine_f.filter((pl.col("Z") == product[0]) & (pl.col("A") == product[1]))
        .select("px", "py", "pz", "energy")
        .collect()
        .to_numpy()
    )
    product_vectors = vector.array(
        {
            "px": ejectile_coords[:, 0],
            "py": ejectile_coords[:, 1],
            "pz": ejectile_coords[:, 2],
            "E": ejectile_coords[:, 3],
        }
    )

    cm_ang_det = product_vectors.boostCM_of(beam_vectors + target_vector).theta

    np.save(write_path / "cm_ang.npy", cm_ang_det)


if __name__ == "__main__":
    find_sim_cm(kine_path, sim_directory, run_min, run_max)
