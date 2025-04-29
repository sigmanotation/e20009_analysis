from spyral.trace.get_legacy_event import GET_DATA_TRACE_START, GET_DATA_TRACE_STOP
from spyral.core.legacy_beam_pads import LEGACY_BEAM_PADS
from spyral.core.run_stacks import form_run_string

from pathlib import Path

import click
import scipy
import numpy as np
import polars as pl
import h5py as h5


@click.command()
@click.argument("workspace_path", type=click.Path(exists=True))
@click.argument("traces_path", type=click.Path(exists=True))
@click.argument("run_min", type=int)
@click.argument("run_max", type=int)
def main(workspace_path: Path, traces_path: Path, run_min: int, run_max: int):
    """
    Calculates the micromegas and window time buckets of beam events in a run to find its drift velocity using downscale beam events.
    The error of both these edges is found as well. The results are output in a csv file in the workspace.

    The PointcloudLegacyPhase must be run on the data before the drift velocity is calculated.

    Arguments
    ---------
    workspace_path: Path
        Path to workspace where attpc_spyral results are stored.
    traces_path: Path
        Path to where HDF5 files are stored.
    run_min: int
        Minimum run to calculate the drift velocity of.
    run_max: int
        Maximum run to calculate the drift velocity of.
    """
    # Make dictionary to store results
    results: dict[str, list] = {
        "run": [],
        "average_micromegas_tb": [],
        "average_micromegas_tb_error": [],
        "average_window_tb": [],
        "average_window_tb_error": [],
    }

    # Analyze each run
    for run in range(run_min, run_max + 1):
        df = None
        try:
            df: pl.DataFrame = pl.read_parquet(
                Path(workspace_path) / "beam_events" / f"{form_run_string(run)}.parquet"
            )
        except Exception:
            continue

        # Make lists of edges
        micromegas_edges: list = []
        window_edges: list = []

        # Apply gates to find desired beam events
        df: pl.DataFrame = df.filter(
            (pl.col("ic_multiplicity") == 1) & (pl.col("ic_sca_multiplicity") == 1)
        )
        df = df.filter(
            (
                abs(
                    pl.col("ic_centroid").list.get(0)
                    - pl.col("ic_sca_centroid").list.get(0)
                )
                <= 10
            )
            & (pl.col("ic_amplitude").list.get(0) > 700)
        )
        events: np.array = df.select(pl.col("event")).to_numpy().flatten()

        # Open files
        trace_file = h5.File(Path(traces_path) / f"{form_run_string(run)}.h5", "r")
        trace_group: h5.Group = trace_file["get"]

        # Analyze gated beam events
        for event in events:
            # Make reconstructed mesh signal
            event_data: h5.Dataset = trace_group[f"evt{event}_data"]
            mesh = [
                trace[GET_DATA_TRACE_START:GET_DATA_TRACE_STOP]
                for trace in event_data
                if trace[4] in LEGACY_BEAM_PADS
            ]
            mesh = np.sum(mesh, axis=0)

            # Perform a moving average smoothing via the convolution theorem
            window = np.arange(-256.0, 256.0, 1.0)
            fil = np.fft.ifftshift(
                np.sinc(window / 80)
            )  # Size of points taken for average is denominator
            transformed = np.fft.fft2(mesh, axes=(0,))
            mesh = np.real(np.fft.ifft2(transformed * fil, axes=(0,)))

            # Find edges of reconstructed mesh signal
            pks, props = scipy.signal.find_peaks(
                mesh, distance=400, prominence=300, width=(300, 400), rel_height=0.85
            )
            if pks.shape[0] == 1:
                micromegas_edges.append(int(np.floor(props["left_ips"][0])))
                window_edges.append(int(np.floor(props["right_ips"][0])))

        # Find average edges and append to results
        results["run"].append(run)
        results["average_micromegas_tb"].append(np.average(micromegas_edges))
        results["average_micromegas_tb_error"].append(scipy.stats.sem(micromegas_edges))
        results["average_window_tb"].append(np.average(window_edges))
        results["average_window_tb_error"].append(scipy.stats.sem(window_edges))

        print(f"Run {run} done.")

    # Write the results to a DataFrame
    results_df = pl.DataFrame(results)
    results_df.write_csv(Path(workspace_path) / "drift_velocity.csv")


if __name__ == "__main__":
    main()
