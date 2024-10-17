from spyral.core.phase import PhaseLike, PhaseResult
from spyral.core.run_stacks import form_run_string
from spyral.core.status_message import StatusMessage
from spyral.core.config import GetParameters, PadParameters
from spyral.correction import (
    generate_electron_correction,
    create_electron_corrector,
    ElectronCorrector,
)
from spyral.core.spy_log import spyral_warn, spyral_error, spyral_info
from spyral.core.constants import (
    INVALID_EVENT_NAME,
    INVALID_EVENT_NUMBER,
    INVALID_PAD_ID,
    NUMBER_OF_TIME_BUCKETS,
)
from spyral.core.hardware_id import HardwareID
from spyral.core.hardware_id import hardware_id_from_array
from spyral.core.pad_map import PadMap
from spyral.trace.get_legacy_event import (
    preprocess_traces,
    GET_DATA_TRACE_START,
    GET_DATA_TRACE_STOP,
)
from spyral.trace.get_event import GetEvent
from spyral.trace.get_trace import GetTraceError
from spyral.trace.peak import Peak
from spyral.interpolate import BilinearInterpolator
from spyral.phases.schema import TRACE_SCHEMA, POINTCLOUD_SCHEMA
from spyral.phases.pointcloud_legacy_phase import get_event_range
from spyral.correction.generate import interpolate_initial_rho

# Import e20009 specific data classes
from e20009_phases.config import ICParameters, DetectorParameters

import h5py as h5
import numpy as np
import polars as pl

from scipy import signal
from pathlib import Path
from multiprocessing import SimpleQueue
from contourpy import contour_generator

"""
Changes from attpc_spyral package base code (circa July 29, 2024):
    - PointcloudLegacyPhase takes modified config classes found in the config.py file. These include
      the ICParameters class (called FRIBParameters in the attpc_spyral package) and modified 
      DetectorParameters. See that file for more information on their attributes.
    - generate_electron_correction uses the average window and micromegas time buckets calculated from 
      all runs in the drift velocity file.
    - PointcloudLegacyPhase run method now takes the window and micromegas time buckets to calculate
      the drift velocity from the indicated file in DetectorParameters. It adds IC SCA information as
      attributes to the output HDF5 file. It also writes information related to the IC and IC SCA traces
      to a parquet file in the workspace.
    - GetLegacyEvent load_traces method extracts IC SCA and downscale beam traces.
    - GetTrace find_peaks method takes a new parameter called min_width. The floor is now taken for both 
      inflection points.
    - GetTrace has a new method called remove_peaks to remove invalid IC and IC SCA peaks. It is used in
      the PointcloudLegacyPhase run method.
    - PointCloud does not accept traces that have 10 or more peaks (points).
    - PointCloud calibrate_z_position method does not take ic_correction as a parameter.
"""


class PointcloudLegacyPhase(PhaseLike):
    """The legacy point cloud phase, inheriting from PhaseLike

    The goal of the legacy point cloud phase is to convert legacy (pre-FRIBDAQ) AT-TPC
    trace data into point clouds. It uses a combination of Fourier transform baseline
    removal and scipy.signal.find_peaks to extract signals from the traces. PointcloudLegacyPhase
    is expected to be the first phase in the Pipeline.

    Parameters
    ----------
    get_params: GetParameters
        Parameters controlling the GET-DAQ signal analysis
    ic_params: ICParameters
        Parameters related to the IC and CoBo 10
    detector_params: DetectorParameters
        Parameters describing the detector
    pad_params: PadParameters
        Parameters describing the pad plane mapping

    Attributes
    ----------
    get_params: GetParameters
        Parameters controlling the GET-DAQ signal analysis
    ic_params: ICParameters
        Parameters related to the IC and CoBo 10
    det_params: DetectorParameters
        Parameters describing the detector
    pad_map: PadMap
        Map which converts trace ID to pad ID

    """

    def __init__(
        self,
        get_params: GetParameters,
        ic_params: ICParameters,
        detector_params: DetectorParameters,
        pad_params: PadParameters,
    ):
        super().__init__(
            "PointcloudLegacy",
            incoming_schema=None,
            outgoing_schema=None,
        )
        self.get_params = get_params
        self.ic_params = ic_params
        self.det_params = detector_params
        self.pad_map = PadMap(pad_params)

    def create_assets(self, workspace_path: Path) -> bool:
        asset_path = self.get_asset_storage_path(workspace_path)
        garf_path = Path(self.det_params.garfield_file_path)
        self.electron_correction_path = asset_path / f"{garf_path.stem}.npy"

        if (
            not self.electron_correction_path.exists()
            and self.det_params.do_garfield_correction
        ):
            generate_electron_correction(
                garf_path,
                self.electron_correction_path,
                self.det_params,
            )
        return True

    def construct_artifact(
        self, payload: PhaseResult, workspace_path: Path
    ) -> PhaseResult:
        result = PhaseResult(
            artifact_path=self.get_artifact_path(workspace_path)
            / f"{form_run_string(payload.run_number)}.h5",
            successful=True,
            run_number=payload.run_number,
        )
        return result

    def run(
        self,
        payload: PhaseResult,
        workspace_path: Path,
        msg_queue: SimpleQueue,
        rng: np.random.Generator,
    ) -> PhaseResult:
        trace_path = payload.artifact_path
        if not trace_path.exists():
            spyral_warn(
                __name__,
                f"Run {payload.run_number} does not exist for phase 1, skipping.",
            )
            return PhaseResult.invalid_result(payload.run_number)

        # Open files
        result = self.construct_artifact(payload, workspace_path)
        trace_file = h5.File(trace_path, "r")
        point_file = h5.File(result.artifact_path, "w")

        min_event, max_event = get_event_range(trace_file)

        # Load electric field correction
        corrector: ElectronCorrector | None = None
        if self.det_params.do_garfield_correction:
            corrector = create_electron_corrector(self.electron_correction_path)

        # Some checks for existance
        event_group = trace_file["get"]
        if not isinstance(event_group, h5.Group):
            spyral_error(
                __name__,
                f"GET event group does not exist in run {payload.run_number}, phase 1 cannot be run!",
            )
            return PhaseResult.invalid_result(payload.run_number)

        # Load drift velocity information
        dv_lf: pl.LazyFrame = pl.scan_csv(self.det_params.drift_velocity_path)
        dv_df: pl.DataFrame = dv_lf.filter(
            pl.col("run") == payload.run_number
        ).collect()
        if dv_df.shape[0] > 1:
            spyral_error(
                __name__,
                f"Multiple drift velocities found for run {payload.run_number}, phase 1 cannot be run!",
            )
            return PhaseResult.invalid_result(payload.run_number)
        elif dv_df.shape[0] == 0:
            spyral_error(
                __name__,
                f"No drift velocity found for run {payload.run_number}, phase 1 cannot be run!",
            )
            return PhaseResult.invalid_result(payload.run_number)
        mm_tb: float = dv_df.get_column("average_micromegas_tb")[0]
        w_tb: float = dv_df.get_column("average_window_tb")[0]

        # Beam event results
        beam_events: dict[str, list] = {
            "event": [],
            "ic_amplitude": [],
            "ic_centroid": [],
            "ic_multiplicity": [],
            "ic_sca_centroid": [],
            "ic_sca_multiplicity": [],
        }

        cloud_group = point_file.create_group("cloud")
        cloud_group.attrs["min_event"] = min_event
        cloud_group.attrs["max_event"] = max_event

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

        msg = StatusMessage(
            self.name, 1, total, payload.run_number
        )  # We always increment by 1

        # Process the data
        for idx in range(min_event, max_event + 1):
            count += 1
            if count > flush_val:
                count = 0
                msg_queue.put(msg)

            event_data: h5.Dataset
            event_name = f"evt{idx}_data"
            if event_name not in event_group:
                continue
            else:
                event_data = event_group[event_name]  # type: ignore

            event = GetLegacyEvent(
                event_data, idx, self.get_params, self.ic_params, rng
            )

            pc = PointCloud()
            pc.load_cloud_from_get_event(event, self.pad_map)
            pc.calibrate_z_position(
                mm_tb,
                w_tb,
                self.det_params.detector_length,
                corrector,
            )

            pc_dataset = cloud_group.create_dataset(
                f"cloud_{pc.event_number}", shape=pc.cloud.shape, dtype=np.float64
            )

            # default IC settings
            pc_dataset.attrs["ic_amplitude"] = -1.0
            pc_dataset.attrs["ic_integral"] = -1.0
            pc_dataset.attrs["ic_centroid"] = -1.0
            pc_dataset.attrs["ic_multiplicity"] = -1.0

            # default IC SCA settings
            pc_dataset.attrs["ic_sca_centroid"] = -1.0
            pc_dataset.attrs["ic_sca_multiplicity"] = -1.0

            # Set IC if present; take first non-garbage peak
            if event.ic_trace is not None:
                # No way to disentangle multiplicity
                for peak in event.ic_trace.get_peaks():
                    pc_dataset.attrs["ic_amplitude"] = peak.amplitude
                    pc_dataset.attrs["ic_integral"] = peak.integral
                    pc_dataset.attrs["ic_centroid"] = peak.centroid
                    pc_dataset.attrs["ic_multiplicity"] = (
                        event.ic_trace.get_number_of_peaks()
                    )
                    break

            # Set IC SCA if present; take first non-garbage peak
            if event.ic_sca_trace is not None:
                # No way to disentangle multiplicity
                for peak in event.ic_sca_trace.get_peaks():
                    pc_dataset.attrs["ic_sca_centroid"] = peak.centroid
                    pc_dataset.attrs["ic_sca_multiplicity"] = (
                        event.ic_sca_trace.get_number_of_peaks()
                    )
                    break

            # Record beam event information
            if event.beam_ds_trace is not None:
                if event.beam_ds_trace.get_number_of_peaks() == 1:
                    beam_events["event"].append(idx)
                    beam_events["ic_amplitude"].append(
                        [peak.amplitude for peak in event.ic_trace.get_peaks()]
                    )
                    beam_events["ic_centroid"].append(
                        [peak.centroid for peak in event.ic_trace.get_peaks()]
                    )
                    beam_events["ic_multiplicity"].append(
                        event.ic_trace.get_number_of_peaks()
                    )
                    beam_events["ic_sca_centroid"].append(
                        [peak.centroid for peak in event.ic_sca_trace.get_peaks()]
                    )
                    beam_events["ic_sca_multiplicity"].append(
                        event.ic_sca_trace.get_number_of_peaks()
                    )

            pc_dataset[:] = pc.cloud

        # Write beam events results to a dataframe
        df = pl.DataFrame(beam_events)
        df.write_parquet(
            workspace_path
            / "beam_events"
            / f"{form_run_string(payload.run_number)}.parquet"
        )

        spyral_info(__name__, "Phase 1 complete")
        return result


class GetLegacyEvent:
    """Class representing a legacy event in the GET DAQ

    Contains traces (GetTraces) from the AT-TPC pad plane as well
    as external signals in CoBo 10. At this time, we only support extraction
    of the IC from CoBo 10.

    Parameters
    ----------
    raw_data: h5py.Dataset
        The hdf5 Dataset that contains trace data
    event_number: int
        The event number
    get_params: GetParameters
        Configuration parameters controlling the GET signal analysis
    ic_params: ICParameters
        Configuration parameters related to the IC and CoBo 10
    rng: numpy.random.Generator
        A random number generator for use in the signal analysis

    Attributes
    ----------
    traces: list[GetTrace]
        The pad plane traces from the event
    ic_trace: GetTrace | None
        Trace of IC signal
    ic_sca_trace: GetTrace | None
        Trace of IC SCA signal
    beam_ds_trace: GetTrace | None
        Trace of downscale beam signal
    name: str
        The event name
    number:
        The event number

    Methods
    -------
    GetEvent(raw_data: h5py.Dataset, event_number: int, params: GetParameters, rng: numpy.random.Generator)
        Construct the event and process traces
    load_traces(raw_data: h5py.Dataset, event_number: int, params: GetParameters, rng: numpy.random.Generator)
        Process traces
    is_valid() -> bool
        Check if the event is valid
    """

    def __init__(
        self,
        raw_data: h5.Dataset,
        event_number: int,
        get_params: GetParameters,
        ic_params: ICParameters,
        rng: np.random.Generator,
    ):
        self.traces: list[GetTrace] = []
        self.ic_trace: GetTrace | None = None
        self.ic_sca_trace: GetTrace | None = None
        self.beam_ds_trace: GetTrace | None = None
        self.name: str = INVALID_EVENT_NAME
        self.number: int = INVALID_EVENT_NUMBER
        self.load_traces(raw_data, event_number, get_params, ic_params, rng)

    def load_traces(
        self,
        raw_data: h5.Dataset,
        event_number: int,
        get_params: GetParameters,
        ic_params: ICParameters,
        rng: np.random.Generator,
    ):
        """Process the traces

        Parameters
        ----------
        raw_data: h5py.Dataset
            The hdf5 Dataset that contains trace data
        event_number: int
            The event number
        get_params: GetParameters
            Configuration parameters controlling the GET signal analysis
        ic_params: ICParameters
            Configuration parameters related to the IC and CoBo 10
        rng: numpy.random.Generator
            A random number generator for use in the signal analysis
        """
        self.name = str(raw_data.name)
        self.number = event_number
        trace_matrix = preprocess_traces(
            raw_data[:, GET_DATA_TRACE_START:GET_DATA_TRACE_STOP].copy(),
            get_params.baseline_window_scale,
        )
        self.traces = [
            GetTrace(
                trace_matrix[idx], hardware_id_from_array(row[0:5]), get_params, rng
            )
            for idx, row in enumerate(raw_data)
        ]
        # Legacy data where external data was stored in CoBo 10 (IC, mesh, downscale beam)
        for trace in self.traces:
            # Extract IC
            if (
                trace.hw_id.cobo_id == 10
                and trace.hw_id.aget_id == 1
                and trace.hw_id.aget_channel == 0
            ):
                self.ic_trace = trace
                self.ic_trace.find_peaks(ic_params, rng, rel_height=0.5, min_width=4.0)  # type: ignore
                # Remove peaks outside of active time window of AT-TPC
                self.ic_trace.remove_peaks(ic_params.low_accept, ic_params.high_accept)

            # Extract IC SCA
            elif (
                trace.hw_id.cobo_id == 10
                and trace.hw_id.aget_id == 2
                and trace.hw_id.aget_channel == 34
            ):
                self.ic_sca_trace = trace
                self.ic_sca_trace.find_peaks(ic_params, rng, rel_height=0.5)
                # Remove peaks outside of active time window of AT-TPC
                self.ic_sca_trace.remove_peaks(
                    ic_params.low_accept, ic_params.high_accept
                )

            # Extract beam downscale beam trace
            elif (
                trace.hw_id.cobo_id == 10
                and trace.hw_id.aget_id == 3
                and trace.hw_id.aget_channel == 34
            ):
                self.beam_ds_trace = trace
                self.beam_ds_trace.find_peaks(ic_params, rng, rel_height=0.8)

        # Remove CoBo 10 from our normal traces
        self.traces = [trace for trace in self.traces if trace.hw_id.cobo_id != 10]

    def is_valid(self) -> bool:
        return self.name != INVALID_EVENT_NAME and self.number != INVALID_EVENT_NUMBER


class GetTrace:
    """A single trace from the GET DAQ data

    Represents a raw signal from the AT-TPC pad plane through the GET data acquisition.

    Parameters
    ----------
    data: ndarray
        The trace data
    id: HardwareID
        The HardwareID for the pad this trace came from
    params: GetParameters
        Configuration parameters controlling the GET signal analysis
    rng: numpy.random.Generator
        A random number generator for use in the signal analysis

    Attributes
    ----------
    trace: ndarray
        The trace data
    peaks: list[Peak]
        The peaks found in the trace
    hw_id: HardwareID
        The hardware ID for the pad this trace came from

    Methods
    -------
    GetTrace(data: ndarray, id: HardwareID, params: GetParameters, rng: numpy.random.Generator)
        Construct the GetTrace and find peaks
    set_trace_data(data: ndarray, id: HardwareID, params: GetParameters, rng: numpy.random.Generator)
        Set the trace data and find peaks
    is_valid() -> bool:
        Check if the trace is valid
    get_pad_id() -> int
        Get the pad id for this trace
    find_peaks(params: GetParameters, rng: numpy.random.Generator, rel_height: float, min_width: float)
        Find the peaks in the trace
    get_number_of_peaks() -> int
        Get the number of peaks found in the trace
    get_peaks(params: GetParameters) -> list[Peak]
        Get the peaks found in the trace
    remove_peaks(low_cut: int, high_cut: int) -> list[Peak]
        Remove peaks above and below the indicated cuts
    """

    def __init__(
        self,
        data: np.ndarray,
        id: HardwareID,
        params: GetParameters,
        rng: np.random.Generator,
    ):
        self.trace: np.ndarray = np.empty(0, dtype=np.int32)
        self.peaks: list[Peak] = []
        self.hw_id: HardwareID = HardwareID()
        if isinstance(data, np.ndarray) and id.pad_id != INVALID_PAD_ID:
            self.set_trace_data(data, id, params, rng)

    def set_trace_data(
        self,
        data: np.ndarray,
        id: HardwareID,
        params: GetParameters,
        rng: np.random.Generator,
    ):
        """Set trace data and find peaks

        Parameters
        ----------
        data: ndarray
            The trace data
        id: HardwareID
            The HardwareID for the pad this trace came from
        params: GetParameters
            Configuration parameters controlling the GET signal analysis
        rng: numpy.random.Generator
            A random number generator for use in the signal analysis
        """
        data_shape = np.shape(data)
        if data_shape[0] != NUMBER_OF_TIME_BUCKETS:
            raise GetTraceError(
                f"GetTrace was given data that did not have the correct shape! Expected 512 time buckets, instead got {data_shape[0]}"
            )

        self.trace = data.astype(np.int32)  # Widen the type and sign it
        self.hw_id = id
        self.find_peaks(params, rng)

    def is_valid(self) -> bool:
        """Check if the trace is valid

        Returns
        -------
        bool
            If True the trace is valid
        """
        return self.hw_id.pad_id != INVALID_PAD_ID and isinstance(
            self.trace, np.ndarray
        )

    def get_pad_id(self) -> int:
        """Get the pad id for this trace

        Returns
        -------
        int
            The ID number for the pad this trace came from
        """
        return self.hw_id.pad_id

    def find_peaks(
        self,
        params: GetParameters,
        rng: np.random.Generator,
        rel_height: float = 0.95,
        min_width: float = 1.0,
    ):
        """Find the peaks in the trace data

        The goal is to determine the centroid location of a signal peak within a given pad trace. Use the find_peaks
        function of scipy.signal to determine peaks. We then use this info to extract peak amplitudes, and integrated charge.

        Note: A random number generator is used to smear the centroids by within their identified time bucket. A time bucket
        is essentially a bin in time over which the signal is sampled. As such, the peak is identified to be on the interval
        [centroid, centroid+1). We sample over this interval to make the data represent this uncertainty.

        Parameters
        ----------
        params: GetParameters
            Configuration paramters controlling the GET signal analysis
        rng: numpy.random.Generator
            A random number generator for use in the signal analysis
        rel_height: float
            The relative height at which the left and right ips points are evaluated. Typically this is
            not needed to be modified, but for some legacy data is necessary
        min_width: float
            The minimum width of the peak. It is not inherently evaluated at the base of the peak, but is
            found according to a formula related to the prominence and relative height. See SciPy docs for more
        """

        if self.is_valid() == False:
            return

        self.peaks.clear()

        pks, props = signal.find_peaks(
            self.trace,
            distance=params.peak_separation,
            prominence=params.peak_prominence,
            width=(min_width, params.peak_max_width),
            rel_height=rel_height,
        )
        for idx, p in enumerate(pks):
            peak = Peak()
            peak.centroid = float(p) + rng.random()
            peak.amplitude = float(self.trace[p])
            peak.positive_inflection = int(np.floor(props["left_ips"][idx]))
            peak.negative_inflection = int(np.floor(props["right_ips"][idx]))
            peak.integral = np.sum(
                np.abs(self.trace[peak.positive_inflection : peak.negative_inflection])
            )
            if peak.amplitude > params.peak_threshold:
                self.peaks.append(peak)

    def get_number_of_peaks(self) -> int:
        """Get the number of peaks found in the trace

        Returns
        -------
        int
            Number of found peaks
        """
        return len(self.peaks)

    def get_peaks(self) -> list[Peak]:
        """Get the peaks found in the trace

        Returns
        -------
        list[Peak]
            The peaks found in the trace
        """
        return self.peaks

    def remove_peaks(self, low_cut: int, high_cut: int) -> list[Peak]:
        """Remove all peaks below and above the indicated cutoffs.

        Parameters
        -------
        low_cut: int
            Remove all peaks below this threshold
        high_cut: int
            Remove all peaks above this threshold
        """
        self.peaks = [
            peak for peak in self.peaks if (low_cut <= peak.centroid <= high_cut)
        ]


class PointCloud:
    """Representation of a AT-TPC event

    A PointCloud is a geometric representation of an event in the AT-TPC
    The GET traces are converted into points in space within the AT-TPC

    Attributes
    ----------
    event_number: int
        The event number
    cloud: ndarray
        The Nx8 array of points in AT-TPC space
        Each row is [x,y,z,amplitude,integral,pad id,time,scale]

    Methods
    -------
    PointCloud()
        Create an empty point cloud
    load_cloud_from_get_event(event: GetEvent, pmap: PadMap, corrector: ElectronCorrector)
        Load a point cloud from a GetEvent
    load_cloud_from_hdf5_data(data: ndarray, event_number: int)
        Load a point cloud from an hdf5 file dataset
    is_valid() -> bool
        Check if the point cloud is valid
    retrieve_spatial_coordinates() -> ndarray
        Get the positional data from the point cloud
    calibrate_z_position(micromegas_tb: float, window_tb: float, detector_length: float)
        Calibrate the cloud z-position from the micromegas and window time references
    remove_illegal_points(detector_length: float)
        Remove any points which lie outside the legal detector bounds in z
    sort_in_z()
        Sort the internal point cloud array by z-position
    """

    def __init__(self):
        self.event_number: int = INVALID_EVENT_NUMBER
        self.cloud: np.ndarray = np.empty(0, dtype=np.float64)

    def load_cloud_from_get_event(
        self,
        event: GetEvent | GetLegacyEvent,
        pmap: PadMap,
    ):
        """Load a point cloud from a GetEvent

        Loads the points from the signals in the traces and applies
        the pad relative gain correction and the pad time correction

        Parameters
        ----------
        event: GetEvent
            The GetEvent whose data should be loaded
        pmap: PadMap
            The PadMap used to get pad correction values
        """
        self.event_number = event.number
        count = 0
        for trace in event.traces:
            count += trace.get_number_of_peaks()
        self.cloud = np.zeros((count, 8))
        idx = 0
        for trace in event.traces:
            # Skip traces with no peaks and those with 10 or more peaks
            if (trace.get_number_of_peaks() == 0) or trace.get_number_of_peaks() >= 10:
                continue

            pid = trace.hw_id.pad_id
            check = pmap.get_pad_from_hardware(trace.hw_id)
            if check is None:
                spyral_warn(
                    __name__,
                    f"When checking pad number of hardware: {trace.hw_id}, recieved None!",
                )
                continue
            if (
                check != pid
            ):  # This is dangerous! We trust the pad map over the merged data!
                pid = check

            pad = pmap.get_pad_data(check)
            if pad is None or pmap.is_beam_pad(check):
                continue
            for peak in trace.get_peaks():
                self.cloud[idx, 0] = pad.x  # X-coordinate, geometry
                self.cloud[idx, 1] = pad.y  # Y-coordinate, geometry
                self.cloud[idx, 2] = (
                    peak.centroid + pad.time_offset
                )  # Z-coordinate, time with correction until calibrated with calibrate_z_position()
                self.cloud[idx, 3] = peak.amplitude
                self.cloud[idx, 4] = peak.integral
                self.cloud[idx, 5] = trace.hw_id.pad_id
                self.cloud[idx, 6] = (
                    peak.centroid + pad.time_offset
                )  # Time bucket with correction
                self.cloud[idx, 7] = pad.scale
                idx += 1
        self.cloud = self.cloud[self.cloud[:, 3] != 0.0]

    def load_cloud_from_hdf5_data(self, data: np.ndarray, event_number: int):
        """Load a point cloud from an hdf5 file dataset

        Parameters
        ----------
        data: ndarray
            This should be a copy of the point cloud data from the hdf5 file
        event_number: int
            The event number
        """
        self.event_number: int = event_number
        self.cloud = data

    def is_valid(self) -> bool:
        """Check if the PointCloud is valid

        Returns
        -------
        bool
            True if the PointCloud is valid
        """
        return self.event_number != INVALID_EVENT_NUMBER

    def retrieve_spatial_coordinates(self) -> np.ndarray:
        """Get only the spatial data from the point cloud


        Returns
        -------
        ndarray
            An Nx3 array of the spatial data of the PointCloud
        """
        return self.cloud[:, 0:3]

    def calibrate_z_position(
        self,
        micromegas_tb: float,
        window_tb: float,
        detector_length: float,
        efield_correction: ElectronCorrector | None = None,
    ):
        """Calibrate the cloud z-position from the micromegas and window time references

        Also applies the electric field correction if given
        Trims any points beyond the bounds of the detector (0 to detector length)

        Parameters
        ----------
        micromegas_tb: float
            The micromegas time reference in GET Time Buckets
        window_tb: float
            The window time reference in GET Time Buckets
        detector_length: float
            The detector length in mm
        efield_correction: ElectronCorrector | None
            The optional Garfield electric field correction to the electron drift
        """
        # Maybe use mm as the reference because it is more stable?
        for idx, point in enumerate(self.cloud):
            self.cloud[idx][2] = (
                (window_tb - point[6]) / (window_tb - micromegas_tb) * detector_length
            )
            if efield_correction is not None:
                self.cloud[idx] = efield_correction.correct_point(self.cloud[idx])

    def remove_illegal_points(self, detector_length: float = 1000.0):
        """Remove any points which lie outside the legal detector bounds in z

        Parameters
        ----------
        detector_length: float
            The length of the detector in the same units as the point cloud data
            (typically mm)

        """
        mask = np.logical_and(
            self.cloud[:, 2] < detector_length, self.cloud[:, 2] > 0.0
        )
        self.cloud = self.cloud[mask]

    def sort_in_z(self):
        """Sort the internal point cloud array by the z-coordinate"""
        indicies = np.argsort(self.cloud[:, 2])
        self.cloud = self.cloud[indicies]


def generate_electron_correction(
    garf_file_path: Path, output_path: Path, params: DetectorParameters
):
    """Generate the grid for the correction to electrons drifting the AT-TPC field

    Need to convert garfield correction data into regularly spaced interpolation scheme
    over rho and z with the correct units

    Garfield data format:
    [x_initial, y_initial, x_final, y_final, z_final, time] all in cm
    x=z, y=rho, x=transverse

    Parameters
    ----------
    garf_file_path: pathlib.Path
        Path to a file containing the data generated by Garfield
    output_path: pathlib.Path
        Path to a numpy file for the interpolation grid to be saved to
    params: DetectorParameters
        Configuration parameters for physical detector properties
    """

    # Use average window and micromegas time buckets from all runs in the drift velocity file
    dv_df: pl.DataFrame = pl.read_csv(params.drift_velocity_path)
    w_tb_avg = dv_df.select(pl.mean("average_window_tb"))[0, 0]
    mm_tb_avg = dv_df.select(pl.mean("average_micromegas_tb"))[0, 0]

    garfield_data: np.ndarray = np.loadtxt(garf_file_path, dtype=float)

    chunk_size = 55  # garfield data in 55 row chunks (steps in rho)
    chunk_midpoint_index = 27  # index of the chunk midpoint
    n_chunks = int(len(garfield_data) / chunk_size)  # number of chunks (steps in z)

    z_steps = np.linspace(30.0, 1000.0, 98)
    gz_min = 30.0
    gz_max = 1000.0
    gz_bins = 98
    rho_steps = np.linspace(-270.0, 270.0, 55)
    grho_min = -270.0
    grho_max = 270.0
    grho_bins = 55

    rho_garf_points, z_garf_points = np.meshgrid(rho_steps, z_steps)

    rho_final: np.ndarray = np.zeros((n_chunks, chunk_size))
    misc_final: np.ndarray = np.zeros((n_chunks, chunk_size, 3))
    for chunk in range(n_chunks):
        for row in range(chunk_size):
            # Convert distances to mm
            misc_final[chunk, row, 0] = (
                garfield_data[chunk * chunk_size + row, 2] * 10.0
            )  # z
            rho_final[chunk, row] = (
                garfield_data[chunk * chunk_size + row, 3] * 10.0
            )  # radial
            misc_final[chunk, row, 1] = (
                garfield_data[chunk * chunk_size + row, 4] * 10.0
            )  # transverse
            # Time in nanoseconds
            misc_final[chunk, row, 2] = garfield_data[
                chunk * chunk_size + row, 5
            ]  # time/z initial

    # Correct the time for the midpoint and convert to mm
    for chunk in misc_final:
        mid_val = chunk[chunk_midpoint_index, 2]
        chunk[:, 2] -= mid_val
        chunk[:, 2] *= (
            params.detector_length / (w_tb_avg - mm_tb_avg) * params.get_frequency * 0.001
        )

    interp = BilinearInterpolator(
        gz_min, gz_max, gz_bins, grho_min, grho_max, grho_bins, misc_final
    )
    contour = contour_generator(z_garf_points, rho_garf_points, rho_final)

    rho_bin_min = 0.0
    rho_bin_max = 275.0
    rho_bins = 276

    z_bin_min = 0.0
    z_bin_max = 1000.0
    z_bins = 1001

    rho_points = np.linspace(rho_bin_min, rho_bin_max, rho_bins)
    z_points = np.linspace(z_bin_min, z_bin_max, z_bins)

    correction_grid = np.zeros((276, 1001, 3))

    for ridx, r in enumerate(rho_points):
        for zidx, z in enumerate(z_points):
            # rescale z to garfield
            zg = (1.0 - z * 0.001) * 970.0 + 30.0  # Garfield first point is at 30.0 mm
            rho_cor = interpolate_initial_rho(contour, z, r) - r
            correction_grid[ridx, zidx, 0] = rho_cor
            others = interp.interpolate(zg, rho_cor)
            correction_grid[ridx, zidx, 1] = others[1]
            correction_grid[ridx, zidx, 2] = others[2]

    np.save(output_path, correction_grid)
