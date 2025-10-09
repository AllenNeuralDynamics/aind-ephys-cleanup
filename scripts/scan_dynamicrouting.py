
import json
import logging
import pathlib
import re
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr
from matplotlib import pyplot as plt
from scipy import signal
from tqdm.auto import tqdm

import aind_ephys_cleanup
from aind_ephys_cleanup.models import DatData

logger = logging.getLogger(__name__)
ROOT = pathlib.Path(
    "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot"
)
RESULTS = pathlib.Path(__file__).parent / "results.json"

lock = threading.Lock()

SAMPLING_RATE = {
    aind_ephys_cleanup.models.RecordingType.AP: 30000,
    aind_ephys_cleanup.models.RecordingType.LFP: 2500,
    aind_ephys_cleanup.models.RecordingType.NIDAQ: 30000,
}

N_CHANNELS = {
    aind_ephys_cleanup.models.RecordingType.AP: 384,  # Neuropixels 1.0
    aind_ephys_cleanup.models.RecordingType.LFP: 384,  # Neuropixels 1.0
    aind_ephys_cleanup.models.RecordingType.NIDAQ: 8,
}

DEFAULT_N_CHANNELS = 384  # Neuropixels 1.0

class OutOfBoundsError(ValueError):
    """Custom exception for window limits out of data range."""
    pass

def _preprocess_window(
    window: np.ndarray,
    sampling_rate: float,
    cutoff_freq: float = 1000.0,
    filter_order: int = 4,
) -> np.ndarray:
    """
    Preprocess window by median subtraction and low-pass filtering.
    
    Processing steps:
    1. Subtract median across channels at each timepoint (removes DC offset/drift)
    2. Apply low-pass Butterworth filter if cutoff < Nyquist (removes high-frequency noise)
    
    Args:
        window: Array of shape (n_samples, n_channels)
        sampling_rate: Sampling rate in Hz
        cutoff_freq: Cutoff frequency in Hz (default 5000 Hz)
        filter_order: Filter order (default 4)
        
    Returns:
        Preprocessed array of same shape
    """
    # Step 1: Median subtraction
    centered = window - np.median(window, axis=1, keepdims=True)

    # Step 2: Low-pass filter (only if cutoff is valid)
    nyquist = sampling_rate / 2

    if cutoff_freq >= nyquist:
        logger.debug(
            f"Skipping lowpass filter: cutoff {cutoff_freq} Hz >= Nyquist {nyquist} Hz"
        )
        return centered

    normalized_cutoff = cutoff_freq / nyquist
    sos = signal.butter(filter_order, normalized_cutoff, btype='low', output='sos')

    filtered = np.zeros_like(centered)
    for ch_idx in range(centered.shape[1]):
        filtered[:, ch_idx] = signal.sosfiltfilt(sos, centered[:, ch_idx])

    return filtered


def process_one_file(dat_file: pathlib.Path) -> None:
    dat_info = aind_ephys_cleanup.get_dat_info(dat_file)
    with lock:
        if RESULTS.exists():
            results = json.loads(RESULTS.read_text())
        else:
            results = []
        results.append(dat_info.model_dump(mode='json'))
        RESULTS.write_text(json.dumps(results, indent=4))

def main(delete_previous: bool = False) -> None:
    if delete_previous and RESULTS.exists():
        RESULTS.unlink()
    dat_files = ROOT.glob("**/*.dat")
    with ThreadPoolExecutor() as executor:
        executor.map(process_one_file, dat_files)
    update_session_ids()
    write_matches()

def update_session_ids() -> None:
    data = json.loads(RESULTS.read_text())
    dat_info = [aind_ephys_cleanup.models.DatData(**item) for item in data]
    updated_dat_info = aind_ephys_cleanup.populate_session_ids(dat_info)
    updated_data = [d.model_dump(mode='json') for d in updated_dat_info]
    RESULTS.with_stem(RESULTS.stem + "_updated").write_text(json.dumps(updated_data, indent=4))

def write_matches() -> None:
    updated_dat_info = json.loads((RESULTS.with_stem(RESULTS.stem + "_updated")).read_text())
    matches = aind_ephys_cleanup.get_matches([aind_ephys_cleanup.models.DatData(**item) for item in updated_dat_info if item['session_id'] is not None])
    RESULTS.with_stem("matches").write_text(json.dumps([[model.model_dump(mode='json') for model in pair] for pair in matches], indent=4))

def get_split_recording_session_ids(dat_info: Iterable[aind_ephys_cleanup.models.DatData], duration_threshold_sec: int = 60) -> list[str]:
    """From a list of DatData, find the sessions that have multiple recordings that are over a
    threshold recording duration, implying that the experiment was split over multiple files. For
    MindScope data, those sessions have not been uploaded correctly, so their raw data should not be deleted.
    """
    session_ids = {d.session_id for d in dat_info if d.session_id is not None}
    session_id_to_models: dict[str, list[DatData]] = {sid: [m for m in dat_info if m.session_id == sid] for sid in session_ids}
    def _get_record_node(model: DatData) -> str:
        match = re.search(r"Record Node \d+", model.path)
        assert match is not None
        return match.group(0)
    n_channels = 384 # assuming Neuropixels 1.0 for MindScope projects
    logger.warning(f"Assuming {n_channels} channels for all devices")
    split_session_ids_to_durations = {}
    for session_id, models in session_id_to_models.items():
        device_node_to_models: dict[tuple[str, str], list[DatData]] = {}
        # find the device with the most recordings on the same record node (if device is recorded
        # on multiple on record nodes we don't want to consider those separate recordings)
        for model in models:
            if model.device_type == aind_ephys_cleanup.models.RecordingType.NIDAQ:
                # NI-DAQ recordings have configurable sampling rates, so hard to determine duration
                continue
            record_node = _get_record_node(model)
            key = (model.device_name, record_node)
            device_node_to_models.setdefault(key, []).append(model)
        most_recordings: list[DatData] = sorted(device_node_to_models.values(), key=len, reverse=True)[0]

        durations_sec = [
            m.size / n_channels / SAMPLING_RATE[m.device_type]
            for m in sorted(most_recordings, key=lambda x: x.path)
        ]
        if len([d for d in durations_sec if d > duration_threshold_sec]) > 1:
            split_session_ids_to_durations[session_id] = durations_sec
    return sorted(split_session_ids_to_durations.keys())


# =============================================================================
# Data Loading and Comparison Functions
# =============================================================================
# The functions below provide modular utilities for loading and comparing
# data from local .dat files and S3 zarr files. They support both visual
# comparison (via save_figure) and quantitative validation (via
# validate_upload_correlation).
# =============================================================================


def load_local_data(
    dat_info: aind_ephys_cleanup.models.DatData,
) -> np.memmap | None:
    """Load local .dat file as memory-mapped array.
    
    Args:
        dat_info: DatData model containing path and metadata
        
    Returns:
        Memory-mapped array reshaped to (n_samples, n_channels), or None if file doesn't exist
    """
    dat_path = pathlib.Path(dat_info.path)
    if not dat_path.exists():
        logger.warning(f"Local file does not exist: {dat_info.path}")
        return None
    return np.memmap(dat_info.path, dtype="int16", mode="r").reshape(-1, N_CHANNELS[dat_info.device_type])


def load_zarr_data(zarr_info: aind_ephys_cleanup.models.ZarrData) -> zarr.Array:
    """Load zarr array from S3.
    
    Args:
        zarr_info: ZarrData model containing path and zarr key
        
    Returns:
        Zarr array
    """
    return zarr.open(zarr_info.path, mode="r")[zarr_info.zarr_key]


def _extract_single_window(
    data: np.ndarray | zarr.Array,
    start_time_sec: float,
    duration_sec: float,
    sampling_rate: int,
    preprocess: bool = True,
) -> np.ndarray:
    """Extract a single data window from a recording with optional preprocessing.
    
    Strict extraction - raises ValueError if window is out of bounds.
    Caller is responsible for bounds checking and handling negative times.
    
    Args:
        data: Array of shape (n_samples, n_channels)
        start_time_sec: Start time in seconds (must be >= 0 and valid)
        duration_sec: Duration in seconds
        sampling_rate: Sampling rate in Hz
        preprocess: Whether to apply median subtraction and lowpass filtering
        
    Returns:
        Data window of shape (n_samples_window, n_channels).
        
    Raises:
        ValueError: If window is out of bounds
    """
    max_samples = data.shape[0]
    max_time_sec = max_samples / sampling_rate
    end_time_sec = start_time_sec + duration_sec

    # Strict bounds check
    if start_time_sec < 0 or end_time_sec > max_time_sec:
        raise OutOfBoundsError(
            f"Window [{start_time_sec:.2f}:{end_time_sec:.2f}]s is outside "
            f"valid range [0:{max_time_sec:.2f}]s"
        )

    # Extract window exactly as requested
    start_sample = round(start_time_sec * sampling_rate)
    end_sample = round(end_time_sec * sampling_rate)

    window = np.array(data[start_sample:end_sample, :])

    # Apply preprocessing if requested
    if preprocess and window.size > 0:
        window = _preprocess_window(window, sampling_rate)

    return window


def get_paired_data_windows(
    data1: np.ndarray | zarr.Array,
    data2: np.ndarray | zarr.Array,
    start_times_sec: Sequence[float],
    duration_sec: float,
    sampling_rate: int,
) -> list[tuple[np.ndarray, np.ndarray, str, bool]]:
    """Extract paired data windows from two recordings, ensuring alignment.
    
    Uses _extract_single_window for consistent preprocessing. Handles bounds
    checking and negative times before extraction.
    
    Args:
        data1: First array of shape (n_samples, n_channels) (e.g., local data)
        data2: Second array of shape (n_samples, n_channels) (e.g., S3 data)
        start_times_sec: Start times for each window in seconds.
            Negative values are relative to end.
        duration_sec: Duration of each window in seconds
        sampling_rate: Sampling rate in Hz
        
    Returns:
        List of tuples (window1, window2, window_key, is_valid) where:
        - window1: Data window from first array (preprocessed)
        - window2: Data window from second array (preprocessed)
        - window_key: String describing time range (e.g., "[0.00:0.01]s")
        - is_valid: True if both windows extracted successfully, False otherwise
    """
    paired_windows = []
    assert data1.shape[0] == data2.shape[0], "Data arrays must have same number of samples"
    n_samples = data1.shape[0]
    max_time = n_samples / sampling_rate

    for t0 in start_times_sec:
        # Handle negative times (relative to end)
        if t0 < 0:
            t0_abs = max_time + t0
        else:
            t0_abs = t0

        t1 = t0_abs + duration_sec
        window_key = f"[{t0_abs:.2f}:{t1:.2f}]s"

        # Check bounds before extraction
        if t0_abs < 0 or t1 > max_time:
            n_channels_1 = data1.shape[1] if len(data1.shape) > 1 else 1
            n_channels_2 = data2.shape[1] if len(data2.shape) > 1 else 1
            empty1 = np.array([]).reshape(0, n_channels_1)
            empty2 = np.array([]).reshape(0, n_channels_2)
            paired_windows.append((empty1, empty2, window_key, False))
            continue

        # Extract windows using strict extraction function
        try:
            window1 = _extract_single_window(data1, t0_abs, duration_sec, sampling_rate)
            window2 = _extract_single_window(data2, t0_abs, duration_sec, sampling_rate)

            # Verify shapes match
            if window1.shape != window2.shape:
                logger.warning(
                    f"Window {window_key} shape mismatch: "
                    f"data1={window1.shape}, data2={window2.shape}"
                )
                paired_windows.append((window1, window2, window_key, False))
                continue

            paired_windows.append((window1, window2, window_key, True))

        except OutOfBoundsError:
            n_channels_1 = data1.shape[1] if len(data1.shape) > 1 else 1
            n_channels_2 = data2.shape[1] if len(data2.shape) > 1 else 1
            empty1 = np.array([]).reshape(0, n_channels_1)
            empty2 = np.array([]).reshape(0, n_channels_2)
            paired_windows.append((empty1, empty2, window_key, False))

    return paired_windows


def validate_upload_correlation(
    dat_info: aind_ephys_cleanup.models.DatData,
    zarr_info: aind_ephys_cleanup.models.ZarrData,
    correlation_threshold: float = 0.99,
    duration_sec: float = 0.01,
    start_times_sec: Sequence[float] | None = None,
) -> tuple[bool, dict[str, float]]:
    """Validate S3 upload by computing correlations between local and S3 data windows.
    
    Invalid windows (out of bounds, empty, NaN correlations) are skipped entirely
    and not included in results. If all windows are invalid, validation fails.
    
    Args:
        dat_info: DatData model for local .dat file
        zarr_info: ZarrData model for S3 zarr file
        correlation_threshold: Minimum correlation required for validation to pass
        duration_sec: Duration of each comparison window in seconds
        start_times_sec: Start times for comparison windows.
            Defaults to (0, 10, 100, 1000, -2*duration)
        
    Returns:
        Tuple of (is_valid, correlations_dict) where:
        - is_valid: True if all valid correlations >= threshold and at least
          one valid window exists
        - correlations_dict: Maps window time ranges to correlation values
          (only includes valid windows, no NaN values)
    """
    if start_times_sec is None:
        start_times_sec = (0, 10, 100, 1000, -2 * duration_sec)

    # Get sampling rate and n_channels
    sampling_rate = SAMPLING_RATE[dat_info.device_type]

    # Load data
    local_data = load_local_data(dat_info)
    if local_data is None:
        return False, {}

    try:
        s3_data = load_zarr_data(zarr_info)
    except Exception as e:
        logger.error(f"Failed to load S3 data from {zarr_info.path}: {e!r}")
        return False, {}

    # Get paired windows from both sources - this ensures alignment
    paired_windows = get_paired_data_windows(
        local_data, s3_data, start_times_sec, duration_sec, sampling_rate
    )

    # Compute correlations for valid windows only
    correlations = {}
    all_valid = True

    for local_win, s3_win, window_key, is_valid in paired_windows:
        # Skip invalid windows (already logged by get_paired_data_windows)
        if not is_valid:
            continue

        # Windows are already preprocessed by _extract_single_window
        # Flatten arrays and compute correlation
        local_flat = local_win.flatten()
        s3_flat = s3_win.flatten()

        # Compute Pearson correlation
        correlation = np.corrcoef(local_flat, s3_flat)[0, 1]

        # Skip windows with NaN correlation (e.g., constant data)
        if np.isnan(correlation):
            logger.warning(
                f"Skipping window {window_key}: NaN correlation "
                "(window may contain constant values)"
            )
            continue

        # Store valid correlation
        correlations[window_key] = float(correlation)

        # Check if correlation meets threshold
        if correlation < correlation_threshold:
            all_valid = False
            logger.warning(
                f"Low correlation {correlation:.4f} < {correlation_threshold} "
                f"for window {window_key} in {dat_info.path}"
            )

    # Check if we have any valid correlations
    if len(correlations) == 0:
        logger.error(
            f"No valid windows for validation of {dat_info.path} - "
            "all windows were skipped (out of bounds, empty, or NaN)"
        )
        return False, {}

    is_valid = all_valid

    if is_valid:
        logger.info(
            f"Upload validation PASSED for {dat_info.path}: "
            f"all correlations >= {correlation_threshold}"
        )
    else:
        logger.warning(
            f"Upload validation FAILED for {dat_info.path}: "
            f"correlations = {correlations}"
        )

    return is_valid, correlations


def compute_shifted_correlations(
    dat_info: aind_ephys_cleanup.models.DatData,
    zarr_info: aind_ephys_cleanup.models.ZarrData,
    duration_sec: float = 0.01,
    start_times_sec: Sequence[float] | None = None,
    offset_sec: float = 1.0,
) -> dict[str, float]:
    """Compute shifted correlations using MISALIGNED windows.
    
    This computes correlations between windows from different time points
    to establish a shifted "chance" correlation level. Useful for setting
    appropriate correlation thresholds.
    
    For example, if offset_sec=1.0, compares:
    - Local window at [0s, 0.01s] vs S3 window at [1s, 1.01s]
    - Local window at [10s, 10.01s] vs S3 window at [11s, 11.01s]
    
    Args:
        dat_info: DatData model for local .dat file
        zarr_info: ZarrData model for S3 zarr file
        duration_sec: Duration of each comparison window in seconds
        start_times_sec: Start times for windows. Defaults to (0, 10, 100, 1000)
        n_channels: Number of channels in the recording
        offset_sec: Time offset between local and S3 windows (seconds)
        
    Returns:
        Dictionary mapping window descriptions to shifted correlation values
    """
    if start_times_sec is None:
        start_times_sec = (0, 10, 100, 1000)

    # Get sampling rate and n_channels
    sampling_rate = SAMPLING_RATE[dat_info.device_type]
    n_channels = N_CHANNELS[dat_info.device_type]

    # Load data
    local_data = load_local_data(dat_info)
    if local_data is None:
        return {}

    try:
        s3_data = load_zarr_data(zarr_info)
    except Exception as e:
        logger.error(f"Failed to load S3 data from {zarr_info.path}: {e!r}")
        return {}

    # Create offset times for S3 (misaligned windows)
    s3_start_times = tuple(t + offset_sec for t in start_times_sec)

    # Extract windows from both datasets
    shifted_correlations = {}
    max_time = local_data.shape[0] / sampling_rate

    for t0_local, t0_s3 in zip(start_times_sec, s3_start_times):
        # Handle negative times
        t0_local_abs = max_time + t0_local if t0_local < 0 else t0_local
        t0_s3_abs = max_time + t0_s3 if t0_s3 < 0 else t0_s3

        # Extract individual windows with bounds checking
        try:
            local_win = _extract_single_window(
                local_data, t0_local_abs, duration_sec, sampling_rate
            )
            s3_win = _extract_single_window(
                s3_data, t0_s3_abs, duration_sec, sampling_rate
            )
        except OutOfBoundsError:
            continue

        # Verify shapes match
        if local_win.shape != s3_win.shape:
            logger.warning(
                f"shifted window shape mismatch: "
                f"local={local_win.shape}, s3={s3_win.shape}"
            )
            continue

        # Windows are already preprocessed by _extract_single_window
        # Flatten and compute correlation
        local_flat = local_win.flatten()
        s3_flat = s3_win.flatten()

        correlation = np.corrcoef(local_flat, s3_flat)[0, 1]

        # Skip NaN
        if np.isnan(correlation):
            continue

        # Store with descriptive key showing the misalignment
        window_key = (
            f"local[{t0_local:.2f}:{t0_local + duration_sec:.2f}]s vs "
            f"s3[{t0_s3:.2f}:{t0_s3 + duration_sec:.2f}]s"
        )
        shifted_correlations[window_key] = float(correlation)

    if shifted_correlations:
        mean_shifted = np.mean(list(shifted_correlations.values()))
        logger.info(
            f"shifted (misaligned) correlation for {dat_info.path}: "
            f"mean = {mean_shifted:.6f} across {len(shifted_correlations)} window pairs"
        )

    return shifted_correlations


def save_figure(
    dat_info: aind_ephys_cleanup.models.DatData,
    zarr_info: aind_ephys_cleanup.models.ZarrData,
) -> None:
    """Create comparison plots between local .dat and S3 zarr data.

    Args:
        dat_info: DatData model for local .dat file
        zarr_info: ZarrData model for S3 zarr file
        n_channels: Number of channels in the recording
    """
    duration_sec = 0.01
    start_times_sec = (0, 10, 100, 1000, -2 * duration_sec)

    # Get sampling rate
    sampling_rate = SAMPLING_RATE[dat_info.device_type]

    # Load data using shared functions
    local_data = load_local_data(dat_info)
    if local_data is None:
        logger.warning(
            f"Skipping figure for {dat_info.path} because local file does not exist"
        )
        return

    try:
        s3_data = load_zarr_data(zarr_info)
    except Exception as e:
        logger.error(f"Failed to load S3 data from {zarr_info.path}: {e!r}")
        return

    # Get paired data windows - ensures alignment between local and S3
    paired_windows = get_paired_data_windows(
        local_data, s3_data, start_times_sec, duration_sec, sampling_rate
    )

    # Create figure path
    fig_filename = (
        dat_info.path.split('workgroups')[1]
        .replace('/', '_')
        .replace('\\', '_')
        .replace('.dat', '')
        .replace('.', '-')
        + ".png"
    )
    fig_path = pathlib.Path(__file__).parent / "for_deletion" / fig_filename
    fig_path.parent.mkdir(exist_ok=True, parents=True)

    # Create plots - filter to only valid windows
    valid_windows = [
        (local_win, s3_win, window_key, is_valid)
        for local_win, s3_win, window_key, is_valid in paired_windows
        if is_valid
    ]

    if not valid_windows:
        logger.warning(
            f"No valid windows for plotting {dat_info.path} - all windows were invalid"
        )
        return

    # Compute correlations for each valid window pair
    correlations = []
    for local_win, s3_win, _, _ in valid_windows:
        # Windows are already preprocessed by _extract_single_window
        local_flat = local_win.flatten()
        s3_flat = s3_win.flatten()
        corr = np.corrcoef(local_flat, s3_flat)[0, 1]
        correlations.append(corr)

    n_plots = len(valid_windows)
    fig, axes = plt.subplots(2, n_plots, figsize=(15, 10))

    # Handle case where there's only one valid window
    if n_plots == 1:
        axes = axes.reshape(2, 1)

    for plot_idx, (local_win, s3_win, window_key, _) in enumerate(valid_windows):
        correlation = correlations[plot_idx]

        for data_idx, window in enumerate((local_win, s3_win)):
            ax: plt.Axes = axes[data_idx, plot_idx]
            plt.sca(ax)

            # Windows are already preprocessed by _extract_single_window
            ax.imshow(
                window.T,  # Transpose for (channels, samples) display
                aspect="auto",
                origin="lower",
            )

            if plot_idx == 0:
                ax.set_ylabel("s3" if data_idx == 1 else "isilon")
            else:
                ax.yaxis.set_visible(False)

            # Add correlation to title of top subplot only
            if data_idx == 0:
                title = f"Corr: {correlation:.6f}\n{'isilon'} {window_key}"
            else:
                title = f"{'s3'} {window_key}"

            ax.set_title(title, fontsize=10)
            ax.xaxis.set_visible(False)

    # Add overall statistics to main title
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    main_title = (
        f"{dat_info.path}\n"
        f"Mean Correlation: {mean_corr:.6f} ± {std_corr:.6f} "
        f"(n={len(correlations)} windows)"
    )
    fig.suptitle(main_title, fontsize=8)
    fig.savefig(fig_path.as_posix())
    plt.close(fig)


def get_file_size(dat_info: aind_ephys_cleanup.models.DatData) -> int:
    return pathlib.Path(dat_info.path).stat().st_size


def compute_validation_metrics(
    dat_info: aind_ephys_cleanup.models.DatData,
    zarr_info: aind_ephys_cleanup.models.ZarrData,
    duration_sec: float = 0.01,
    start_times_sec: Sequence[float] | None = None,
    shifted_sec: float = 1.0,
    split_session_ids: Iterable[str] | None = None,
) -> dict:
    """Compute comprehensive validation metrics for a matched pair.
    
    Args:
        dat_info: DatData model for local .dat file
        zarr_info: ZarrData model for S3 zarr file
        duration_sec: Duration of each comparison window in seconds
        start_times_sec: Start times for windows. Defaults to (0, 10, 100, 1000, -2*duration)
        shifted_sec: Time offset for shifted (misaligned) correlations
        split_session_ids: Set of session IDs that have split recordings (should not be deleted)
        
    Returns:
        Dictionary containing:
        - aligned_correlations: List of correlation values for aligned windows
        - shifted_correlations: List of correlation values for misaligned windows
        - min_aligned: Minimum aligned correlation
        - mean_aligned: Mean aligned correlation
        - std_aligned: Std of aligned correlations
        - mean_shifted: Mean shifted correlation
        - std_shifted: Std of shifted correlations
        - shifted_threshold: mean_shifted + 1*std_shifted
        - all_above_threshold: Whether all aligned > shifted_threshold
        - safe_to_delete: Boolean recommendation
        - file_path: Local file path
        - session_id: Session ID
        - device_name: Device name
    """
    if start_times_sec is None:
        start_times_sec = (0, 10, 100, 1000, -2 * duration_sec)

    if split_session_ids is None:
        split_session_ids = set()

    # Get aligned correlations
    _, aligned_corr_dict = validate_upload_correlation(
        dat_info, zarr_info,
        correlation_threshold=0.0,  # Don't filter, get all
        duration_sec=duration_sec,
        start_times_sec=start_times_sec,
    )

    # Get shifted (misaligned) correlations
    shifted_corr_dict = compute_shifted_correlations(
        dat_info, zarr_info,
        duration_sec=duration_sec,
        start_times_sec=start_times_sec,
        offset_sec=shifted_sec,
    )

    # Extract values
    aligned_values = list(aligned_corr_dict.values()) if aligned_corr_dict else []
    shifted_values = list(shifted_corr_dict.values()) if shifted_corr_dict else []

    # Start with all fields from dat_info model
    metrics = dat_info.model_dump(mode='json')

    # Add correlation metrics
    metrics.update({
        "aligned_correlations": aligned_values,
        "shifted_correlations": shifted_values,
    })

    if shifted_values:
        metrics["shifted_threshold"] = min(1.0, float(np.mean(shifted_values) + 1 * np.std(shifted_values)))
    else:
        metrics["shifted_threshold"] = None

    # Determine if safe to delete
    safe_to_delete = False
    all_above_threshold = False

    # Check if session is split (should never delete split sessions)
    if dat_info.session_id in split_session_ids:
        safe_to_delete = False
        metrics["all_above_threshold"] = None
        metrics["safe_to_delete"] = False
        metrics["reason"] = "split_recording_session"
        return metrics

    if aligned_values and shifted_values:
        abs_threshold = 0.7  # Loose absolute threshold
        all_above_shifted_threshold = min(aligned_values) >= metrics["shifted_threshold"]
        metrics["all_above_threshold"] = all_above_shifted_threshold and min(aligned_values) >= abs_threshold

        # Safe to delete if:
        # 1. All aligned correlations > shifted + 1*std AND
        # 2. Min aligned correlation > loose absolute threshold
        safe_to_delete = metrics["all_above_threshold"] 
        if not all_above_shifted_threshold:
            metrics["reason"] = "aligned_below_shifted_threshold"
        elif min(aligned_values) < abs_threshold:
            metrics["reason"] = "aligned_below_absolute_threshold"
    else:
        metrics["all_above_threshold"] = None

    metrics["safe_to_delete"] = safe_to_delete
    if safe_to_delete:
        metrics["size_gb"] = get_file_size(dat_info) / 1e9

    return metrics


def validate_all_matches_with_metrics(
    output_filename: str = "validation_metrics.json",
    append_existing: bool = True,
    progress_bar: bool = True,
) -> list[dict]:
    """Compute validation metrics for all matched pairs and save to JSON incrementally.
    
    Results are written to JSON after each file is processed, so progress is saved
    even if the script is interrupted.
    
    Args:
        output_filename: Name of output JSON file (saved in same dir as matches.json)
        append_existing: Whether to append to existing results file
        progress_bar: Whether to display a progress bar
        
    Returns:
        List of validation metric dictionaries
    """
    matches_data = json.loads((RESULTS.with_stem("matches")).read_text())
    updated_dat_info = [
        aind_ephys_cleanup.models.DatData(**item)
        for item in json.loads(
            (RESULTS.with_stem(RESULTS.stem + "_updated")).read_text()
        )
    ]
    split = get_split_recording_session_ids(updated_dat_info)

    output_path = RESULTS.parent / output_filename

    # Load existing results if file exists
    if append_existing and output_path.exists():
        logger.info(f"Loading existing results from {output_path}")
        all_metrics = json.loads(output_path.read_text())
    else:
        all_metrics = []

    # Build set of already-processed file paths for quick lookup
    processed_paths = {m["file_path"] for m in all_metrics if "file_path" in m}

    # Optionally wrap iterator with progress bar
    iterator = enumerate(matches_data)
    if progress_bar:
        iterator = tqdm(
            enumerate(matches_data),
            desc="Validating matches",
            total=len(matches_data),
        )

    for idx, (dat_info_dict, zarr_info_dict) in iterator:
        dat_info = aind_ephys_cleanup.models.DatData(**dat_info_dict)
        zarr_info = aind_ephys_cleanup.models.ZarrData(**zarr_info_dict)

        file_path = str(dat_info.path)

        # Skip if already processed
        if file_path in processed_paths:
            if not progress_bar:
                logger.info(
                    f"[{idx+1}/{len(matches_data)}] Skipping {file_path} - already processed"
                )
            continue

        if not progress_bar:
            logger.info(f"[{idx+1}/{len(matches_data)}] Computing metrics for {file_path}")

        try:
            metrics = compute_validation_metrics(
                dat_info, zarr_info, split_session_ids=split
            )
        except Exception as e:
            logger.error(f"Failed to compute metrics for {file_path}: {e!r}")
            metrics = {
                "file_path": file_path,
                "session_id": dat_info.session_id,
                "device_name": dat_info.device_name,
                "safe_to_delete": False,
                "error": str(e),
            }

        all_metrics.append(metrics)
        processed_paths.add(file_path)

        # Write to JSON after each file (incremental save)
        output_path.write_text(json.dumps(all_metrics, indent=2))

    logger.info(f"Saved validation metrics to {output_path}")

    # Print summary
    total = len(all_metrics)
    safe_count = sum(1 for m in all_metrics if m.get("safe_to_delete", False))
    unsafe_count = total - safe_count

    print("\n" + "="*60)
    print("VALIDATION METRICS SUMMARY")
    print("="*60)
    print(f"Total files evaluated: {total}")
    print(f"Safe to delete: {safe_count}")
    print(f"NOT safe to delete: {unsafe_count}")
    print(f"\nMetrics saved to: {output_path}")

    if safe_count > 0:
        safe_files = [m for m in all_metrics if m.get("safe_to_delete", False)]
        total_size = sum(
            get_file_size(aind_ephys_cleanup.models.DatData(**dat_dict))
            for dat_dict, _ in matches_data
            if str(dat_dict.get("path")) in [m["file_path"] for m in safe_files]
        )
        print(f"Total size safe to delete: {total_size / 1e12:.2f} TB")

    print("="*60)

    return all_metrics


if __name__ == "__main__":
    # Compute validation metrics for all matched pairs
    validate_all_matches_with_metrics()
