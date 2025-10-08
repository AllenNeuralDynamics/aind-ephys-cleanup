
import json
import logging
import pathlib
import re
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from matplotlib import pyplot as plt
import numpy as np
import zarr

import aind_ephys_cleanup
from aind_ephys_cleanup.models import DatData

logger = logging.getLogger(__name__)
ROOT = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot")
RESULTS = pathlib.Path(__file__).parent / "results.json"

lock = threading.Lock()

SAMPLING_RATE = {
    aind_ephys_cleanup.models.RecordingType.AP: 30000,
    aind_ephys_cleanup.models.RecordingType.LFP: 2500,
}

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


def save_figure(dat_info: aind_ephys_cleanup.models.DatData, zarr_info: aind_ephys_cleanup.models.ZarrData) -> None:

    sampling_rate = SAMPLING_RATE[dat_info.device_type]
    duration_sec = 0.01
    start_times_sec = (0, 10, 100, 1000, -2 * duration_sec)
    if dat_info.device_type == aind_ephys_cleanup.models.RecordingType.NIDAQ:
        logger.info(f"Skipping figure for {dat_info.path} because it is a NI-DAQmx file")
        return
    if not pathlib.Path(dat_info.path).exists():
        logger.warning(f"Skipping figure for {dat_info.path} because local file does not exist")
        return

    # fig_path = pathlib.Path(__file__).parent / "for_deletion" / (zarr_info.path.replace('ecephys/', '').replace('/ecephys_compressed', '').split('ecephys_')[1].replace('/','_').removesuffix('.zarr') + ".png")
    fig_path = pathlib.Path(__file__).parent / "for_deletion" / (dat_info.path.split('workgroups')[1].replace('/','_').replace('\\','_').replace('.dat', '').replace('.', '-') + ".png")
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    s3_data = zarr.open(zarr_info.path, mode="r")[zarr_info.zarr_key]
    local_data = np.memmap(dat_info.path, dtype="int16", mode="r").reshape(-1, 384)

    fig, axes = plt.subplots(2, len(start_times_sec), figsize=(15, 10))
    for data_idx, data in enumerate((local_data, s3_data)):
        if data is None:
            continue
        for time_idx, t0 in enumerate(start_times_sec):
            if t0 < 0:
                t0 = (data.shape[0] / sampling_rate) + t0
            t1 = t0 + duration_sec
            d = data[round(t0 * sampling_rate): round(t1  * sampling_rate), :].T
            ax: plt.Axes = axes[data_idx, time_idx]
            plt.sca(ax)
            im = ax.imshow(
                d,
                aspect="auto",
                origin="lower",
                # cmap="bwr",
            )
            # im.set_clim(-500, 500)
            if time_idx == 0:
                ax.set_ylabel("raw")
            else:
                ax.yaxis.set_visible(False)
            ax.set_title(
                f"{'s3' if data is s3_data else 'isilon'} [{t0}:{t0 + duration_sec}] s",
                fontsize=10,
            )
            ax.xaxis.set_visible(False)
        fig.suptitle(dat_info.path, fontsize=8)
    fig.savefig(fig_path.as_posix())
    plt.close(fig)

def get_file_size(dat_info: aind_ephys_cleanup.models.DatData) -> int:
    return pathlib.Path(dat_info.path).stat().st_size

if __name__ == "__main__":
    # plot()
    # main()
    # update_session_ids()
    # write_matches()
    updated_dat_info = [aind_ephys_cleanup.models.DatData(**item) for item in json.loads((RESULTS.with_stem(RESULTS.stem + "_updated")).read_text())]
    split = get_split_recording_session_ids(updated_dat_info)
    print(split)
    size = 0
    for dat_info, zarr_info in json.loads((RESULTS.with_stem("matches")).read_text()):
        dat_info = aind_ephys_cleanup.models.DatData(**dat_info)
        zarr_info = aind_ephys_cleanup.models.ZarrData(**zarr_info)
        if dat_info.session_id in split:
            logger.info(f"Skipping figure for {dat_info.path} because session {dat_info.session_id} has split recordings")
            continue
        save_figure(dat_info, zarr_info)
        size += get_file_size(dat_info)
    print(f"Total size of files with figures: {size / 1e9:.2f} GB")