import concurrent.futures
import contextlib
import datetime
import logging
import pathlib
from collections.abc import Iterable

import aind_session
import npc_io
import npc_session
import numpy as np
import upath
import zarr

import aind_ephys_cleanup.models as models

logger = logging.getLogger(__name__)



def _get_aind_session_from_subject_date_time(
    subject_id: int, date: str, time: str, search_before = datetime.timedelta(minutes=1), search_after = datetime.timedelta(minutes=1)
) -> aind_session.Session:
    """
    
    >>> 
    """
    matches = aind_session.get_sessions(
        subject_id=subject_id,
        platform='ecephys',
        start_date=datetime.datetime.fromisoformat(f"{date} {time}") - search_before,
        end_date=datetime.datetime.fromisoformat(f"{date} {time}") + search_after,
    )
    if not matches:
        raise ValueError(f"No matching AIND session found for {subject_id=} {date=} {time=}. Data likely has not been uploaded to Code Ocean.")
    if len(matches) > 1:
        raise AssertionError(f"Multiple matching AIND sessions found for {subject_id=} {date=} {time=}: {matches=}")
    return matches[0]

def populate_session_ids(dat_info: Iterable[models.DatData]) -> tuple[models.DatData, ...]:
    """For a list of DatData objects, populate the session_id field if it is missing by searching on
    Code Ocean / S3 using the subject_id, date and start_time associated with a .dat file.

    Notes: 
     - data may not have a discoverable AIND session if it hasn't been uploaded, in which case the
       session_id will remain None.
     - this function uses a thread pool to speed up lookups, as each lookup requires a web call.
     - if input models already have session_id populated, they will be validated and updated if incorrect.
    """
    if isinstance(dat_info, models.DatData):
        dat_info = [dat_info]
    dat_info = tuple(dat_info)  # Ensure we can iterate multiple times
    info_to_session_id: dict[tuple[int, str, str], str] = {}
    unique_session_info = set()
    for d in dat_info:
        if d.session_id is not None:
            with contextlib.suppress(ValueError):
                npc_session.extract_aind_session_id(d.session_id)
                info_to_session_id[(d.subject_id, d.date, d.start_time)] = d.session_id
                continue
        unique_session_info.add((d.subject_id, d.date, d.start_time))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_info = {}
        for info in unique_session_info:
            future = executor.submit(_get_aind_session_from_subject_date_time, *info)
            future_to_info[future] = info
        for future in concurrent.futures.as_completed(future_to_info):
            try:
                session = future.result()
            except ValueError:
                # No matching session found
                continue
            info = future_to_info[future]
            info_to_session_id[info] = session.id
    updated_models = []
    for d in dat_info:
        if (d.subject_id, d.date, d.start_time) not in info_to_session_id:
            # No matching session found, leave unmodified
            updated_models.append(d)
            continue
        else:
            updated_models.append(
                d.model_copy(
                    update={"session_id": info_to_session_id[(d.subject_id, d.date, d.start_time)]}
                )
            )
    return tuple(updated_models)

def get_settings_xml_path(dat_or_zarr_path: str) -> upath.UPath:
    """
    
    >>> path = "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_796012_20250715/DRpilot_796012_20250715/Record Node 104/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeD-LFP/continuous.dat"
    >>> get_settings_xml_path(path).as_posix()
    '//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_796012_20250715/DRpilot_796012_20250715/Record Node 104/settings.xml'
    
    >>> path = 's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_compressed/experiment1_Record Node 103#Neuropix-PXI-100.ProbeD-LFP.zarr'
    >>> get_settings_xml_path(path).as_posix()
    's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 103/settings.xml'
    """
    path = npc_io.from_pathlike(dat_or_zarr_path)
    if 'ecephys_compressed' in path.as_posix():
        record_node = path.stem.split('#')[0].split('_')[1]
        return next(npc_io.from_pathlike(f"{path.parent.as_posix().replace('ecephys_compressed', 'ecephys_clipped')}/{record_node}").glob('settings*.xml'))
    return next(path.parent.parent.parent.parent.parent.glob('settings*.xml'))


def get_dat_info(path: str | pathlib.Path | upath.UPath) -> models.DatData:
    """Get information about a single .dat file.
    
    Examples:
    
    >>> path = "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_796012_20250715/DRpilot_796012_20250715/Record Node 104/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeD-LFP/continuous.dat"
    >>> get_dat_info(path).model_dump(mode='json')
    {'dtype': 'int16', 'size': 6726224256, 'path': '//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_796012_20250715/DRpilot_796012_20250715/Record Node 104/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeD-LFP/continuous.dat', 'session_id': None, 'subject_id': 796012, 'date': '2025-07-15', 'settings_xml_path': '//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_796012_20250715/DRpilot_796012_20250715/Record Node 104/settings.xml', 'start_time': '13:54:38', 'device_name': 'Neuropix-PXI-100.ProbeD-LFP', 'device_type': 'LFP'}
    """
    path = npc_io.from_pathlike(path)
    if path.protocol != "":
        raise ValueError("Only reading local file paths is supported for .dat files.")
    arr = np.memmap(
        path,
        dtype=np.int16,
        mode="r",
    )
    return models.DatData(
        path=path.as_posix(),
        size=arr.size,
        dtype=str(arr.dtype),
    )


def get_zarr_info(path: str | pathlib.Path | upath.UPath) -> tuple[models.ZarrData, ...]:
    """Get information about all arrays in a Zarr store.
    
    
    Examples:
    
    >>> path = 's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_compressed/experiment1_Record Node 103#Neuropix-PXI-100.ProbeD-LFP.zarr'
    >>> zarr_info = get_zarr_info(path)
    >>> len(zarr_info)
    1
    >>> zarr_info[0].model_dump(mode='json')
    {'dtype': 'int16', 'size': 4574202624, 'path': 's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_compressed/experiment1_Record Node 103#Neuropix-PXI-100.ProbeD-LFP.zarr', 'segment_name': 'traces_seg0', 'session_id': 'ecephys_670248_2023-08-03_12-04-15', 'subject_id': 670248, 'date': '2023-08-03', 'settings_xml_path': 's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 103/settings.xml', 'start_time': '12:04:36', 'device_name': 'Neuropix-PXI-100.ProbeD-LFP', 'device_type': 'LFP'}
    """
    path = npc_io.from_pathlike(path)
    if path.suffix != ".zarr":
        logger.warning(f"path suffix is {path.suffix!r}. Expected .zarr.")
    return tuple(
        models.ZarrData(
            path=path.as_posix(),
            size=arr.size,
            dtype=str(arr.dtype),
            segment_name=name,
        )
        for name, arr in zarr.open(path, mode="r").items()
        if str(name).startswith("traces_seg")
    )

if __name__ == "__main__":
    import doctest
    doctest.testmod()
