import concurrent.futures
import contextlib
import datetime
import logging
import pathlib
import re
from collections.abc import Iterable

import aind_session
import npc_io
import npc_session
import numpy as np
import tqdm
import upath
import zarr

import aind_ephys_cleanup.models as models

logger = logging.getLogger(__name__)

def get_matches(dat_info: Iterable[models.DatData], progress_bar: bool = False) -> list[tuple[models.DatData, models.ZarrData]]:
    if isinstance(dat_info, models.DatData):
        dat_info = [dat_info]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_dat = {
            executor.submit(get_matching_zarr_info, d): d for d in dat_info
        }
        matches = []
        futures_iter = concurrent.futures.as_completed(future_to_dat)
        if progress_bar:
            futures_iter = tqdm.tqdm(futures_iter, total=len(future_to_dat), desc="Finding matches")
        for future in futures_iter:
            dat = future_to_dat[future]
            try:
                zarr_info = future.result()
            except Exception as exc:
                logger.error(f"Error finding match for {dat.path}: {exc!r}")
            else:
                if zarr_info is not None:
                    matches.append((dat, zarr_info))
    return matches

def _get_datetime(info: models.DatData | models.ZarrData) -> datetime.datetime:
    return datetime.datetime.fromisoformat(f"{info.date} {info.start_time}")

def is_match(dat_info: models.DatData, zarr_info: models.ZarrData) -> bool | None:
    """Check if a DatData and ZarrData object match based on subject_id, date, start_time,
    device_name and session_id.

    - None is returned if any of the fields required for matching are not present.
    """
    if dat_info.session_id is None:
        return None
    for field in ('device_name', 'session_id', 'size'):
        if getattr(dat_info, field) != getattr(zarr_info, field):
            return False
    # in the rare case that we picked up the settings.xml from another record node, start times may mismatch by a few seconds
    if _get_datetime(dat_info) - _get_datetime(zarr_info) > datetime.timedelta(seconds=3):
        return False
    return True

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

def get_matching_zarr_info(dat_info: models.DatData) -> models.ZarrData | None:
    """Given a DatData object, find the matching ZarrData object by searching compressed ephys data
    on S3 using the session_id, device_name and size.
    
    - session_id must be populated in the input dat_info
    - if multiple matching Zarr arrays are present, the one with size closest to the .dat file is
      returned.
    """
    if dat_info.session_id is None:
        raise ValueError("dat_info must have session_id populated to search for matching zarr upload.")
    session = aind_session.Session(dat_info.session_id)
    zarr_paths = list(session.ecephys.compressed_dir.glob(f"*{dat_info.device_name}.zarr"))
    # we discover the Zarr paths available for the device, rather than trying to reconstruct the
    # path from the experiment/recording index, because a) the naming scheme may change in
    # future, b) for MindScope data, the data are re-organized before upload, filtering out dummy
    # recordings, so the single Zarr may not have the original experiment/recording index
    # - so this method is more general, but we must be more careful to ensure data size and content match
    if len(zarr_paths) == 0:
        logger.info(f"No matching Zarr upload found for {dat_info.path}")
    if len(zarr_paths) > 1:
        experiment = re.search(r"(experiment\d+)(?=/)", dat_info.path)
        if experiment is None:
            raise ValueError(f"Could not determine experiment number from path: {dat_info.path}")
        record_node = re.search(r"Record Node \d+", dat_info.path)
        single_path = next((p for p in zarr_paths if p.stem.startswith(f"{experiment.group(0)}_{record_node.group(0)}")), None)
        if single_path is None:
            raise ValueError(f"Multiple matching Zarr uploads found for {dat_info.path}, and could not disambiguate using experiment number and record node: {[p.name for p in zarr_paths]}")
    else:
        single_path = zarr_paths[0]
    zarr_info = get_zarr_info(single_path)
    closest_match = sorted(zarr_info, key=lambda z: abs(z.size - dat_info.size))[0] # array with size closest to the .dat file
    if not is_match(dat_info, closest_match):
        logger.warning(f"Found Zarr upload {closest_match.path} for {dat_info.path}, but it does not appear to match.")
        return None
    return closest_match

def populate_session_ids(dat_info: Iterable[models.DatData], progress_bar: bool = False) -> tuple[models.DatData, ...]:
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
        futures_iter = concurrent.futures.as_completed(future_to_info)
        if progress_bar:
            futures_iter = tqdm.tqdm(futures_iter, total=len(future_to_info), desc="Populating session IDs")
        for future in futures_iter:
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
                    update={"session_id": info_to_session_id[(d.subject_id, d.date, d.start_time)]},
                    deep=True,
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
    {'dtype': 'int16', 'size': 6726224256, 'path': '//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_796012_20250715/DRpilot_796012_20250715/Record Node 104/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeD-LFP/continuous.dat', 'session_id': None, 'subject_id': 796012, 'date': '2025-07-15', 'settings_xml_path': '//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_796012_20250715/DRpilot_796012_20250715/Record Node 104/settings.xml', 'start_time': '13:54:38', 'device_name': 'Neuropix-PXI-100.ProbeD-LFP'}
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
    {'dtype': 'int16', 'size': 4574202624, 'path': 's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_compressed/experiment1_Record Node 103#Neuropix-PXI-100.ProbeD-LFP.zarr', 'session_id': 'ecephys_670248_2023-08-03_12-04-15', 'subject_id': 670248, 'date': '2023-08-03', 'settings_xml_path': 's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 103/settings.xml', 'start_time': '12:04:36', 'device_name': 'Neuropix-PXI-100.ProbeD-LFP', 'segment_name': 'traces_seg0'}
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
