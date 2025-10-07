import logging
import pathlib

import npc_io
import numpy as np
import upath
import zarr

import aind_ephys_cleanup.models as models

logger = logging.getLogger(__name__)


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
