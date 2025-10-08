import json
import random
from pathlib import Path

from aind_ephys_cleanup import models, utils

known_uploaded = models.DatData(
    **{
        'dtype': 'int16', 
        'size': 89079281280, 
        'path': '//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_744279_20250113/DRpilot_744279_20250113/Record Node 108/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeB-AP/continuous.dat',
        'session_id': 'ecephys_744279_2025-01-13_13-26-43',
    }
)
def test_populate_session_ids() -> None:

    # Use a copy of known_uploaded with session_id stripped out
    dat_info = [known_uploaded.model_copy(update={"session_id": None})]
    assert not any(d.session_id for d in dat_info), "session_id should be None in every model before running test"
    
    updated_dat_info = utils.populate_session_ids(dat_info)

    # Check that the function returns the same number of items
    assert len(updated_dat_info) == len(dat_info)
    assert any(d.session_id is not None for d in updated_dat_info), "No session_ids were populated"

def test_matching_zarr_info() -> None:
    matching_zarr = utils.get_matching_zarr_info(known_uploaded)
    assert utils.is_match(known_uploaded, matching_zarr)

if __name__ == "__main__":
    # test_populate_session_ids()
    test_matching_zarr_info()
