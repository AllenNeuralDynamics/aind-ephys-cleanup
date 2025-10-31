import datetime
import logging
import pathlib
import polars as pl
import npc_lims
import tqdm
logger = logging.getLogger(__name__)

DRYRUN = True
if not DRYRUN: # user confirmation on terminal
    confirm = input(f"Are you sure you want to delete data? Type 'yes' to confirm: ")
    if confirm.lower() != 'yes':
        print("Aborting deletion")
        exit(0)
        
df = pl.read_json("C:/Users/ben.hardcastle/github/aind-ephys-cleanup/scripts/validation_metrics.json")
safe_to_delete = df.filter(pl.col('safe_to_delete'))
print(f"{safe_to_delete['size_gb'].sum(): .2f} GB potentially safe to delete -- must verify split recordings from npc_lims")


sessions_to_delete = []
for session in tqdm.tqdm(safe_to_delete['session_id'].unique().to_list(), desc="Checking sessions against npc_lims for split recordings"):
    # check no overlap with split sessions in npc_lims
    # but first check that we're not looking at a surface channel recording (whose session_id would
    # be normalized by npc_lims to give the same as the non-surface channel)
    if not df.filter(pl.col('session_id') == session, pl.col('path').str.contains('surface')).is_empty():
        # session (with timestamp) corresponds to surface channel recording
        continue
    issues = npc_lims.get_session_issues(session)
    if issues and "https://github.com/AllenInstitute/npc_lims/issues/5" in issues and session != 'ecephys_686176_2023-12-06_13-03-34':
        logger.warning(f"{session} is a known split recording (on npc_lims) but was marked safe to delete -- method for finding split recordings is faulty!")
        #! Edge case currently not handled:
        # If a session was split AND we already deleted one set of recordings we don't detect it as
        # a split recording eg ecephys_686176_2023-12-06_13-03-34 / DRpilot_686176_20231206
        continue
    else:
        sessions_to_delete.append(session)
assert 'ecephys_686176_2023-12-06_13-03-34' not in sessions_to_delete, f"known split session found in sessions_to_delete: {sessions_to_delete}"

deleted_gb = 0
for row in safe_to_delete.filter(pl.col('session_id').is_in(sessions_to_delete)).to_dicts():
    path = pathlib.Path(row['path'])
    if not path.exists():
        continue
    assert min(row['aligned_correlations']) >= 0.7
    if not DRYRUN:
        print(f"Deleting {row['path']}")
        path.unlink()
        # dump path to deleted.txt
        with open(pathlib.Path(__file__).parent / f"deleted_{datetime.date.today()}.txt", "a") as f:
            f.write(f"{row['path']}\n")
    deleted_gb += row['size_gb']
print(f"{'DRYRUN: ' if DRYRUN else ''}Deleted {deleted_gb: .2f} GB in total")


# import npc_sync
# import matplotlib.pyplot as plt
# import numpy as np
# plt.hist(
#     np.diff(npc_sync.SyncDataset("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_746439_20250130/20250130T134910.h5").get_rising_edges('barcode_ephys', units='seconds'))
    
# )
# plt.savefig('hist.png')