
import json
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor
import polars as pl
import aind_ephys_cleanup
import altair as alt
ROOT = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot")
RESULTS = pathlib.Path(__file__).parent / "results.json"
lock = threading.Lock()

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
        
if __name__ == "__main__":
    # plot()
    main()
