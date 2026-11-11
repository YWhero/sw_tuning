"""Polling must not mark a provisionally published SAM job as dead."""
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mask_workflow import Workflow
from labeling_storage import atomic_json, read_json


def test_poll_waits_for_submission(tmp_path):
    class Dataset:
        lock = threading.RLock()
        def latest(self, index): return read_json(tmp_path / 'latest.json', {})
        def job_dir(self, index, job): return tmp_path / job
        def directory(self, index): return tmp_path
    (tmp_path / 'test').mkdir()
    atomic_json(tmp_path / 'latest.json', {'job_id':'test','status':'running'})
    atomic_json(tmp_path / 'test/progress.json', {'status':'running'})
    ds = Dataset()
    workflow = Workflow(ds, {})
    started = threading.Event()
    def poll():
        started.set()
        return workflow.current(1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with ds.lock:
            future = pool.submit(poll)
            assert started.wait(2)
            # Simulate the interval between publishing a job and finishing submit.
            assert not threading.Event().wait(.1)
            atomic_json(tmp_path / 'latest.json', {'job_id':'test','status':'awaiting_review'})
            atomic_json(tmp_path / 'test/progress.json', {'status':'awaiting_review'})
        assert future.result(timeout=2)['status'] == 'awaiting_review'
