"""One persistent SAM model, owned by the unified labeling dashboard."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import queue
import signal
import socketserver
import threading
import traceback
import uuid

from labeling_storage import atomic_json, read_json
from propagate_masks import load_predictor, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', required=True)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--parent', type=int, required=True)
    args = parser.parse_args()
    # Kill this GPU owner even if tmux/the dashboard is killed abruptly.
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != args.parent:
        return
    root = Path(__file__).resolve().parents[1]
    runtime = read_json(args.runtime)
    for key in ('source', 'dependencies', 'checkpoint'):
        runtime[key] = str((root / runtime[key]).resolve())
    path = Path(args.socket)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    print('Loading shared SAM model...', flush=True)
    predictor = load_predictor(runtime)
    print('SAM ready; model stays loaded until dashboard shutdown.', flush=True)
    instance = uuid.uuid4().hex
    pending = set()
    lock = threading.Lock()
    jobs = queue.Queue()

    def work():
        while True:
            job = jobs.get()
            try:
                atomic_json(Path(job['directory'])/'progress.json',
                            {'status':'running','stage':'preparing','completed':0,'total':job['frames']})
                # Every job gets fresh inference state; only the model is shared.
                run(job, predictor=predictor)
            except Exception as exc:
                atomic_json(Path(job['directory'])/'progress.json', {'status':'failed','error':str(exc)})
                with (Path(job['directory'])/'worker.log').open('a') as log:
                    traceback.print_exc(file=log)
                traceback.print_exc()
            finally:
                with lock:
                    pending.discard(job['job_id'])
                jobs.task_done()
                import torch
                torch.cuda.empty_cache()  # releases scratch space, keeps model weights

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                request = json.loads(self.rfile.readline(65536))
                if request.get('command') == 'submit':
                    job_path = Path(request['job']).resolve()
                    job = read_json(job_path)
                    if job_path != Path(job['directory']).resolve()/'job.json':
                        raise ValueError('Job directory mismatch')
                    for key in ('source','dependencies','checkpoint'):
                        if Path(job['runtime'][key]).resolve() != Path(runtime[key]):
                            raise ValueError('Job uses a different SAM runtime')
                    for key in ('gpu','model_config','checkpoint_sha256'):
                        if job['runtime'].get(key, 0 if key=='gpu' else None) != runtime.get(key, 0 if key=='gpu' else None):
                            raise ValueError('Job uses a different SAM runtime')
                    with lock:
                        if job['job_id'] not in pending:
                            atomic_json(Path(job['directory'])/'progress.json',
                                        {'status':'running','stage':'queued','completed':0,'total':job['frames']})
                            pending.add(job['job_id'])
                            jobs.put(job)
                elif request.get('command') != 'status':
                    raise ValueError('Unknown command')
                with lock:
                    result = {'ready':True,'pid':os.getpid(),'instance':instance,'jobs':list(pending)}
            except Exception as exc:
                result = {'error':str(exc)}
            self.wfile.write(json.dumps(result).encode()+b'\n')

    threading.Thread(target=work, daemon=True).start()
    server = socketserver.ThreadingUnixStreamServer(str(path), Handler)
    os.chmod(path, 0o600)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
