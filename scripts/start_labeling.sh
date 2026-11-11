#!/usr/bin/env bash
# Start the unified three-camera labeler for a LeRobot v2.1 dataset.
set -euo pipefail
if [[ "${1:-}" == --stop ]]; then
    for session in refilling-label-dashboard refilling-label-cam_left_head refilling-label-cam_left_wrist refilling-label-cam_right_wrist; do
        if tmux has-session -t "=$session" 2>/dev/null; then
            tmux kill-session -t "=$session"
        fi
    done
    echo 'Labeling servers stopped; their shared SAM worker exits with the dashboard.'
    exit 0
fi
if [[ $# -ne 1 || "$1" == --help || "$1" == -h ]]; then
    echo "Usage: bash $0 /path/to/lerobot_dataset"
    echo 'Stop and unload SAM: bash start_labeling.sh --stop'
    echo 'Required cameras: cam_left_head, cam_left_wrist, cam_right_wrist'
    echo 'Open http://localhost:8769/ (forward port 8769 in VS Code).'
    [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]] && exit 0
    exit 2
fi
LABEL_DATASET="$(realpath -e -- "$1")"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v tmux >/dev/null || { echo 'tmux is required.' >&2; exit 1; }
source ./scripts/activate-conda.sh
python - "$LABEL_DATASET" <<'PY'
import fcntl
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

sys.path.insert(0, 'tools')
from labeling_storage import Dataset, atomic_json
root = Path.cwd()
dataset = Path(sys.argv[1]).resolve()
logs = root / 'logs'
logs.mkdir(exist_ok=True)
lock = (logs / 'labeling-start.lock').open('w')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit('Another labeling launcher is running. Try again shortly.')


def get(port, path):
    with urlopen(f'http://127.0.0.1:{port}{path}', timeout=5) as response:
        return json.load(response)


def has_session(name):
    return subprocess.run(['tmux', 'has-session', '-t', '='+name],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


# Validate before stopping any existing server or changing configuration.
modality_path = dataset / 'meta/modality.json'
modality = json.loads(modality_path.read_text()) if modality_path.exists() else {}
services = []
for camera, port in [('cam_left_head',8766), ('cam_left_wrist',8767), ('cam_right_wrist',8768)]:
    key = modality.get('video', {}).get(camera, {}).get('original_key', 'observation.images.rgb.'+camera)
    data = Dataset(dataset, key)
    if not data.episodes:
        raise SystemExit(f'Empty dataset: {dataset}')
    for episode, metadata in data.episodes.items():
        if not data.video(episode).is_file():
            raise SystemExit(f'Missing {camera} video: {data.video(episode)}')
        data.instruction(episode, 0)
        data.instruction(episode, metadata['length']-1)
    session = 'refilling-label-'+camera
    current = None
    try:
        current = get(port, '/api/state')
    except Exception:
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise SystemExit(f'Port {port} is occupied by an unrecognized/unresponsive service.')
    reuse = current is not None and Path(current['dataset']).resolve() == dataset and current['camera'] == key
    if current and not reuse:
        if not has_session(session):
            raise SystemExit(f'Port {port} is not managed by this launcher; stop its server first.')
        if any(e.get('status') == 'running' for e in current['episodes']):
            raise SystemExit(f'SAM is running on {current["dataset"]}; wait for completion before changing datasets.')
    services.append((camera, key, port, session, reuse))


def start(session, args, log):
    command = f'cd {shlex.quote(str(root))} && exec {shlex.join(args)} >> {shlex.quote(str(log))} 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)


for camera, key, port, session, reuse in services:
    if reuse:
        continue
    if has_session(session):
        subprocess.run(['tmux','kill-session','-t','='+session],check=True)
    config = logs / ('labeling_active_'+camera+'.json')
    atomic_json(config, {'dataset':str(dataset), 'camera':key, 'gpu':0, 'render_review_video':False, 'resident_socket':str(logs/'labeling_sam.sock')})
    start(session, [sys.executable,'-u','tools/labeling_app.py','--config',str(config),'--port',str(port)],
          logs / ('labeling_refilling_'+camera+'.log'))

session = 'refilling-label-dashboard'
if not has_session(session):
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',8769)) == 0:
            raise SystemExit('Port 8769 is already occupied. Check its service before retrying.')
    start(session, [sys.executable,'-u','tools/refilling_labeling_dashboard.py'], logs/'labeling_refilling_dashboard.log')

for attempt in range(90):
    try:
        from labeling_sam_client import request
        assert request(logs/'labeling_sam.sock',command='status')['ready']
        for route, (_, key, _, _, _) in zip(('head','left','right'),services):
            state = get(8769, f'/{route}/api/state')
            assert Path(state['dataset']).resolve() == dataset and state['camera'] == key
        print(f'Dataset: {dataset}\nLabeling ready: http://localhost:8769/\nForward port 8769 in VS Code. Refresh the page after changing datasets.')
        break
    except Exception as exc:
        if attempt == 89:
            raise SystemExit(f'Startup check failed: {exc}. See logs/labeling_refilling*.log')
        time.sleep(1)
PY
