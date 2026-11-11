"""Add sub_tasks from subtasks.parquet for the legacy arms recipe.

Requires dataset-specific meta/modality.json; never guesses robot channel ranges.
Original episodes.jsonl is backed up before updating. Run explicitly:
    python tools/prepare_dataset.py data/<dataset>
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

def episode_parquet_path(root: Path, info: dict, episode_index: int) -> Path:
    chunk = episode_index // info.get("chunks_size", 1000)
    pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    return root / pattern.format(episode_chunk=chunk, episode_index=episode_index)


def build_sub_tasks(df: pd.DataFrame, subtask_map: dict[int, str]) -> list[dict]:
    """연속된 subtask_index 구간을 [{start, end, text}] 목록으로 묶는다."""
    idx = df["subtask_index"].to_numpy()
    segments = []
    start = 0
    for i in range(1, len(idx) + 1):
        if i == len(idx) or idx[i] != idx[start]:
            segments.append(
                {"start": int(start), "end": int(i), "text": subtask_map[int(idx[start])]}
            )
            start = i
    return segments


def prepare(root: Path) -> None:
    meta = root / "meta"
    episodes_path = meta / "episodes.jsonl"
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]

    if not (meta / "modality.json").exists():
        raise ValueError(f"Missing {meta / 'modality.json'}; supply this dataset's channel metadata first")

    if all("sub_tasks" in ep for ep in episodes):
        print(f"  episodes.jsonl 에 sub_tasks 필드가 이미 있음 — 건너뜀")
        return

    subtask_df = pd.read_parquet(meta / "subtasks.parquet")
    subtask_map = dict(zip(subtask_df["subtask_index"], subtask_df["subtask"]))
    info = json.loads((meta / "info.json").read_text())

    for ep in episodes:
        if "sub_tasks" in ep:
            continue
        df = pd.read_parquet(episode_parquet_path(root, info, ep["episode_index"]))
        ep["sub_tasks"] = build_sub_tasks(df, subtask_map)

    shutil.copy(episodes_path, episodes_path.with_suffix(".jsonl.bak"))
    episodes_path.write_text("".join(json.dumps(ep) + "\n" for ep in episodes))
    n_seg = sum(len(ep["sub_tasks"]) for ep in episodes)
    print(f"  sub_tasks 생성 완료: 에피소드 {len(episodes)}개, 구간 {n_seg}개 (원본은 episodes.jsonl.bak)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        root = Path(arg)
        print(f"처리 중: {root}")
        prepare(root)
