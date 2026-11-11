"""Remove a clicked connected component; temporal erasure fails on ambiguity."""

import cv2
import numpy as np


def component(mask, point):
    x, y = map(int, point)
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        raise ValueError("Point outside image")
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    value = labels[y, x]
    if value == 0:
        raise ValueError("Click inside a mask component")
    return labels == value


def erase_component(objects, frames, anchor, point, scope, progress=None):
    ids = [key for key, masks in objects.items() if masks[anchor, int(point[1]), int(point[0])]]
    if len(ids) != 1:
        raise ValueError("Erasure is ambiguous; select a non-overlapping component")
    key = ids[0]
    selected = component(objects[key][anchor], point)
    masks = objects[key]
    masks[anchor][selected] = 0
    if scope == "frame":
        return objects
    for direction in (1, -1):
        previous = selected.copy()
        old_frame = frames[anchor]
        for i in range(anchor + direction, len(masks) if direction > 0 else -1, direction):
            new_frame = frames[i]
            old_gray, new_gray = (
                cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY),
            )
            flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
            yy, xx = np.mgrid[: previous.shape[0], : previous.shape[1]].astype(np.float32)
            warped = cv2.remap(
                previous.astype(np.uint8), xx - flow[..., 0], yy - flow[..., 1], cv2.INTER_NEAREST
            ).astype(bool)
            if not warped.any():
                raise ValueError("Component tracking lost; use frame-only correction")
            _, labels = cv2.connectedComponents(masks[i].astype(np.uint8), connectivity=8)
            values = labels[warped]
            values = values[values > 0]
            if not len(values):
                raise ValueError(
                    "Component not visible; temporal erasure requires manual correction"
                )
            counts = np.bincount(values)
            label = counts.argmax()
            selected_next = labels == label
            overlap = (selected_next & warped).sum()
            if overlap / max(1, selected_next.sum()) < 0.5 or overlap / warped.sum() < 0.5:
                raise ValueError("Component merge/loss detected; refuse whole-video erasure")
            masks[i][selected_next] = 0
            previous, old_frame = selected_next, new_frame
            if progress:
                progress(i)
    return objects
