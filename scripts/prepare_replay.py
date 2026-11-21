"""Extract explicit past/current samples; optional fixed crop is diagnostic only."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2


CAMERAS = {"cam0": "head", "cam1": "right_wrist", "cam2": "left_wrist"}


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(dataset, episode, start, end, period, output, crop_box=None):
    if not all(math.isfinite(v) for v in (start, end, period)) or not 0 <= start < end:
        raise ValueError("Expected finite start < end")
    if end - start > 12 or period < 0.2:
        raise ValueError("At most 12 seconds, sampling period >= 0.2 seconds")
    if type(episode) is not int or episode < 0:
        raise ValueError("Expected nonnegative episode")
    if crop_box is not None and (
        len(crop_box) != 4 or not all(math.isfinite(v) for v in crop_box)
        or not 0 <= crop_box[0] < crop_box[2] <= 1
        or not 0 <= crop_box[1] < crop_box[3] <= 1
    ):
        raise ValueError("Crop must be normalized finite xyxy")
    info = json.loads((dataset / "meta/info.json").read_text())
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid dataset FPS")
    indices = sorted({round((start + n * period) * fps)
                      for n in range(math.floor((end - start) / period + 1e-8) + 1)})
    if any(i / fps > end + 1e-8 for i in indices):
        raise ValueError("Rounded frame exceeds observation cutoff")
    output.mkdir(parents=True, exist_ok=False)
    rows, sources = [], {}
    for camera, view in CAMERAS.items():
        key = "observation.images." + camera
        path = dataset / info["video_path"].format(
            episode_chunk=episode // info.get("chunks_size", 1000),
            video_key=key, episode_index=episode,
        )
        sources[camera] = {"path": str(path), "sha256": file_sha(path)}
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened() or indices[-1] >= capture.get(cv2.CAP_PROP_FRAME_COUNT):
                raise ValueError(f"Unavailable video/window: {camera}")
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, pixels = capture.read()
                if not ok:
                    raise ValueError(f"Missing frame {camera}:{index}")
                # Keep dataset-native image dimensions and pixel provenance.
                name = f"{camera}-{index}.png"
                if not cv2.imwrite(str(output / name), pixels):
                    raise ValueError("Image write failed")
                row = {"view": view, "path": name, "source_camera": camera,
                       "source_frame": index, "source_timestamp_s": index / fps,
                       "width": pixels.shape[1], "height": pixels.shape[0],
                       "sha256": file_sha(output / name)}
                rows.append(row)
                if camera == "cam0" and crop_box is not None:
                    h, w = pixels.shape[:2]
                    box = [math.floor(crop_box[0] * w), math.floor(crop_box[1] * h),
                           math.ceil(crop_box[2] * w), math.ceil(crop_box[3] * h)]
                    name = f"head-crop-{index}.png"
                    if not cv2.imwrite(str(output / name), pixels[box[1]:box[3], box[0]:box[2]]):
                        raise ValueError("Crop write failed")
                    rows.append({**row, "view": "head_crop", "path": name,
                                 "width": box[2] - box[0], "height": box[3] - box[1],
                                 "sha256": file_sha(output / name),
                                 "crop": {"parent_path": row["path"], "xyxy_pixels": box,
                                          "source_size": [w, h], "mode": "fixed_diagnostic",
                                          "uses_future_coordinates": False}})
        finally:
            capture.release()
    rows.sort(key=lambda r: (r["source_timestamp_s"], r["view"]))
    for i, row in enumerate(rows):
        row["i"] = i
    manifest = {"schema_version": 1, "purpose": "debug_replay_not_independent_test",
                "dataset": str(dataset), "episode": episode, "fps": fps,
                "cutoff_s": end, "sampling_period_s": period,
                "timestamp_basis": "dataset_frame_index_divided_by_fps_not_live_ROS",
                "camera_mapping": CAMERAS, "sources": sources, "frames": rows,
                "roi_status": "fixed_diagnostic" if crop_box is not None else "no_crop"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--episode", required=True, type=int)
    p.add_argument("--start", required=True, type=float)
    p.add_argument("--end", required=True, type=float)
    p.add_argument("--period", type=float, default=0.5)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--fixed-crop", type=float, nargs=4)
    a = p.parse_args()
    m = prepare(a.dataset.expanduser().resolve(), a.episode, a.start, a.end,
                a.period, a.output.expanduser().resolve(), a.fixed_crop)
    print(json.dumps({"frames": len(m["frames"]), "roi_status": m["roi_status"]}))


if __name__ == "__main__":
    main()
