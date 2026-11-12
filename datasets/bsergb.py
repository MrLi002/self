"""BS-ERGB training data loader.

The BS-ERGB release stores one RGB image per timestamp and one event file for
each adjacent image interval. Every skip-N window (endpoints ``start`` and
``end = start + N + 1``) is one training sample.

``target_mode="all"`` (default): a window returns **all N intermediate frames**
as targets, e.g. skip=3 with endpoints (0, 4) returns targets 1, 2 and 3 together
with one event split per target. The network is then run on the K = N targets of
the window (they are flattened into the batch dimension by the training script),
so stage 1 reconstructs K gray images and stages 2/3 predict K frames.

``target_mode="random"``: the previous behaviour, one randomly chosen target per
read (K = 1).

Every sample is a dict with K = number of targets::

    endpoints : (6, H, W)          I_0 and I_1, OpenCV BGR in [0, 1]
    gts       : (K, 3, H, W)       ground-truth intermediate frames (ascending time)
    voxels    : (K, 3*bins, H, W)  per target [E_0->t, E_t->1, E_1->t]
    masks     : (K, 1, H, W)       per target event mask
    t         : (K,)               normalised target times (target-start)/(end-start)

Windows with different K cannot be stacked into one batch; use
``SkipGroupedBatchSampler`` when batch_size > 1.
"""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from util.event import EventSequence, load_events
from util.utils_func import events_to_channels, process_mask
from util.voxelization import to_voxel_grid


CropBox = Tuple[int, int, int, int]
WindowIndex = Tuple[int, int, int, int]
TARGET_MODES = ("all", "random")


def _natural_name_key(path: Path) -> Tuple[object, ...]:
    """Sort numeric filenames numerically while keeping general names stable."""

    parts: List[object] = []
    token = ""
    is_digit: Optional[bool] = None
    for char in path.name:
        char_is_digit = char.isdigit()
        if is_digit is not None and char_is_digit != is_digit:
            parts.append((0, int(token)) if is_digit else (1, token.lower()))
            token = ""
        token += char
        is_digit = char_is_digit
    if token:
        parts.append((0, int(token)) if is_digit else (1, token.lower()))
    return tuple(parts)


def _list_files(folder: Path, suffixes: Iterable[str]) -> List[Path]:
    suffix_set = {suffix.lower() for suffix in suffixes}
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in suffix_set),
        key=_natural_name_key,
    )


def _load_event_chunks(paths: Sequence[Path], crop_box: CropBox) -> List[np.ndarray]:
    """Load and crop each interval file once (one array per adjacent-frame interval)."""
    return [load_events(str(path), bsergb=True, size=crop_box) for path in paths]


def _sequence_from_chunks(chunks: Sequence[np.ndarray], height: int, width: int) -> Optional[EventSequence]:
    """Concatenate interval chunks into one time-sorted EventSequence (None if empty)."""
    non_empty = [chunk for chunk in chunks if chunk.size]
    if not non_empty:
        return None
    features = np.concatenate(non_empty, axis=0)  # always a fresh copy
    if features.shape[0] > 1 and np.any(np.diff(features[:, 2]) < 0):
        features = features[np.argsort(features[:, 2], kind="stable")]
    return EventSequence(features, height, width)


def _load_sequence(paths: Sequence[Path], height: int, width: int, crop_box: CropBox) -> Optional[EventSequence]:
    """Load, crop, and concatenate an interval of BS-ERGB event files."""
    return _sequence_from_chunks(_load_event_chunks(paths, crop_box), height, width)


def _voxel_or_zeros(sequence: Optional[EventSequence], bins: int, height: int, width: int) -> torch.Tensor:
    if sequence is None or len(sequence) == 0 or sequence.duration() <= 0:
        return torch.zeros((bins, height, width), dtype=torch.float32)
    return to_voxel_grid(sequence, bins)


def _event_channels_or_zeros(sequence: Optional[EventSequence], height: int, width: int) -> torch.Tensor:
    if sequence is None or len(sequence) == 0:
        return torch.zeros((2, height, width), dtype=torch.float32)
    return events_to_channels(sequence).float()


class BSERGBTrainDataset(Dataset):
    """One sample per BS-ERGB temporal window, with all (or one random) targets.

    Expected layout::

        data_root/
          scene_name/
            images/*.png
            events/*.npz

    Images remain in OpenCV BGR channel order to match the existing DSER++
    inference scripts and released checkpoints.
    """

    def __init__(
        self,
        data_root: str | os.PathLike[str],
        bins: int = 8,
        skips: Sequence[int] = (1, 3),
        crop_size: int | Sequence[int] = 256,
        scenes: Optional[Sequence[str]] = None,
        augment: bool = True,
        rotate: bool = True,
        excluded_frame_indices: Optional[Dict[str, Sequence[int]]] = None,
        target_mode: str = "all",
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root).expanduser().resolve()
        self.bins = int(bins)
        self.skips = tuple(sorted({int(skip) for skip in skips}))
        self.augment = bool(augment)
        self.rotate = bool(rotate)
        self.target_mode = str(target_mode)
        self.excluded_frame_indices = {
            key: set(value) for key, value in (excluded_frame_indices or {}).items()
        }

        if isinstance(crop_size, int):
            self.crop_height = self.crop_width = crop_size
        else:
            if len(crop_size) != 2:
                raise ValueError("crop_size must be an int or a two-element sequence")
            self.crop_height, self.crop_width = map(int, crop_size)

        if self.target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}")
        if self.bins <= 0:
            raise ValueError("bins must be positive")
        if not self.skips or any(skip <= 0 for skip in self.skips):
            raise ValueError("skips must contain positive integers")
        if self.crop_height <= 0 or self.crop_width <= 0:
            raise ValueError("crop_size must be positive")
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"BS-ERGB root does not exist: {self.data_root}")

        scene_names = list(scenes) if scenes else sorted(
            path.name for path in self.data_root.iterdir() if path.is_dir()
        )
        if not scene_names:
            raise RuntimeError(f"No scene directories found under {self.data_root}")

        self.scene_metadata: List[Dict[str, object]] = []
        self.samples: List[WindowIndex] = []
        for scene_name in scene_names:
            self._register_scene(scene_name)

        if not self.samples:
            raise RuntimeError(
                "No valid BS-ERGB training samples were found. Check the scene "
                "layout, skip values, and excluded frame indices."
            )

    # ------------------------------------------------------------------ indexing
    def _register_scene(self, scene_name: str) -> None:
        scene_root = self.data_root / scene_name
        image_folder = scene_root / "images"
        event_folder = scene_root / "events"
        if not image_folder.is_dir() or not event_folder.is_dir():
            raise FileNotFoundError(
                f"Scene {scene_name!r} must contain images/ and events/ directories"
            )

        image_paths = _list_files(image_folder, (".png", ".jpg", ".jpeg"))
        event_paths = _list_files(event_folder, (".npz",))
        if len(image_paths) < 3:
            raise RuntimeError(f"Scene {scene_name!r} contains fewer than three images")
        if len(event_paths) < len(image_paths) - 1:
            raise RuntimeError(
                f"Scene {scene_name!r} has {len(image_paths)} images but only "
                f"{len(event_paths)} event intervals"
            )

        first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Failed to read image: {image_paths[0]}")
        height, width = first.shape[:2]
        if height < self.crop_height or width < self.crop_width:
            raise ValueError(
                f"Scene {scene_name!r} has resolution {width}x{height}, smaller "
                f"than crop {self.crop_width}x{self.crop_height}"
            )

        scene_index = len(self.scene_metadata)
        self.scene_metadata.append(
            {"name": scene_name, "images": image_paths, "events": event_paths,
             "height": height, "width": width}
        )

        excluded = self.excluded_frame_indices.get(scene_name, set())
        for skip in self.skips:
            endpoint_offset = skip + 1
            for start in range(len(image_paths) - endpoint_offset):
                end = start + endpoint_offset
                if any(frame in excluded for frame in range(start, end + 1)):
                    continue
                self.samples.append((scene_index, start, end, skip))

    def __len__(self) -> int:
        return len(self.samples)

    def num_targets(self, index: int) -> int:
        """K for sample ``index`` (skip in 'all' mode, 1 in 'random' mode)."""
        return self.samples[index][3] if self.target_mode == "all" else 1

    @property
    def mean_targets(self) -> float:
        return float(np.mean([self.num_targets(i) for i in range(len(self))]))

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _read_image(path: Path) -> torch.Tensor:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0

    def _choose_crop(self, height: int, width: int) -> CropBox:
        if self.augment:
            top = random.randint(0, height - self.crop_height)
            left = random.randint(0, width - self.crop_width)
        else:
            top = (height - self.crop_height) // 2
            left = (width - self.crop_width) // 2
        return top, left, top + self.crop_height, left + self.crop_width

    @staticmethod
    def _apply_spatial_transform(tensors: Sequence[torch.Tensor], vertical_flip: bool,
                                 horizontal_flip: bool, rotations: int) -> List[torch.Tensor]:
        """Apply the same flip/rotation to every tensor (any rank, spatial dims last)."""
        out = []
        for tensor in tensors:
            if vertical_flip:
                tensor = torch.flip(tensor, dims=(-2,))
            if horizontal_flip:
                tensor = torch.flip(tensor, dims=(-1,))
            if rotations:
                tensor = torch.rot90(tensor, rotations, dims=(-2, -1))
            out.append(tensor.contiguous())
        return out

    def __getitem__(self, index: int) -> Dict[str, object]:
        scene_index, start, end, skip = self.samples[index]
        if self.target_mode == "all":
            targets = list(range(start + 1, end))
        else:
            targets = [random.randrange(start + 1, end)]

        metadata = self.scene_metadata[scene_index]
        image_paths = metadata["images"]
        event_paths = metadata["events"]
        height, width = int(metadata["height"]), int(metadata["width"])
        h, w = self.crop_height, self.crop_width

        crop_box = self._choose_crop(height, width)
        top, left, bottom, right = crop_box

        def read(frame: int) -> torch.Tensor:
            return self._read_image(image_paths[frame])[:, top:bottom, left:right]

        endpoints = torch.cat((read(start), read(end)), dim=0)               # (6, h, w)
        gts = torch.stack([read(target) for target in targets], dim=0)        # (K, 3, h, w)

        # every interval file of the window is read once and shared by all targets
        chunks = _load_event_chunks(event_paths[start:end], crop_box)
        voxel_list, mask_list = [], []
        for target in targets:
            split = target - start                       # number of intervals in [start, target]
            events_0t = _sequence_from_chunks(chunks[:split], h, w)
            events_t1 = _sequence_from_chunks(chunks[split:], h, w)
            event_counts = torch.cat(
                (_event_channels_or_zeros(events_0t, h, w), _event_channels_or_zeros(events_t1, h, w)), dim=0)
            mask_list.append(process_mask(event_counts).float())
            events_1t = None if events_t1 is None else events_t1.copy().reverse()
            voxel_list.append(torch.cat((
                _voxel_or_zeros(events_0t, self.bins, h, w),
                _voxel_or_zeros(events_t1, self.bins, h, w),
                _voxel_or_zeros(events_1t, self.bins, h, w),
            ), dim=0).float())
        voxels = torch.stack(voxel_list, dim=0)                               # (K, 3*bins, h, w)
        masks = torch.stack(mask_list, dim=0)                                 # (K, 1, h, w)
        times = torch.tensor([(target - start) / float(end - start) for target in targets],
                             dtype=torch.float32)                             # (K,)

        if self.augment:
            vertical_flip = random.random() < 0.5
            horizontal_flip = random.random() < 0.5
            if self.rotate and h == w:
                rotations = random.randrange(4)
            elif self.rotate:
                rotations = random.randrange(2) * 2
            else:
                rotations = 0
            endpoints, gts, voxels, masks = self._apply_spatial_transform(
                (endpoints, gts, voxels, masks), vertical_flip, horizontal_flip, rotations)

        return {
            "endpoints": endpoints,
            "gts": gts,
            "voxels": voxels,
            "masks": masks,
            "t": times,
            "scene": str(metadata["name"]),
            "start_index": start,
            "end_index": end,
            "target_indices": torch.tensor(targets, dtype=torch.long),
            "skip": skip,
        }


class SkipGroupedBatchSampler(Sampler[List[int]]):
    """Shuffled batches in which all samples have the same number of targets K.

    Needed for batch_size > 1 with target_mode='all' (skip=1 -> K=1, skip=3 -> K=3).
    Within each group the order is reshuffled every epoch, then the batches of all
    groups are interleaved in random order.
    """

    def __init__(self, dataset: BSERGBTrainDataset, batch_size: int, drop_last: bool = False,
                 generator: Optional[torch.Generator] = None) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.generator = generator
        groups: Dict[int, List[int]] = defaultdict(list)
        for index in range(len(dataset)):
            groups[dataset.num_targets(index)].append(index)
        self.groups = dict(groups)

    def _num_batches(self, size: int) -> int:
        return size // self.batch_size if self.drop_last else math.ceil(size / self.batch_size)

    def __len__(self) -> int:
        return sum(self._num_batches(len(indices)) for indices in self.groups.values())

    def __iter__(self) -> Iterator[List[int]]:
        batches: List[List[int]] = []
        for indices in self.groups.values():
            order = torch.randperm(len(indices), generator=self.generator).tolist()
            shuffled = [indices[i] for i in order]
            for begin in range(0, len(shuffled), self.batch_size):
                batch = shuffled[begin:begin + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        for i in torch.randperm(len(batches), generator=self.generator).tolist():
            yield batches[i]


def build_bsergb_train_dataset(data_root: str | os.PathLike[str], bins: int = 8, skips: Sequence[int] = (1, 3), crop_size: int | Sequence[int] = 256, scenes: Optional[Sequence[str]] = None, augment: bool = True, target_mode: str = "all") -> BSERGBTrainDataset:
    """Convenience builder used by train_bsergb.py."""

    return BSERGBTrainDataset(
        data_root=data_root,
        bins=bins,
        skips=skips,
        crop_size=crop_size,
        scenes=scenes,
        augment=augment,
        target_mode=target_mode,
    )
