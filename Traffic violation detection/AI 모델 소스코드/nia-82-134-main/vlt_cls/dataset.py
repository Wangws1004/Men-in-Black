# coding: utf-8
import os
from typing import Any, Tuple

import torch
from PIL import Image


class VltDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.classes, self.class_to_idx = self.find_classes()
        self.samples = self.make_dataset()
        self.targets = [s[1] for s in self.samples]

    def find_classes(self):
        classes = sorted(e.name for e in os.scandir(self.data_dir) if e.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def make_dataset(self):
        def is_valid_file(fname):
            return fname.lower().endswith("jpg")

        instances = []
        available_classes = set()
        for target_class in sorted(self.classes):
            class_idx = self.class_to_idx[target_class]
            target_dir = os.path.join(self.data_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if is_valid_file(fname):
                        path = os.path.join(root, fname)
                        instances.append((path, class_idx))
                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(self.classes) - available_classes
        if empty_classes:
            msg = (
                f"Found no valid file for the classes "
                f"{', '.join(sorted(empty_classes))}. "
            )
            print(msg)

        return instances

    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        path, target = self.samples[index]
        sample = self.pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return os.path.basename(path).replace(".jpg", ""), sample, target

    def __len__(self) -> int:
        return len(self.samples)
