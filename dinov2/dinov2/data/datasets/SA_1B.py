# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import cv2
import logging
import os
import glob
import numpy as np

from .extended import ExtendedVisionDataset
from typing import Callable, List, Optional, Tuple, Union, Any

logger = logging.getLogger("dinov2")




class SA_1B(ExtendedVisionDataset):

    def __init__(
        self,
        *,
        root: str= "SA-1B",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root_dir = glob.glob('./SA-1B/*.jpg')

    def __len__(self) -> int:
        return len(self.root_dir)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        image = cv2.imread(self.root_dir[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = 0
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
