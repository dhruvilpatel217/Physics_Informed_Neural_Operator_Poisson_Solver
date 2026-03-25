# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5MapStyleDataset(Dataset):
    """Simple map-style HDF5 dataset for Poisson Solver"""

    def __init__(
        self,
        file_path,
        device: Union[str, torch.device] = "cuda",
    ):
        self.file_path = file_path
        with h5py.File(file_path, "r") as f:
            self.keys = list(f.keys())
            print(f"HDF5 keys found: {self.keys}")

        # Set up device
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and device.index == None:
            device = torch.device("cuda:0")
        self.device = device

    def __len__(self):
        with h5py.File(self.file_path, "r") as f:
            return len(f[self.keys[0]])

    def __getitem__(self, idx):
        data = {}
        with h5py.File(self.file_path, "r") as f:
            for key in self.keys:
                data[key] = np.array(f[key][idx])

        # rho_norm = np.max(np.abs(data["rho"])) #global max
        # phi_norm = np.max(np.abs(data["potential"]))
        rho_norm = 5.83456e+11  
        phi_norm = 4.12442e+00
        original_eps = 8.854e-12

        invar = torch.from_numpy(
            (data["rho"][:, :257, :257]).astype(np.float32)
        ) / rho_norm                                  

        # ── CHANGE 4: Output is phi instead of sol ────────────────────────────
        outvar = torch.from_numpy(
            (data["potential"][:, :257, :257]).astype(np.float32)
        ) / phi_norm                                    

        # Your Poisson dataset is 257×257.
        x = np.linspace(0, 1, 257)
        y = np.linspace(0, 1, 257)

        xx, yy = np.meshgrid(x, y)
        x_invar = torch.from_numpy(xx.astype(np.float32)).view(1, 257, 257)
        y_invar = torch.from_numpy(yy.astype(np.float32)).view(1, 257, 257)

        if self.device.type == "cuda":
            invar   = invar.cuda()
            outvar  = outvar.cuda()
            x_invar = x_invar.cuda()
            y_invar = y_invar.cuda()

        return invar, outvar, x_invar, y_invar
