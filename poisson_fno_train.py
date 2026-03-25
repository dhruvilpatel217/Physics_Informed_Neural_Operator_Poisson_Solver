# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain
from typing import Dict
import os
import glob
from sympy import Symbol, Function

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from physicsnemo.utils.logging import LaunchLogger
from physicsnemo.utils.checkpoint import save_checkpoint
from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch
from omegaconf import DictConfig
from physicsnemo.utils.checkpoint import load_checkpoint
from torch.utils.data import DataLoader

from utils import HDF5MapStyleDataset


def keep_last_n(file_pattern, n=3):
            # Find all files matching the pattern and sort them by creation/epoch time
            files = sorted(glob.glob(file_pattern), key=os.path.getmtime)
            if len(files) > n:
                for f in files[:-n]:
                    try:
                        os.remove(f)
                    except OSError:
                        pass


def validation_step(graph, dataloader, epoch):
    """Validation Step"""

    with torch.no_grad():
        loss_epoch = 0
        for data in dataloader:
            invar, outvar, x_invar, y_invar = data
            out = graph.forward(
                {"rho_prime": invar[:, 0].unsqueeze(dim=1), "x": x_invar, "y": y_invar}
            )

            deepo_out_phi = out["phi"]

            loss_epoch += F.mse_loss(outvar, deepo_out_phi)

        # convert data to numpy
        outvar = outvar.detach().cpu().numpy()
        predvar = deepo_out_phi.detach().cpu().numpy()

        # plotting
        fig, ax = plt.subplots(1, 3, figsize=(25, 5))

        d_min = np.min(outvar[0, 0])
        d_max = np.max(outvar[0, 0])

        im = ax[0].imshow(outvar[0, 0], vmin=d_min, vmax=d_max)
        plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(predvar[0, 0], vmin=d_min, vmax=d_max)
        plt.colorbar(im, ax=ax[1])
        im = ax[2].imshow(np.abs(predvar[0, 0] - outvar[0, 0]))
        plt.colorbar(im, ax=ax[2])

        ax[0].set_title("True")
        ax[1].set_title("Pred")
        ax[2].set_title("Difference")

        fig.savefig(f"results_{epoch}.png")
        plt.close()
        return loss_epoch / len(dataloader)


class MdlsSymWrapper(Arch):
    """
    Wrapper model to convert PhysicsNeMo model to PhysicsNeMo-Sym model.

    PhysicsNeMo Sym relies on the inputs/outputs of the model being dictionary of tensors.
    This wrapper converts the input dictionary of tensors to a tensor inputs that can
    be processed by the PhysicsNeMo model that operate on tensors. Appropriate
    transformations are performed in the forward pass of the model to translate between
    these two input/output definitions.

    These transformations can differ based on the models. For e.g. typically for a fully
    connected network, the input tensors are combined by concatenating them along
    appropriate dimension before passing them as an input to the PhysicsNeMo model.
    During the output, the process is reversed, the output tensor from pytorch model is
    split across appropriate dimensions and then converted to a dictionary with
    appropriate keys to produce the final output.

    Having the model wrapped in a wrapper like this allows gradient computation using
    the PhysicsNeMo Sym's optimized gradient computing backend.

    For more details on PhysicsNeMo Sym models, refer:
    https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/tutorials/simple_training_example.html#using-custom-models-in-physicsnemo
    For more details on Key class, refer:
    https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.html#module-physicsnemo.sym.key
    """

    def __init__(
        self,
        input_keys=[Key("rho"), Key("x"), Key("y")],
        output_keys=[Key("phi")],
        trunk_net=None,
        branch_net=None,
    ):
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
        )

        self.branch_net = branch_net
        self.trunk_net = trunk_net

    def forward(self, dict_tensor: Dict[str, torch.Tensor]):
        # Concatenate x, y inputs to feeed in the trunk network which has a MLP
        xy_input_shape = dict_tensor["x"].shape
        xy = self.concat_input(
            {
                rho: dict_tensor[rho].view(xy_input_shape[0], -1, 1) for rho in ["x", "y"]
            },  # flatten the coordinate dimensions
            ["x", "y"],
            detach_dict=self.detach_key_dict,
            dim=-1,  # concat along the last dimension to form the feature vector.
        )
        fc_out = self.trunk_net(xy)

        # Pass the rho-prime for the FNO input
        fno_out = self.branch_net(dict_tensor["rho_prime"])

        # reshape the fc_out
        fc_out = fc_out.view(
            xy_input_shape[0], -1, xy_input_shape[-2], xy_input_shape[-1]
        )

        # multiply the outputs of branch and trunk networks to get the final output
        out = fc_out * fno_out

        return self.split_output(
            out, self.output_key_dict, dim=1
        )  # Split along the channel dimension to get a dictionary of tensors


@hydra.main(version_base="1.3", config_path="conf", config_name="config_deeponet.yaml")
def main(cfg: DictConfig):
    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    LaunchLogger.initialize()

    # Use Diffusion equation for the Poisson PDE
    rho_norm = 5.83456e+11  
    phi_norm = 4.12442e+00  

    original_eps = 8.854e-12
    coeff = (phi_norm / (original_eps*rho_norm))
    x, y = Symbol("x"), Symbol("y")
    rho_var = Function('rho')(x, y)

    # Create the source term Q = coeff * rho
    # print(coeff)
    Q_scaled = coeff * rho_var

    # poison = Diffusion(T='phi', D=eps, Q='rho', dim=2, time=False)
    poison = Diffusion(T='phi', D=1.0, Q=Q_scaled, dim=2, time=False)
    poison.pprint()
    
    # forcing_fn = 1.0 * 4.49996e00 * 3.88433e-03  # after scaling
    # darcy = Diffusion(T="u", time=False, dim=2, D="k", Q=forcing_fn)
    # darcy.pprint()

    dataset = HDF5MapStyleDataset(
        to_absolute_path("./train.hdf5"), device=device
    )
    validation_dataset = HDF5MapStyleDataset(
        to_absolute_path("./validation.hdf5"), device=device
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=False)
    
    model_branch = FNO(
        in_channels=cfg.model.fno.in_channels,
        out_channels=cfg.model.fno.out_channels,
        decoder_layers=cfg.model.fno.decoder_layers,
        decoder_layer_size=cfg.model.fno.decoder_layer_size,
        dimension=cfg.model.fno.dimension,
        latent_channels=cfg.model.fno.latent_channels,
        num_fno_layers=cfg.model.fno.num_fno_layers,
        num_fno_modes=cfg.model.fno.num_fno_modes,
        padding=cfg.model.fno.padding,
    )

    model_trunk = FullyConnected(
        in_features=cfg.model.fc.in_features,
        out_features=cfg.model.fc.out_features,
        layer_size=cfg.model.fc.layer_size,
        num_layers=cfg.model.fc.num_layers,
    )

    # Poisson solver (changed)
    model = MdlsSymWrapper(
        input_keys=[Key("rho_prime"), Key("x"), Key("y")],
        output_keys=[Key("phi")],
        trunk_net=model_trunk,
        branch_net=model_branch,
    ).to(device)

    # darcy = Diffusion(T="u", time=False, dim=2, D="k", Q=forcing_fn)
    # Sym internally registers the residual output as:
    # "diffusion" + "_" + "u"  =  "diffusion_u"


    # Poisson (changed)
    phy_informer = PhysicsInformer(
        required_outputs=["diffusion_phi"],
        equations=poison,
        grad_method="autodiff",
        device=device,
    )

    optimizer = torch.optim.Adam(
        chain(model_branch.parameters(), model_trunk.parameters()),
        betas=(0.9, 0.999),
        lr=cfg.start_lr,
        weight_decay=0.0,
        fused=True if torch.cuda.is_available() else False,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    # # 5. Load Trained Weights if they exist
    # ckpt_path = to_absolute_path("./outputs_poisson/checkpoints")
    
    # # This will check for existing .pt or .mdlus files in the directory
    # # It safely skips if the directory is empty (first run)
    # try:
    #     # We pass model_branch and model_trunk because they were passed to save_checkpoint
    #     load_checkpoint(
    #         ckpt_path, 
    #         models=[model_branch, model_trunk], 
    #         optimizer=optimizer, 
    #         scheduler=scheduler, 
    #         device=device
    #     )
    #     print(">>> Resuming training from last saved checkpoint.")
    # except Exception as e:
    #     print(f">>> No valid checkpoint found or error loading: {e}")
    #     print(">>> Starting training from scratch.")

    for epoch in range(cfg.max_epochs):
        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader),
            epoch_alert_freq=10,
        ) as log:
            for data in dataloader:
                optimizer.zero_grad()
                invar = data[0][:, 0].unsqueeze(dim=1)
                outvar = data[1]

                coords = torch.stack([data[2], data[3]], dim=1).requires_grad_(True)
                # compute forward pass
                out = model.forward(
                    {
                        "rho_prime": invar,
                        "x": coords[:, 0:1],
                        "y": coords[:, 1:2],
                    }
                )

                # --- NEW: HARD ENFORCEMENT OF BOUNDARY CONDITIONS ---
                # This ensures the boundary is strictly 0.0
                # Assuming out["phi"] shape is [B, 1, 257, 257]
                # mask = torch.ones_like(out["phi"])
                # mask[:, :, 0, :] = 0   # Top
                # mask[:, :, -1, :] = 0  # Bottom
                # mask[:, :, :, 0] = 0   # Left
                # mask[:, :, :, -1] = 0  # Right
                
                # Apply mask to the output
                # out["phi"] = out["phi"] * mask

                residuals = phy_informer.forward(
                    {
                        "coordinates": coords,
                        "phi": out["phi"],
                        "rho": invar,
                    }
                )
                pde_out_arr = residuals["diffusion_phi"]

                # Boundary condition
                pde_out_arr = F.pad(
                    pde_out_arr[..., 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0
                )
                loss_pde = F.l1_loss(pde_out_arr, torch.zeros_like(pde_out_arr))

                # Compute data loss
                deepo_out_phi = out["phi"]
                loss_data = F.mse_loss(outvar, deepo_out_phi) 

                # Compute total loss
                loss = loss_data + cfg.physics_weight * loss_pde

                # Backward pass and optimizer and learning rate update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                log.log_minibatch(
                    {"loss_data": loss_data.detach(), "loss_pde": loss_pde.detach()}
                )

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        with LaunchLogger("valid", epoch=epoch) as log:
            error = validation_step(model, validation_dataloader, epoch)
            log.log_epoch({"Validation error": error})

        save_checkpoint(
            "./checkpoints",
            models=[model_branch, model_trunk],
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )

        # 2. Cleanup: Keep only the most recent 3 epochs
        # PhysicsNeMo typically names the main checkpoint as 'checkpoint.0.{epoch}.pt'
        checkpoint_dir = "./checkpoints"
        all_pt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint.0.*.pt"))
        
        # Sort by epoch number extracted from the filename to be safe
        # Filename example: ./checkpoints/checkpoint.0.12.pt -> 12
        all_pt_files.sort(key=lambda x: int(x.split('.')[-2]))

        if len(all_pt_files) > 3:
            # Identify the files belonging to epochs we want to delete
            files_to_delete = all_pt_files[:-3]
            
            for f in files_to_delete:
                # Extract the epoch number of the old checkpoint
                old_epoch = f.split('.')[-2]
                
                # A. Delete the .pt file
                if os.path.exists(f):
                    os.remove(f)
                
                # B. Delete all model weights (.mdlus) associated with this specific old epoch
                # Pattern matches FNO.0.{old_epoch}.mdlus and FullyConnected.0.{old_epoch}.mdlus
                model_files = glob.glob(os.path.join(checkpoint_dir, f"*.0.{old_epoch}.mdlus"))
                for m in model_files:
                    if os.path.exists(m):
                        os.remove(m)
            
            print(f">>> Cleanup: Kept epochs {[f.split('.')[-2] for f in all_pt_files[-3:]]}")

        # 3. Cleanup Results Images: Keep only the last 3
        results_images = sorted(glob.glob("./results_*.png"), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if len(results_images) > 3:
            for img in results_images[:-3]:
                os.remove(img)


if __name__ == "__main__":
    main()
