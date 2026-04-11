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

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from physicsnemo.utils.logging import LaunchLogger
from physicsnemo.utils.checkpoint import save_checkpoint, load_checkpoint
from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from sympy import symbols, Function
from skimage.metrics import structural_similarity as ssim_metric

from utils1 import HDF5MapStyleDataset


def collect_epoch_weight_files(checkpoint_dir, epoch):
    epoch_str = str(epoch)
    patterns = [
        os.path.join(checkpoint_dir, f"checkpoint.0.{epoch_str}.pt"),
        os.path.join(checkpoint_dir, f"*.0.{epoch_str}.mdlus"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(files)


def append_final_metric_summary(log_path, epochs, metric_history, epoch_weight_files):
    if not epochs:
        return

    last_epoch  = epochs[-1]
    has_previous = len(epochs) > 1

    lines = []
    lines.append("\n===== Final Metric Epoch Summary =====")
    lines.append("Rule: best epoch is minimum metric value among epochs excluding the last epoch.")

    for metric_name, values in metric_history.items():
        last_value   = values[-1]
        last_weights = epoch_weight_files.get(last_epoch, [])
        last_weights_str = ", ".join(last_weights) if last_weights else "N/A"

        lines.append(f"Metric: {metric_name}")
        lines.append(
            f"  Last Epoch: {last_epoch} | Value: {last_value:.10e} | Weights: {last_weights_str}"
        )

        if has_previous:
            prev_values  = values[:-1]
            best_idx     = int(np.argmin(np.asarray(prev_values)))
            best_epoch   = epochs[best_idx]
            best_value   = prev_values[best_idx]
            best_weights = epoch_weight_files.get(best_epoch, [])
            best_weights_str = ", ".join(best_weights) if best_weights else "N/A"
            lines.append(
                f"  Best Previous Epoch: {best_epoch} | Value: {best_value:.10e} | Weights: {best_weights_str}"
            )
        else:
            lines.append("  Best Previous Epoch: N/A (need at least 2 total epochs)")

    lines.append("===== End Final Metric Epoch Summary =====\n")

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def validation_step(graph, dataloader, epoch):
    """Validation Step with MSE, SSIM, APE metrics"""

    with torch.no_grad():
        loss_epoch = 0.0
        mse_epoch  = 0.0
        ssim_epoch = 0.0
        ape_epoch  = 0.0
        count      = 0

        for data in dataloader:
            invar, outvar, x_invar, y_invar = data
            out = graph.forward(
                {"rho_prime": invar[:, 0].unsqueeze(dim=1), "x": x_invar, "y": y_invar}
            )
            deepo_out_phi = out["phi"]

            loss_epoch += F.mse_loss(outvar, deepo_out_phi).item()

            # Convert to numpy
            true_np = outvar.detach().cpu().numpy()
            pred_np = deepo_out_phi.detach().cpu().numpy()

            # --- MSE ---
            mse_epoch += np.mean((true_np - pred_np) ** 2)

            # --- SSIM via skimage ---
            batch_ssim = 0.0
            for b in range(true_np.shape[0]):
                sample_ssim = 0.0
                for c in range(true_np.shape[1]):
                    t          = true_np[b, c]
                    p          = pred_np[b, c]
                    data_range = t.max() - t.min()
                    if data_range == 0:
                        data_range = 1.0
                    sample_ssim += ssim_metric(t, p, data_range=data_range)
                batch_ssim += sample_ssim / true_np.shape[1]
            ssim_epoch += batch_ssim / true_np.shape[0]

            # --- APE (true=0 replaced by 1 in denominator) ---
            true_denom  = np.where(true_np == 0, 1.0, true_np)
            ape_epoch  += np.mean(np.abs(true_np - pred_np) / np.abs(true_denom)) * 100

            count += 1

        # Average over all batches
        mse_val  = mse_epoch  / count
        ssim_val = ssim_epoch / count
        ape_val  = ape_epoch  / count

        # Use last batch for visualisation
        outvar_np  = outvar.detach().cpu().numpy()
        predvar_np = deepo_out_phi.detach().cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(25, 5))
        d_min = np.min(outvar_np[0, 0])
        d_max = np.max(outvar_np[0, 0])

        im = ax[0].imshow(outvar_np[0, 0], vmin=d_min, vmax=d_max)
        plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(predvar_np[0, 0], vmin=d_min, vmax=d_max)
        plt.colorbar(im, ax=ax[1])
        im = ax[2].imshow(np.abs(predvar_np[0, 0] - outvar_np[0, 0]))
        plt.colorbar(im, ax=ax[2])

        ax[0].set_title("True")
        ax[1].set_title("Pred")
        ax[2].set_title("Difference")

        fig.savefig(f"results_{epoch}.png")
        plt.close()

        return loss_epoch / count, mse_val, ssim_val, ape_val


class MdlsSymWrapper(Arch):
    """
    Wrapper model to convert PhysicsNeMo model to PhysicsNeMo-Sym model.
    """

    def __init__(
        self,
        input_keys=[Key("rho"), Key("x"), Key("y")],
        output_keys=[Key("rho_prime"),Key("phi")],
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
        xy_input_shape = dict_tensor["x"].shape
        xy = self.concat_input(
            {
                rho: dict_tensor[rho].view(xy_input_shape[0], -1, 1) for rho in ["x", "y"]
            },
            ["x", "y"],
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        fc_out = self.trunk_net(xy)
        fno_out = self.branch_net(dict_tensor["rho_prime"])

        fc_out = fc_out.view(
            xy_input_shape[0], -1, xy_input_shape[-2], xy_input_shape[-1]
        )
        out = fc_out * fno_out

        return self.split_output(
            out, self.output_key_dict, dim=1
        )


@hydra.main(version_base="1.3", config_path="conf1", config_name="config_deeponet.yaml")
def main(cfg: DictConfig):
    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    LaunchLogger.initialize()

    # Use Diffusion equation for the Poisson PDE
    # rho_norm     = 1.16147e-08  #case2/out0
    # phi_norm     = 6.55763e-01
    original_eps = 8.854e-12    
    rho_norm     = 1.23223e-08  #complete data
    phi_norm     = 9.21588e-01
    
    coeff = (rho_norm / (original_eps * phi_norm))
    forcing_scale = float(coeff) # We will use this to normalize the PDE residual

    x, y = symbols('x y')
    rho1 = Function('rho')(x, y)
    rho1 = rho1 * coeff

    poison = Diffusion(T="phi", D=1.0, Q=rho1, dim=2, time=False)
    poison.pprint()

    dataset = HDF5MapStyleDataset(
        to_absolute_path("./train1.hdf5"), device=device
    )
    validation_dataset = HDF5MapStyleDataset(
        to_absolute_path("./validation1.hdf5"), device=device
    )

    # Increased batch size to 8 per config alignment
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

    model = MdlsSymWrapper(
        input_keys=[Key("rho_prime"), Key("x"), Key("y")],
        output_keys=[Key("rho"),Key("phi")],
        trunk_net=model_trunk,
        branch_net=model_branch,
    ).to(device)

    phy_informer = PhysicsInformer(
        required_outputs=["diffusion_phi"],
        equations=poison,
        grad_method="autodiff", # Or "finite_difference" if you prefer
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

    # ── NEW: Checkpoint Resumption Logic ─────────────────────────────────────
    start_epoch = 0
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    pt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint.0.*.pt"))
    
    if pt_files:
        pt_files.sort(key=lambda x: int(x.split('.')[-2]))
        latest_ckpt = pt_files[-1]
        loaded_epoch = int(latest_ckpt.split('.')[-2])
        
        print(f"\n===========================================")
        print(f"Found existing weights: {latest_ckpt}")
        print(f"Attempting to resume training...")
        
        try:
            load_checkpoint(
                checkpoint_dir,
                models=[model_branch, model_trunk],
                optimizer=optimizer,
                scheduler=scheduler,
                device=device
            )
            # --- ADD THESE LINES TO OVERRIDE THE LR ---
            new_lr = 0.001  # Set your desired new learning rate here
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Success! Resuming from epoch {start_epoch} with manual LR: {new_lr}")
            # ------------------------------------------
            start_epoch = loaded_epoch + 1
            print(f"Success! Resuming from epoch {start_epoch} / {cfg.max_epochs}")
        except Exception as e:
            print(f"Failed to load checkpoint natively: {e}")
            print("Starting training from scratch.")
            start_epoch = 0
        print(f"===========================================\n")
    else:
        print("\nNo existing checkpoints found. Starting training from scratch.\n")
    # ─────────────────────────────────────────────────────────────────────────

    # ── Metric history for plotting ───────────────────────────────────────────
    epoch_list         = []
    val_error_list     = []
    mse_list           = []
    ssim_list          = []
    ape_list           = []
    pde_list           = []
    epoch_weight_files = {}
    # ─────────────────────────────────────────────────────────────────────────

    for epoch in range(start_epoch, cfg.max_epochs):

        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader),
            epoch_alert_freq=10,
        ) as log:
            pde_loss_train = 0.0   
            train_count    = 0     

            for data in dataloader:
                optimizer.zero_grad()
                invar = data[0][:, 0].unsqueeze(dim=1)
                outvar = data[1]

                coords = torch.stack([data[2], data[3]], dim=1).requires_grad_(True)
                
                out = model.forward(
                    {
                        "rho_prime": invar,
                        "x": coords[:, 0:1],
                        "y": coords[:, 1:2],
                    }
                )

                # # Inside training loop
                # pred_phi = out["phi"]
                # # Select only the edge pixels
                # top_edge = pred_phi[:, :, 0, :]
                # bottom_edge = pred_phi[:, :, -1, :]
                # left_edge = pred_phi[:, :, :, 0]
                # right_edge = pred_phi[:, :, :, -1]

                # loss_bc = F.mse_loss(top_edge, torch.zeros_like(top_edge)) + \
                #           F.mse_loss(bottom_edge, torch.zeros_like(bottom_edge)) + \
                #           F.mse_loss(left_edge, torch.zeros_like(left_edge)) + \
                #           F.mse_loss(right_edge, torch.zeros_like(right_edge))

                # --- HARD ENFORCEMENT OF BOUNDARY CONDITIONS ---
                mask = torch.ones_like(out["phi"])
                mask[:, :, 0, :] = 0   # Top
                mask[:, :, -1, :] = 0  # Bottom
                mask[:, :, :, 0] = 0   # Left
                mask[:, :, :, -1] = 0  # Right
                
                out["phi"] = out["phi"] * mask

                residuals = phy_informer.forward(
                    {
                        "coordinates": coords,
                        "phi": out["phi"],
                        "rho": out["rho"],
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
                deepo_out_rho = out["rho"]
                loss_data = F.mse_loss(outvar, deepo_out_phi) + + F.mse_loss(
                    data[0][:, 0].unsqueeze(dim=1), deepo_out_rho
                )

                # Compute total loss
                loss = loss_data + (cfg.physics_weight * loss_pde) 

                loss.backward()
                optimizer.step()
                scheduler.step()  

                pde_loss_train += loss_pde.detach().item()   
                train_count    += 1                              

                # log.log_minibatch(
                #     {"loss_data": loss_data.detach(), "loss_pde": loss_pde.detach()}
                # )
                log.log_minibatch({
                    "loss_phi": F.mse_loss(deepo_out_rho, outvar).detach(),
                    "loss_rho": F.mse_loss(deepo_out_phi, data[0][:, 0].unsqueeze(dim=1)).detach(),
                    "loss_data": loss_data.detach(),
                    "loss_pde": loss_pde.detach()
                })
            
            # Step the scheduler at the END of the epoch
            
            
            pde_epoch_avg = pde_loss_train / train_count
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # Validation loop
        with LaunchLogger("valid", epoch=epoch) as log:
            error, mse_val, ssim_val, ape_val = validation_step(
                model, validation_dataloader, epoch
            )
            log.log_epoch({
                "Validation error" : error,
                "MSE"              : mse_val,
                "SSIM"             : ssim_val,
                "APE (%)"          : ape_val,
            })

        # ── Append metrics and save updated plots ─────────────────────────────
        epoch_list.append(epoch)
        val_error_list.append(float(error))
        mse_list.append(float(mse_val))
        ssim_list.append(float(ssim_val))
        ape_list.append(float(ape_val))
        pde_list.append(float(pde_epoch_avg))

        fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        axes[0].plot(epoch_list, mse_list,  marker='o', color='steelblue', linewidth=2)
        axes[0].set_title("MSE vs Epoch",      fontsize=14)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE")
        axes[0].grid(True)

        axes[1].plot(epoch_list, ssim_list, marker='o', color='seagreen',  linewidth=2)
        axes[1].set_title("SSIM vs Epoch",     fontsize=14)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("SSIM")
        axes[1].grid(True)

        axes[2].plot(epoch_list, ape_list,  marker='o', color='tomato',    linewidth=2)
        axes[2].set_title("APE (%) vs Epoch",  fontsize=14)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("APE (%)")
        axes[2].grid(True)

        axes[3].plot(epoch_list, pde_list,  marker='o', color='purple',    linewidth=2)
        axes[3].set_title("PDE Loss vs Epoch", fontsize=14)
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("PDE Loss")
        axes[3].grid(True)

        plt.tight_layout()
        plt.savefig("metrics_plot.png", dpi=150)
        plt.close()
        # ─────────────────────────────────────────────────────────────────────

        save_checkpoint(
            "./checkpoints",
            models=[model_branch, model_trunk],
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )

        checkpoint_dir = "./checkpoints"
        epoch_weight_files[epoch] = collect_epoch_weight_files(checkpoint_dir, epoch)

        all_pt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint.0.*.pt"))
        all_pt_files.sort(key=lambda x: int(x.split('.')[-2]))

        if len(all_pt_files) > 3:
            files_to_delete = all_pt_files[:-3]
            for f in files_to_delete:
                old_epoch = f.split('.')[-2]
                if os.path.exists(f):
                    os.remove(f)
                model_files = glob.glob(
                    os.path.join(checkpoint_dir, f"*.0.{old_epoch}.mdlus")
                )
                for m in model_files:
                    if os.path.exists(m):
                        os.remove(m)
            print(f">>> Cleanup: Kept epochs {[f.split('.')[-2] for f in all_pt_files[-3:]]}")

        results_images = sorted(
            glob.glob("./results_*.png"),
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        if len(results_images) > 3:
            for img in results_images[:-3]:
                os.remove(img)

    # End-of-run summary
    metric_history = {
        "Validation error" : val_error_list,
        "MSE"              : mse_list,
        "SSIM"             : ssim_list,
        "APE (%)"          : ape_list,
    }

    run_log_paths = [
        os.path.join(os.getcwd(), "poisson_train.log"),
        os.path.join(os.getcwd(), ".log"),
    ]

    for run_log_path in run_log_paths:
        append_final_metric_summary(
            log_path=run_log_path,
            epochs=epoch_list,
            metric_history=metric_history,
            epoch_weight_files=epoch_weight_files,
        )


if __name__ == "__main__":
    main()

