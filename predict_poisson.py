import os
import argparse
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity as ssim_metric

from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch
from physicsnemo.utils.checkpoint import load_checkpoint

# ==========================================
# GLOBAL PRECISION SETTING
# ==========================================
torch.set_default_dtype(torch.float64)

# ==========================================
# CONSTANTS (Must match training!)
# ==========================================
GRID_SIZE = 256
RHO_NORM  = np.float64(1.232227672511348e-08)
PHI_NORM  = np.float64(9.215881670158015e-01)


class MdlsSymWrapper(Arch):
    """
    Wrapper model for float64 precision inference.
    Exactly matches the architecture used in training and predict_validation_full.py.
    """
    def __init__(self, input_keys, output_keys, trunk_net, branch_net, num_basis=1):
        super().__init__(input_keys=input_keys, output_keys=output_keys)
        self.branch_net = branch_net.to(torch.float64)
        self.trunk_net  = trunk_net.to(torch.float64)
        self.num_basis  = num_basis

    def forward(self, dict_tensor: Dict[str, torch.Tensor]):
        xy_input_shape = dict_tensor["x"].shape

        # Ensure inputs are cast to double precision
        xy = self.concat_input(
            {k: dict_tensor[k].view(xy_input_shape[0], -1, 1).to(torch.float64) for k in ["x", "y"]},
            ["x", "y"],
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        fc_out = self.trunk_net(xy)

        # Branch network input cast to double
        fno_out = self.branch_net(dict_tensor["rho_prime"].to(torch.float64))

        fc_out = fc_out.view(
            xy_input_shape[0], -1, xy_input_shape[-2], xy_input_shape[-1]
        )

        if self.num_basis > 1:
            out = (fc_out * fno_out).sum(dim=1, keepdim=True)
        else:
            out = fc_out * fno_out

        phi = out[:, 0:1, :, :]

        # Strictly Enforce Physical Dirichlet Boundary Condition
        boundary_mean = (
            phi[:, :, 0,  :].mean(dim=-1) +
            phi[:, :, -1, :].mean(dim=-1) +
            phi[:, :, :,  0].mean(dim=-1) +
            phi[:, :, :, -1].mean(dim=-1)
        ) / 4.0

        phi = phi - boundary_mean.unsqueeze(-1).unsqueeze(-1)

        # Hard-zero boundaries
        phi[:, :, 0,  :] = 0.0
        phi[:, :, -1, :] = 0.0
        phi[:, :, :,  0] = 0.0
        phi[:, :, :, -1] = 0.0

        return self.split_output(phi, self.output_key_dict, dim=1)


def save_sample_figure(rho_grid, phi_pred, phi_true, error_abs, sample_idx, save_path, rmse, ape, ssim):
    """Save a 4-panel figure: Input rho, Predicted phi, True phi, Absolute error."""
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))

    im0 = axes[0].imshow(rho_grid, cmap="RdBu_r")
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title("Input rho")

    vmin = min(phi_pred.min(), phi_true.min())
    vmax = max(phi_pred.max(), phi_true.max())

    im1 = axes[1].imshow(phi_pred, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title("Predicted phi")

    im2 = axes[2].imshow(phi_true, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im2, ax=axes[2])
    axes[2].set_title("True phi")

    im3 = axes[3].imshow(error_abs, cmap="hot")
    plt.colorbar(im3, ax=axes[3])
    axes[3].set_title("Absolute error")

    fig.suptitle(f"Grid {sample_idx} | RMSE: {rmse:.4e} | APE: {ape:.4f}% | SSIM: {ssim:.6f}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Predict on specific grid indices from HDF5.")
    parser.add_argument(
        "--grids", type=int, nargs='+', required=True,
        help="List of grid indices to predict on (e.g., --grids 692 695 915)"
    )
    parser.add_argument(
        "--validation_hdf5", type=str, default="./testing_64.hdf5",
        help="Path to the testing HDF5 dataset."
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="./outputs_poisson_v5/checkpoints",
        help="Path to the directory containing trained model checkpoints."
    )
    parser.add_argument(
        "--config", type=str, default="./conf/config_deeponet.yaml",
        help="Path to the Hydra YAML config used during training."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./predictions_selected_grids",
        help="Directory to store the resulting .txt files and images."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Double Precision)")

    # ── 1. Load Config ────────────────────────────────────────────────────────
    cfg = OmegaConf.load(os.path.abspath(args.config))
    num_basis = int(cfg.model.get("num_basis", 1))

    # ── 2. Build Model (explicit float64 on all sub-models) ───────────────────
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
    ).to(device).to(torch.float64)

    model_trunk = FullyConnected(
        in_features=cfg.model.fc.in_features,
        out_features=cfg.model.fc.out_features,
        layer_size=cfg.model.fc.layer_size,
        num_layers=cfg.model.fc.num_layers,
    ).to(device).to(torch.float64)

    model = MdlsSymWrapper(
        input_keys=[Key("rho_prime"), Key("x"), Key("y")],
        output_keys=[Key("phi")],
        trunk_net=model_trunk,
        branch_net=model_branch,
        num_basis=num_basis,
    ).to(device).to(torch.float64)

    # ── 3. Load Weights ──────────────────────────────────────────────────────
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    print(f"Loading checkpoint from: {ckpt_dir}")
    load_checkpoint(ckpt_dir, models=[model_branch, model_trunk], device=device)
    model.eval()

    # ── 4. Coordinate Grids ──────────────────────────────────────────────────
    lin = np.linspace(0, 1, GRID_SIZE, dtype=np.float64)
    xx, yy = np.meshgrid(lin, lin)
    x_t = torch.from_numpy(xx).view(1, 1, GRID_SIZE, GRID_SIZE).to(device, dtype=torch.float64)
    y_t = torch.from_numpy(yy).view(1, 1, GRID_SIZE, GRID_SIZE).to(device, dtype=torch.float64)

    # ── 5. Setup Output Directory ────────────────────────────────────────────
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}\n")

    # ── 6. Process Specific Grids ────────────────────────────────────────────
    with h5py.File(os.path.abspath(args.validation_hdf5), "r") as h5f:
        total_samples = h5f["rho"].shape[0]

        for grid_idx in args.grids:
            if grid_idx < 0 or grid_idx >= total_samples:
                print(f"⚠️ Warning: Grid {grid_idx} is out of bounds (0 to {total_samples-1}). Skipping.")
                continue

            print(f"Processing Grid {grid_idx}...")
            
            # Create subfolder for this specific grid
            grid_dir = os.path.join(output_dir, f"grid_{grid_idx:04d}")
            os.makedirs(grid_dir, exist_ok=True)

            # Load raw physical data as float64
            rho_raw  = h5f["rho"][grid_idx].squeeze().astype(np.float64)
            phi_true = h5f["potential"][grid_idx].squeeze().astype(np.float64)

            # Save input and expected as txt
            np.savetxt(os.path.join(grid_dir, "rho_input.txt"), rho_raw, fmt="%.6e")
            np.savetxt(os.path.join(grid_dir, "phi_expected.txt"), phi_true, fmt="%.6e")

            # Normalize rho for model input
            rho_tensor = (
                torch.from_numpy(rho_raw / RHO_NORM)
                .unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float64)
            )

            # Forward pass
            with torch.no_grad():
                out = model({"rho_prime": rho_tensor, "x": x_t, "y": y_t})

            # Denormalize phi
            phi_pred = out["phi"][0, 0].cpu().numpy() * PHI_NORM

            # Save predicted phi
            np.savetxt(os.path.join(grid_dir, "phi_pred.txt"), phi_pred, fmt="%.6e")

            # Metrics
            error     = phi_pred - phi_true
            error_abs = np.abs(error)
            rmse      = float(np.sqrt(np.mean(error_abs ** 2)))

            denom = np.where(np.abs(phi_true) < 1e-15, 1.0, np.abs(phi_true))
            ape   = float(np.mean(error_abs / denom) * 100)

            pred_norm = phi_pred / PHI_NORM
            true_norm = phi_true / PHI_NORM
            data_range = true_norm.max() - true_norm.min()
            if data_range == 0:
                data_range = 1.0
            sample_ssim = float(ssim_metric(true_norm, pred_norm, data_range=data_range))

            print(f"  RMSE: {rmse:.4e} | APE: {ape:.4f}% | SSIM: {sample_ssim:.6f}")

            # Save image
            img_path = os.path.join(grid_dir, f"result.png")
            save_sample_figure(
                rho_grid=rho_raw,
                phi_pred=phi_pred,
                phi_true=phi_true,
                error_abs=error_abs,
                sample_idx=grid_idx,
                save_path=img_path,
                rmse=rmse,
                ape=ape,
                ssim=sample_ssim
            )
            
    print(f"\nAll selected grids processed successfully!")

if __name__ == "__main__":
    main()
