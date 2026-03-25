# predict_poisson.py
# ─────────────────────────────────────────────────────────────────────────────
# Single-sample Poisson solver inference.
#
# Accepts a 2D text file of doubles (space-separated, 257x257) as the charge
# density (rho) input, internally wraps it into a temporary HDF5 file,
# runs the exact same DeepONet model used during training, and produces:
#   1. A 2D text file of the predicted potential  (like potential_sample)
#   2. A 4-panel PNG visualisation (rho | predicted phi | expected phi | error)
#      or a 2-panel PNG if no expected potential is given.
#
# Usage:
#   python predict_poisson.py                             (defaults below)
#   python predict_poisson.py --rho_file my_rho.txt
#   python predict_poisson.py --rho_file rho_sample --expected_file potential_sample
#   python predict_poisson.py --ckpt_dir ./outputs_poisson/checkpoints
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import tempfile
from typing import Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.utils.checkpoint import load_checkpoint
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch


# ── Normalization constants (identical to training) ──────────────────────────
RHO_NORM = 5.83456e+11
PHI_NORM = 4.12442e+00
GRID_SIZE = 257


# ── Model wrapper — exact copy from poisson_fno_train.py ────────────────────
class MdlsSymWrapper(Arch):
    """
    DeepONet wrapper: FNO branch (rho → features) * FC trunk (x,y → features).
    Identical to the class used during training so that checkpoint weights
    load without any shape or key mismatches.
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
        # Concatenate x, y inputs for the trunk MLP
        xy_input_shape = dict_tensor["x"].shape
        xy = self.concat_input(
            {
                rho: dict_tensor[rho].view(xy_input_shape[0], -1, 1)
                for rho in ["x", "y"]
            },
            ["x", "y"],
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        fc_out = self.trunk_net(xy)

        # FNO branch processes the charge density
        fno_out = self.branch_net(dict_tensor["rho_prime"])

        # Reshape trunk output to spatial grid
        fc_out = fc_out.view(
            xy_input_shape[0], -1, xy_input_shape[-2], xy_input_shape[-1]
        )

        # Element-wise product of branch and trunk outputs
        out = fc_out * fno_out

        return self.split_output(out, self.output_key_dict, dim=1)


def load_rho_from_textfile(filepath: str) -> np.ndarray:
    """Load a 257x257 space-separated text file of doubles."""
    rho = np.loadtxt(filepath)
    if rho.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(
            f"Expected rho shape ({GRID_SIZE}, {GRID_SIZE}), got {rho.shape}"
        )
    return rho


def rho_to_temp_hdf5(rho: np.ndarray, tmp_dir: str) -> str:
    """
    Convert a raw 2D rho array into a single-sample HDF5 file with the same
    layout the training dataset uses: keys 'rho' and 'potential', each with
    shape (1, 1, 257, 257).  The potential is filled with zeros (placeholder)
    because we only need the rho for inference.
    """
    tmp_path = os.path.join(tmp_dir, "single_input.hdf5")
    rho_4d = rho[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,257,257)
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("rho", data=rho_4d)
        # Placeholder potential (zeros) — not used during prediction
        f.create_dataset("potential", data=np.zeros_like(rho_4d))
    return tmp_path


def build_coordinate_grids(device: torch.device):
    """Build the (x, y) meshgrid tensors used by the trunk network."""
    lin = np.linspace(0, 1, GRID_SIZE)
    xx, yy = np.meshgrid(lin, lin)
    x_t = torch.from_numpy(xx.astype(np.float32)).view(1, 1, GRID_SIZE, GRID_SIZE).to(device)
    y_t = torch.from_numpy(yy.astype(np.float32)).view(1, 1, GRID_SIZE, GRID_SIZE).to(device)
    return x_t, y_t


def build_model(cfg, device: torch.device):
    """Instantiate branch (FNO) + trunk (FC) + wrapper, matching training."""
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
    ).to(device)

    model_trunk = FullyConnected(
        in_features=cfg.model.fc.in_features,
        out_features=cfg.model.fc.out_features,
        layer_size=cfg.model.fc.layer_size,
        num_layers=cfg.model.fc.num_layers,
    ).to(device)

    model = MdlsSymWrapper(
        input_keys=[Key("rho_prime"), Key("x"), Key("y")],
        output_keys=[Key("phi")],
        trunk_net=model_trunk,
        branch_net=model_branch,
    ).to(device)

    return model, model_branch, model_trunk


def save_potential_textfile(phi: np.ndarray, filepath: str):
    """Save the 257x257 predicted potential as a space-separated text file."""
    np.savetxt(filepath, phi, fmt="%.6e")
    print(f"Predicted potential saved to: {filepath}")


def visualize(rho: np.ndarray, phi_pred: np.ndarray,
              phi_expected, save_path: str):
    """
    Generate the output image.
    If phi_expected is provided  → 4-panel: rho | pred | expected | error
    Otherwise                    → 2-panel: rho | pred
    """
    has_expected = phi_expected is not None

    ncols = 4 if has_expected else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7.5 * ncols, 6))

    # Panel 1: Input charge density
    im0 = axes[0].imshow(rho, cmap="RdBu_r")
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title("Input Charge Density (ρ)")

    # Panel 2: Predicted potential
    im1 = axes[1].imshow(phi_pred, cmap="viridis")
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title("Predicted Potential (φ)")

    if has_expected:
        # Use shared color limits from the expected potential
        vmin, vmax = phi_expected.min(), phi_expected.max()
        axes[1].images[0].set_clim(vmin, vmax)

        # Panel 3: Expected potential
        im2 = axes[2].imshow(phi_expected, vmin=vmin, vmax=vmax, cmap="viridis")
        plt.colorbar(im2, ax=axes[2])
        axes[2].set_title("Expected Potential (φ)")

        # Panel 4: Absolute error
        diff = np.abs(phi_pred - phi_expected)
        rel_err = diff.mean() / (np.abs(phi_expected).mean() + 1e-12)
        im3 = axes[3].imshow(diff, cmap="hot")
        plt.colorbar(im3, ax=axes[3])
        axes[3].set_title(f"|Error|  —  Mean rel err: {rel_err:.4%}")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualisation saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run single-sample Poisson solver inference on a 2D rho text file."
    )
    parser.add_argument(
        "--rho_file", type=str, default="rho_sample",
        help="Path to the input rho text file (257x257, space-separated doubles)."
    )
    parser.add_argument(
        "--expected_file", type=str, default=None,
        help="Optional path to the expected potential text file for comparison."
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="./outputs_poisson/checkpoints",
        help="Path to the directory containing trained model checkpoints."
    )
    parser.add_argument(
        "--config", type=str, default="./conf/config_deeponet.yaml",
        help="Path to the Hydra YAML config used during training."
    )
    parser.add_argument(
        "--output_potential", type=str, default="predicted_potential.txt",
        help="Path for the output predicted-potential text file."
    )
    parser.add_argument(
        "--output_image", type=str, default="prediction_result.png",
        help="Path for the output visualisation PNG."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model config ────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)

    # ── Read the input rho text file ─────────────────────────────────────────
    print(f"Reading rho from: {args.rho_file}")
    rho_raw = load_rho_from_textfile(args.rho_file)

    # ── Convert to temporary HDF5 (backend conversion as requested) ──────────
    with tempfile.TemporaryDirectory() as tmp_dir:
        hdf5_path = rho_to_temp_hdf5(rho_raw, tmp_dir)
        print(f"Temporary HDF5 created at: {hdf5_path}")

        # ── Normalise rho and prepare model input tensor ─────────────────────
        rho_tensor = (
            torch.from_numpy(rho_raw.astype(np.float32))
            .unsqueeze(0).unsqueeze(0)           # (1, 1, 257, 257)
            / RHO_NORM
        ).to(device)

        x_coord, y_coord = build_coordinate_grids(device)

        # ── Build and load model ─────────────────────────────────────────────
        model, model_branch, model_trunk = build_model(cfg, device)

        print(f"Loading checkpoint from: {args.ckpt_dir}")
        load_checkpoint(
            args.ckpt_dir,
            models=[model_branch, model_trunk],
            device=device,
        )
        model.eval()
        print("Checkpoint loaded successfully.")

        # ── Inference ────────────────────────────────────────────────────────
        with torch.no_grad():
            out = model(
                {"rho_prime": rho_tensor, "x": x_coord, "y": y_coord}
            )

    # ── Denormalise the prediction back to physical units ────────────────────
    phi_pred = out["phi"][0, 0].cpu().numpy() * PHI_NORM
    print(f"Predicted potential range: [{phi_pred.min():.6e}, {phi_pred.max():.6e}]")

    # ── Save predicted potential as a 2D text file ───────────────────────────
    save_potential_textfile(phi_pred, args.output_potential)

    # ── Load expected potential if provided ──────────────────────────────────
    phi_expected = None
    if args.expected_file is not None:
        print(f"Reading expected potential from: {args.expected_file}")
        phi_expected = load_rho_from_textfile(args.expected_file)

    # ── Generate visualisation ───────────────────────────────────────────────
    visualize(rho_raw, phi_pred, phi_expected, args.output_image)

    print("Done.")


if __name__ == "__main__":
    main()
