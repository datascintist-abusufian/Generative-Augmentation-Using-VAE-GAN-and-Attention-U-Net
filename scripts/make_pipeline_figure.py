#!/usr/bin/env python
"""Publication-style overview figure (IEEE TMI double column) built from real run artifacts.

Every image in the figure is produced by running the trained fold-local models on cached slices:
  (a) real slice -> shared geometric transform -> mask-conditioned VAE (latent perturbation)
      -> conditional refiner G -> QC gate (SSIM, anatomy Dice, set-level FID from the QC JSON)
  (b) real + QC-passed synthetic pairs -> residual Attention U-Net -> prediction / GT / attention gate

Usage:
  python scripts/make_pipeline_figure.py --config configs/default.yaml --seed 42 --fold 0
Outputs PDF + SVG (vector, for the manuscript) and a 600-dpi PNG (for the README).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

from cardiac_aug.config import load_config  # noqa: E402
from cardiac_aug.evaluation.explainability import attention_gate_map  # noqa: E402
from cardiac_aug.models import AttentionUNet, ConditionalRefiner, MaskConditionedVAE  # noqa: E402

# Okabe-Ito (colour-blind safe): LV, Myo, RV
CLASS_COLOURS = {1: "#D55E00", 2: "#009E73", 3: "#0072B2"}
CLASS_NAMES = {1: "LV", 2: "Myo", 3: "RV"}
INK, MUTED = "#222222", "#6b6b6b"
ENC, DEC, LAT, GATE = "#d6e4f0", "#f5e0cc", "#e4e4e4", "#f2c14e"

plt.rcParams.update(
    {
        "font.family": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# ----------------------------------------------------------------------------- data + inference
def load_slices(manifest, patient_ids):
    out = []
    for record in manifest:
        if record["patient_id"] in patient_ids:
            data = np.load(record["path"])
            for i in range(len(data["images"])):
                out.append((record["patient_id"], str(data["phases"][i]), data["images"][i], data["masks"][i]))
    return out


def pick_slice(slices, prefer_phase="ED"):
    """Most complete slice: all three structures present, largest foreground, ED preferred."""

    def score(item):
        _, phase, _, mask = item
        present = sum(int((mask == c).any()) for c in (1, 2, 3))
        return (present, phase == prefer_phase, int((mask > 0).sum()))

    return max(slices, key=score)


def fixed_transform(image, mask, angle_deg=12.0, scale=1.06, flip=True):
    """A deterministic instance of the paired geometric transform used in training."""
    if flip:
        image, mask = torch.flip(image, [-1]), torch.flip(mask, [-1])
    a = math.radians(angle_deg)
    theta = image.new_tensor([[scale * math.cos(a), -scale * math.sin(a), 0], [scale * math.sin(a), scale * math.cos(a), 0]])
    grid = F.affine_grid(theta[None], (1, 1, *image.shape[-2:]), align_corners=False)
    image = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
    mask = F.grid_sample(mask.float()[None, None], grid, mode="nearest", align_corners=False)[0, 0].long()
    return image, mask


def dice(prediction, target):
    values = []
    for c in (1, 2, 3):
        a, b = prediction == c, target == c
        if a.sum() + b.sum():
            values.append(2 * np.logical_and(a, b).sum() / (a.sum() + b.sum()))
    return float(np.mean(values)) if values else 0.0


def load_segmenter(path, config, device):
    model = AttentionUNet(num_classes=config.data.num_classes, base=config.model.base_channels).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model"])
    return model.eval()


@torch.no_grad()
def run_models(config, fold_dir, seed, device):
    manifest = json.loads((Path(config.data.cache_dir) / "manifest.json").read_text())["patients"]
    split = json.loads((fold_dir / "split.json").read_text())
    gen = fold_dir / "generative"

    vae = MaskConditionedVAE(
        config.data.image_size, config.data.num_classes, config.model.latent_dim, config.model.base_channels
    ).to(device)
    vae.load_state_dict(torch.load(gen / "vae" / "final.pt", map_location=device, weights_only=True)["model"])
    vae.eval()
    refiner = ConditionalRefiner(config.data.num_classes, config.model.base_channels).to(device)
    refiner.load_state_dict(torch.load(gen / "hybrid" / "final.pt", map_location=device, weights_only=True)["generator"])
    refiner.eval()
    baseline = load_segmenter(fold_dir / "conditions" / "baseline" / "best.pt", config, device)
    hybrid = load_segmenter(fold_dir / "conditions" / "hybrid_qc" / "best.pt", config, device)
    qc = json.loads((gen / "hybrid_qc.qc.json").read_text())

    # (a) synthesis from a TRAINING patient (the only patients generators ever see)
    pid, phase, image, mask = pick_slice(load_slices(manifest, set(split["train"])))
    x = torch.from_numpy(image).float()[None].to(device)
    y = torch.from_numpy(mask.astype(np.int64)).to(device)
    xt, yt = fixed_transform(x, y)
    torch.manual_seed(seed)
    coarse = vae.perturb(xt[None], yt[None], config.model.latent_noise_std)
    refined = refiner(coarse, yt[None])
    xt_np, yt_np = xt[0].cpu().numpy(), yt.cpu().numpy()
    refined_np = refined[0, 0].cpu().numpy()
    ssim = structural_similarity(refined_np, xt_np, data_range=1.0)
    anatomy = dice(baseline(refined).argmax(1)[0].cpu().numpy(), yt_np)

    # (b) segmentation of an unseen TEST patient
    tpid, tphase, timage, tmask = pick_slice(load_slices(manifest, set(split["test"])))
    tx = torch.from_numpy(timage).float()[None, None].to(device)
    prediction = hybrid(tx).argmax(1)[0].cpu().numpy()
    attention = attention_gate_map(hybrid, tx)[0, 0].cpu().numpy()

    return dict(
        x=image, y=mask, xt=xt_np, yt=yt_np, coarse=coarse[0, 0].cpu().numpy(), refined=refined_np,
        ssim=ssim, anatomy=anatomy, qc=qc, source=f"{pid} {phase}",
        tx=timage, ty=tmask, prediction=prediction, attention=attention,
        test_dice=dice(prediction, tmask), test=f"{tpid} {tphase}",
    )


# ----------------------------------------------------------------------------- drawing helpers
def crop_box(mask, margin=0.55):
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return slice(None), slice(None)
    cy, cx = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    half = max(rows.max() - rows.min(), cols.max() - cols.min()) / 2 * (1 + margin) + 2
    n = mask.shape[0]
    half = min(half, n / 2)
    y0 = int(np.clip(round(cy - half), 0, n - 2 * half))
    x0 = int(np.clip(round(cx - half), 0, n - 2 * half))
    size = int(round(2 * half))
    return slice(y0, y0 + size), slice(x0, x0 + size)


class Canvas:
    def __init__(self, width_in=7.16, height_in=4.45):
        self.W, self.H = width_in * 100, height_in * 100
        self.fig = plt.figure(figsize=(width_in, height_in))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def text(self, x, y, s, size=7, **kw):
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        kw.setdefault("color", INK)
        kw.setdefault("zorder", 8)
        if "$" in s:  # typographic prime renders tighter than mathtext's "'"
            s = s.replace("'", "\u2032")
        return self.ax.text(x, y, s, fontsize=size, **kw)

    def box(self, x, y, w, h, fc="white", ec=INK, lw=0.6, ls="-", r=4, z=1):
        self.ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
        )

    def arrow(self, p, q, ls="-", color=INK, lw=0.8, rad=0.0, z=5):
        self.ax.add_patch(
            FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color, ls=ls,
                            connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z)
        )

    def trapezoid(self, x, y, w, h_left, h_right, fc, label=None):
        cy = y
        pts = [(x, cy - h_left / 2), (x + w, cy - h_right / 2), (x + w, cy + h_right / 2), (x, cy + h_left / 2)]
        self.ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=INK, lw=0.6, zorder=2))
        if label:
            self.text(x + w / 2, cy, label, size=8)

    def image(self, x, y, s, img, mask=None, crop=None, overlay="fill", cmap="gray", heat=None, frame=INK):
        crop = crop or (slice(None), slice(None))
        extent = (x, x + s, y, y + s)
        self.ax.imshow(img[crop], cmap=cmap, vmin=0, vmax=1, extent=extent, interpolation="lanczos", zorder=3)
        if heat is not None:
            self.ax.imshow(heat[crop], cmap="magma", alpha=0.55, extent=extent, interpolation="bilinear", zorder=4)
        if mask is not None:
            m = mask[crop]
            if overlay in ("fill", "mask"):
                colours = [(0, 0, 0, 0)] + [matplotlib.colors.to_rgba(CLASS_COLOURS[c], 0.45 if overlay == "fill" else 1.0) for c in (1, 2, 3)]
                self.ax.imshow(np.ma.masked_where(m == 0, m), cmap=ListedColormap(colours), vmin=0, vmax=3,
                               extent=extent, interpolation="nearest", zorder=4)
            if overlay in ("fill", "contour"):
                rows = np.linspace(y + s, y, m.shape[0])
                cols = np.linspace(x, x + s, m.shape[1])
                for c in (1, 2, 3):
                    if (m == c).any():
                        self.ax.contour(cols, rows, (m == c).astype(float), levels=[0.5], colors=CLASS_COLOURS[c],
                                        linewidths=0.6, zorder=5)
        self.ax.add_patch(Rectangle((x, y), s, s, fill=False, ec=frame, lw=0.6, zorder=6))

    def mask_image(self, x, y, s, mask, crop=None):
        blank = np.zeros(mask.shape, dtype=float)
        self.image(x, y, s, blank, mask=mask, crop=crop, overlay="mask")

    def panel_label(self, x, y, letter, title):
        self.text(x, y, f"({letter})", size=8.5, ha="left", weight="bold")
        self.text(x + 18, y, title, size=8, ha="left", weight="bold")


# ----------------------------------------------------------------------------- figure
def draw(r, config, out_stem, watermark=None):
    c = Canvas()
    q = config.qc
    s = 58

    # ======================= (a) synthesis =======================
    c.panel_label(6, 432, "a", "Mask-conditioned hybrid VAE–GAN synthesis (training patients only)")
    top, bot = 340, 272  # image rows (bottom-left y)
    mid = (top + bot + s) / 2

    crop_src = crop_box(r["y"])
    c.image(8, top, s, r["x"], crop=crop_src)
    c.mask_image(8, bot, s, r["y"], crop=crop_src)
    c.text(8 + s / 2, top + s + 7, "real slice $x$")
    c.text(8 + s / 2, bot - 7, "mask $y$")

    c.arrow((70, top + s / 2), (94, top + s / 2))
    c.arrow((70, bot + s / 2), (94, bot + s / 2))
    c.text(82, mid, "$\\mathcal{T}$", size=9)
    c.text(82, mid - 11, "shared", size=5.5, color=MUTED)

    crop_t = crop_box(r["yt"])
    c.image(98, top, s, r["xt"], crop=crop_t)
    c.mask_image(98, bot, s, r["yt"], crop=crop_t)
    c.text(98 + s / 2, top + s + 7, "$x'=\\mathcal{T}(x)$")
    c.text(98 + s / 2, bot - 7, "$y'=\\mathcal{T}(y)$")

    # VAE block
    vx, vw = 168, 170
    c.box(vx, 255, vw, 160, fc="#fafafa", ec=MUTED, ls="--", lw=0.6)
    c.text(vx + vw / 2, 405, "Mask-conditioned VAE", size=7.5, weight="bold")
    c.arrow((158, top + s / 2), (180, mid + 8), rad=-0.15)
    c.arrow((158, bot + s / 2), (180, mid - 8), rad=0.15)
    c.trapezoid(180, mid, 38, 92, 40, ENC, "$E_\\phi$")
    c.text(199, mid + 54, "$x'\\oplus y'$", size=6.5, color=MUTED)
    c.arrow((218, mid), (226, mid))
    c.box(226, mid - 24, 44, 48, fc=LAT, r=3, z=2)
    c.text(248, mid + 12, "$\\mu,\\,\\sigma$", size=7)
    c.text(248, mid, "$z=\\mu+\\varepsilon$", size=6.5)
    c.text(248, mid - 13, "$\\varepsilon\\sim\\mathcal{N}(0,\\sigma_p^2 I)$", size=5.2)
    c.arrow((270, mid), (280, mid))
    c.trapezoid(280, mid, 38, 40, 92, DEC, "$D_\\theta$")
    c.box(244, 266, 44, 22, fc=ENC, r=3, z=2)
    c.text(266, 277, "$E_y(y')$", size=6.5)
    c.arrow((158, bot + 12), (244, 277), color=MUTED, lw=0.6)
    c.arrow((288, 277), (296, mid - 34), color=MUTED, lw=0.6, rad=0.3)
    c.text(vx + vw / 2, 262, "$\\ell_1(D_\\theta(z,y'),x')+\\beta\\,\\mathrm{KL}$", size=6, color=MUTED)

    # coarse output
    c.arrow((318, mid), (348, mid))
    c.image(350, mid - s / 2, s, r["coarse"], crop=crop_t)
    c.text(350 + s / 2, mid + s / 2 + 7, "coarse $\\hat{x}$")

    # refiner block
    gx, gw = 420, 104
    c.box(gx, 255, gw, 160, fc="#fafafa", ec=MUTED, ls="--", lw=0.6)
    c.text(gx + gw / 2, 405, "Conditional refiner", size=7.5, weight="bold")
    c.arrow((408, mid), (430, mid))
    ux, uy = gx + 12, mid + 4
    for i, (h, dy) in enumerate([(30, 0), (22, -14), (14, -26)]):
        c.ax.add_patch(Rectangle((ux + i * 9, uy - h / 2 + dy), 6, h, fc=ENC, ec=INK, lw=0.4, zorder=3))
        c.ax.add_patch(Rectangle((ux + 70 - i * 9, uy - h / 2 + dy), 6, h, fc=DEC, ec=INK, lw=0.4, zorder=3))
    c.ax.add_patch(Rectangle((ux + 35, uy - 40), 6, 10, fc=LAT, ec=INK, lw=0.4, zorder=3))
    for i, dy in enumerate((0, -14)):
        c.ax.plot([ux + 6 + i * 9, ux + 70 - i * 9], [uy + dy + 6, uy + dy + 6], color=MUTED, lw=0.4, ls=":", zorder=2)
    c.text(gx + gw / 2, mid + 34, "U-Net generator $G(\\hat{x},y')$", size=6.5)
    c.arrow((ux + 76, mid), (gx + gw + 12, mid))
    c.box(gx + 12, 274, 80, 30, fc="#f0e6f2", r=3, z=2)
    c.text(gx + 52, 295, "PatchGAN $D(\\cdot,y')$", size=6.2)
    for i in range(4):
        for j in range(2):
            c.ax.add_patch(Rectangle((gx + 36 + i * 8, 278 + j * 6), 6, 5, fc="#d8c3dd" if (i + j) % 2 else "white",
                                     ec=INK, lw=0.3, zorder=3))
    c.arrow((gx + gw - 2, mid - 2), (gx + 93, 292), ls=(0, (2, 1.5)), color=MUTED, lw=0.6, rad=-0.35)
    c.text(gx + gw / 2, 262, "$\\mathcal{L}_{\\mathrm{adv}}+\\lambda\\,\\ell_1$  (training only)", size=5.6, color=MUTED)

    # refined output
    c.image(536, mid - s / 2, s, r["refined"], mask=r["yt"], crop=crop_t, overlay="contour")
    c.text(536 + s / 2, mid + s / 2 + 7, "synthetic $(\\tilde{x},\\,y')$")

    # QC gate
    qx, qw = 606, 106
    c.arrow((596, mid), (qx, mid))
    c.box(qx, mid - 52, qw, 104, fc="#fff8e6", ec=INK, lw=0.7)
    c.text(qx + qw / 2, mid + 42, "Quality-control gate", size=7, weight="bold")
    ok = lambda b: "$\\checkmark$" if b else "$\\times$"  # noqa: E731
    ssim_ok = q.ssim_min <= r["ssim"] <= q.ssim_max
    ana_ok = r["anatomy"] >= q.anatomy_dice_min
    fid = r["qc"].get("final_dataset_fid", float("nan"))
    fid_ok = bool(r["qc"].get("fid_passed", False))
    lines = [
        ("SSIM$(\\tilde{x},x')$", f"= {r['ssim']:.2f}  $\\in[{q.ssim_min:.2f},\\,{q.ssim_max:.3g}]$", ssim_ok),
        ("Dice$(S_0(\\tilde{x}),y')$", f"= {r['anatomy']:.2f}  $\\geq {q.anatomy_dice_min:.2f}$", ana_ok),
        ("set-level FID", f"= {fid:.1f}  $\\leq {q.fid_max:g}$", fid_ok),
    ]
    for i, (name, val, passed) in enumerate(lines):
        yy = mid + 22 - i * 24
        c.text(qx + 6, yy + 4, name, size=6.2, ha="left")
        c.text(qx + 6, yy - 6, val, size=5.6, ha="left", color=MUTED)
        c.text(qx + qw - 8, yy, ok(passed), size=9, color="#1b7f3b" if passed else "#b42318")
    c.text(qx + qw / 2, mid - 45, f"retained {r['qc'].get('retained_count', '?')} / {r['qc'].get('candidate_count', '?')} candidates",
           size=5.5, color=MUTED)
    c.arrow((qx + qw / 2, mid - 52), (qx + qw / 2, 262))
    c.text(qx + qw / 2, 256, "to training set (b)", size=5.8, color=MUTED)

    # ======================= (b) segmentation =======================
    c.ax.plot([6, c.W - 6], [244, 244], color="#cccccc", lw=0.5)
    c.panel_label(6, 228, "b", "Segmentation with a residual Attention U-Net (evaluation on held-out test patients)")

    ts = 50
    c.text(66, 205, "training set  $\\mathcal{D}_{\\mathrm{real}}\\cup\\mathcal{D}_{\\mathrm{syn}}$", size=6.5)
    c.image(12, 142, ts, r["xt"], mask=r["yt"], crop=crop_t, overlay="contour")
    c.image(70, 142, ts, r["refined"], mask=r["yt"], crop=crop_t, overlay="contour")
    c.text(12 + ts / 2, 135, "real", size=6, color=MUTED)
    c.text(70 + ts / 2, 135, "synthetic (QC-passed)", size=6, color=MUTED)

    crop_test = crop_box(r["ty"])
    c.image(41, 40, ts, r["tx"], crop=crop_test)
    c.text(41 + ts / 2, 33, "test slice", size=6, color=MUTED)

    # Attention U-Net schematic
    enc_x = [158, 188, 218, 248]
    dec_x = [392, 362, 332, 302]
    ys = [175, 145, 115, 85]
    hs = [56, 44, 34, 26]
    bw = 14
    c.text(290, 205, "Residual Attention U-Net  $S_\\psi$", size=7.5, weight="bold")
    c.arrow((124, 167), (enc_x[0], ys[0]), rad=0.0)
    c.text(140, 180, "train", size=5.5, color=MUTED)
    c.arrow((93, 65), (enc_x[0], ys[0] - 20), rad=0.25)
    c.text(118, 100, "test", size=5.5, color=MUTED)
    for i in range(4):
        c.ax.add_patch(Rectangle((enc_x[i], ys[i] - hs[i] / 2), bw, hs[i], fc=ENC, ec=INK, lw=0.5, zorder=3))
        c.ax.add_patch(Rectangle((dec_x[i], ys[i] - hs[i] / 2), bw, hs[i], fc=DEC, ec=INK, lw=0.5, zorder=3))
        gate_x = (enc_x[i] + bw + dec_x[i]) / 2
        gy = ys[i] + hs[i] / 2 - 5 if i else ys[i] + 12
        c.ax.plot([enc_x[i] + bw, gate_x - 6], [gy, gy], color=MUTED, lw=0.6, zorder=2)
        c.arrow((gate_x + 6, gy), (dec_x[i], gy), color=MUTED, lw=0.6)
        c.ax.add_patch(Circle((gate_x, gy), 6, fc=GATE, ec=INK, lw=0.5, zorder=4))
        c.text(gate_x, gy, "AG", size=4.2, zorder=9)
        if i < 3:
            c.arrow((enc_x[i] + bw, ys[i] - hs[i] / 2 + 4), (enc_x[i + 1], ys[i + 1]), lw=0.6)
            c.arrow((dec_x[i + 1] + bw, ys[i + 1]), (dec_x[i], ys[i] - hs[i] / 2 + 4), lw=0.6)
    c.ax.add_patch(Rectangle((275 - bw / 2, 52), bw, 20, fc=LAT, ec=INK, lw=0.5, zorder=3))
    c.arrow((enc_x[3] + bw, ys[3] - hs[3] / 2 + 4), (275 - bw / 2, 62), lw=0.6)
    c.arrow((275 + bw / 2, 62), (dec_x[3], ys[3] - hs[3] / 2 + 4), lw=0.6)
    c.text(275, 44, "bottleneck", size=5.5, color=MUTED)
    c.ax.add_patch(Rectangle((418, ys[0] - 10), 8, 20, fc="#cfe8d6", ec=INK, lw=0.5, zorder=3))
    c.arrow((dec_x[0] + bw, ys[0]), (418, ys[0]), lw=0.6)
    c.text(422, ys[0] - 18, "1×1", size=5, color=MUTED)
    # mini legend for the network
    lx, ly = 140, 20
    for k, (fc, lab) in enumerate([(ENC, "residual conv + max-pool"), (DEC, "up-conv + residual conv"), (GATE, "attention gate")]):
        if fc == GATE:
            c.ax.add_patch(Circle((lx + k * 118 + 4, ly), 4, fc=fc, ec=INK, lw=0.4))
        else:
            c.ax.add_patch(Rectangle((lx + k * 118, ly - 4), 8, 8, fc=fc, ec=INK, lw=0.4))
        c.text(lx + k * 118 + 12, ly, lab, size=5.5, ha="left", color=MUTED)

    # outputs
    os_ = 70
    ox = [446, 534, 622]
    oy = 100
    c.arrow((426, ys[0]), (ox[0] + 10, oy + os_ + 2), rad=-0.25)
    c.image(ox[0], oy, os_, r["tx"], mask=r["prediction"], crop=crop_test)
    c.image(ox[1], oy, os_, r["tx"], mask=r["ty"], crop=crop_test)
    c.image(ox[2], oy, os_, r["tx"], crop=crop_test, heat=r["attention"])
    c.text(ox[0] + os_ / 2, oy + os_ + 8, "prediction $S_\\psi(x)$")
    c.text(ox[1] + os_ / 2, oy + os_ + 8, "ground truth")
    c.text(ox[2] + os_ / 2, oy + os_ + 8, "finest attention gate")
    c.text(ox[0] + os_ / 2, oy - 8, f"mean Dice = {r['test_dice']:.3f}", size=6, color=MUTED)
    for k, cl in enumerate((1, 2, 3)):
        lx = ox[0] + 30 + k * 62
        c.ax.add_patch(Rectangle((lx, 60), 9, 9, fc=CLASS_COLOURS[cl], ec="none", alpha=0.8))
        c.text(lx + 13, 64.5, {1: "LV cavity", 2: "myocardium", 3: "RV cavity"}[cl], size=6, ha="left")
    c.text(ox[1] + os_ / 2, 40, f"source: {r['source']} (train)   ·   test: {r['test']}", size=5, color=MUTED)

    if watermark:
        c.text(c.W / 2, c.H / 2, watermark, size=26, color="#c62828", alpha=0.18, rotation=18, weight="bold", zorder=20)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 600})):
        c.fig.savefig(out.with_suffix(ext), facecolor="white", **kw)
    plt.close(c.fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", default=None, help="output path stem (default: <run>/summary/figures/pipeline)")
    parser.add_argument("--watermark", default=None, help="diagonal label, e.g. for layout previews")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu")
    fold_dir = Path(config.output_dir) / f"seed_{args.seed}" / f"fold_{args.fold}"
    results = run_models(config, fold_dir, args.seed, device)
    stem = args.output or Path(config.output_dir) / "summary" / "figures" / "pipeline"
    out = draw(results, config, stem, args.watermark)
    print(f"Wrote {out}.pdf/.svg/.png  |  SSIM={results['ssim']:.3f}  anatomy Dice={results['anatomy']:.3f}  "
          f"test Dice={results['test_dice']:.3f}")


if __name__ == "__main__":
    main()
