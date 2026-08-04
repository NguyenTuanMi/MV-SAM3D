"""
Post-processing Layer 1: rescale a generated mesh to exact real-world dimensions
taken from a product spec sheet (width / height / depth in meters).

Why this exists
----------------
SAM3D-Objects decodes meshes into a canonical, unit-normalized space
(roughly a +-0.5 bounding box). Real-world scale is normally recovered via
a metric-depth estimate (Depth Anything V3) + camera pose optimization,
which is only as accurate as that per-scene depth estimate. When a ground
truth dimension is available (e.g. an IKEA product page states
60cm x 45cm x 75cm), we should use it directly instead of trusting the
estimated scale -- it's exact, not estimated.

This module is intentionally standalone: it operates on the exported GLB
(or an in-memory trimesh object), so it works regardless of which
inference script produced it (run_inference.py, run_inference_weighted.py,
or a future one).
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import trimesh

logger = logging.getLogger("scale_to_spec")

# If any two axis-wise scale factors disagree by more than this fraction,
# we flag the object for manual review instead of silently forcing an
# anisotropic fit -- large disagreement usually means the reconstruction's
# proportions are wrong, not that non-uniform scaling is the correct fix.
DEFAULT_ANISOTROPY_WARN_THRESHOLD = 0.20


@dataclass
class ScaleResult:
    input_path: str
    output_path: str
    # Axis convention: X = width, Y = height (up), Z = depth. If your mesh
    # uses a different up-axis, pass axis_map to ScaleToSpec.
    original_bbox_m: list
    target_bbox_m: list
    scale_factors: list          # [sx, sy, sz] actually applied
    applied_scale: float         # single scalar actually applied (see mode)
    mode: str                    # "isotropic" or "anisotropic"
    anisotropy_ratio: float      # max(scale)/min(scale) - 1.0
    flagged_for_review: bool
    notes: str = ""


def _axis_index(name: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[name.lower()]


def rescale_mesh_to_spec(
    mesh_path: Union[str, Path],
    width_m: float,
    height_m: float,
    depth_m: float,
    output_path: Optional[Union[str, Path]] = None,
    axis_map: Optional[dict] = None,
    mode: str = "isotropic",
    anisotropy_warn_threshold: float = DEFAULT_ANISOTROPY_WARN_THRESHOLD,
) -> ScaleResult:
    """
    Rescale a mesh/GLB so its bounding box matches spec-sheet dimensions.

    Args:
        mesh_path: path to the generated .glb (or any trimesh-loadable mesh)
        width_m, height_m, depth_m: target real-world dimensions in meters,
            taken directly from the product spec sheet.
        output_path: where to write the rescaled glb. Defaults to
            "<stem>_scaled.glb" next to the input.
        axis_map: maps semantic axis -> mesh axis, e.g. {"width": "x",
            "height": "z", "depth": "y"} if your pipeline's up-axis is Z
            instead of glTF's default Y-up. Defaults to the identity
            mapping (width=x, height=y, depth=z), which matches SAM3D's
            / DA3's exported orientation. IMPORTANT: verify this
            assumption on a few objects before trusting it in bulk --
            if it's wrong, you will silently scale the wrong axis.
        mode: "isotropic" (recommended default) scales all axes by a
            single factor derived from the most reliable axis (largest
            target dimension, on the theory that the largest axis is
            least affected by view-dependent reconstruction noise) --
            this preserves the reconstructed proportions and only
            corrects overall size. "anisotropic" scales each axis
            independently to force an exact bbox match, which can
            distort geometry if the reconstruction's aspect ratio is off.
        anisotropy_warn_threshold: if the three independent per-axis
            scale factors disagree by more than this fraction, the
            result is flagged for manual review (this is computed
            regardless of `mode`, since it's diagnostic information
            about reconstruction quality either way).

    Returns:
        ScaleResult with the applied transform and a review flag.
    """
    mesh_path = Path(mesh_path)
    axis_map = axis_map or {"width": "x", "height": "y", "depth": "z"}

    scene_or_mesh = trimesh.load(mesh_path, force="scene")
    # Work on a combined mesh for bbox math, but transform the actual
    # scene graph so multi-node GLBs (materials, multiple primitives)
    # are preserved.
    if isinstance(scene_or_mesh, trimesh.Scene):
        combined = scene_or_mesh.to_geometry()
    else:
        combined = scene_or_mesh

    bbox_min, bbox_max = combined.bounds
    extent = bbox_max - bbox_min  # [ex, ey, ez] in mesh's own (unitless) space

    targets_by_semantic = {"width": width_m, "height": height_m, "depth": depth_m}
    target_extent = np.zeros(3)
    for semantic, mesh_axis in axis_map.items():
        target_extent[_axis_index(mesh_axis)] = targets_by_semantic[semantic]

    # avoid divide-by-zero on degenerate axes
    safe_extent = np.where(extent < 1e-9, 1e-9, extent)
    per_axis_scale = target_extent / safe_extent

    max_s, min_s = per_axis_scale.max(), per_axis_scale.min()
    anisotropy_ratio = (max_s / min_s) - 1.0 if min_s > 0 else float("inf")
    flagged = anisotropy_ratio > anisotropy_warn_threshold

    if mode == "anisotropic":
        applied = per_axis_scale
        applied_scalar = float(np.mean(per_axis_scale))
    else:
        # isotropic: anchor on the axis with the largest target dimension,
        # since it's typically the axis measured most reliably (both by
        # the spec sheet and by the reconstruction).
        anchor_axis = int(np.argmax(target_extent))
        s = per_axis_scale[anchor_axis]
        applied = np.array([s, s, s])
        applied_scalar = float(s)

    transform = np.eye(4)
    transform[0, 0], transform[1, 1], transform[2, 2] = applied
    if isinstance(scene_or_mesh, trimesh.Scene):
        scene_or_mesh.apply_transform(transform)
        out_mesh = scene_or_mesh
    else:
        scene_or_mesh.apply_transform(transform)
        out_mesh = scene_or_mesh

    output_path = Path(output_path) if output_path else mesh_path.with_name(
        mesh_path.stem + "_scaled.glb"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_mesh.export(str(output_path))

    result = ScaleResult(
        input_path=str(mesh_path),
        output_path=str(output_path),
        original_bbox_m=extent.tolist(),
        target_bbox_m=target_extent.tolist(),
        scale_factors=applied.tolist(),
        applied_scale=applied_scalar,
        mode=mode,
        anisotropy_ratio=float(anisotropy_ratio),
        flagged_for_review=bool(flagged),
    )

    if flagged:
        msg = (
            f"[scale_to_spec] WARNING: {mesh_path.name} shows {anisotropy_ratio:.1%} "
            f"disagreement between per-axis scale factors (threshold "
            f"{anisotropy_warn_threshold:.0%}). This usually means the "
            f"reconstructed proportions don't match the real object -- "
            f"flagging for manual review rather than silently forcing a fit."
        )
        logger.warning(msg)
        result.notes = msg

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Rescale a generated GLB to exact spec-sheet dimensions."
    )
    parser.add_argument("mesh_path", type=str, help="Path to input .glb")
    parser.add_argument("--width_m", type=float, required=True)
    parser.add_argument("--height_m", type=float, required=True)
    parser.add_argument("--depth_m", type=float, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--mode", choices=["isotropic", "anisotropic"], default="isotropic"
    )
    parser.add_argument(
        "--anisotropy_warn_threshold", type=float, default=DEFAULT_ANISOTROPY_WARN_THRESHOLD
    )
    parser.add_argument(
        "--report_json", type=str, default=None,
        help="Optional path to dump the ScaleResult as JSON (for pipeline logging/QA)."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = rescale_mesh_to_spec(
        mesh_path=args.mesh_path,
        width_m=args.width_m,
        height_m=args.height_m,
        depth_m=args.depth_m,
        output_path=args.output_path,
        mode=args.mode,
        anisotropy_warn_threshold=args.anisotropy_warn_threshold,
    )
    print(json.dumps(asdict(result), indent=2))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()