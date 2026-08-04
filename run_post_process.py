"""
Standalone orchestrator: run scale -> mass -> sim-export on an *existing*
result.glb, without re-running the (expensive, GPU-bound) SAM3D inference.
Useful while you're tuning spec dimensions, density tables, or friction
defaults -- no need to regenerate geometry each time.

Usage:
    python postprocess/run_postprocess.py \\
        --glb_path outputs/bread/result.glb \\
        --image_path data/bread/images/1.png \\
        --name bread --category "loaf of bread" \\
        --width_m 0.28 --height_m 0.09 --depth_m 0.12 \\
        --estimate_mass --export_sim_ready
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from scale_to_spec import rescale_mesh_to_spec
from estimate_mass import estimate_mass
from export_sim_ready import export_sim_ready


def main():
    parser = argparse.ArgumentParser(description="Run the full scale->mass->sim-export chain on an existing GLB.")
    parser.add_argument("--glb_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True,
                         help="Reference image for mass estimation (any view of the object)")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--category", type=str, default="unknown object")
    parser.add_argument("--width_m", type=float, required=True)
    parser.add_argument("--height_m", type=float, required=True)
    parser.add_argument("--depth_m", type=float, required=True)
    parser.add_argument("--spec_weight_kg", type=float, default=None)
    parser.add_argument("--estimate_mass", action="store_true")
    parser.add_argument("--export_sim_ready", action="store_true")
    parser.add_argument("--vlm_base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    glb_path = Path(args.glb_path)
    output_dir = Path(args.output_dir) if args.output_dir else glb_path.parent

    print(f"[1/3] Scaling {glb_path.name} to {args.width_m}x{args.height_m}x{args.depth_m} m ...")
    scale_result = rescale_mesh_to_spec(
        mesh_path=glb_path, width_m=args.width_m, height_m=args.height_m,
        depth_m=args.depth_m, output_path=output_dir / f"{args.name}_scaled.glb",
    )
    print(json.dumps(asdict(scale_result), indent=2))
    if scale_result.flagged_for_review:
        print(f"  ⚠ {scale_result.notes}")

    mass_result = None
    if args.estimate_mass or args.spec_weight_kg is not None:
        print(f"\n[2/3] Estimating mass ...")
        mass_result = estimate_mass(
            image_path=Path(args.image_path), width_m=args.width_m,
            height_m=args.height_m, depth_m=args.depth_m, category=args.category,
            spec_weight_kg=args.spec_weight_kg, vlm_base_url=args.vlm_base_url,
            vlm_model=args.vlm_model,
        )
        print(json.dumps(asdict(mass_result), indent=2))
    else:
        print("\n[2/3] Skipped (pass --estimate_mass or --spec_weight_kg)")

    if args.export_sim_ready:
        if mass_result is None:
            print("\n[3/3] Skipped: no mass available")
        else:
            print(f"\n[3/3] Exporting URDF/MJCF ...")
            asset = export_sim_ready(
                name=args.name, scaled_mesh_path=Path(scale_result.output_path),
                mass_kg=mass_result.mass_kg, material=mass_result.material,
                output_dir=output_dir,
            )
            print(json.dumps(asdict(asset), indent=2))
    else:
        print("\n[3/3] Skipped (pass --export_sim_ready)")


if __name__ == "__main__":
    main()