"""
Post-processing Layer 3: package a scaled mesh + estimated mass into a
simulation-ready URDF and/or MJCF asset.

Deliberately NOT a port of PhysX-Anything's exporter: that tool is built
for part-decomposed, jointed objects (drawers, hinges, multi-body). Our
objects at this stage (box, plate, bread...) are simple single-body rigid
objects, so a lightweight single-link template is easier to build, debug,
and trust than adapting a multi-part articulation exporter.

Pipeline position: run this AFTER scale_to_spec.py (so the mesh already
has correct real-world dimensions) and estimate_mass.py (so you have a
mass_kg + material to plug in).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

logger = logging.getLogger("export_sim_ready")

# Coarse friction defaults by material -- same spirit as the density table
# in estimate_mass.py: good enough for a demo, not a materials database.
FRICTION_BY_MATERIAL = {
    "wood_solid": 0.5, "wood_particleboard": 0.5, "plastic": 0.35,
    "ceramic": 0.4, "glass": 0.2, "metal_steel": 0.4, "metal_aluminum": 0.4,
    "cardboard": 0.6, "fabric_foam": 0.7, "bread_baked_dough": 0.6,
    "unknown": 0.5,
}


@dataclass
class SimReadyAsset:
    name: str
    mesh_path: str
    collision_mesh_path: str
    mass_kg: float
    material: str
    friction: float
    inertia_diag: list      # [ixx, iyy, izz] about center of mass, principal axes
    center_of_mass: list
    urdf_path: Optional[str] = None
    mjcf_path: Optional[str] = None


def _make_collision_mesh(mesh: trimesh.Trimesh, target_face_count: int = 2000) -> trimesh.Trimesh:
    """
    Simplified collision proxy. Real deployments should swap this for a
    proper convex decomposition (VHACD / CoACD) so concave objects (e.g. a
    bowl) get multiple convex hulls instead of one hull that fills in the
    concavity. This function uses a single convex hull as a dependency-free
    placeholder -- flagged clearly so it isn't mistaken for the real thing.
    """
    hull = mesh.convex_hull
    return hull


def _compute_inertia(mesh: trimesh.Trimesh, mass_kg: float):
    """
    Uses trimesh's mass_properties, which assumes uniform density derived
    from the given mass and the mesh's own volume. Returns inertia in the
    mesh's principal axes (diagonal), plus center of mass, both of which
    URDF/MJCF expect.
    """
    mesh = mesh.copy()
    if not mesh.is_watertight:
        logger.warning(
            "[export_sim_ready] Mesh is not watertight -- trimesh's volume/"
            "inertia estimate may be inaccurate. Consider using the convex "
            "hull for inertia purposes if this matters for your sim."
        )
    density = mass_kg / max(mesh.volume, 1e-9)
    mesh.density = density
    props = mesh.mass_properties
    # Diagonalize to principal axes so URDF's simple ixx/iyy/izz (with the
    # object's local frame aligned to those axes) is valid.
    inertia_tensor = props["inertia"]
    eigvals, eigvecs = np.linalg.eigh(inertia_tensor)
    return eigvals, props["center_mass"], eigvecs


URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="{name}_link">
    <inertial>
      <origin xyz="{com_x} {com_y} {com_z}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{mesh_relpath}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{collision_relpath}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""

MJCF_TEMPLATE = """<mujoco model="{name}">
  <asset>
    <mesh name="{name}_visual" file="{mesh_relpath}"/>
    <mesh name="{name}_collision" file="{collision_relpath}"/>
  </asset>
  <worldbody>
    <body name="{name}" pos="0 0 0">
      <freejoint/>
      <inertial pos="{com_x} {com_y} {com_z}" mass="{mass}"
                diaginertia="{ixx} {iyy} {izz}"/>
      <geom type="mesh" mesh="{name}_visual" contype="0" conaffinity="0" group="1"/>
      <geom type="mesh" mesh="{name}_collision" friction="{friction} 0.005 0.0001"
            group="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def export_sim_ready(
    name: str,
    scaled_mesh_path: Path,
    mass_kg: float,
    material: str = "unknown",
    output_dir: Optional[Path] = None,
    export_urdf: bool = True,
    export_mjcf: bool = True,
) -> SimReadyAsset:
    scaled_mesh_path = Path(scaled_mesh_path)
    output_dir = Path(output_dir) if output_dir else scaled_mesh_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_or_mesh = trimesh.load(scaled_mesh_path, force="mesh")
    collision_mesh = _make_collision_mesh(scene_or_mesh)
    collision_path = output_dir / f"{name}_collision.obj"
    collision_mesh.export(str(collision_path))

    eigvals, com, _ = _compute_inertia(scene_or_mesh, mass_kg)
    friction = FRICTION_BY_MATERIAL.get(material, FRICTION_BY_MATERIAL["unknown"])

    asset = SimReadyAsset(
        name=name,
        mesh_path=str(scaled_mesh_path),
        collision_mesh_path=str(collision_path),
        mass_kg=mass_kg,
        material=material,
        friction=friction,
        inertia_diag=eigvals.tolist(),
        center_of_mass=com.tolist(),
    )

    fmt_kwargs = dict(
        name=name,
        mesh_relpath=scaled_mesh_path.name,
        collision_relpath=collision_path.name,
        mass=f"{mass_kg:.6f}",
        com_x=f"{com[0]:.6f}", com_y=f"{com[1]:.6f}", com_z=f"{com[2]:.6f}",
        ixx=f"{eigvals[0]:.8f}", iyy=f"{eigvals[1]:.8f}", izz=f"{eigvals[2]:.8f}",
        friction=f"{friction:.3f}",
    )

    if export_urdf:
        urdf_path = output_dir / f"{name}.urdf"
        urdf_path.write_text(URDF_TEMPLATE.format(**fmt_kwargs))
        asset.urdf_path = str(urdf_path)

    if export_mjcf:
        mjcf_path = output_dir / f"{name}.xml"
        mjcf_path.write_text(MJCF_TEMPLATE.format(**fmt_kwargs))
        asset.mjcf_path = str(mjcf_path)

    metadata_path = output_dir / f"{name}_sim_metadata.json"
    metadata_path.write_text(json.dumps(asdict(asset), indent=2))

    return asset


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export a scaled mesh + mass to URDF/MJCF.")
    parser.add_argument("scaled_mesh_path", type=str)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--mass_kg", type=float, required=True)
    parser.add_argument("--material", type=str, default="unknown")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asset = export_sim_ready(
        name=args.name,
        scaled_mesh_path=Path(args.scaled_mesh_path),
        mass_kg=args.mass_kg,
        material=args.material,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(asdict(asset), indent=2))


if __name__ == "__main__":
    main()