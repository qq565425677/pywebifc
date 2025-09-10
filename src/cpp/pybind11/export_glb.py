#!/usr/bin/env python3
"""
Export a GLB from an IFC using the pywebifc bindings.

Pipeline:
  - open IFC with pywebifc
  - call build_gltf_like(model_id[, types]) to get scenes/nodes/meshes/materials
  - pack into a valid GLB (glTF 2.0) without external deps

Usage:
  python -m pybind11.export_glb input.ifc output.glb [--types 123 456] [--normals] [--winding {as-is,flip,auto}] [--metallicFactor 0.5] [--roughnessFactor 0.8] [--warnEmpty]
  python pybind11/export_glb.py input.ifc output.glb [--normals] [--winding {as-is,flip,auto}] [--metallicFactor 0.5] [--roughnessFactor 0.8] [--warnEmpty]

Notes:
  - By default exports POSITION + INDICES. Use --normals to also export NORMALs
    if present in the glTF-like input. (No UVs.)
  - Each primitive becomes TRIANGLES with uint16/uint32 indices.
  - --winding can flip triangle index order (CW<->CCW). Default 'auto' attempts
    a quick orientation check (signed volume estimate) and flips if negative.
  - Materials map to pbrMetallicRoughness(baseColorFactor). If
    --metallicFactor/--roughnessFactor are provided, they are written to GLB;
    otherwise these fields are omitted (glTF defaults apply).
"""
import argparse
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import sys
import traceback
import numpy as np
import logging
from pygltflib import (
    GLTF2,
    Scene as GLTFScene,
    Node as GLTFNode,
    Mesh as GLTFMesh,
    Primitive as GLTFPrimitive,
    Buffer as GLTFBuffer,
    BufferView as GLTFBufferView,
    Accessor as GLTFAccessor,
    Asset as GLTFAsset,
    PbrMetallicRoughness as GLTFPBR,
    Material as GLTFMaterial,
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    UNSIGNED_SHORT,
    UNSIGNED_INT,
)

here = Path(__file__).resolve().parent

COMPONENT_TYPE_DTYPES = {
    FLOAT: np.float32,
    UNSIGNED_SHORT: np.uint16,
    UNSIGNED_INT: np.uint32,
}

TYPE_NUM_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


class BinBuilder:
    """Builds a single GLB binary buffer with 4-byte alignment and tracks
    bufferViews/accessors. Keeps JSON-free to avoid external deps.

    Usage:
      - add_view(data: bytes, target: Optional[int]) -> int (bufferView index)
      - add_accessor(...) -> int (accessor index)
      - blob, buffer_views, accessors attributes are used to assemble GLB
    """

    def __init__(self) -> None:
        self.blob = bytearray()
        self.buffer_views: List[Any] = []  # Dict or GLTFBufferView
        self.accessors: List[Any] = []  # Dict or GLTFAccessor

    def _align4(self):
        pad = (-len(self.blob)) % 4
        if pad:
            self.blob.extend(b"\x00" * pad)

    def add_view(self, data_bytes: bytes, target: Optional[int]) -> int:
        self._align4()
        byte_offset = len(self.blob)
        self.blob.extend(data_bytes)
        bv = GLTFBufferView(
            buffer=0,
            byteOffset=byte_offset,
            byteLength=len(data_bytes),
            target=target,
        )
        self.buffer_views.append(bv)
        return len(self.buffer_views) - 1

    def add_accessor(
        self,
        buffer_view: int,
        component_type: int,
        count: int,
        type_str: str,
        byte_offset: int = 0,
        minv: Optional[List[float]] = None,
        maxv: Optional[List[float]] = None,
    ) -> int:
        acc = GLTFAccessor(
            bufferView=buffer_view,
            byteOffset=byte_offset,
            componentType=component_type,
            count=int(count),
            type=type_str,
        )
        if minv is not None:
            acc.min = minv
        if maxv is not None:
            acc.max = maxv
        self.accessors.append(acc)
        return len(self.accessors) - 1


def _ensure_float32_xyz(x: Any) -> Tuple[np.ndarray, int]:
    """Return (flat_float32, vertex_count).

    Accepts list/tuple or numpy arrays with shape (N,3) or (3N,).
    Produces a contiguous float32 1D array of length 3N.
    """
    if x is None:
        return np.empty((0,), dtype=np.float32), 0
    arr = np.asarray(x)
    if arr.size == 0:
        return np.empty((0,), dtype=np.float32), 0
    if arr.ndim == 2 and arr.shape[1] == 3:
        vcount = int(arr.shape[0])
        flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
        return flat, vcount
    flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    if flat.size % 3 != 0:
        raise ValueError("POSITION/NORMAL array length must be multiple of 3")
    return flat, flat.size // 3


def _ensure_uint32_indices(x: Any) -> np.ndarray:
    if x is None:
        return np.empty((0,), dtype=np.uint32)
    arr = np.asarray(x)
    if arr.size == 0:
        return np.empty((0,), dtype=np.uint32)
    return np.ascontiguousarray(arr, dtype=np.uint32).reshape(-1)


def _estimate_orientation_signed_volume(
    pos_f32_flat: np.ndarray,
    idx_u32: np.ndarray,
    max_tris: int = 2000,
    random_state: int = 0,
) -> float:
    """Estimate mesh orientation via signed volume (sum over tetrahedra).

    Returns a signed value proportional to volume; negative often indicates CW winding
    relative to the origin. Uses up to `max_tris` triangles for speed (random sampling).

    Args:
        pos_f32_flat: Flat (N*3,) float32 vertex array.
        idx_u32: Flat (M*3,) uint32 index array.
        max_tris: Maximum number of triangles to sample.
        random_state: Optional seed for reproducible random sampling.
    """
    if pos_f32_flat.size == 0 or idx_u32.size < 3:
        return 0.0

    tri = idx_u32.reshape(-1, 3)
    n_tris = len(tri)

    if n_tris > max_tris:
        # 随机抽样 max_tris 个三角形
        rng = np.random.default_rng(random_state)
        choice = rng.choice(n_tris, size=max_tris, replace=False)
        tri = tri[choice]

    # Gather positions; compute v0 · (v1 × v2)
    p = pos_f32_flat.reshape(-1, 3).astype(np.float64, copy=False)
    v0 = p[tri[:, 0]]
    v1 = p[tri[:, 1]]
    v2 = p[tri[:, 2]]

    cross = np.cross(v1, v2)
    signed = float(np.einsum("ij,ij->i", v0, cross).sum())
    return signed


def _flip_winding_u32(idx_u32: np.ndarray) -> np.ndarray:
    if idx_u32.size == 0:
        return idx_u32
    tri = idx_u32.reshape(-1, 3).copy()
    tri[:, [1, 2]] = tri[:, [2, 1]]
    return tri.reshape(-1)


def gltf_like_to_glb(
    g: Dict[str, Any],
    out_path: str,
    include_normals: bool = False,
    winding: str = "auto",  # as-is | flip | auto (default: auto)
    metallic_factor: Optional[float] = None,
    roughness_factor: Optional[float] = None,
    warn_empty: bool = False,
) -> None:

    gltf = GLTF2(
        scene=0,
        scenes=g["scenes"],
    )

    # Materials
    materials_in = g.get("materials", [])
    materials_out: List[GLTFMaterial] = []
    for m in materials_in:
        base = m.get("baseColorFactor", [0.78, 0.78, 0.78, 1.0])
        pbr = GLTFPBR(baseColorFactor=base)
        if metallic_factor is not None:
            pbr.metallicFactor = float(metallic_factor)
        if roughness_factor is not None:
            pbr.roughnessFactor = float(roughness_factor)
        materials_out.append(GLTFMaterial(pbrMetallicRoughness=pbr))
    gltf.materials = materials_out

    # Build a single binary buffer via BinBuilder
    binb = BinBuilder()
    gltf_meshes: List[GLTFMesh] = []
    for mesh_idx, mesh in enumerate(g.get("meshes", [])):
        prims_out: List[GLTFPrimitive] = []
        for prim_idx, prim in enumerate(mesh.get("primitives", [])):
            points = prim.get("points", None)
            normals = prim.get("normals", None)
            faces = prim.get("faces", None)
            material_idx = prim.get("material")
            # Convert to NumPy arrays
            pos_f32, vcount = _ensure_float32_xyz(points)
            idx_u32 = _ensure_uint32_indices(faces)
            if vcount == 0 or idx_u32.size == 0:
                if warn_empty:
                    logging.warning(
                        f"Empty primitive skipped (mesh {mesh_idx}, prim {prim_idx})"
                    )
                continue
            # Decide whether to flip winding
            do_flip = False
            if winding == "flip":
                do_flip = True
            elif winding == "auto":
                signed = _estimate_orientation_signed_volume(pos_f32, idx_u32)
                do_flip = signed < 0.0
            if do_flip:
                idx_u32 = _flip_winding_u32(idx_u32)
            # Pack positions (float32)
            bv_pos_idx = binb.add_view(pos_f32.tobytes(order="C"), ARRAY_BUFFER)
            # Pack normals (float32) if requested and length matches
            has_normals = False
            if include_normals and normals is not None:
                nrm_f32, ncount = _ensure_float32_xyz(normals)
                has_normals = ncount == vcount
                if not has_normals and warn_empty:
                    logging.warning(
                        f"NORMAL count mismatch; ignoring normals (mesh {mesh_idx}, prim {prim_idx})"
                    )
                if has_normals:
                    # If we flipped winding, also invert normals to keep lighting consistent
                    if do_flip:
                        nrm_f32 = (-nrm_f32).astype(np.float32, copy=False)
                    bv_nrm_idx = binb.add_view(nrm_f32.tobytes(order="C"), ARRAY_BUFFER)
            # Pick smallest index component type and pack indices
            idx_comp_type = UNSIGNED_INT
            idx_arr = idx_u32
            if idx_u32.size:
                max_idx = int(idx_u32.max())
                if max_idx <= 65535:
                    idx_comp_type = UNSIGNED_SHORT
                    idx_arr = idx_u32.astype(np.uint16, copy=False)

            bv_idx_idx = binb.add_view(idx_arr.tobytes(order="C"), ELEMENT_ARRAY_BUFFER)

            # Accessors
            pos_reshaped = pos_f32.reshape((-1, 3))
            mn = pos_reshaped.min(axis=0).astype(np.float32).tolist()
            mx = pos_reshaped.max(axis=0).astype(np.float32).tolist()
            acc_pos_idx = binb.add_accessor(
                buffer_view=bv_pos_idx,
                component_type=FLOAT,
                count=vcount,
                type_str="VEC3",
                minv=mn,
                maxv=mx,
            )
            if has_normals:
                acc_nrm_idx = binb.add_accessor(
                    buffer_view=bv_nrm_idx,
                    component_type=FLOAT,
                    count=vcount,
                    type_str="VEC3",
                )
            acc_idx_idx = binb.add_accessor(
                buffer_view=bv_idx_idx,
                component_type=idx_comp_type,
                count=idx_arr.size,
                type_str="SCALAR",
            )

            prim_out = GLTFPrimitive()
            prim_out.attributes.POSITION = acc_pos_idx
            if has_normals:
                prim_out.attributes.NORMAL = acc_nrm_idx
            prim_out.indices = acc_idx_idx
            prim_out.mode = 4  # TRIANGLES
            if material_idx is not None:
                prim_out.material = int(material_idx)
            prims_out.append(prim_out)
        if prims_out:
            gltf_meshes.append(GLTFMesh(primitives=prims_out))
        elif warn_empty:
            logging.warning(f"Mesh {mesh_idx} has no valid primitives")
    gltf.meshes = gltf_meshes
    # Nodes and Scenes
    gltf.nodes = [
        GLTFNode(
            name=n.get("name", None),
            mesh=n.get("mesh", None),
            matrix=n.get("matrix", None),
            children=n.get("children", None),
            extras=n.get("extras", None),
        )
        for n in g.get("nodes", [])
    ]
    gltf.scenes = [GLTFScene(nodes=s.get("nodes", [])) for s in g.get("scenes", [])]
    gltf.buffers = [GLTFBuffer(byteLength=len(binb.blob))]
    gltf.bufferViews = binb.buffer_views
    gltf.accessors = binb.accessors
    # Attach binary and save
    gltf.set_binary_blob(bytes(binb.blob))
    gltf.save(
        out_path, asset=GLTFAsset(version="2.0", generator="pywebifc-glb-exporter")
    )


def build_hierarchical_nodes(
    g: Dict[str, Any], hierarchy: Dict[str, Any]
) -> Dict[str, Any]:
    # Build a fresh nodes/scenes list representing the IFC spatial tree.
    grouped_node_geoms = g.get("grouped_node_geoms", {})

    children_map = hierarchy.get("children", {})
    names_map = hierarchy.get("names", {})
    roots = hierarchy.get("roots", [])

    nodes: List[Dict[str, Any]] = []

    def make_node(
        name: Optional[str] = None,
        *,
        mesh: Optional[int] = None,
        matrix: Optional[List[float]] = None,
        children: Optional[List[int]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> int:
        n: Dict[str, Any] = {}
        if name is not None:
            n["name"] = name
        if mesh is not None:
            n["mesh"] = mesh
        if matrix is not None:
            n["matrix"] = matrix
        if children:
            n["children"] = children
        if extras:
            n["extras"] = extras
        nodes.append(n)
        return len(nodes) - 1

    def build_subtree(elt_id: int) -> int:
        # Returns node index for elt_id (spatial container or element)
        child_ids = children_map.get(elt_id, [])
        built_child_indices: List[int] = []

        # First build spatial/element sub-children
        for cid in child_ids:
            built_child_indices.append(build_subtree(cid))

        # If this is an element with geometry placements, attach them as leaf nodes
        geo_nodes: List[int] = []
        for id, geo in grouped_node_geoms.get(elt_id, {}).items():
            mesh_idx = geo.get("mesh")
            matrix = geo.get("matrix")
            name = f"#{id}"
            extras = {"id": id}
            geo_nodes.append(
                make_node(name=name, mesh=mesh_idx, matrix=matrix, extras=extras)
            )

        element_children: List[int] = []
        element_children.extend(built_child_indices)
        element_children.extend(geo_nodes)

        # Name: prefer IfcRoot.Name, fallback to #id
        name = names_map.get(elt_id, f"#{elt_id}")
        extras = {"id": elt_id}
        children = element_children if element_children else None
        idx = make_node(name=name, children=children, extras=extras)
        return idx

    scenes: List[Dict[str, List[int]]] = []
    for r in roots:
        scenes.append({"nodes": [build_subtree(r)]})
    return {"nodes": nodes, "scenes": scenes}


def ensure_build_path_on_sys_path() -> None:
    candidate = (here.parent / "build" / "pybind11").resolve()
    print(f"Checking for built module in: {candidate}")
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def import_pywebifc():
    try:
        import pywebifc  # type: ignore

        return pywebifc
    except Exception:
        ensure_build_path_on_sys_path()
        try:
            import pywebifc  # type: ignore

            return pywebifc
        except Exception as e:
            print("Failed to import pywebifc. Ensure the module is built.")
            traceback.print_exc()
            sys.exit(1)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Export IFC to GLB via pywebifc")
    ap.add_argument("ifc", help="Path to IFC file")
    ap.add_argument("out", help="Output .glb path")
    ap.add_argument(
        "--types",
        type=int,
        nargs="*",
        default=None,
        help="Optional IFC type codes to include",
    )
    ap.add_argument(
        "--normals",
        action="store_true",
        help="Include NORMAL attribute in GLB if available",
    )
    ap.add_argument(
        "--winding",
        choices=["as-is", "flip", "auto"],
        default="auto",
        help="Triangle winding: keep as-is, force flip, or auto-detect (default)",
    )
    ap.add_argument(
        "--metallicFactor",
        type=float,
        default=None,
        help="Optional metallicFactor (0..1). If omitted, not written",
    )
    ap.add_argument(
        "--roughnessFactor",
        type=float,
        default=None,
        help="Optional roughnessFactor (0..1). If omitted, not written",
    )
    ap.add_argument(
        "--warnEmpty",
        action="store_true",
        help="Log warnings when primitives are empty or normals mismatch",
    )
    args = ap.parse_args(argv)

    w = import_pywebifc()

    mid = w.open_model(args.ifc)
    try:
        # Optional logging setup for warnings
        if args.warnEmpty:
            import logging as _logging

            if not _logging.getLogger().handlers:
                _logging.basicConfig(
                    level=_logging.WARNING, format="%(levelname)s: %(message)s"
                )

        data = w.build_gltf_like(mid, args.types)
        # Build IFC spatial hierarchy and assemble hierarchical nodes in Python
        hierarchy = w.build_spatial_hierarchy(mid)
        assembled = build_hierarchical_nodes(data, hierarchy)
        data["nodes"] = assembled["nodes"]
        data["scenes"] = assembled["scenes"]

        # Important: build_gltf_like now returns NumPy views on C++ memory.
        # Close the model only after binary packing is done.
        gltf_like_to_glb(
            data,
            args.out,
            include_normals=args.normals,
            winding=args.winding,
            metallic_factor=args.metallicFactor,
            roughness_factor=args.roughnessFactor,
            warn_empty=args.warnEmpty,
        )
    finally:
        w.close_model(mid)
    print(f"Wrote GLB: {args.out}")


if __name__ == "__main__":
    main()
