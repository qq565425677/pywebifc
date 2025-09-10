#!/usr/bin/env python3
"""
Export a GLB from an IFC using the pywebifc bindings.

Pipeline:
  - open IFC with pywebifc
  - call build_gltf_like(model_id[, types]) to get scenes/nodes/meshes/materials
  - pack into a valid GLB (glTF 2.0) without external deps

Usage:
  python -m pybind11.export_glb input.ifc output.glb [--types 123 456] [--normals] [--metallicFactor 0.5] [--roughnessFactor 0.8]
  python pybind11/export_glb.py input.ifc output.glb [--normals] [--metallicFactor 0.5] [--roughnessFactor 0.8]

Notes:
  - By default exports POSITION + INDICES. Use --normals to also export NORMALs
    if present in the glTF-like input. (No UVs.)
  - Each primitive becomes TRIANGLES with uint32 indices.
  - Materials map to pbrMetallicRoughness(baseColorFactor). If
    --metallicFactor/--roughnessFactor are provided, they are written to GLB;
    otherwise these fields are omitted (glTF defaults apply).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import sys
import traceback
import numpy as np

here = Path(__file__).resolve().parent


def _align4(n: int) -> int:
    return (n + 3) & ~3


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


def gltf_like_to_glb(
    g: Dict[str, Any],
    out_path: str,
    include_normals: bool = False,
    metallic_factor: Optional[float] = None,
    roughness_factor: Optional[float] = None,
) -> None:
    # Set up base glTF document
    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "pywebifc-export-glb"},
        "scene": 0,
        "scenes": [],
        "nodes": [],
        "meshes": [],
        "materials": [],
        "buffers": [],
        "bufferViews": [],
        "accessors": [],
    }

    # Copy scenes and nodes from input (caller may replace with hierarchical ones)
    gltf["nodes"] = g.get("nodes", [])
    gltf["scenes"] = g.get("scenes", [])

    # Materials: map to pbrMetallicRoughness
    materials_out: List[Dict[str, Any]] = []
    for m in g.get("materials", []):
        base = m.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        pbr: Dict[str, Any] = {"baseColorFactor": base}
        if metallic_factor is not None:
            pbr["metallicFactor"] = float(metallic_factor)
        if roughness_factor is not None:
            pbr["roughnessFactor"] = float(roughness_factor)
        materials_out.append({"pbrMetallicRoughness": pbr})
    gltf["materials"] = materials_out

    # Build a single binary buffer; append per-primitive blocks
    bin_blob = bytearray()
    accessors: List[Dict[str, Any]] = []
    buffer_views: List[Dict[str, Any]] = []

    def add_block(data_bytes: bytes, target: Optional[int]) -> tuple[int, int]:
        # returns (bufferViewIndex, accessorByteOffset)
        byte_offset = len(bin_blob)
        bin_blob.extend(data_bytes)
        # 4-byte align buffer after each block
        pad_len = _align4(len(bin_blob)) - len(bin_blob)
        if pad_len:
            bin_blob.extend(b"\x00" * pad_len)
        bv = {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(data_bytes),
        }
        if target is not None:
            bv["target"] = target
        buffer_views.append(bv)
        return len(buffer_views) - 1, byte_offset

    meshes_out: List[Dict[str, Any]] = []
    for mesh in g.get("meshes", []):
        prims_out: List[Dict[str, Any]] = []
        for prim in mesh.get("primitives", []):
            points = prim.get("points", None)
            normals = prim.get("normals", None)
            faces = prim.get("faces", None)
            material_idx = prim.get("material")

            # Convert to NumPy arrays
            pos_f32, vcount = _ensure_float32_xyz(points)
            idx_u32 = _ensure_uint32_indices(faces)
            if vcount == 0 or idx_u32.size == 0:
                continue

            # Pack positions (float32)
            bv_pos_idx, bv_pos_off = add_block(
                pos_f32.tobytes(order="C"), 34962
            )  # ARRAY_BUFFER

            # Pack normals (float32) if requested and length matches
            has_normals = False
            if include_normals and normals is not None:
                nrm_f32, ncount = _ensure_float32_xyz(normals)
                has_normals = ncount == vcount
                if has_normals:
                    bv_nrm_idx, _ = add_block(nrm_f32.tobytes(order="C"), 34962)

            # Pack indices (uint32)
            bv_idx_idx, bv_idx_off = add_block(
                idx_u32.tobytes(order="C"), 34963
            )  # ELEMENT_ARRAY_BUFFER

            # Accessors
            pos_reshaped = pos_f32.reshape((-1, 3))
            mn = pos_reshaped.min(axis=0).astype(np.float32).tolist()
            mx = pos_reshaped.max(axis=0).astype(np.float32).tolist()

            acc_pos = {
                "bufferView": bv_pos_idx,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": vcount,
                "type": "VEC3",
                "min": mn,
                "max": mx,
            }
            acc_idx = {
                "bufferView": bv_idx_idx,
                "byteOffset": 0,
                "componentType": 5125,  # UNSIGNED_INT
                "count": int(idx_u32.size),
                "type": "SCALAR",
            }
            acc_pos_idx = len(accessors)
            accessors.append(acc_pos)
            if has_normals:
                acc_nrm = {
                    "bufferView": bv_nrm_idx,
                    "byteOffset": 0,
                    "componentType": 5126,  # FLOAT
                    "count": vcount,
                    "type": "VEC3",
                }
                acc_nrm_idx = len(accessors)
                accessors.append(acc_nrm)

            acc_idx_idx = len(accessors)
            accessors.append(acc_idx)

            prim_out = {
                "attributes": {"POSITION": acc_pos_idx},
                "indices": acc_idx_idx,
                "mode": 4,  # TRIANGLES
            }
            if has_normals:
                prim_out["attributes"]["NORMAL"] = acc_nrm_idx
            if material_idx is not None:
                prim_out["material"] = int(material_idx)

            prims_out.append(prim_out)

        if prims_out:
            meshes_out.append({"primitives": prims_out})

    gltf["meshes"] = meshes_out
    # Update node.mesh indices are already matching gltf["meshes"] order
    # because we preserved the order in build_gltf_like.

    gltf["buffers"].append({"byteLength": len(bin_blob)})
    gltf["bufferViews"] = buffer_views
    gltf["accessors"] = accessors

    # Create GLB
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # pad JSON to 4-byte alignment with spaces
    json_pad = _align4(len(json_bytes)) - len(json_bytes)
    if json_pad:
        json_bytes += b" " * json_pad

    # GLB header
    # magic 'glTF' (0x46546C67), version 2, total length after composing
    def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
        return struct.pack("<I", len(payload)) + chunk_type + payload

    json_chunk = _chunk(b"JSON", json_bytes)
    bin_chunk = _chunk(b"BIN\x00", bytes(bin_blob)) if bin_blob else b""
    total_len = 12 + len(json_chunk) + len(bin_chunk)
    header = struct.pack("<III", 0x46546C67, 2, total_len)

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(json_chunk)
        if bin_chunk:
            f.write(bin_chunk)


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
    args = ap.parse_args(argv)

    w = import_pywebifc()

    mid = w.open_model(args.ifc)
    try:
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
            metallic_factor=args.metallicFactor,
            roughness_factor=args.roughnessFactor,
        )
    finally:
        w.close_model(mid)
    print(f"Wrote GLB: {args.out}")


if __name__ == "__main__":
    main()
