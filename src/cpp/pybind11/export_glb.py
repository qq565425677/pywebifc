#!/usr/bin/env python3
"""
Export a GLB from an IFC using the pywebifc bindings.

Pipeline:
  - open IFC with pywebifc
  - call build_gltf_like(model_id[, types]) to get scenes/nodes/meshes/materials
  - pack into a valid GLB (glTF 2.0) without external deps

Usage:
  python -m pybind11.export_glb input.ifc output.glb [--types 123 456] [--normals] [--winding {as-is,flip,auto}] [--metallicFactor 0.5] [--roughnessFactor 0.8] [--noClean]
  python pybind11/export_glb.py input.ifc output.glb [--normals] [--winding {as-is,flip,auto}] [--metallicFactor 0.5] [--roughnessFactor 0.8] [--noClean]

Notes:
    - Underlying C++ now always provides an interleaved (N,6) float array per primitive:
        [x y z nx ny nz]. Normals may be zeroed if not computed.
        When --normals is passed, exporter extracts columns 3..5 as NORMAL attribute.
        Without --normals only POSITION (first 3 floats) is used. No separate prim['normals'] key.
  - Each primitive becomes TRIANGLES with uint16/uint32 indices.
  - --winding can flip triangle index order (CW<->CCW). Default 'auto' attempts
    a quick orientation check (signed volume estimate) and flips if negative.
  - Materials map to pbrMetallicRoughness(baseColorFactor). If
    --metallicFactor/--roughnessFactor are provided, they are written to GLB;
    otherwise these fields are omitted (glTF defaults apply).
  - When normals are not exported, a lightweight NumPy-based clean runs per
    primitive to deduplicate identical vertices and faces, drop degenerate
    triangles, and compact unused vertices (similar to pyvista.clean but exact
    float matches, no tolerance-based merging). Use --noClean to disable.
"""
import argparse
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import sys
import traceback
import numpy as np
import time
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

from meshoptimizer import (
    optimize_vertex_cache,
    optimize_overdraw,
    optimize_vertex_fetch,
    simplify,
)


def ensure_build_path_on_sys_path() -> None:
    here = Path(__file__).resolve().parent
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


w = import_pywebifc()

# Set default web-ifc log level to 'warn' to suppress startup info spam.
# Users can override via --log-level. Best-effort: only if API exists.
try:
    if hasattr(w, "set_log_level_name"):
        w.set_log_level_name("warn")
    elif hasattr(w, "set_log_level"):
        # spdlog: 0=trace,1=debug,2=info,3=warn,4=err,5=critical,6=off
        w.set_log_level(3)
except Exception:
    pass

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


class Timer:
    def __init__(self, name=None, verbose=True):
        self.name = name
        self.verbose = verbose
        self.start = 0.0
        self.end = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        if self.verbose:
            label = f"{self.name}: " if self.name else ""
            print(f"{label}{self.elapsed:.6f} 秒")


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
    max_tris: int | None = None,
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

    if max_tris is not None and n_tris > max_tris:
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


def _clean_mesh_numpy(
    pos_f32_flat: np.ndarray,
    idx_u32: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lightweight mesh clean: deduplicate vertices and faces with NumPy.

    - Removes duplicate vertices (exact float32 matches)
    - Remaps indices accordingly
    - Drops degenerate triangles (with repeated vertex indices)
    - Drops duplicate triangles (ignoring winding by sorting vertex ids)
    - Compacts to only used vertices

    Args:
        pos_f32_flat: Flat (N*3,) float32 positions array
        idx_u32: Flat (M*3,) uint32 indices array

    Returns:
        (clean_pos_flat_float32, clean_idx_flat_uint32)
    """
    if pos_f32_flat.size == 0 or idx_u32.size == 0:
        return pos_f32_flat, idx_u32

    # Shapes
    pos = pos_f32_flat.reshape(-1, 3)
    tri = idx_u32.reshape(-1, 3)

    # 1) Deduplicate vertices (exact matches) and get inverse map
    # unique_pos[inverse[i]] == pos[i]
    unique_pos, inverse = np.unique(pos, axis=0, return_inverse=True)

    # Remap indices to unique vertex space
    tri = inverse[tri]

    # 2) Remove degenerate triangles (any repeated vertex within the triangle)
    mask_non_degen = (
        (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    )
    if not np.all(mask_non_degen):
        tri = tri[mask_non_degen]
        if tri.size == 0:
            # No triangles left; early out with compacted empty mesh
            return unique_pos.astype(np.float32, copy=False).reshape(-1), np.empty(
                (0,), dtype=np.uint32
            )

    # 3) Remove duplicate faces ignoring winding by sorting indices within each tri
    tri_sorted = np.sort(tri, axis=1)
    _, unique_face_idx = np.unique(tri_sorted, axis=0, return_index=True)
    if unique_face_idx.size != tri.shape[0]:
        unique_face_idx.sort()
        tri = tri[unique_face_idx]

    # 4) Compact to only used vertices
    used_old = np.unique(tri.reshape(-1))
    # Map old->new
    remap = np.full(unique_pos.shape[0], -1, dtype=np.int64)
    remap[used_old] = np.arange(used_old.size, dtype=np.int64)
    tri_compact = remap[tri]
    pos_compact = unique_pos[used_old]

    # Return flattened arrays with correct dtypes
    return (
        np.ascontiguousarray(pos_compact, dtype=np.float32).reshape(-1),
        np.ascontiguousarray(tri_compact, dtype=np.uint32).reshape(-1),
    )


def gltf_like_to_glb(
    g: Dict[str, Any],
    out_path: str | None = None,
    include_normals: bool = False,
    winding: str = "auto",  # as-is | flip | auto (default: auto)
    metallic_factor: Optional[float] = None,
    roughness_factor: Optional[float] = None,
    clean: bool = True,
) -> GLTF2:

    # ----- Helpers (local, keep namespace clean) -----
    def compute_winding_flip(pos_f32: np.ndarray, idx_u32: np.ndarray) -> bool:
        if winding == "flip":
            return True
        if winding == "as-is":
            return False
        # auto: estimate signed volume; sample for large meshes for speed xxxxxxxxxx
        # Keep behavior stable by defaulting to full pass for smaller meshes xxxxxxxxxx
        max_tris = None
        # do not sample, use all
        # tri_count = idx_u32.size // 3
        # if tri_count > 500_000:
        #     max_tris = 200_000
        signed = _estimate_orientation_signed_volume(
            pos_f32, idx_u32, max_tris=max_tris
        )
        return signed < 0.0

    def pack_vertices(
        pos_f32: np.ndarray,
        interleaved_src: Optional[np.ndarray],
        do_flip: bool,
    ) -> tuple[int, Optional[int]]:
        """Create bufferViews/accessors for POSITION (and NORMAL if present).

        Returns (acc_pos_idx, acc_nrm_idx_or_none).
        """
        # Common min/max for POSITION
        if pos_f32.size == 0:
            raise ValueError("Cannot pack empty position array")

        pos3 = pos_f32.reshape(-1, 3)
        mn = pos3.min(axis=0).astype(np.float32).tolist()
        mx = pos3.max(axis=0).astype(np.float32).tolist()
        vcount = int(pos3.shape[0])

        # Interleaved zero-copy path
        if include_normals and interleaved_src is not None:
            src = interleaved_src
            if do_flip:
                src = src.copy()
                src[:, 3:6] *= -1.0
            bv_idx = binb.add_view(
                np.ascontiguousarray(src, dtype=np.float32).tobytes(order="C"),
                ARRAY_BUFFER,
            )
            try:
                binb.buffer_views[bv_idx].byteStride = 6 * 4
            except Exception:
                raise ValueError("Failed to set byteStride for interleaved buffer view")
            acc_pos = binb.add_accessor(
                buffer_view=bv_idx,
                component_type=FLOAT,
                count=vcount,
                type_str="VEC3",
                byte_offset=0,
                minv=mn,
                maxv=mx,
            )
            acc_nrm = binb.add_accessor(
                buffer_view=bv_idx,
                component_type=FLOAT,
                count=vcount,
                type_str="VEC3",
                byte_offset=12,
            )
            return acc_pos, acc_nrm

        # Non-interleaved path; optionally include normals as separate view
        acc_nrm = None
        bv_pos = binb.add_view(pos_f32.tobytes(order="C"), ARRAY_BUFFER)
        acc_pos = binb.add_accessor(
            buffer_view=bv_pos,
            component_type=FLOAT,
            count=vcount,
            type_str="VEC3",
            minv=mn,
            maxv=mx,
        )
        return acc_pos, acc_nrm

    def pack_indices(idx_u32: np.ndarray) -> tuple[int, int, np.ndarray]:
        """Choose smallest component type, add view + accessor. Returns (acc_idx, comp_type, packed_arr)."""
        if idx_u32.size == 0:
            # Should not happen here; caller checks empties
            raise ValueError("Cannot pack empty index array")
        max_idx = int(idx_u32.max())
        if max_idx <= 65535:
            comp = UNSIGNED_SHORT
            arr = idx_u32.astype(np.uint16, copy=False)
        else:
            comp = UNSIGNED_INT
            arr = idx_u32
        bv_idx = binb.add_view(arr.tobytes(order="C"), ELEMENT_ARRAY_BUFFER)
        acc_idx = binb.add_accessor(
            buffer_view=bv_idx,
            component_type=comp,
            count=arr.size,
            type_str="SCALAR",
        )
        return acc_idx, comp, arr

    # ----- Build GLTF skeleton -----
    gltf = GLTF2(scene=0, scenes=g["scenes"])

    # Materials
    mats_in = g.get("materials", [])
    mats_out: List[GLTFMaterial] = []
    for m in mats_in:
        base = m.get("baseColorFactor", [0.78, 0.78, 0.78, 1.0])
        pbr = GLTFPBR(baseColorFactor=base)
        if metallic_factor is not None:
            pbr.metallicFactor = float(metallic_factor)
        if roughness_factor is not None:
            pbr.roughnessFactor = float(roughness_factor)
        mats_out.append(GLTFMaterial(pbrMetallicRoughness=pbr))
    gltf.materials = mats_out

    # Binary builder and meshes
    binb = BinBuilder()
    gltf_meshes: List[GLTFMesh] = []
    for mesh_idx, mesh in enumerate(g.get("meshes", [])):
        prims_out: List[GLTFPrimitive] = []
        for prim_idx, prim in enumerate(mesh.get("primitives", [])):
            interleaved_src = prim.get("points", None)
            faces = prim.get("faces", None)
            material_idx = prim.get("material", None)
            if any(x is None for x in (interleaved_src, faces, material_idx)):
                raise ValueError(
                    f"Primitive missing required data (mesh {mesh_idx}, prim {prim_idx})"
                )
            interleaved_src = np.ascontiguousarray(interleaved_src, dtype=np.float32)
            faces = np.ascontiguousarray(faces, dtype=np.uint32).reshape(-1)

            # 0) Optimize indices and vertices for better GPU performance
            optimize_vertex_cache(faces, faces)  # vertex_count is automatically derived
            optimize_overdraw(faces, faces, interleaved_src, vertex_positions_stride=24)
            optimize_vertex_fetch(interleaved_src, faces, interleaved_src)

            # 1) Positions, normals, counts
            points = np.ascontiguousarray(
                interleaved_src[:, :3], dtype=np.float32
            ).reshape(-1)
            if points.size == 0 or faces.size == 0:
                logging.warning(
                    "Empty primitive skipped (mesh %d, prim %d)", mesh_idx, prim_idx
                )
                continue

            # 2) Optional clean (only when normals are not exported)
            if (not include_normals) and clean:
                points, faces = w.clean_mesh(points, faces)
                if points.size == 0 or faces.size == 0:
                    logging.warning(
                        "Primitive became empty after clean (mesh %d, prim %d)",
                        mesh_idx,
                        prim_idx,
                    )
                    continue

            # 3) Winding
            do_flip = compute_winding_flip(points, faces)
            if do_flip:
                faces = _flip_winding_u32(faces)

            # 4) Pack vertices and indices
            acc_pos_idx, acc_nrm_idx = pack_vertices(points, interleaved_src, do_flip)
            acc_idx_idx, _, _ = pack_indices(faces)

            # 5) Assemble primitive
            prim_out = GLTFPrimitive()
            prim_out.attributes.POSITION = acc_pos_idx
            if acc_nrm_idx is not None:
                prim_out.attributes.NORMAL = acc_nrm_idx
            prim_out.indices = acc_idx_idx
            prim_out.mode = 4  # TRIANGLES
            prim_out.material = int(material_idx)
            prims_out.append(prim_out)

        if prims_out:
            gltf_meshes.append(GLTFMesh(primitives=prims_out))
        else:
            logging.warning("Mesh %d has no valid primitives", mesh_idx)

    # Finalize GLTF structure
    gltf.meshes = gltf_meshes
    gltf.buffers = [GLTFBuffer(byteLength=len(binb.blob))]
    gltf.bufferViews = binb.buffer_views
    gltf.accessors = binb.accessors
    gltf.set_binary_blob(bytes(binb.blob))

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

    if out_path:
        gltf.save(
            out_path, asset=GLTFAsset(version="2.0", generator="pywebifc-glb-exporter")
        )
    return gltf


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


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Export IFC to GLB via pywebifc")
    ap.add_argument("ifc", help="Path to IFC file")
    ap.add_argument("out", help="Output .glb path")
    ap.add_argument(
        "--log-level",
        choices=["trace", "debug", "info", "warn", "error", "critical", "off"],
        default=None,
        help="Set web-ifc log level (default: warn)",
    )
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
        "--noClean",
        dest="clean",
        action="store_false",
        default=True,
        help="Disable per-primitive mesh clean when not exporting normals",
    )
    args = ap.parse_args(argv)

    # Apply requested log level before opening model (both web-ifc and Python logging)
    if args.log_level:
        try:
            if hasattr(w, "set_log_level_name"):
                w.set_log_level_name(args.log_level)
            elif hasattr(w, "set_log_level"):
                _map = {
                    "trace": 0,
                    "debug": 1,
                    "info": 2,
                    "warn": 3,
                    "error": 4,
                    "critical": 5,
                    "off": 6,
                }
                w.set_log_level(_map[args.log_level])
        except Exception:
            pass

        # Configure Python logging level to match
        py_level = {
            "trace": logging.DEBUG,
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warn": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }.get(args.log_level, None)
        if args.log_level == "off":
            logging.disable(logging.CRITICAL)
        elif py_level is not None and not logging.getLogger().handlers:
            logging.basicConfig(level=py_level, format="%(levelname)s: %(message)s")

    with Timer("Open IFC"):
        mid = w.open_model(args.ifc)
    try:

        with Timer("Build GLTF-like"):
            # Prefer to avoid building normals unless requested, and share buffers
            # to speed up C++->Python transfer.
            data = w.build_gltf_like(
                mid, args.types, include_normals=args.normals, share_buffers=True
            )
        # Build IFC spatial hierarchy and assemble hierarchical nodes in Python
        with Timer("Build hierarchical nodes"):
            hierarchy = w.build_spatial_hierarchy(mid)
        with Timer("Assemble hierarchical nodes"):
            assembled = build_hierarchical_nodes(data, hierarchy)
        data["nodes"] = assembled["nodes"]
        data["scenes"] = assembled["scenes"]

        # Important: build_gltf_like now returns NumPy views on C++ memory.
        # Close the model only after binary packing is done.
        with Timer("Pack GLB"):
            gltf_like_to_glb(
                data,
                args.out,
                include_normals=args.normals,
                winding=args.winding,
                metallic_factor=args.metallicFactor,
                roughness_factor=args.roughnessFactor,
                clean=args.clean,
            )
    finally:
        w.close_model(mid)
    print(f"Wrote GLB: {args.out}")


if __name__ == "__main__":
    main()
