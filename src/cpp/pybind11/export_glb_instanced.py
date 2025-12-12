#!/usr/bin/env python3
"""
Enhanced GLB exporter with EXT_mesh_gpu_instancing support.

This version analyzes geometry reuse patterns and uses GPU instancing 
when beneficial for performance and file size.
"""
import argparse
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
import sys
import traceback
import numpy as np
import time
import logging
from collections import defaultdict
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

# Import pywebifc (same logic as original)
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

# Set default log level to suppress startup noise
try:
    if hasattr(w, "set_log_level_name"):
        w.set_log_level_name("warn")
    elif hasattr(w, "set_log_level"):
        w.set_log_level(3)  # warn level
except Exception:
    pass

# Same utility classes as original
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
    """Enhanced binary builder with instancing support."""
    
    def __init__(self) -> None:
        self.blob = bytearray()
        self.buffer_views: List[Any] = []
        self.accessors: List[Any] = []

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

def analyze_instancing_opportunities(grouped_node_geoms: Dict) -> Dict[str, List]:
    """
    分析实例化机会。返回按 (mesh_idx, material_idx) 分组的实例列表。
    
    Returns:
        instances_by_mesh: {f"{mesh_idx}_{material_idx}": [{"matrix": [...], "node_id": id}, ...]}
    """
    instances_by_mesh = defaultdict(list)
    
    for node_id, geometries in grouped_node_geoms.items():
        for geom_id, geom_data in geometries.items():
            mesh_idx = geom_data.get("mesh")
            matrix = geom_data.get("matrix")
            
            if mesh_idx is not None and matrix is not None:
                # 用mesh索引作为分组key（材质信息已经烘焙到mesh中）
                key = f"mesh_{mesh_idx}"
                instances_by_mesh[key].append({
                    "matrix": matrix,
                    "node_id": node_id,
                    "geom_id": geom_id
                })
    
    return instances_by_mesh

def matrix_to_trs(matrix: List[float]) -> Tuple[List[float], List[float], List[float]]:
    """
    将 glTF 列主序 4x4 矩阵分解为 Translation, Rotation(quaternion XYZW), Scale。

    关键点：
    - glTF 矩阵在 JSON 中按列主序存储；这里用 reshape(..., order='F') 正确还原。
    - 支持非均匀缩放：分别对每列归一化得到纯旋转矩阵。
    - 使用稳健的旋转矩阵->四元数转换（涵盖 trace<=0 的情况）。
    """
    import math

    arr = np.array(matrix, dtype=np.float64)
    # 两种读法：列主序(glTF规范)与行主序（部分来源可能如此）
    M_col = arr.reshape((4, 4), order="F")
    M_row = arr.reshape((4, 4), order="C")

    def decompose(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = M[:3, 3].astype(np.float64)
        RS = M[:3, :3].astype(np.float64)
        sx = float(np.linalg.norm(RS[:, 0])); sy = float(np.linalg.norm(RS[:, 1])); sz = float(np.linalg.norm(RS[:, 2]))
        eps = 1e-12
        nx = sx if abs(sx) > eps else 1.0
        ny = sy if abs(sy) > eps else 1.0
        nz = sz if abs(sz) > eps else 1.0
        R = RS.copy()
        R[:, 0] /= nx; R[:, 1] /= ny; R[:, 2] /= nz
        if np.linalg.det(R) < 0:
            i = int(np.argmax([abs(sx), abs(sy), abs(sz)]))
            if i == 0:
                sx = -sx; R[:, 0] *= -1.0
            elif i == 1:
                sy = -sy; R[:, 1] *= -1.0
            else:
                sz = -sz; R[:, 2] *= -1.0
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        return t, np.array([sx, sy, sz], dtype=np.float64), R

    tF, sF, RF = decompose(M_col)
    tC, sC, RC = decompose(M_row)

    # 选择更“可信”的分解：优先位移范数较大，其次缩放非零的方案
    def score(t: np.ndarray, s: np.ndarray) -> float:
        return float(np.linalg.norm(t)) + 1e-3 * float(np.linalg.norm(s))

    use_col = score(tF, sF) >= score(tC, sC)
    t = tF if use_col else tC
    sv = sF if use_col else sC
    R = RF if use_col else RC

    # 此时 R 已为最接近的正交旋转矩阵，s 为对应缩放向量
    eps = 1e-12

    # 旋转矩阵 -> 四元数（XYZW）
    trace = float(np.trace(R))
    if trace > 0.0:
        tmp = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * tmp
        qx = (R[2, 1] - R[1, 2]) / tmp
        qy = (R[0, 2] - R[2, 0]) / tmp
        qz = (R[1, 0] - R[0, 1]) / tmp
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            tmp = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            qx = 0.25 * tmp
            qy = (R[0, 1] + R[1, 0]) / tmp if tmp != 0 else 0.0
            qz = (R[0, 2] + R[2, 0]) / tmp if tmp != 0 else 0.0
            qw = (R[2, 1] - R[1, 2]) / tmp if tmp != 0 else 1.0
        elif R[1, 1] > R[2, 2]:
            tmp = math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            qx = (R[0, 1] + R[1, 0]) / tmp if tmp != 0 else 0.0
            qy = 0.25 * tmp
            qz = (R[1, 2] + R[2, 1]) / tmp if tmp != 0 else 0.0
            qw = (R[0, 2] - R[2, 0]) / tmp if tmp != 0 else 1.0
        else:
            tmp = math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            qx = (R[0, 2] + R[2, 0]) / tmp if tmp != 0 else 0.0
            qy = (R[1, 2] + R[2, 1]) / tmp if tmp != 0 else 0.0
            qz = 0.25 * tmp
            qw = (R[1, 0] - R[0, 1]) / tmp if tmp != 0 else 1.0

    # 归一化四元数，顺序为 XYZW（glTF）
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm_q = float(np.linalg.norm(q))
    if norm_q > eps:
        q /= norm_q

    translation = t.astype(np.float32).tolist()
    rotation = q.astype(np.float32).tolist()
    scale = [float(sv[0]), float(sv[1]), float(sv[2])]

    return translation, rotation, scale

def create_instanced_node(
    mesh_idx: int, 
    instances: List[Dict], 
    binb: BinBuilder,
    instance_threshold: int = 3
) -> Optional[Dict]:
    """
    为给定的mesh和实例列表创建GPU实例化节点。
    
    Args:
        mesh_idx: 要实例化的mesh索引
        instances: 实例列表，每个包含matrix等信息
        binb: 二进制构建器
        instance_threshold: 最小实例数阈值，少于此数量不使用实例化
        
    Returns:
        包含EXT_mesh_gpu_instancing扩展的节点字典，或None（如果不适合实例化）
    """
    if len(instances) < instance_threshold:
        return None
    
    # 分解为 TRANSLATION / ROTATION / SCALE（与大多数引擎对 EXT_mesh_gpu_instancing 的实现兼容）
    instance_count = len(instances)
    translations: List[float] = []
    rotations: List[float] = []
    scales: List[float] = []
    for inst in instances:
        t, r, s = matrix_to_trs(inst["matrix"])  # t: vec3, r: quat(xyzw), s: vec3
        translations.extend(t)
        rotations.extend(r)
        scales.extend(s)

    trans_arr = np.asarray(translations, dtype=np.float32)
    rot_arr = np.asarray(rotations, dtype=np.float32)
    scale_arr = np.asarray(scales, dtype=np.float32)

    bv_t = binb.add_view(trans_arr.tobytes(order="C"), ARRAY_BUFFER)
    acc_t = binb.add_accessor(bv_t, FLOAT, instance_count, "VEC3")
    bv_r = binb.add_view(rot_arr.tobytes(order="C"), ARRAY_BUFFER)
    acc_r = binb.add_accessor(bv_r, FLOAT, instance_count, "VEC4")
    bv_s = binb.add_view(scale_arr.tobytes(order="C"), ARRAY_BUFFER)
    acc_s = binb.add_accessor(bv_s, FLOAT, instance_count, "VEC3")

    # 准备拾取映射：实例索引 -> 原始节点/几何 ID
    orig_node_ids = [inst["node_id"] for inst in instances]
    orig_geom_ids = [inst["geom_id"] for inst in instances]

    # 创建实例化节点（EXT_mesh_gpu_instancing）
    node = {
        "mesh": mesh_idx,
        "name": f"Instanced_Mesh_{mesh_idx}",
        "extensions": {
            "EXT_mesh_gpu_instancing": {
                "attributes": {
                    "TRANSLATION": acc_t,
                    "ROTATION": acc_r,
                    "SCALE": acc_s,
                }
            }
        },
        "extras": {
            "instance_count": instance_count,
            "original_nodes": orig_node_ids,
            "original_geoms": orig_geom_ids,
        }
    }
    
    return node

def gltf_like_to_glb_instanced(
    g: Dict[str, Any],
    out_path: str | None = None,
    include_normals: bool = False,
    winding: str = "auto",
    metallic_factor: Optional[float] = None,
    roughness_factor: Optional[float] = None,
    clean: bool = True,
    instance_threshold: int = 3,
    use_instancing: bool = True,
    require_instancing: bool = False,
) -> GLTF2:
    """
    增强版GLB导出，支持EXT_mesh_gpu_instancing。
    
    Args:
        instance_threshold: 最小实例数阈值，少于此数量的不使用GPU实例化
        use_instancing: 是否启用GPU实例化
    """
    
    # 导入原始脚本的辅助函数
    from export_glb import (_ensure_float32_xyz, _ensure_uint32_indices, 
                            _estimate_orientation_signed_volume, _flip_winding_u32)

    gltf = GLTF2(scene=0)
    
    # 1. Materials (same as original)
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
    
    # 2. Build Meshes, Buffers, and Accessors
    binb = BinBuilder()
    gltf_meshes: List[GLTFMesh] = []
    
    # 使用与 export_glb.py 完全相同的健壮的 mesh 构建逻辑
    for mesh_idx, mesh in enumerate(g.get("meshes", [])):
        prims_out: List[GLTFPrimitive] = []
        for prim_idx, prim in enumerate(mesh.get("primitives", [])):
            points = prim.get("points", None)
            normals = prim.get("normals", None)
            faces = prim.get("faces", None)
            material_idx = prim.get("material")
            
            pos_f32, vcount = _ensure_float32_xyz(points)
            idx_u32 = _ensure_uint32_indices(faces)
            
            if vcount == 0 or idx_u32.size == 0:
                logging.warning(f"Empty primitive skipped (mesh {mesh_idx}, prim {prim_idx})")
                continue
            
            if (not include_normals) and clean:
                pos_f32, idx_u32 = w.clean_mesh(pos_f32, idx_u32)
                vcount = 0 if pos_f32.size == 0 else pos_f32.size // 3
                if vcount == 0 or idx_u32.size == 0:
                    logging.warning(f"Primitive became empty after clean (mesh {mesh_idx}, prim {prim_idx})")
                    continue
            
            do_flip = False
            if winding == "flip":
                do_flip = True
            elif winding == "auto":
                signed = _estimate_orientation_signed_volume(pos_f32, idx_u32)
                do_flip = signed < 0.0
            
            if do_flip:
                idx_u32 = _flip_winding_u32(idx_u32)
            
            bv_pos_idx = binb.add_view(pos_f32.tobytes(order="C"), ARRAY_BUFFER)
            
            has_normals = False
            bv_nrm_idx: int | None = None
            if include_normals and normals is not None:
                nrm_f32, ncount = _ensure_float32_xyz(normals)
                has_normals = ncount == vcount
                if has_normals:
                    if do_flip:
                        nrm_f32 = (-nrm_f32).astype(np.float32, copy=False)
                    bv_nrm_idx = binb.add_view(nrm_f32.tobytes(order="C"), ARRAY_BUFFER)
            
            idx_comp_type = UNSIGNED_INT
            idx_arr = idx_u32
            if idx_u32.size:
                max_idx = int(idx_u32.max())
                if max_idx <= 65535:
                    idx_comp_type = UNSIGNED_SHORT
                    idx_arr = idx_u32.astype(np.uint16, copy=False)
            
            bv_idx_idx = binb.add_view(idx_arr.tobytes(order="C"), ELEMENT_ARRAY_BUFFER)
            
            pos_reshaped = pos_f32.reshape((-1, 3))
            mn = pos_reshaped.min(axis=0).astype(np.float32).tolist()
            mx = pos_reshaped.max(axis=0).astype(np.float32).tolist()
            
            acc_pos_idx = binb.add_accessor(
                buffer_view=bv_pos_idx, component_type=FLOAT, count=vcount, type_str="VEC3", minv=mn, maxv=mx
            )
            
            acc_nrm_idx = None
            if has_normals:
                assert bv_nrm_idx is not None
                acc_nrm_idx = binb.add_accessor(
                    buffer_view=bv_nrm_idx, component_type=FLOAT, count=vcount, type_str="VEC3"
                )
            
            acc_idx_idx = binb.add_accessor(
                buffer_view=bv_idx_idx, component_type=idx_comp_type, count=idx_arr.size, type_str="SCALAR"
            )
            
            prim_out = GLTFPrimitive()
            prim_out.attributes.POSITION = acc_pos_idx
            if has_normals and acc_nrm_idx is not None:
                prim_out.attributes.NORMAL = acc_nrm_idx
            prim_out.indices = acc_idx_idx
            prim_out.mode = 4
            if material_idx is not None:
                prim_out.material = int(material_idx)
            prims_out.append(prim_out)
            
        if prims_out:
            gltf_meshes.append(GLTFMesh(primitives=prims_out))
        else:
            # 保持索引一致性，但要确保不引用空mesh
            gltf_meshes.append(GLTFMesh(primitives=[]))
            logging.warning(f"Mesh {mesh_idx} has no valid primitives and will be empty.")

    gltf.meshes = gltf_meshes

    # 3. Build Nodes and Scenes (Instanced or Traditional)
    grouped_node_geoms = g.get("grouped_node_geoms", {})
    
    if use_instancing:
        instances_by_mesh = analyze_instancing_opportunities(grouped_node_geoms)
        
        nodes = []
        scene_nodes = []
        instanced_meshes = set()
        
        for mesh_key, instances in instances_by_mesh.items():
            mesh_idx = int(mesh_key.split('_')[1])
            if len(instances) >= instance_threshold and gltf.meshes[mesh_idx].primitives:
                instanced_node = create_instanced_node(mesh_idx, instances, binb, instance_threshold)
                if instanced_node:
                    nodes.append(GLTFNode(**instanced_node))
                    scene_nodes.append(len(nodes) - 1)
                    instanced_meshes.add(mesh_idx)
                    print(f"使用GPU实例化: Mesh {mesh_idx}, {len(instances)} 个实例")
        
        # Fallback for non-instanced nodes
        for node_id, geometries in grouped_node_geoms.items():
            for geom_id, geom_data in geometries.items():
                mesh_idx = geom_data.get("mesh")
                if mesh_idx not in instanced_meshes and gltf.meshes[mesh_idx].primitives:
                    node = GLTFNode(
                        mesh=mesh_idx,
                        name=f"#{geom_id}",
                        matrix=geom_data.get("matrix"),
                        extras={"id": geom_id}
                    )
                    nodes.append(node)
                    scene_nodes.append(len(nodes) - 1)
        
        gltf.nodes = nodes
        gltf.scenes = [GLTFScene(nodes=scene_nodes)]
        
        if any(n.extensions for n in nodes):
            gltf.extensionsUsed = ["EXT_mesh_gpu_instancing"]
            if require_instancing:
                gltf.extensionsRequired = ["EXT_mesh_gpu_instancing"]
    
    else:
        # Traditional node-based export (fallback)
        nodes = []
        scene_nodes = []
        for node_id, geometries in grouped_node_geoms.items():
            for geom_id, geom_data in geometries.items():
                mesh_idx = geom_data.get("mesh")
                if gltf.meshes[mesh_idx].primitives:
                    node = GLTFNode(
                        mesh=mesh_idx,
                        name=f"#{geom_id}",
                        matrix=geom_data.get("matrix"),
                        extras={"id": geom_id}
                    )
                    nodes.append(node)
                    scene_nodes.append(len(nodes) - 1)
        gltf.nodes = nodes
        gltf.scenes = [GLTFScene(nodes=scene_nodes)]

    # 4. Finalize GLB
    gltf.buffers = [GLTFBuffer(byteLength=len(binb.blob))]
    gltf.bufferViews = binb.buffer_views
    gltf.accessors = binb.accessors
    gltf.set_binary_blob(bytes(binb.blob))
    
    if out_path:
        gltf.save(out_path, asset=GLTFAsset(
            version="2.0", 
            generator="pywebifc-instanced-glb-exporter"
        ))
    
    return gltf

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Export IFC to GLB with GPU instancing support")
    ap.add_argument("ifc", help="Path to IFC file")
    ap.add_argument("out", help="Output .glb path")
    ap.add_argument(
        "--instance-threshold", 
        type=int, 
        default=3,
        help="Minimum instances required for GPU instancing (default: 3)"
    )
    ap.add_argument(
        "--no-instancing",
        dest="use_instancing",
        action="store_false",
        default=True,
        help="Disable GPU instancing (fallback to node-based)"
    )
    ap.add_argument(
        "--require-instancing",
        dest="require_instancing",
        action="store_true",
        default=False,
        help="Mark EXT_mesh_gpu_instancing as required (default: not required)",
    )
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
    
    # Apply requested log level
    if args.log_level:
        try:
            if hasattr(w, "set_log_level_name"):
                w.set_log_level_name(args.log_level)
            elif hasattr(w, "set_log_level"):
                level_map = {"trace": 0, "debug": 1, "info": 2, "warn": 3, "error": 4, "critical": 5, "off": 6}
                w.set_log_level(level_map[args.log_level])
        except Exception:
            pass
    
    with Timer("Open IFC"):
        mid = w.open_model(args.ifc)
    
    try:
        with Timer("Build GLTF-like"):
            data = w.build_gltf_like(
                mid, args.types, include_normals=args.normals, share_buffers=True
            )
        
        with Timer("Build hierarchical nodes"):
            hierarchy = w.build_spatial_hierarchy(mid)
        
        # 注意：这里不使用原版的hierarchical nodes，而是直接分析grouped_node_geoms
        
        with Timer("Pack GLB with instancing"):
            gltf_like_to_glb_instanced(
                data,
                args.out,
                include_normals=args.normals,
                winding=args.winding,
                metallic_factor=args.metallicFactor,
                roughness_factor=args.roughnessFactor,
                clean=args.clean,
                instance_threshold=args.instance_threshold,
                use_instancing=args.use_instancing,
                require_instancing=args.require_instancing,
            )
            
    finally:
        w.close_model(mid)
    
    print(f"Wrote GLB with GPU instancing: {args.out}")

if __name__ == "__main__":
    main()