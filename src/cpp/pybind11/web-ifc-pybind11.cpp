/* Minimal pybind11 bindings for web-ifc
 * Implements only a few basic functions to validate build/run.
 */

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <unordered_set>
#include <cstring>
#include <array>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../version.h"
#include "../web-ifc/modelmanager/ModelManager.h"
#include "../web-ifc/parsing/IfcLoader.h"
#include "../web-ifc/geometry/IfcGeometryProcessor.h"
#include "../web-ifc/geometry/representation/geometry.h"
#include "../web-ifc/geometry/representation/IfcGeometry.h"

namespace py = pybind11;

// For this minimal binding, keep threading disabled by default.
static std::unique_ptr<webifc::manager::ModelManager> g_manager;

static webifc::manager::ModelManager &manager()
{
    if (!g_manager)
    {
        g_manager = std::make_unique<webifc::manager::ModelManager>(false /* mt_enabled */);
    }
    return *g_manager;
}

// Create a model with default LoaderSettings and return model ID.
static uint32_t CreateModel()
{
    webifc::manager::LoaderSettings settings; // defaults
    return manager().CreateModel(settings);
}

// Helper: build LoaderSettings from a Python dict/kwargs
static webifc::manager::LoaderSettings LoaderSettingsFromDict(py::dict kwargs)
{
    webifc::manager::LoaderSettings s;

    auto get_bool = [&](const char *k, bool &out) {
        py::str key(k);
        if (kwargs.contains(key))
            out = py::cast<bool>(kwargs[key]);
    };
    auto get_u16 = [&](const char *k, uint16_t &out) {
        py::str key(k);
        if (kwargs.contains(key))
            out = static_cast<uint16_t>(py::cast<int>(kwargs[key]));
    };
    auto get_u32 = [&](const char *k, uint32_t &out) {
        py::str key(k);
        if (kwargs.contains(key))
            out = static_cast<uint32_t>(py::cast<long long>(kwargs[key]));
    };
    auto get_double = [&](const char *k, double &out) {
        py::str key(k);
        if (kwargs.contains(key))
            out = py::cast<double>(kwargs[key]);
    };

    get_bool("coordinate_to_origin", s.COORDINATE_TO_ORIGIN);
    get_u16("circle_segments", s.CIRCLE_SEGMENTS);
    get_u32("tape_size", s.TAPE_SIZE);
    get_u32("memory_limit", s.MEMORY_LIMIT);
    get_u16("linewriter_buffer", s.LINEWRITER_BUFFER);
    get_double("tolerance_plane_intersection", s.TOLERANCE_PLANE_INTERSECTION);
    get_double("tolerance_plane_deviation", s.TOLERANCE_PLANE_DEVIATION);
    get_double("tolerance_back_deviation_distance", s.TOLERANCE_BACK_DEVIATION_DISTANCE);
    get_double("tolerance_inside_outside_perimeter", s.TOLERANCE_INSIDE_OUTSIDE_PERIMETER);
    get_double("tolerance_scalar_equality", s.TOLERANCE_SCALAR_EQUALITY);
    {
        // plane_refit_iterations is uint16_t but we accept int
        py::str key("plane_refit_iterations");
        if (kwargs.contains(key))
            s.PLANE_REFIT_ITERATIONS = static_cast<uint16_t>(py::cast<int>(kwargs[key]));
    }
    get_u16("boolean_union_threshold", s.BOOLEAN_UNION_THRESHOLD);

    return s;
}

// Create a model with custom settings provided via kwargs (overload for Python)
static uint32_t CreateModelWithSettings(py::kwargs kwargs)
{
    webifc::manager::LoaderSettings settings = LoaderSettingsFromDict(py::dict(kwargs));
    return manager().CreateModel(settings);
}

// Convenience: open an IFC model directly from a file path.
// Returns modelID on success.
static uint32_t OpenModel(const std::string &path)
{
    webifc::manager::LoaderSettings settings; // defaults
    uint32_t modelID = manager().CreateModel(settings);
    auto *loader = manager().GetIfcLoader(modelID);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to open IFC file: " + path);
    }
    loader->LoadFile(in);
    return modelID;
}

// Open a model with custom settings provided via kwargs (overload for Python)
static uint32_t OpenModelWithSettings(const std::string &path, py::kwargs kwargs)
{
    webifc::manager::LoaderSettings settings = LoaderSettingsFromDict(py::dict(kwargs));
    uint32_t modelID = manager().CreateModel(settings);
    auto *loader = manager().GetIfcLoader(modelID);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to open IFC file: " + path);
    }
    loader->LoadFile(in);
    return modelID;
}


static void CloseModel(uint32_t modelID)
{
    manager().CloseModel(modelID);
}

static void CloseAllModels()
{
    manager().CloseAllModels();
}

static bool IsModelOpen(uint32_t modelID)
{
    return manager().IsModelOpen(modelID);
}

static uint32_t GetMaxExpressID(uint32_t modelID)
{
    if (!manager().IsModelOpen(modelID))
        throw std::runtime_error("Model not open");
    return manager().GetIfcLoader(modelID)->GetMaxExpressId();
}

static uint32_t GetLineType(uint32_t modelID, uint32_t expressID)
{
    if (!manager().IsModelOpen(modelID))
        throw std::runtime_error("Model not open");
    return manager().GetIfcLoader(modelID)->GetLineType(expressID);
}

static std::vector<uint32_t> GetAllLines(uint32_t modelID)
{
    if (!manager().IsModelOpen(modelID))
        throw std::runtime_error("Model not open");
    return manager().GetIfcLoader(modelID)->GetAllLines();
}

static std::string GetVersion()
{
    return std::string(WEB_IFC_VERSION_NUMBER);
}

// Set global log level for web-ifc (via ModelManager), using spdlog levels.
// 0=trace,1=debug,2=info,3=warn,4=err,5=critical,6=off
static void SetLogLevel(uint8_t level)
{
    // Ensure manager exists so SetLogLevel also sets pattern once
    (void)manager();
    g_manager->SetLogLevel(level);
}

static void SetLogLevelName(const std::string &name)
{
    std::string s;
    s.resize(name.size());
    std::transform(name.begin(), name.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    uint8_t lvl = 2; // default info
    if (s == "trace")
        lvl = 0;
    else if (s == "debug")
        lvl = 1;
    else if (s == "info")
        lvl = 2;
    else if (s == "warn" || s == "warning")
        lvl = 3;
    else if (s == "err" || s == "error")
        lvl = 4;
    else if (s == "critical" || s == "crit")
        lvl = 5;
    else if (s == "off" || s == "none")
        lvl = 6;
    SetLogLevel(lvl);
}

// Convert a FlatMesh for a single element to a lightweight Python dict
// with placement transforms and geometry references.
static py::dict GetFlatMeshPy(uint32_t modelID, uint32_t expressID)
{
    if (!manager().IsModelOpen(modelID))
        throw std::runtime_error("Model not open");

    auto *geomProc = manager().GetGeometryProcessor(modelID);
    webifc::geometry::IfcFlatMesh flat = geomProc->GetFlatMesh(expressID);

    // Ensure vertex data is prepared for referenced geometries (fills fvertexData)
    for (auto &pg : flat.geometries)
    {
        auto &g = geomProc->GetGeometry(pg.geometryExpressID);
        g.GetVertexData();
        (void)g.GetIndexData();
    }

    py::list py_geoms;
    for (const auto &pg : flat.geometries)
    {
        py::dict item;
        item["geometry_express_id"] = py::int_(pg.geometryExpressID);
        // matrix as 16-length row-major array from stored flatTransformation
        py::list mat;
        for (int i = 0; i < 16; ++i)
            mat.append(pg.flatTransformation[i]);
        item["matrix"] = std::move(mat);
        // color as RGBA if available; color channels are doubles [0,1]
        py::list col;
        col.append(pg.color.r);
        col.append(pg.color.g);
        col.append(pg.color.b);
        col.append(pg.color.a);
        item["color_rgba"] = std::move(col);
        py_geoms.append(std::move(item));
    }

    py::dict result;
    result["express_id"] = py::int_(flat.expressID);
    result["geometries"] = std::move(py_geoms);
    return result;
}

// Build a glTF-like scene graph (scenes, nodes, meshes, materials) entirely
// in Python data structures for easy debugging/assembly on the Python side.
// - Each mesh has one primitive: points (float32 list), faces (uint32 index list), material index
// - Each node references a mesh index, has a name "#<id>", and a 4x4 matrix (length-16 list)
// - Materials are deduplicated by RGBA color pulled from placements; simple unlit-like baseColor
static py::dict BuildGLTFLike(
    uint32_t modelID,
    std::optional<std::vector<uint32_t>> optTypes,
    bool includeNormals = true,
    bool shareBuffers = false)
{
    if (!manager().IsModelOpen(modelID))
        throw std::runtime_error("Model not open");

    auto *loader = manager().GetIfcLoader(modelID);
    auto *geomProc = manager().GetGeometryProcessor(modelID);

    // Collect express IDs
    std::vector<uint32_t> expressIds;
    if (optTypes && !optTypes->empty())
    {
        for (auto t : *optTypes)
        {
            auto ids = loader->GetExpressIDsWithType(t);
            expressIds.insert(expressIds.end(), ids.begin(), ids.end());
        }
    }
    else
    {
        for (auto t : manager().GetSchemaManager().GetIfcElementList())
        {
            if (t == webifc::schema::IFCOPENINGELEMENT || t == webifc::schema::IFCSPACE || t == webifc::schema::IFCOPENINGSTANDARDCASE)
                continue;
            auto ids = loader->GetExpressIDsWithType(t);
            expressIds.insert(expressIds.end(), ids.begin(), ids.end());
        }
    }

    // Outputs
    py::dict groupedNodeGeoms; // express_id -> geometry_express_id -> meshIdx, matrix
    py::list meshes;           // mesh objects
    py::list materials;        // material objects (simple baseColor only)

    // Dedup maps
    // Deduplicate meshes by (geometryExpressID, materialIndex) so the same geometry
    // with different materials becomes distinct meshes.
    std::unordered_map<uint64_t, uint32_t> geomMatToMeshIdx; // (geom<<32)|mat -> mesh index
    std::unordered_map<std::string, uint32_t> colorToMatIdx; // "r,g,b,a" -> material index

    // Cache extracted arrays per geometry so multiple materials can reuse
    // the same position/normal/index arrays without re-extracting/copying.
    struct CachedGeomArrays {
        py::array points;   // keep generic to allow zero-copy views
        py::array normals;  // may be empty when includeNormals=false
        py::array faces;
    };
    std::unordered_map<uint32_t, CachedGeomArrays> geomArraysCache; // geometryExpressID -> arrays

    auto get_material_index = [&](const webifc::geometry::IfcPlacedGeometry &pg) -> uint32_t
    {
        // Quantize to 6 decimals to keep keys stable
        auto key = std::to_string((double)llround(pg.color.r * 1e6) / 1e6) + "," +
                   std::to_string((double)llround(pg.color.g * 1e6) / 1e6) + "," +
                   std::to_string((double)llround(pg.color.b * 1e6) / 1e6) + "," +
                   std::to_string((double)llround(pg.color.a * 1e6) / 1e6);
        auto it = colorToMatIdx.find(key);
        if (it != colorToMatIdx.end())
            return it->second;

        py::dict mat;
        // Minimal PBR-like dict: only baseColorFactor
        py::list baseColor;
        baseColor.append(pg.color.r);
        baseColor.append(pg.color.g);
        baseColor.append(pg.color.b);
        baseColor.append(pg.color.a);
        mat["baseColorFactor"] = std::move(baseColor);

        uint32_t newIdx = static_cast<uint32_t>(materials.size());
        materials.append(std::move(mat));
        colorToMatIdx.emplace(key, newIdx);
        return newIdx;
    };

    auto ensure_mesh_for_geometry = [&](uint32_t geometryExpressID, uint32_t materialIndex) -> uint32_t
    {
        uint64_t key = (static_cast<uint64_t>(geometryExpressID) << 32) | static_cast<uint64_t>(materialIndex);
        auto it = geomMatToMeshIdx.find(key);
        if (it != geomMatToMeshIdx.end())
            return it->second;

        // Prepare or reuse geometry data arrays
        if (!geomArraysCache.contains(geometryExpressID))
        {
            auto &geom = geomProc->GetGeometry(geometryExpressID);
            geom.GetVertexData();              // fills float buffer
            const auto &fv = geom.fvertexData; // interleaved [x y z nx ny nz] per vertex
            const auto &idx = geom.indexData;  // 3 per triangle

            const py::ssize_t item_stride = static_cast<py::ssize_t>(webifc::geometry::VERTEX_FORMAT_SIZE_FLOATS);
            const py::ssize_t row_stride_bytes = item_stride * static_cast<py::ssize_t>(sizeof(float));

            if (shareBuffers)
            {
                // Zero-copy NumPy views into C++ buffers with proper strides.
                // NOTE: The underlying memory must remain valid until the GLB pack.
                // Exporter keeps the model open until after packing, so this is safe.
                float *base_ptr = const_cast<float *>(fv.data());
                // points view: offset 0
                py::array points(
                    py::buffer_info(
                        base_ptr,
                        sizeof(float),
                        py::format_descriptor<float>::format(),
                        2,
                        {static_cast<py::ssize_t>(geom.numPoints), static_cast<py::ssize_t>(3)},
                        {row_stride_bytes, static_cast<py::ssize_t>(sizeof(float))}));

                py::array normals;
                if (includeNormals)
                {
                    // normals view: offset +3 floats
                    py::array tmp(
                        py::buffer_info(
                            base_ptr + 3,
                            sizeof(float),
                            py::format_descriptor<float>::format(),
                            2,
                            {static_cast<py::ssize_t>(geom.numPoints), static_cast<py::ssize_t>(3)},
                            {row_stride_bytes, static_cast<py::ssize_t>(sizeof(float))}));
                    normals = std::move(tmp);
                }

                // faces view: contiguous indices
                py::array faces(
                    py::buffer_info(
                        const_cast<uint32_t *>(idx.data()),
                        sizeof(uint32_t),
                        py::format_descriptor<uint32_t>::format(),
                        1,
                        {static_cast<py::ssize_t>(idx.size())},
                        {static_cast<py::ssize_t>(sizeof(uint32_t))}));

                geomArraysCache.emplace(geometryExpressID, CachedGeomArrays{std::move(points), std::move(normals), std::move(faces)});
            }
            else
            {
                // Copying path (previous behavior), but skip normals if not requested
                py::array_t<float> points({static_cast<py::ssize_t>(geom.numPoints), static_cast<py::ssize_t>(3)});
                {
                    auto p = points.mutable_unchecked<2>();
                    const float *src = fv.data();
                    for (size_t i = 0; i < geom.numPoints; ++i)
                    {
                        const float *v = src + i * item_stride;
                        p(i, 0) = v[0];
                        p(i, 1) = v[1];
                        p(i, 2) = v[2];
                    }
                }

                py::array normals; // may stay empty
                if (includeNormals)
                {
                    py::array_t<float> nrm({static_cast<py::ssize_t>(geom.numPoints), static_cast<py::ssize_t>(3)});
                    auto n = nrm.mutable_unchecked<2>();
                    const float *src = fv.data();
                    for (size_t i = 0; i < geom.numPoints; ++i)
                    {
                        const float *v = src + i * item_stride + 3;
                        n(i, 0) = v[0];
                        n(i, 1) = v[1];
                        n(i, 2) = v[2];
                    }
                    normals = std::move(nrm);
                }

                py::array_t<uint32_t> faces({static_cast<py::ssize_t>(idx.size())});
                if (!idx.empty())
                {
                    std::memcpy(faces.mutable_data(), idx.data(), idx.size() * sizeof(uint32_t));
                }

                geomArraysCache.emplace(geometryExpressID, CachedGeomArrays{std::move(points), std::move(normals), std::move(faces)});
            }
        }

        // Primitive
        py::dict prim;
        prim["material"] = py::int_(materialIndex);
        // Reuse cached arrays (no move), so subsequent meshes for the same
        // geometry can share buffers without re-extraction/copy.
        const auto &cached = geomArraysCache.at(geometryExpressID);
        prim["points"] = cached.points;
        prim["faces"] = cached.faces;
        if (includeNormals && !cached.normals.is_none())
        {
            prim["normals"] = cached.normals;
        }

        py::list prims;
        prims.append(std::move(prim));

        py::dict mesh;
        mesh["primitives"] = std::move(prims);

        uint32_t meshIdx = static_cast<uint32_t>(meshes.size());
        meshes.append(std::move(mesh));
        geomMatToMeshIdx.emplace(key, meshIdx);
        return meshIdx;
    };

    // Build nodes from flat meshes
    for (auto id : expressIds)
    {
        webifc::geometry::IfcFlatMesh flat = geomProc->GetFlatMesh(id);
        if (flat.geometries.empty())
            continue;
        groupedNodeGeoms[py::int_(id)] = py::dict();
        for (auto &pg : flat.geometries)
        {
            uint32_t matIdx = get_material_index(pg);
            uint32_t meshIdx = ensure_mesh_for_geometry(pg.geometryExpressID, matIdx);

            py::dict geoNode;
            geoNode["mesh"] = py::int_(meshIdx);

            py::list mat;
            for (int i = 0; i < 16; ++i)
                mat.append(pg.flatTransformation[i]);
            geoNode["matrix"] = std::move(mat);

            groupedNodeGeoms[py::int_(id)][py::int_(pg.geometryExpressID)] = std::move(geoNode);
        }
    }

    py::dict out;
    out["grouped_node_geoms"] = std::move(groupedNodeGeoms);
    out["meshes"] = std::move(meshes);
    out["materials"] = std::move(materials);
    return out;
}

// Helper: best-effort IfcRoot.Name. Falls back to empty string if missing.
static std::string GetIfcRootNameSafely(webifc::parsing::IfcLoader *loader, uint32_t expressID)
{
    if (!loader)
        throw std::runtime_error("Loader is null");
    // IfcRoot attributes are first in the flattened list. Name is the third
    // argument after GlobalId and OwnerHistory (index 2). OwnerHistory may be
    // $, but position remains the same.
    loader->MoveToArgumentOffset(expressID, 2);
    auto tk = loader->GetTokenType();
    if (tk == webifc::parsing::IfcTokenType::STRING || tk == webifc::parsing::IfcTokenType::LABEL)
    {
        loader->StepBack();
        try
        {
            return loader->GetDecodedStringArgument();
        }
        catch (...)
        {
            return std::string();
        }
    }
    return std::string();
}

// Build spatial/decomposition hierarchy via IfcRelAggregates and
// IfcRelContainedInSpatialStructure. Returns dict with:
// {
//   'roots': [int],
//   'children': { str(parentId): [int, ...] },
//   'names': { str(id): str },
//   'types': { str(id): int }
// }
static py::dict BuildSpatialHierarchy(uint32_t modelID)
{
    if (!manager().IsModelOpen(modelID))
        throw std::runtime_error("Model not open");

    auto *loader = manager().GetIfcLoader(modelID);

    std::unordered_map<uint32_t, std::vector<uint32_t>> children;
    std::unordered_set<uint32_t> allNodes;

    // 1) Aggregation: parent -> children (e.g., Project->Site->Building->Storey)
    {
        auto rels = loader->GetExpressIDsWithType(webifc::schema::IFCRELAGGREGATES);
        for (auto relID : rels)
        {
            loader->MoveToArgumentOffset(relID, 4);
            // order here: RelatingObject, then RelatedObjects (SET)
            uint32_t parent = loader->GetRefArgument();
            auto set = loader->GetSetArgument();
            for (auto tape : set)
            {
                uint32_t child = loader->GetRefArgument(tape);
                children[parent].push_back(child);
                allNodes.insert(child);
            }
            allNodes.insert(parent);
        }
    }

    // 2) Containment: RelatingStructure -> RelatedElements (SET)
    {
        auto rels = loader->GetExpressIDsWithType(webifc::schema::IFCRELCONTAINEDINSPATIALSTRUCTURE);
        for (auto relID : rels)
        {
            // For IfcRelContainedInSpatialStructure the order is typically
            // RelatedElements (SET) followed by RelatingStructure.
            loader->MoveToArgumentOffset(relID, 4);
            auto set = loader->GetSetArgument();
            uint32_t parent = loader->GetRefArgument();
            for (auto tape : set)
            {
                uint32_t child = loader->GetRefArgument(tape);
                children[parent].push_back(child);
                allNodes.insert(child);
            }
            allNodes.insert(parent);
        }
    }

    // 3) Roots: prefer IfcProject(s); if none, compute nodes that are never a child
    std::vector<uint32_t> roots = loader->GetExpressIDsWithType(webifc::schema::IFCPROJECT);
    if (roots.empty())
    {
        std::unordered_set<uint32_t> asChild;
        for (auto &kv : children)
        {
            for (auto c : kv.second)
                asChild.insert(c);
        }
        for (auto &kv : children)
        {
            uint32_t p = kv.first;
            if (!asChild.contains(p))
                roots.push_back(p);
        }
        // if still empty, as a last resort include all nodes
        if (roots.empty())
        {
            for (auto id : allNodes)
                roots.push_back(id);
        }
    }

    // 4) Names and types for all encountered nodes
    std::unordered_map<uint32_t, std::string> names;
    std::unordered_map<uint32_t, uint32_t> types;
    auto add_name_type = [&](uint32_t id)
    {
        if (!names.contains(id))
        {
            auto nm = GetIfcRootNameSafely(loader, id);
            if (nm.empty())
            {
                nm = std::string("#") + std::to_string(id);
            }
            names.emplace(id, std::move(nm));
            types.emplace(id, loader->GetLineType(id));
        }
    };

    for (auto r : roots)
        add_name_type(r);
    for (auto &kv : children)
    {
        add_name_type(kv.first);
        for (auto c : kv.second)
            add_name_type(c);
    }

    // 5) Convert to Python dicts/lists
    py::list py_roots;
    for (auto r : roots)
        py_roots.append(py::int_(r));

    py::dict py_children;
    for (auto &kv : children)
    {
        py::list lst;
        for (auto c : kv.second)
            lst.append(py::int_(c));
        py_children[py::int_(kv.first)] = std::move(lst);
    }

    py::dict py_names;
    for (auto &kv : names)
    {
        py_names[py::int_(kv.first)] = kv.second;
    }

    py::dict py_types;
    for (auto &kv : types)
    {
        py_types[py::int_(kv.first)] = py::int_(kv.second);
    }

    py::dict out;
    out["roots"] = std::move(py_roots);
    out["children"] = std::move(py_children);
    out["names"] = std::move(py_names);
    out["types"] = std::move(py_types);
    return out;
}

// ---------------------------------------------------------------------
// Mesh utilities: C++ implementation of clean_mesh exposed to Python
// ---------------------------------------------------------------------
static py::tuple CleanMesh(py::array pos_f32_flat, py::array idx_u32_flat)
{
    // Request buffers with expected dtypes and C layout
    auto pos = py::array_t<float, py::array::c_style | py::array::forcecast>(pos_f32_flat);
    auto idx = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>(idx_u32_flat);

    auto bpos = pos.request();
    auto bidx = idx.request();

    const size_t npos = static_cast<size_t>(bpos.size);
    const size_t nidx = static_cast<size_t>(bidx.size);

    if (npos == 0 || nidx == 0)
    {
        // Return copies to ensure ownership from Python side
        auto out_pos = py::array_t<float>(npos);
        auto out_idx = py::array_t<uint32_t>(nidx);
        if (npos)
            std::memcpy(out_pos.mutable_data(), bpos.ptr, npos * sizeof(float));
        if (nidx)
            std::memcpy(out_idx.mutable_data(), bidx.ptr, nidx * sizeof(uint32_t));
        return py::make_tuple(out_pos, out_idx);
    }

    if (npos % 3 != 0)
        throw std::runtime_error("pos_f32_flat length must be a multiple of 3");
    if (nidx % 3 != 0)
        throw std::runtime_error("idx_u32_flat length must be a multiple of 3");

    const size_t num_vertices = npos / 3;
    const size_t num_tris = nidx / 3;

    const float *pos_ptr = static_cast<const float *>(bpos.ptr);
    const uint32_t *idx_ptr = static_cast<const uint32_t *>(bidx.ptr);

    // 1) Deduplicate vertices (exact float32 matches using bitwise equality)
    struct Arr3U32Hash
    {
        size_t operator()(const std::array<uint32_t, 3> &a) const noexcept
        {
            // Simple mix of three 32-bit values
            size_t h = static_cast<size_t>(a[0]) * 0x9E3779B185EBCA87ULL;
            h ^= static_cast<size_t>(a[1]) + 0x9E3779B185EBCA87ULL + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(a[2]) + 0x9E3779B185EBCA87ULL + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct Arr3U32Eq
    {
        bool operator()(const std::array<uint32_t, 3> &a, const std::array<uint32_t, 3> &b) const noexcept
        {
            return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
        }
    };

    std::unordered_map<std::array<uint32_t, 3>, uint32_t, Arr3U32Hash, Arr3U32Eq> vert_map;
    vert_map.reserve(num_vertices * 2 + 1);

    std::vector<float> unique_pos;
    unique_pos.reserve(npos);

    std::vector<uint32_t> inverse(num_vertices);

    for (size_t i = 0; i < num_vertices; ++i)
    {
        // Read 3 floats, reinterpret as 3 uint32_t (bitwise equality)
        std::array<uint32_t, 3> key;
        std::memcpy(&key[0], pos_ptr + 3 * i + 0, sizeof(uint32_t));
        std::memcpy(&key[1], pos_ptr + 3 * i + 1, sizeof(uint32_t));
        std::memcpy(&key[2], pos_ptr + 3 * i + 2, sizeof(uint32_t));

        auto it = vert_map.find(key);
        if (it == vert_map.end())
        {
            uint32_t new_idx = static_cast<uint32_t>(unique_pos.size() / 3);
            vert_map.emplace(key, new_idx);
            unique_pos.push_back(pos_ptr[3 * i + 0]);
            unique_pos.push_back(pos_ptr[3 * i + 1]);
            unique_pos.push_back(pos_ptr[3 * i + 2]);
            inverse[i] = new_idx;
        }
        else
        {
            inverse[i] = it->second;
        }
    }

    // 2) Remap indices into unique vertex space and drop degenerate triangles
    std::vector<std::array<uint32_t, 3>> tris;
    tris.reserve(num_tris);
    for (size_t t = 0; t < num_tris; ++t)
    {
        uint32_t a = inverse[idx_ptr[3 * t + 0]];
        uint32_t b = inverse[idx_ptr[3 * t + 1]];
        uint32_t c = inverse[idx_ptr[3 * t + 2]];
        if (a == b || b == c || a == c)
            continue; // degenerate
        tris.push_back({a, b, c});
    }

    if (tris.empty())
    {
        // Return compacted empty mesh with unique vertices (as Python did)
        auto out_pos = py::array_t<float>(unique_pos.size());
        if (!unique_pos.empty())
            std::memcpy(out_pos.mutable_data(), unique_pos.data(), unique_pos.size() * sizeof(float));
        auto out_idx = py::array_t<uint32_t>(0);
        return py::make_tuple(out_pos, out_idx);
    }

    // 3) Remove duplicate faces ignoring winding by sorting indices within each tri
    std::unordered_set<std::array<uint32_t, 3>, Arr3U32Hash, Arr3U32Eq> seen_faces;
    seen_faces.reserve(tris.size() * 2 + 1);
    std::vector<std::array<uint32_t, 3>> dedup_tris;
    dedup_tris.reserve(tris.size());

    for (auto &tri : tris)
    {
        std::array<uint32_t, 3> key = tri;
        // sort ascending
        if (key[0] > key[1]) std::swap(key[0], key[1]);
        if (key[1] > key[2]) std::swap(key[1], key[2]);
        if (key[0] > key[1]) std::swap(key[0], key[1]);
        if (seen_faces.insert(key).second)
        {
            dedup_tris.push_back(tri); // keep original winding
        }
    }

    // 4) Compact to only used vertices; follow Python behavior of sorted unique indices
    std::vector<uint32_t> used;
    used.reserve(unique_pos.size() / 3);
    {
        std::unordered_set<uint32_t> used_set;
        used_set.reserve(dedup_tris.size() * 3);
        for (auto &tri : dedup_tris)
        {
            used_set.insert(tri[0]);
            used_set.insert(tri[1]);
            used_set.insert(tri[2]);
        }
        used.assign(used_set.begin(), used_set.end());
        std::sort(used.begin(), used.end());
    }

    // Build remap oldUnique -> compact [0..K)
    std::vector<uint32_t> remap(unique_pos.size() / 3, UINT32_MAX);
    for (uint32_t i = 0; i < static_cast<uint32_t>(used.size()); ++i)
    {
        remap[used[i]] = i;
    }

    // Remap triangles
    std::vector<uint32_t> out_idx_vec;
    out_idx_vec.reserve(dedup_tris.size() * 3);
    for (auto &tri : dedup_tris)
    {
        out_idx_vec.push_back(remap[tri[0]]);
        out_idx_vec.push_back(remap[tri[1]]);
        out_idx_vec.push_back(remap[tri[2]]);
    }

    // Build compact positions
    std::vector<float> out_pos_vec;
    out_pos_vec.reserve(used.size() * 3);
    for (auto u : used)
    {
        out_pos_vec.push_back(unique_pos[3 * u + 0]);
        out_pos_vec.push_back(unique_pos[3 * u + 1]);
        out_pos_vec.push_back(unique_pos[3 * u + 2]);
    }

    // Convert to NumPy arrays
    py::array_t<float> out_pos(out_pos_vec.size());
    if (!out_pos_vec.empty())
        std::memcpy(out_pos.mutable_data(), out_pos_vec.data(), out_pos_vec.size() * sizeof(float));

    py::array_t<uint32_t> out_idx(out_idx_vec.size());
    if (!out_idx_vec.empty())
        std::memcpy(out_idx.mutable_data(), out_idx_vec.data(), out_idx_vec.size() * sizeof(uint32_t));

    return py::make_tuple(out_pos, out_idx);
}

PYBIND11_MODULE(pywebifc, m)
{
    m.doc() = R"doc(
Minimal Python bindings for web-ifc core.

This module exposes a small set of functions for basic I/O and
introspection. APIs are intentionally minimal to validate end-to-end
functionality; more features can be added as needed.
)doc";

    m.def(
        "get_version",
        &GetVersion,
        R"doc(Return the web-ifc core version string.

Examples
--------
>>> import pywebifc as w
>>> w.get_version()
'x.y.z'
)doc");

    // Model lifecycle
    m.def(
        "create_model",
        &CreateModel,
        R"doc(Create an empty in-memory model and return its integer model ID.

Notes
-----
Use together with other functions that accept a ``model_id``. Call
``close_model(model_id)`` when done to free resources.
)doc");

    // Overload: create_model(**settings)
    m.def(
        "create_model",
        &CreateModelWithSettings,
        R"doc(Create a model with custom loader/geometry settings via keyword args.

Accepted keyword args (all optional)
------------------------------------
- coordinate_to_origin: bool (default False)
- circle_segments: int (default 12)
- boolean_union_threshold: int (default 150)
- tolerance_plane_intersection: float
- tolerance_plane_deviation: float
- tolerance_back_deviation_distance: float
- tolerance_inside_outside_perimeter: float
- tolerance_scalar_equality: float
- plane_refit_iterations: int
- tape_size: int
- memory_limit: int
- linewriter_buffer: int

Returns
-------
int
    The created model's ID.
)doc");

    // Logging controls
    m.def(
        "set_log_level",
        &SetLogLevel,
        py::arg("level"),
        R"doc(Set web-ifc log level by numeric spdlog level.

Levels
------
0=trace, 1=debug, 2=info, 3=warn, 4=err, 5=critical, 6=off
)doc");

    m.def(
        "set_log_level_name",
        &SetLogLevelName,
        py::arg("name"),
        R"doc(Set web-ifc log level by name.

Accepted values: 'trace', 'debug', 'info', 'warn', 'error', 'critical', 'off'
)doc");

    m.def(
        "open_model",
        &OpenModel,
        py::arg("path"),
        R"doc(Open an IFC file from a filesystem path and return its ``model_id``.

Parameters
----------
path : str
    Absolute or relative path to an IFC STEP file.

Returns
-------
int
    The opened model's ID.

Raises
------
RuntimeError
    If the file cannot be opened for reading.

Examples
--------
>>> import pywebifc as w
>>> mid = w.open_model("/path/to/model.ifc")
>>> w.is_model_open(mid)
True
)doc");

    // Overload: open_model(path, **settings)
    m.def(
        "open_model",
        &OpenModelWithSettings,
        py::arg("path"),
        R"doc(Open an IFC file with custom settings via keyword args. 

Accepted keyword args (all optional)
------------------------------------
- coordinate_to_origin: bool (default False)
- circle_segments: int (default 12)
- boolean_union_threshold: int (default 150)
- tolerance_plane_intersection: float
- tolerance_plane_deviation: float
- tolerance_back_deviation_distance: float
- tolerance_inside_outside_perimeter: float
- tolerance_scalar_equality: float
- plane_refit_iterations: int
- tape_size: int
- memory_limit: int
- linewriter_buffer: int

Returns
-------
int
    The opened model's ID.
)doc");

    m.def(
        "close_model",
        &CloseModel,
        py::arg("model_id"),
        R"doc(Close a specific model by its ``model_id`` and free resources.)doc");

    m.def(
        "close_all_models",
        &CloseAllModels,
        R"doc(Close all currently open models and release their resources.)doc");

    m.def(
        "is_model_open",
        &IsModelOpen,
        py::arg("model_id"),
        R"doc(Check whether a model with ``model_id`` is currently open.)doc");

    // Introspection helpers
    m.def(
        "get_max_express_id",
        &GetMaxExpressID,
        py::arg("model_id"),
        R"doc(Return the maximum EXPRESS line ID in the model.)doc");

    m.def(
        "get_line_type",
        &GetLineType,
        py::arg("model_id"),
        py::arg("express_id"),
        R"doc(Return the IFC type code (uint) for a given EXPRESS line ID.)doc");

    m.def(
        "get_all_lines",
        &GetAllLines,
        py::arg("model_id"),
        R"doc(Return a list of all EXPRESS line IDs present in the model.)doc");

    // Geometry accessors for Python-side GLB assembly
    m.def(
        "get_flat_mesh",
        &GetFlatMeshPy,
        py::arg("model_id"),
        py::arg("express_id"),
        R"doc(Returns a dict with placement(s) for an IFC entity: {
  'express_id': int,
  'geometries': [ { 'geometry_express_id': int, 'matrix': [16], 'color_rgba': [4] }, ... ]
})doc");

    m.def(
        "build_gltf_like",
        &BuildGLTFLike,
        py::arg("model_id"),
        py::arg("types").none(true) = py::none(),
        py::arg("include_normals") = true,
        py::arg("share_buffers") = false,
        R"doc(Build a glTF-like scene graph as Python dicts/lists.

Parameters
----------
model_id : int
types : Optional[List[int]]
    Optional schema type codes to include; defaults to all element types
    excluding openings/spaces.
include_normals : bool
    If False, normal arrays are omitted to reduce work/memory.
share_buffers : bool
    If True, return zero-copy NumPy views into C++ geometry buffers
    (lifetime must outlive consumption; safe when model stays open until GLB packing).

Returns
-------
dict
    {
      'scenes': [ {'nodes': [int, ...]} ],
      'nodes': [ {'mesh': int, 'name': str, 'matrix': [float x16]}, ... ],
      'meshes': [ {'primitives': [ {'material': int, 'points': [float], 'normals': [float], 'faces': [int]} ]}, ... ],
      'materials': [ {'baseColorFactor': [float x4]}, ... ]
    }
)doc");

    m.def(
        "build_spatial_hierarchy",
        &BuildSpatialHierarchy,
        py::arg("model_id"),
        R"doc(Build a simple spatial/decomposition hierarchy tree.

Returns
-------
dict
    {
      'roots': [int],
      'children': { str(parentId): [int, ...] },
      'names': { str(id): str },
      'types': { str(id): int }
    }

Notes
-----
Children combine IfcRelAggregates (decomposition) and
IfcRelContainedInSpatialStructure (spatial containment) edges.
)doc");

    // Convenience: mirror version as module attribute
    m.attr("__version__") = GetVersion();

    // ---------------------------------------------------------------------
    // Mesh utilities
    // ---------------------------------------------------------------------

    // Clean a triangle mesh by deduplicating identical vertices and faces.
    // - pos_f32_flat: flat float32 positions (len = 3*N)
    // - idx_u32_flat: flat uint32 triangle indices (len = 3*M)
    // Returns (clean_pos_flat_float32, clean_idx_flat_uint32)
    m.def(
        "clean_mesh",
        &CleanMesh,
        py::arg("pos_f32_flat"),
        py::arg("idx_u32_flat"),
        R"doc(
Clean a triangle mesh by deduplicating vertices and faces using exact float matches.

Parameters
----------
pos_f32_flat : numpy.ndarray (float32, shape=(3*N,) or (N,3))
    Flat positions array.
idx_u32_flat : numpy.ndarray (uint32, shape=(3*M,) or (M,3))
    Flat triangle index array.

Returns
-------
(numpy.ndarray, numpy.ndarray)
    Tuple of (clean_pos_flat_float32, clean_idx_flat_uint32).

Notes
-----
This performs exact bitwise equality on float32 positions (no tolerance).
Degenerate and duplicate triangles (ignoring winding) are removed, and
vertices are compacted to only those referenced by remaining triangles.
)doc");
}
