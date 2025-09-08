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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
        return 0u;
    return manager().GetIfcLoader(modelID)->GetMaxExpressId();
}

static uint32_t GetLineType(uint32_t modelID, uint32_t expressID)
{
    if (!manager().IsModelOpen(modelID))
        return 0u;
    return manager().GetIfcLoader(modelID)->GetLineType(expressID);
}

static std::vector<uint32_t> GetAllLines(uint32_t modelID)
{
    if (!manager().IsModelOpen(modelID))
        return {};
    return manager().GetIfcLoader(modelID)->GetAllLines();
}

static std::string GetVersion()
{
    return std::string(WEB_IFC_VERSION_NUMBER);
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
static py::dict BuildGLTFLike(uint32_t modelID, std::optional<std::vector<uint32_t>> optTypes)
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
    py::list scenes;    // will contain one scene referencing all nodes
    py::list nodes;     // node objects
    py::list meshes;    // mesh objects
    py::list materials; // material objects (simple baseColor only)

    // Dedup maps
    std::unordered_map<uint32_t, uint32_t> geomIdToMeshIdx;  // geometryExpressID -> mesh index
    std::unordered_map<std::string, uint32_t> colorToMatIdx; // "r,g,b,a" -> material index

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
        auto it = geomIdToMeshIdx.find(geometryExpressID);
        if (it != geomIdToMeshIdx.end())
            return it->second;

        // Prepare geometry data
        auto &geom = geomProc->GetGeometry(geometryExpressID);
        geom.GetVertexData();              // fills float buffer
        const auto &fv = geom.fvertexData; // interleaved [x y z nx ny nz] per vertex
        const auto &idx = geom.indexData;  // 3 per triangle

        // Extract positions only (points)
        py::list points;
        for (size_t i = 0; i < geom.numPoints; ++i)
        {
            points.append(fv[i * webifc::geometry::VERTEX_FORMAT_SIZE_FLOATS + 0]);
            points.append(fv[i * webifc::geometry::VERTEX_FORMAT_SIZE_FLOATS + 1]);
            points.append(fv[i * webifc::geometry::VERTEX_FORMAT_SIZE_FLOATS + 2]);
        }

        // Extract vertex normals if available (nx, ny, nz)
        py::list normals;
        for (size_t i = 0; i < geom.numPoints; ++i)
        {
            normals.append(fv[i * webifc::geometry::VERTEX_FORMAT_SIZE_FLOATS + 3]);
            normals.append(fv[i * webifc::geometry::VERTEX_FORMAT_SIZE_FLOATS + 4]);
            normals.append(fv[i * webifc::geometry::VERTEX_FORMAT_SIZE_FLOATS + 5]);
        }

        // Faces (triangle indices)
        py::list faces;
        for (auto v : idx)
            faces.append(v);

        // Primitive
        py::dict prim;
        prim["material"] = py::int_(materialIndex);
        prim["points"] = std::move(points);
        prim["faces"] = std::move(faces);
        prim["normals"] = std::move(normals);

        py::list prims;
        prims.append(std::move(prim));

        py::dict mesh;
        mesh["primitives"] = std::move(prims);

        uint32_t meshIdx = static_cast<uint32_t>(meshes.size());
        meshes.append(std::move(mesh));
        geomIdToMeshIdx.emplace(geometryExpressID, meshIdx);
        return meshIdx;
    };

    // Build nodes from flat meshes
    py::list sceneNodes;
    for (auto id : expressIds)
    {
        webifc::geometry::IfcFlatMesh flat = geomProc->GetFlatMesh(id);
        for (auto &pg : flat.geometries)
        {
            uint32_t matIdx = get_material_index(pg);
            uint32_t meshIdx = ensure_mesh_for_geometry(pg.geometryExpressID, matIdx);

            py::dict node;
            node["mesh"] = py::int_(meshIdx);
            // Include express_id for easier regrouping on Python side
            node["express_id"] = py::int_(flat.expressID);
            // Name equals the entity's #id per request
            node["name"] = std::string("#") + std::to_string(flat.expressID);

            py::list mat;
            for (int i = 0; i < 16; ++i)
                mat.append(pg.flatTransformation[i]);
            node["matrix"] = std::move(mat);

            uint32_t nodeIdx = static_cast<uint32_t>(nodes.size());
            nodes.append(std::move(node));
            sceneNodes.append(py::int_(nodeIdx));
        }
    }

    // One scene referencing all nodes we created
    py::dict scene;
    scene["nodes"] = std::move(sceneNodes);
    scenes.append(std::move(scene));

    py::dict out;
    out["scenes"] = std::move(scenes);
    out["nodes"] = std::move(nodes);
    out["meshes"] = std::move(meshes);
    out["materials"] = std::move(materials);
    return out;
}

// Helper: best-effort IfcRoot.Name. Falls back to empty string if missing.
static std::string GetIfcRootNameSafely(webifc::parsing::IfcLoader *loader, uint32_t expressID)
{
    if (!loader)
        return std::string();
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
        py_children[std::to_string(kv.first).c_str()] = std::move(lst);
    }

    py::dict py_names;
    for (auto &kv : names)
    {
        py_names[std::to_string(kv.first).c_str()] = kv.second;
    }

    py::dict py_types;
    for (auto &kv : types)
    {
        py_types[std::to_string(kv.first).c_str()] = py::int_(kv.second);
    }

    py::dict out;
    out["roots"] = std::move(py_roots);
    out["children"] = std::move(py_children);
    out["names"] = std::move(py_names);
    out["types"] = std::move(py_types);
    return out;
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
    The created model's ID.

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
        R"doc(Build a glTF-like scene graph as Python dicts/lists.

Parameters
----------
model_id : int
types : Optional[List[int]]
    Optional schema type codes to include; defaults to all element types
    excluding openings/spaces.

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
}
