"""
Type stubs for the `pywebifc` C-extension module.

These provide signatures and docstrings for Pylance/Pyright so that
hover, completion, and type checking work even though the runtime
implementation is in a compiled module built via pybind11.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

__version__: str

def get_version() -> str:
    """Return the web-ifc core version string.

    Examples
    --------
    >>> import pywebifc as w
    >>> w.get_version()
    'x.y.z'
    """
    ...

def create_model() -> int:
    """Create an empty in-memory model and return its integer model ID.

    Notes
    -----
    Use together with other functions that accept a ``model_id``. Call
    ``close_model(model_id)`` when done to free resources.
    """
    ...

def open_model(path: str) -> int:
    """Open an IFC file from a filesystem path and return its ``model_id``.

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
    """
    ...

def close_model(model_id: int) -> None:
    """Close a specific model by its ``model_id`` and free resources."""
    ...

def close_all_models() -> None:
    """Close all currently open models and release their resources."""
    ...

def is_model_open(model_id: int) -> bool:
    """Check whether a model with ``model_id`` is currently open."""
    ...

def get_max_express_id(model_id: int) -> int:
    """Return the maximum EXPRESS line ID in the model."""
    ...

def get_line_type(model_id: int, express_id: int) -> int:
    """Return the IFC type code (uint) for a given EXPRESS line ID."""
    ...

def get_all_lines(model_id: int) -> List[int]:
    """Return a list of all EXPRESS line IDs present in the model."""
    ...

def get_flat_mesh(model_id: int, express_id: int) -> Dict[str, Any]:
    """Return placement(s) for an IFC entity's flattened mesh data.

    Returns a dict like::

        {
          'express_id': int,
          'geometries': [
            {
              'geometry_express_id': int,
              'matrix': List[float],  # length-16, column-major
              'color_rgba': List[float],  # [r,g,b,a]
            },
            ...
          ],
        }
    """
    ...

def build_gltf_like(model_id: int, types: Optional[List[int]] = ...) -> Dict[str, Any]:
    """Build a glTF-like scene graph (Python dicts/lists) ready for GLB packing.

    Returns a dict like::

        {
          'scenes': [ {'nodes': List[int]} ],
          'nodes': [ {'mesh': int, 'name': str, 'matrix': List[float], 'express_id': int} ],
          'meshes': [
            {
              'primitives': [
                {
                  'material': int,
                  'points': List[float],    # xyz per vertex
                  'normals': List[float],   # nx ny nz per vertex
                  'faces': List[int],       # triangle indices
                }
              ]
            }
          ],
          'materials': [ {'baseColorFactor': List[float]} ]
        }

    If `types` is omitted, includes most element types except openings/spaces.
    """
    ...

def build_spatial_hierarchy(model_id: int) -> Dict[str, Any]:
    """Build a simple spatial/decomposition hierarchy tree.
    Returns a dict like::
        {
          'roots': List[int],
          'children': Dict[str, List[int]],
          'names': Dict[str, str],
          'types': Dict[str, int],
        }
    Children combine IfcRelAggregates (decomposition) and
    IfcRelContainedInSpatialStructure (spatial containment) edges.
    """
    ...

def clean_mesh(pos_f32_flat: np.ndarray, idx_u32_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clean a triangle mesh by deduplicating vertices and faces using exact float matches.

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
    Performs exact bitwise equality on float32 positions (no tolerance).
    Removes degenerate and duplicate triangles (ignoring winding) and
    compacts to only used vertices.
    """
    ...
