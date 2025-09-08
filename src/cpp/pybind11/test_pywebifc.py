#!/usr/bin/env python3
"""
Quick test script for the pywebifc extension.

Usage:
  python test_pywebifc.py [optional_ifc_path]

If no path is provided, it uses the default:
  tests/ifcfiles/public/AC20-FZK-Haus.ifc

It attempts to import the built extension from build/pybind11.
"""

import sys
import traceback
from pathlib import Path

here = Path(__file__).resolve().parent


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


def main() -> None:
    default_path = (
        here.parent.parent.parent
        / "tests"
        / "ifcfiles"
        / "public"
        / "AC20-FZK-Haus.ifc"
    ).resolve()
    ifc_path = (sys.argv[1] if len(sys.argv) > 1 else str(default_path)).strip()

    pywebifc = import_pywebifc()
    print("web-ifc version:", pywebifc.get_version())

    print("Opening:", ifc_path)
    model_id = pywebifc.open_model(ifc_path)
    print("Model ID:", model_id)
    print("Is open:", pywebifc.is_model_open(model_id))
    max_id = pywebifc.get_max_express_id(model_id)
    print("Max EXPRESS ID:", max_id)

    lines = pywebifc.get_all_lines(model_id)
    print("Total lines:", len(lines))
    preview = lines[:10]
    print("Sample lines:", preview)
    if preview:
        first_line_type = pywebifc.get_line_type(model_id, preview[0])
        print("First line type:", first_line_type)

    pywebifc.close_model(model_id)
    print("Closed model. Is open:", pywebifc.is_model_open(model_id))


if __name__ == "__main__":
    main()
