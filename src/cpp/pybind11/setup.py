from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
from glob import glob

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self) -> None:
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_cmake(ext)
            else:
                super().build_extension(ext)

    def build_cmake(self, ext: CMakeExtension) -> None:
        build_temp = pathlib.Path(self.build_temp).resolve()
        build_dir = build_temp / "pybind11"
        build_dir.mkdir(parents=True, exist_ok=True)

        # CMake project lives in this directory (contains CMakeLists.txt)
        src_dir = pathlib.Path(__file__).parent.resolve()

        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]

        # Prefer Ninja on non-Windows for faster builds if available
        if os.environ.get("CMAKE_GENERATOR") is None and sys.platform != "win32":
            cmake_args += ["-G", "Ninja"]

        subprocess.check_call(["cmake", str(src_dir)] + cmake_args, cwd=str(build_dir))

        # Build only the Python extension target to avoid unnecessary work
        build_cmd = [
            "cmake",
            "--build",
            ".",
            "--config",
            cfg,
            "--target",
            "pywebifc",
            "-j",
            str(os.cpu_count() or 2),
        ]
        subprocess.check_call(build_cmd, cwd=str(build_dir))

        # Find the compiled extension produced by pybind11_add_module(pywebifc ...)
        patterns = [
            str(build_dir / "pywebifc*.so"),
            str(build_dir / "pywebifc*.pyd"),
            str(build_dir / cfg / "pywebifc*.so"),
            str(build_dir / cfg / "pywebifc*.pyd"),
            str(build_dir / "pybind11" / "pywebifc*.so"),
            str(build_dir / "pybind11" / "pywebifc*.pyd"),
            str(build_dir / "pybind11" / cfg / "pywebifc*.so"),
            str(build_dir / "pybind11" / cfg / "pywebifc*.pyd"),
        ]
        candidates: list[str] = []
        for p in patterns:
            candidates.extend(glob(p))
        if not candidates:
            raise RuntimeError(f"Could not find built extension in {build_dir}")
        candidates.sort(key=lambda p: len(p))
        built_path = pathlib.Path(candidates[0]).resolve()

        dest_path = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_path, dest_path)
        print(f"Copied extension: {built_path} -> {dest_path}")

        # Also install type stubs next to the extension for IDE/type-checkers
        stub_src = src_dir / "stubs" / "pywebifc.pyi"
        if stub_src.exists():
            stub_dst = dest_path.parent / "pywebifc.pyi"
            shutil.copy2(stub_src, stub_dst)
            print(f"Installed stubs: {stub_src} -> {stub_dst}")


setup(
    name="pywebifc",
    version="0.1.0",
    description="Python bindings for web-ifc with tooling",
    ext_modules=[CMakeExtension("pywebifc")],
    cmdclass={"build_ext": CMakeBuild},
    # export_glb.py lives alongside this setup.py
    py_modules=["export_glb"],
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pygltflib",
    ],
    entry_points={
        "console_scripts": [
            "pywebifc-export-glb=export_glb:main",
        ]
    },
)
