# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import subprocess
import sys
import sysconfig
from glob import glob

import pybind11
import torch
import torch.utils.cpp_extension
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DEVICE = os.getenv("DEVICE", "cuda")
PACKAGE_INCLUDES = ["iaxl", "iaxl.*", "kvshrink", "kvshrink.*"]


def _is_cuda() -> bool:
    return DEVICE == "cuda"


def _is_xpu() -> bool:
    return DEVICE == "xpu"


def _is_cpu() -> bool:
    return DEVICE == "cpu"


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

        if _is_xpu():
            self.sycl_ext = torch.utils.cpp_extension.SyclExtension(
                name=name,
                sources=[],
                include_dirs=[
                    os.path.join(sourcedir, "include"),
                    pybind11.get_include(),
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "sycl": ["-O3", "-std=c++17"],
                },
                extra_link_args=["-lsycl"],
            )
        else:
            self.sycl_ext = None


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext: CMakeExtension):
        build_dir = self.build_temp
        os.makedirs(build_dir, exist_ok=True)

        cmake_args = [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        ]

        torch_cmake_prefix = torch.utils.cmake_prefix_path
        pybind11_cmake_dir = pybind11.get_cmake_dir()

        cmake_prefix_paths = [torch_cmake_prefix, pybind11_cmake_dir]
        cmake_args.append(f"-DCMAKE_PREFIX_PATH={';'.join(cmake_prefix_paths)}")

        if _is_cuda():
            cmake_args.append("-DDEVICE=cuda")
        elif _is_xpu():
            cmake_args.append("-DDEVICE=xpu")
        else:
            raise RuntimeError("Please ensure either CUDA or XPU is available.")

        extra_cmake_args = os.environ.get("IAXL_CMAKE_ARGS", "")
        if extra_cmake_args:
            cmake_args.extend(shlex.split(extra_cmake_args))

        cmake_args.append(ext.sourcedir)

        print(f"[INFO] CMake command: {' '.join(cmake_args)}")

        subprocess.check_call(cmake_args, cwd=build_dir)
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", "--", "-j8"],
            cwd=build_dir,
        )


def get_package_data():
    packages = find_packages(include=PACKAGE_INCLUDES)
    package_data = {}
    patterns = ["*.so", "*.sh", "*.patch"]

    for package in packages:
        found_patterns = []
        package_dir = os.path.join(ROOT_DIR, package.replace(".", os.sep))
        for pattern in patterns:
            files = glob(os.path.join(package_dir, pattern))
            if files:
                found_patterns.append(pattern)
        if found_patterns:
            package_data[package] = found_patterns
            print(f"[INFO] Including {found_patterns} for package: {package}")

    print(f"[INFO] Package data: {package_data}")
    return package_data


ext_modules = []
ext_modules.append(CMakeExtension(name="iaxl", sourcedir=ROOT_DIR))

setup(
    name="iaxl",
    version="0.10.0",
    description="intel-accel-for-llm",
    author="bin.yang@intel.com",
    packages=find_packages(include=PACKAGE_INCLUDES),
    python_requires=">=3.10",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    package_data=get_package_data(),
    zip_safe=False,
)
