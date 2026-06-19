from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from sglang.jit_kernel import utils as sgl_jit_utils


PATCH_KERNEL_PATH = Path(__file__).resolve().parent


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


def _resolve_patch_csrc(files: List[str] | None) -> list[str]:
    return [str((PATCH_KERNEL_PATH / "csrc" / f).resolve()) for f in files or []]


def load_patch_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    extra_dependencies: List[str] | None = None,
    build_directory: str | None = None,
    header_only: bool = True,
):
    """Compile a JIT kernel whose sources live under this monkeypatch package."""
    from tvm_ffi.cpp import load, load_inline

    cpp_files_resolved = _resolve_patch_csrc(cpp_files)
    cuda_files_resolved = _resolve_patch_csrc(cuda_files)
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    include_paths = list(extra_include_paths or [])

    for dep in set(extra_dependencies or []):
        if dep not in sgl_jit_utils._REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {dep} is not registered.")
        include_paths += sgl_jit_utils._REGISTERED_DEPENDENCIES[dep]()

    module_name = "sgl_kernel_jit_" + "_".join(str(arg) for arg in args)
    if header_only:
        cpp_sources = [f'#include "{path}"' for path in cpp_files_resolved]
        cpp_sources += [_make_wrapper(tup) for tup in (cpp_wrappers or [])]
        cuda_sources = [f'#include "{path}"' for path in cuda_files_resolved]
        cuda_sources += [_make_wrapper(tup) for tup in (cuda_wrappers or [])]
        with sgl_jit_utils._jit_compile_context():
            return load_inline(
                module_name,
                cpp_sources=cpp_sources,
                cuda_sources=cuda_sources,
                extra_cflags=sgl_jit_utils.DEFAULT_CFLAGS + extra_cflags,
                extra_cuda_cflags=sgl_jit_utils._get_default_target_flags()
                + extra_cuda_cflags,
                extra_ldflags=sgl_jit_utils.DEFAULT_LDFLAGS + extra_ldflags,
                extra_include_paths=sgl_jit_utils.DEFAULT_INCLUDE + include_paths,
                build_directory=build_directory,
            )

    if cpp_wrappers is not None or cuda_wrappers is not None:
        raise ValueError("wrappers are only supported for header-only patch JIT modules")
    with sgl_jit_utils._jit_compile_context():
        return load(
            module_name,
            cpp_files=cpp_files_resolved,
            cuda_files=cuda_files_resolved,
            extra_cflags=sgl_jit_utils.DEFAULT_CFLAGS + extra_cflags,
            extra_cuda_cflags=sgl_jit_utils._get_default_target_flags()
            + extra_cuda_cflags,
            extra_ldflags=sgl_jit_utils.DEFAULT_LDFLAGS + extra_ldflags,
            extra_include_paths=sgl_jit_utils.DEFAULT_INCLUDE + include_paths,
            build_directory=build_directory,
        )

