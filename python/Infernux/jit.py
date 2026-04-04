"""Public JIT helpers for gameplay scripts.

This is the supported import surface for Infernux JIT usage:

    from Infernux.jit import njit, precompile_jit

The implementation delegates to ``Infernux._jit_kernels`` so the engine keeps
one canonical JIT/fallback mechanism.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import subprocess
import sys

import Infernux._jit_kernels as _jit_kernels

_log = logging.getLogger("Infernux.jit")
_JIT_REQUIREMENT = "numba>=0.61.0"
_JIT_CHECK_ENV = "_INFERNUX_JIT_RUNTIME_CHECKED"


def _sync_exports() -> None:
    global JIT_AVAILABLE, njit, precompile
    JIT_AVAILABLE = _jit_kernels.JIT_AVAILABLE
    njit = _jit_kernels.njit
    precompile = _jit_kernels.precompile


def _run_python(args: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000
    return subprocess.run([sys.executable, *args], **kwargs)


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _ensure_pip() -> bool:
    completed = _run_python(["-m", "pip", "--version"], timeout=60)
    if completed.returncode == 0:
        return True

    completed = _run_python(["-m", "ensurepip", "--upgrade"], timeout=600)
    if completed.returncode != 0:
        return False

    completed = _run_python(["-m", "pip", "--version"], timeout=60)
    return completed.returncode == 0


def _install_numba() -> bool:
    completed = _run_python(
        [
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--prefer-binary",
            "--upgrade",
            _JIT_REQUIREMENT,
        ],
        timeout=1800,
    )
    return completed.returncode == 0 and _has_module("numba")


def _reload_jit_exports() -> None:
    global _jit_kernels
    importlib.invalidate_caches()
    _jit_kernels = importlib.reload(_jit_kernels)
    _sync_exports()


_sync_exports()


def ensure_jit_runtime(*, auto_install: bool = True) -> bool:
    """Ensure the current Python runtime can import Numba-backed JIT helpers.

    On editor startup this gives older Hub-created project runtimes one extra
    chance to self-heal: if ``numba`` is missing, try to install it into the
    current interpreter once, then reload the JIT kernel module in-process.

    Failure is non-fatal; the engine continues with pure-Python fallbacks.
    """
    if JIT_AVAILABLE:
        return True

    if _has_module("numba"):
        _reload_jit_exports()
        return JIT_AVAILABLE

    if not auto_install or os.environ.get("INFERNUX_DISABLE_JIT_AUTOINSTALL") == "1":
        return False

    if os.environ.get(_JIT_CHECK_ENV) == "1":
        return False
    os.environ[_JIT_CHECK_ENV] = "1"

    if not _ensure_pip():
        _log.warning("JIT runtime check failed: pip is unavailable in the current Python runtime.")
        return False

    if not _install_numba():
        _log.warning("JIT runtime check failed: unable to install numba into the current Python runtime.")
        return False

    _reload_jit_exports()
    return JIT_AVAILABLE


def precompile_jit() -> None:
    """Compile and cache built-in JIT kernels ahead of time."""
    ensure_jit_runtime(auto_install=False)
    precompile()


__all__ = [
    "JIT_AVAILABLE",
    "ensure_jit_runtime",
    "njit",
    "precompile",
    "precompile_jit",
]