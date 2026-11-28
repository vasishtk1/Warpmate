"""Python access to Warpmate's own inference code (the C++/CUDA library in
src/inference), through ctypes.

    engine = Engine("model/warpmate_net.npz", "model/warpmate_net_int8.npz")
    probs = engine.gpu("sparse", codes, stm)        # NumPy in, NumPy out
    probs = engine.cpu(codes, stm)                  # CPU reference
    engine.gpu_device("sparse", codes_t, stm_t, probs_t)   # torch CUDA tensors

Boards use the dataset's compact form: codes (n, slots) int16 piece codes
padded with PAD (769), and stm (n,) uint8 with 0 = White to move. Outputs are
(n, 3) probabilities (and optionally logits) of loss / draw / win for the
side to move, the same as softmax(model(features(codes, stm))) in PyTorch.

The library is looked for in $WARPMATE_INFER_LIB, then build/.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAD = 769


def library_path() -> Path:
    if os.environ.get("WARPMATE_INFER_LIB"):
        return Path(os.environ["WARPMATE_INFER_LIB"])
    name = "libwarpmate_infer.dylib" if sys.platform == "darwin" else "libwarpmate_infer.so"
    return ROOT / "build" / name


def _load_library(path: Path) -> ctypes.CDLL:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: build it with "
                                "cmake -S . -B build && cmake --build build")
    lib = ctypes.CDLL(str(path))
    c = ctypes
    ptr = c.c_void_p
    lib.wm_create.restype = ptr
    lib.wm_create.argtypes = [c.c_char_p, c.c_char_p, c.c_char_p, c.c_int]
    lib.wm_destroy.argtypes = [ptr]
    lib.wm_has_gpu.argtypes = [ptr]
    lib.wm_has_int8.argtypes = [ptr]
    lib.wm_device_name.argtypes = [c.c_char_p, c.c_int]
    lib.wm_variant_name.restype = c.c_char_p
    lib.wm_variant_name.argtypes = [c.c_int]
    lib.wm_eval_cpu.argtypes = [ptr, c.c_int, ptr, c.c_int, ptr, c.c_int, ptr, ptr,
                                c.c_char_p, c.c_int]
    lib.wm_eval_gpu.argtypes = [ptr, c.c_int, ptr, c.c_int, ptr, c.c_int, ptr, ptr,
                                c.c_char_p, c.c_int]
    lib.wm_eval_gpu_device.argtypes = [ptr, c.c_int, c.c_uint64, c.c_int, c.c_uint64, c.c_int,
                                       c.c_uint64, c.c_uint64, c.c_uint64, c.c_char_p, c.c_int]
    return lib


class Engine:
    """The network's weights loaded into the library (and onto the GPU, if any)."""

    def __init__(self, float_npz: str | Path, int8_npz: str | Path | None = None,
                 lib_path: Path | None = None):
        self._lib = _load_library(lib_path or library_path())
        err = ctypes.create_string_buffer(512)
        handle = self._lib.wm_create(str(float_npz).encode(),
                                     str(int8_npz).encode() if int8_npz else None, err, 512)
        if not handle:
            raise RuntimeError(err.value.decode())
        self._handle = handle
        self.has_gpu = bool(self._lib.wm_has_gpu(handle))
        self.has_int8 = bool(self._lib.wm_has_int8(handle))
        self.variants: list[str] = []
        while (name := self._lib.wm_variant_name(len(self.variants))) is not None:
            self.variants.append(name.decode())

    def __del__(self):
        if getattr(self, "_handle", None):
            self._lib.wm_destroy(self._handle)
            self._handle = None

    @property
    def device_name(self) -> str:
        buf = ctypes.create_string_buffer(256)
        self._lib.wm_device_name(buf, 256)
        return buf.value.decode()

    def variant_id(self, variant: str) -> int:
        if variant not in self.variants:
            raise ValueError(f"unknown variant {variant!r}; available: {self.variants}")
        return self.variants.index(variant)

    # ---- NumPy in, NumPy out ----

    @staticmethod
    def _inputs(codes: np.ndarray, stm: np.ndarray):
        codes = np.ascontiguousarray(codes, dtype=np.int16)
        stm = np.ascontiguousarray(stm, dtype=np.uint8)
        if codes.ndim != 2 or stm.shape != (codes.shape[0],):
            raise ValueError("codes must be (n, slots) and stm (n,)")
        return codes, stm

    def _call(self, fn, first_arg: int, codes, stm, want_logits: bool):
        codes, stm = self._inputs(codes, stm)
        n, slots = codes.shape
        probs = np.empty((n, 3), dtype=np.float32)
        logits = np.empty((n, 3), dtype=np.float32) if want_logits else None
        err = ctypes.create_string_buffer(512)
        status = fn(self._handle, first_arg, codes.ctypes.data, slots, stm.ctypes.data, n,
                    probs.ctypes.data, logits.ctypes.data if want_logits else None, err, 512)
        if status != 0:
            raise RuntimeError(err.value.decode())
        return (probs, logits) if want_logits else probs

    def cpu(self, codes, stm, int8: bool = False, logits: bool = False):
        """CPU reference implementation (32-bit or 8-bit weights)."""
        return self._call(self._lib.wm_eval_cpu, int(int8), codes, stm, logits)

    def gpu(self, variant: str, codes, stm, logits: bool = False):
        """One of the CUDA variants; copies the boards to the GPU and back."""
        return self._call(self._lib.wm_eval_gpu, self.variant_id(variant), codes, stm, logits)

    # ---- torch CUDA tensors, no copies ----

    def gpu_device(self, variant: str, codes, stm, probs, logits=None, stream=None) -> None:
        """Runs on torch CUDA tensors already on the GPU: codes (n, slots)
        int16, stm (n,) uint8, probs (n, 3) float32 (written), logits likewise
        or None. Queued on `stream` (default: torch's current stream), so it
        can be timed with CUDA events and recorded into a CUDA graph."""
        import torch
        n, slots = codes.shape
        for t, dtype in ((codes, torch.int16), (stm, torch.uint8), (probs, torch.float32)):
            if t.dtype != dtype or not t.is_cuda or not t.is_contiguous():
                raise ValueError(f"expected a contiguous CUDA {dtype} tensor, got {t.dtype}")
        if stream is None:
            stream = torch.cuda.current_stream()
        err = ctypes.create_string_buffer(512)
        status = self._lib.wm_eval_gpu_device(
            self._handle, self.variant_id(variant), codes.data_ptr(), slots, stm.data_ptr(), n,
            probs.data_ptr(), logits.data_ptr() if logits is not None else 0,
            stream.cuda_stream, err, 512)
        if status != 0:
            raise RuntimeError(err.value.decode())
