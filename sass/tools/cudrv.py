"""Minimal CUDA driver bindings (ctypes) for loading cubins and timing launches.

Binds the versioned entry points (`cuMemAlloc_v2`, ...) that `cuda.h` maps
the plain names to; the unversioned symbols are the legacy ABI. Uses the
device's primary context.
"""
from __future__ import annotations

import ctypes
from typing import Callable

import numpy as np

P = ctypes.c_void_p
U64 = ctypes.c_uint64


class Driver:
    _SIGNATURES = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetName": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
        "cuDeviceGetAttribute": [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
        "cuDevicePrimaryCtxRetain": [ctypes.POINTER(P), ctypes.c_int],
        "cuCtxSetCurrent": [P],
        "cuCtxSynchronize": [],
        "cuModuleLoadData": [ctypes.POINTER(P), ctypes.c_char_p],
        "cuModuleGetFunction": [ctypes.POINTER(P), P, ctypes.c_char_p],
        "cuMemAlloc_v2": [ctypes.POINTER(U64), ctypes.c_size_t],
        "cuMemFree_v2": [U64],
        "cuMemcpyHtoD_v2": [U64, ctypes.c_char_p, ctypes.c_size_t],
        "cuMemcpyDtoH_v2": [ctypes.c_char_p, U64, ctypes.c_size_t],
        "cuMemsetD8_v2": [U64, ctypes.c_ubyte, ctypes.c_size_t],
        "cuLaunchKernel": [P, *[ctypes.c_uint] * 6, ctypes.c_uint, P, ctypes.POINTER(P), ctypes.POINTER(P)],
        "cuEventCreate": [ctypes.POINTER(P), ctypes.c_uint],
        "cuEventRecord": [P, P],
        "cuEventSynchronize": [P],
        "cuEventElapsedTime": [ctypes.POINTER(ctypes.c_float), P, P],
        "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    }

    def __init__(self, device: int = 0):
        self.lib = ctypes.CDLL("libcuda.so.1")
        for name, argtypes in self._SIGNATURES.items():
            fn = getattr(self.lib, name)
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int
        self.call("cuInit", 0)
        dev = ctypes.c_int()
        self.call("cuDeviceGet", ctypes.byref(dev), device)
        self.dev = dev.value
        ctx = P()
        self.call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), self.dev)
        self.call("cuCtxSetCurrent", ctx)
        self._events = None

    def call(self, name: str, *args) -> None:
        err = getattr(self.lib, name)(*args)
        if err:
            text = ctypes.c_char_p()
            self.lib.cuGetErrorName(err, ctypes.byref(text))
            raise RuntimeError(f"{name}: {text.value.decode() if text.value else err}")

    @property
    def name(self) -> str:
        buf = ctypes.create_string_buffer(128)
        self.call("cuDeviceGetName", buf, 128, self.dev)
        return buf.value.decode()

    def attribute(self, attr: int) -> int:
        """cuDeviceGetAttribute (CU_DEVICE_ATTRIBUTE_* number)."""
        v = ctypes.c_int()
        self.call("cuDeviceGetAttribute", ctypes.byref(v), attr, self.dev)
        return v.value

    def module(self, cubin: bytes) -> P:
        mod = P()
        self.call("cuModuleLoadData", ctypes.byref(mod), cubin)
        return mod

    def function(self, mod: P, kernel: str) -> P:
        fn = P()
        self.call("cuModuleGetFunction", ctypes.byref(fn), mod, kernel.encode())
        return fn

    def load(self, cubin: bytes, kernel: str) -> P:
        return self.function(self.module(cubin), kernel)

    def alloc(self, nbytes: int) -> int:
        ptr = U64()
        self.call("cuMemAlloc_v2", ctypes.byref(ptr), max(nbytes, 1))
        return ptr.value

    def free(self, ptr: int) -> None:
        self.call("cuMemFree_v2", ptr)

    def upload(self, arr: np.ndarray) -> int:
        ptr = self.alloc(arr.nbytes)
        self.write(ptr, arr)
        return ptr

    def write(self, ptr: int, arr: np.ndarray) -> None:
        self.call("cuMemcpyHtoD_v2", ptr, np.ascontiguousarray(arr).tobytes(), arr.nbytes)

    def fill(self, ptr: int, byte: int, nbytes: int) -> None:
        self.call("cuMemsetD8_v2", ptr, byte, nbytes)

    def read(self, ptr: int, like: np.ndarray) -> np.ndarray:
        buf = ctypes.create_string_buffer(like.nbytes)
        self.call("cuMemcpyDtoH_v2", buf, ptr, like.nbytes)
        return np.frombuffer(buf.raw, dtype=like.dtype).reshape(like.shape).copy()

    def launch(self, fn: P, grid, block, params: list[bytes]) -> None:
        bufs = [ctypes.create_string_buffer(p, len(p)) for p in params]
        argv = (P * len(bufs))(*(ctypes.cast(b, P) for b in bufs))
        self.call("cuLaunchKernel", fn, *grid, *block, 0, None, argv, None)

    def sync(self) -> None:
        self.call("cuCtxSynchronize")

    def time_us(self, launch: Callable[[], None], inner: int) -> float:
        """Mean microseconds per launch over `inner` back-to-back launches."""
        if self._events is None:
            self._events = (P(), P())
            for ev in self._events:
                self.call("cuEventCreate", ctypes.byref(ev), 0)
        start, stop = self._events
        self.call("cuEventRecord", start, None)
        for _ in range(inner):
            launch()
        self.call("cuEventRecord", stop, None)
        self.call("cuEventSynchronize", stop)
        ms = ctypes.c_float()
        self.call("cuEventElapsedTime", ctypes.byref(ms), start, stop)
        return ms.value * 1e3 / inner


def scalar(dtype, x) -> bytes:
    return np.array(x, dtype=dtype).tobytes()


def ptr(x: int) -> bytes:
    return scalar(np.uint64, x)
