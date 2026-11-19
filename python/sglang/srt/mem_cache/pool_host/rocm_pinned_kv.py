# SPDX-License-Identifier: Apache-2.0
"""Process-private host KV with fixed physical pages for ROCm HiSparse."""

import ctypes
import logging
import math
import mmap
import os
import weakref
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _verbs():
    lib = ctypes.CDLL("libibverbs.so.1", use_errno=True)
    pointer = ctypes.c_void_p
    functions = (
        ("ibv_get_device_list", ctypes.POINTER(pointer), [ctypes.POINTER(ctypes.c_int)]),
        ("ibv_free_device_list", None, [ctypes.POINTER(pointer)]),
        ("ibv_get_device_name", ctypes.c_char_p, [pointer]),
        ("ibv_open_device", pointer, [pointer]),
        ("ibv_close_device", ctypes.c_int, [pointer]),
        ("ibv_alloc_pd", pointer, [pointer]),
        ("ibv_dealloc_pd", ctypes.c_int, [pointer]),
        ("ibv_reg_mr", pointer, [pointer, pointer, ctypes.c_size_t, ctypes.c_int]),
        ("ibv_dereg_mr", ctypes.c_int, [pointer]),
    )
    for name, result, args in functions:
        function = getattr(lib, name)
        function.restype = result
        function.argtypes = args
    return lib


def _local_hca() -> str:
    devices = [
        name.strip()
        for name in os.environ.get("ROCSHMEM_ALLOWED_IBV_DEVICES", "").split(",")
        if name.strip()
    ]
    rank_device = torch.cuda.current_device()
    if len(devices) == 1:
        return devices[0]
    if rank_device >= len(devices):
        raise ValueError(
            f"Missing local HCA for logical device {rank_device}; set "
            "ROCSHMEM_ALLOWED_IBV_DEVICES to one HCA or one per visible device."
        )
    return devices[rank_device]


class _LocalRegistration:
    """Retain local-write memory regions until the HIP mappings are released."""

    def __init__(self, address: int, size: int, device: str):
        self.pid = os.getpid()
        self.lib = lib = _verbs()
        self.context = None
        self.pd = None
        self.regions = []

        count = ctypes.c_int()
        devices = lib.ibv_get_device_list(ctypes.byref(count))
        if not devices:
            raise OSError(ctypes.get_errno(), "ibv_get_device_list")
        try:
            for i in range(count.value):
                if lib.ibv_get_device_name(devices[i]).decode() == device:
                    self.context = lib.ibv_open_device(devices[i])
                    if not self.context:
                        raise OSError(ctypes.get_errno(), f"ibv_open_device({device})")
                    break
            else:
                raise ValueError(f"Local HCA {device!r} is unavailable")
        finally:
            lib.ibv_free_device_list(devices)

        try:
            self.pd = lib.ibv_alloc_pd(self.context)
            if not self.pd:
                raise OSError(ctypes.get_errno(), f"ibv_alloc_pd({device})")
            chunk = 1 << 30
            for offset in range(0, size, chunk):
                # IBV_ACCESS_LOCAL_WRITE. No remote key is exported and no QP is created.
                region = lib.ibv_reg_mr(
                    self.pd, address + offset, min(chunk, size - offset), 1
                )
                if not region:
                    raise OSError(
                        ctypes.get_errno(),
                        f"ibv_reg_mr({device}, offset={offset}, pool_bytes={size})",
                    )
                self.regions.append(region)
        except BaseException:
            self.close()
            raise

    def close(self):
        if os.getpid() != self.pid:
            return
        while self.regions:
            rc = self.lib.ibv_dereg_mr(self.regions[-1])
            if rc:
                raise OSError(rc, "ibv_dereg_mr")
            self.regions.pop()
        if self.pd:
            rc = self.lib.ibv_dealloc_pd(self.pd)
            if rc:
                raise OSError(rc, "ibv_dealloc_pd")
            self.pd = None
        if self.context:
            rc = self.lib.ibv_close_device(self.context)
            if rc:
                raise OSError(rc, "ibv_close_device")
            self.context = None


class RocmPinnedKVAllocator:
    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        if torch.device(device).type != "cpu":
            raise ValueError(f"Expected host KV on CPU, got {device}")
        numel = math.prod(dims)
        if not numel:
            return torch.empty(dims, dtype=dtype, device=device)
        size = numel * dtype.itemsize
        # Retain 2 MiB virtual alignment for HIP. Back it with ordinary base pages.
        alignment = 2 << 20
        span = (size + alignment - 1) // alignment * alignment
        mapping = mmap.mmap(
            -1,
            span + alignment,
            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        try:
            base = ctypes.addressof(ctypes.c_ubyte.from_buffer(mapping))
            offset = -base % alignment
            mapping.madvise(mmap.MADV_DONTFORK)
            mapping.madvise(mmap.MADV_NOHUGEPAGE, offset, span)
            # First-write prevents a read fault from mapping the shared zero page.
            ctypes.memset(base + offset, 0, span)
            device_name = _local_hca()
            registration = _LocalRegistration(base + offset, size, device_name)
        except BaseException:
            mapping.close()
            raise
        try:
            owner = (ctypes.c_ubyte * size).from_buffer(mapping, offset)
            buffer = torch.frombuffer(owner, dtype=dtype, count=numel).reshape(dims)
            # Views retain this storage owner even after the original Tensor is gone.
            finalizer = weakref.finalize(owner, registration.close)
            finalizer.atexit = False
            buffer._sglang_hisparse_local_mr = registration
        except BaseException:
            registration.close()
            raise
        logger.info(
            "HiSparse host KV pinned on %s: %d bytes, %d local regions",
            device_name,
            size,
            len(registration.regions),
        )
        return buffer
