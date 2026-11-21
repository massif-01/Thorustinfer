"""BF16 projections for parallel Qwen draft blocks; the host pads the last tile."""

import tilelang.language as T
from kernels.cuda.model.floating_core import mma_loop


def projection(n, k):
    bm, bn, bk = 16, 64, 32
    if n % bn or k % bk:
        raise ValueError("Draft projection must have aligned output and input channels")
    product = mma_loop(bm, k, bm, bn, bk)

    @T.prim_func
    def main(
        A: T.Tensor((bm, k), "bfloat16"),
        W: T.Tensor((n, k), "bfloat16"),
        Y: T.Tensor((bm, n), "bfloat16"),
    ):
        with T.Kernel(n // bn, threads=128) as bx:
            acc = T.alloc_fragment((bm, bn), T.float32)
            product(A, W, acc, 0, bx, 0)
            T.copy(acc, Y[0, bx * bn])

    return main
