#!/usr/bin/env python

import torch
import triton
import triton.language as tl


@triton.jit
def divide_by(X, Y):
    xptr = X + tl.arange(0, 1)
    xx = tl.load(xptr)
    yptr = Y + tl.arange(0, 1)
    yy = tl.load(yptr)
    xx = xx // yy
    tl.store(xptr, xx)


def grid(meta):
    return (1,)


def divide_by_two(x):
    if isinstance(x, triton.code_gen.TensorWrapper):
        y_inner = torch.tensor([2], dtype=x.base.dtype, device="cuda")
        y = triton.reinterpret(y_inner, x.dtype)
    else:
        y = torch.tensor([2], dtype=x.dtype, device="cuda")
    divide_by[grid](x, y)


def test_div() -> None:
    assert torch.cuda.is_available()
    # print("testing with x1 as int16")
    x1 = torch.tensor([0xFFFE], dtype=torch.int16, device="cuda")
    divide_by_two(x1)
    assert x1[0] == -1, x1
    # print("testing with x2 as int32")
    x2 = torch.tensor([0xFFFE], dtype=torch.int32, device="cuda")
    divide_by_two(x2)
    assert x2[0] == 0x7FFF
    # print("testing with x3 as uint16-ish")
    x3 = triton.reinterpret(
        torch.tensor([0xFFFE], dtype=torch.int16, device="cuda"), tl.uint16
    )
    divide_by_two(x3)
    assert x3.base[0] == 0x7FFF, x3.base


@triton.jit
def sgn_kernel(X, S, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    xx = tl.load(X + offsets, mask=mask)
    ss = tl.where(xx > 0, 1, tl.where(xx < 0, -1, 0))
    tl.store(S + offsets, ss, mask=mask)


def sgn(x, block_size=32):
    n = len(x.base) if isinstance(x, triton.code_gen.TensorWrapper) else len(x)

    def grid2(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    s = torch.full((n,), 17, dtype=torch.int32, device="cuda")
    sgn_kernel[grid2](x, s, n, BLOCK_SIZE=block_size)
    return s


def test_cast() -> None:
    x = torch.tensor([0xFFFE, 12, -6, 0], dtype=torch.int16, device="cuda")
    s = sgn(x)
    assert all(s == torch.tensor([-1, 1, -1, 0], device="cuda"))

    x = triton.reinterpret(
        torch.tensor([0xFFFE, 12, 0], dtype=torch.int16, device="cuda"), dtype=tl.uint16
    )
    s = sgn(x)
    assert all(s == torch.tensor([1, 1, 0], device="cuda", dtype=torch.int32))


def main():
    test_div()
    test_cast()


if __name__ == "__main__":
    main()
