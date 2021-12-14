import copy
import itertools
from typing import Union

import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl
from triton.code_gen import TensorWrapper

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
float_dtypes = ['float16', 'float32', 'float64']
dtypes = int_dtypes + uint_dtypes + float_dtypes

def numpy_random(shape, dtype_str):
    if isinstance(shape, int):
        shape = (shape, )
    rs = RandomState(seed=17)
    dtype = getattr(np, dtype_str)
    if dtype_str == 'bool':
        return rs.randint(0, 2, shape, dtype=dtype)
    elif dtype_str in int_dtypes or dtype_str in uint_dtypes:
        return rs.randint(1, 32, shape, dtype=dtype)
    elif dtype_str in float_dtypes:
        return rs.normal(0, 1, shape)
    else:
        raise RuntimeError(f'Unknown dtype {dtype_str}')


def numpy_to_triton(x: np.ndarray, device='cuda') -> Union[TensorWrapper, torch.Tensor]:
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip('u')  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return TensorWrapper(torch.tensor(x_signed, device=device), getattr(tl, t))
    else:
        return torch.tensor(x, device=device)


def torch_dtype_name(dtype) -> str:
    if isinstance(dtype, triton.language.dtype):
        return dtype.name
    elif isinstance(dtype, torch.dtype):
        return str(dtype).split('.')[1]  # 'torch.int64' -> 'int64'
    else:
        raise TypeError(f'not a triton or torch dtype: {type(dtype)}')


def triton_to_numpy(x):
    if isinstance(x, TensorWrapper):
        return np.array(x.base.cpu(), dtype=getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        return np.array(x.cpu(), dtype=getattr(np, torch_dtype_name(x.dtype)))
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


def triton_empty_like(x):
    if isinstance(x, TensorWrapper):
        return TensorWrapper(torch.empty_like(x.base), dtype=x.dtype)
    elif isinstance(x, torch.Tensor):
        return torch.empty_like(x)
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")



def patch_kernel(template, to_replace):
    kernel = copy.deepcopy(template)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel


@pytest.mark.parametrize("dtype_x", [dtype_x for dtype_x in dtypes])
def test_empty_kernel(dtype_x, device='cuda'):
    SIZE = 128
    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        pass
    x = numpy_to_triton(numpy_random(SIZE, dtype_str=dtype_x), device=device)
    kernel[(1, )](x, SIZE=SIZE, num_warps=4)


# generic test functions
def _test_unary(dtype_x, expr, torch_expr=None, device='cuda'):
    SIZE = 128
    # define the kernel / launch-grid
    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)  # noqa: F841
        z = GENERATE_TEST_HERE  # noqa: F821
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    # inputs
    x = numpy_random(SIZE, dtype_str=dtype_x)
    if 'log' in expr:
        x = np.abs(x_np) + 0.01
    # reference result
    z_ref = eval(expr if torch_expr is None else torch_expr)
    # triton result
    x_tri = numpy_to_triton(x, device=device)
    z_tri = numpy_to_triton(np.empty_like(z_ref), device=device)
    kernel[(1, )](z_tri, x_tri, SIZE=SIZE, num_warps=4)
    # compare
    np.testing.assert_allclose(z_ref, triton_to_numpy(z_tri), rtol=0.01)


def _test_binary(dtype_x, dtype_y, expr, mode_x='real', mode_y='real', device='cuda'):
    SIZE = 128
    # define the kernel / launch-grid
    @triton.jit
    def kernel(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)  # noqa: F841 
        y = tl.load(Y + off)  # noqa: F841 
        z = GENERATE_TEST_HERE
        print('x', x, 'y', y, 'z', z)
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    # inputs
    x = numpy_random(SIZE, dtype_str=dtype_x)
    y = numpy_random(SIZE, dtype_str=dtype_y)
    if mode_x == 'nan': x[:] = float('nan')
    if mode_y == 'nan': y[:] = float('nan')
    # reference result
    z_ref = eval(expr)
    # triton result
    x_tri = numpy_to_triton(x, device=device)
    y_tri = numpy_to_triton(y, device=device)
    z_tri = numpy_to_triton(np.empty(SIZE, dtype=z_ref.dtype), device=device)
    kernel[(1, )](z_tri, x_tri, y_tri, SIZE=SIZE, num_warps=4)
    # compare
    print('z_tri.dtype', z_tri.dtype, type(z_tri), 'z_ref.dtype', z_ref.dtype, type(z_ref))
    np.testing.assert_allclose(z_ref, triton_to_numpy(z_tri), err_msg=expr, rtol=0.01)


# ---------------
# test binary ops
# ---------------
@pytest.mark.parametrize("dtype_x, dtype_y, expr", [
    (dtype_x, dtype_y, f'x{op}y') \
  for op in ['+', '-', '*', '/', '%'] \
  for dtype_x in dtypes \
  for dtype_y in dtypes
])
def test_bin_op(dtype_x, dtype_y, expr, device='cuda'):
    _test_binary(dtype_x, dtype_y, expr, device=device)


# ---------------
# test bitwise ops
# ---------------
@pytest.mark.parametrize("dtype_x, dtype_y, expr", [
    (dtype_x, dtype_y, f' x {op} y') \
  for op in ['&', '|', '^'] \
  for dtype_x in dtypes \
  for dtype_y in dtypes
])
def test_bitwise_op(dtype_x, dtype_y, expr, device='cuda'):
    if 'float' in dtype_x + dtype_y:
        with pytest.raises(TypeError):
            _test_binary(dtype_x, dtype_y, expr, device=device)
    elif (dtype_x == 'uint64'  and dtype_y in int_dtypes) or (dtype_x in int_dtypes and dtype_y == 'uint64'):
        # TODO(madeleine): Make sure Triton raises an error. This one is from numpy.
        with pytest.raises(TypeError):
            _test_binary(dtype_x, dtype_y, expr, device=device)
    else:
        _test_binary(dtype_x, dtype_y, expr, device=device)


# ---------------
# test compare ops
# ---------------
ops = ['==', '!=', '>', '<', '>=', '<=']
@pytest.mark.parametrize("dtype_x, dtype_y, expr, mode_x, mode_y", \
# real
[
    (dtype_x, dtype_y, f' x {op} y', 'real', 'real') \
    for op in ops \
    for dtype_x in dtypes \
    for dtype_y in dtypes
] + \
# NaNs
[('float32', 'float32', f' x {op} y', mode_x, mode_y) \
    for op in ops
    for mode_x, mode_y in [('nan' , 'real'), 
                           ('real', 'nan'), 
                           ('nan' , 'nan')]

])
def test_compare_op(dtype_x, dtype_y, expr, mode_x, mode_y, device='cuda'):
    _test_binary(dtype_x, dtype_y, expr, mode_x=mode_x, mode_y=mode_y, device=device)


# ---------------
# test unary ops
# ---------------
@pytest.mark.parametrize("dtype_x, expr", [
    (dtype_x, f' -x') for dtype_x in dtypes
] + [\
    (dtype_x, f' ~x') for dtype_x in int_dtypes
     ])
def test_unary_op(dtype_x, expr, device='cuda'):
    _test_unary(dtype_x, expr, device=device)

# ----------------
# test math ops
# ----------------
# @pytest.mark.paramterize("expr", [
#     'exp', 'log', 'cos', 'sin'
# ])

@pytest.mark.parametrize("expr", [
    'exp', 'log', 'cos', 'sin'
])
def test_math_op(expr, device='cuda'):
    _test_unary('float32', f'tl.{expr}(x)', f'torch.{expr}(x) ', device=device)


# ----------------
# test indexing
# ----------------


def make_ptr_str(name, shape):
    rank = len(shape)
    offsets = []
    stride = 1
    for i in reversed(range(rank)):
        idx = ', '.join([':' if ii == i else 'None' for ii in range(rank)])
        offsets += [f'tl.arange(0, {shape[i]})[{idx}]*{stride}']
        stride *= shape[i]
    return f"{name} + {' + '.join(offsets)}"


@pytest.mark.parametrize("expr", [f'x[{s}]' for s in
    ['None, :', ':, None',\
     'None, :, :', ':, :, None']\
])
def test_index1d(expr, device='cuda'):
    dtype = torch.int32
    rank_x = expr.count(':')
    rank_y = expr.count(',') + 1
    shape_x = [32 for _ in range(rank_x)]
    shape_z = [32 for _ in range(rank_y)]

    # Triton kernel
    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        m = tl.arange(0, SIZE)
        n = tl.arange(0, SIZE)
        x = tl.load(X_PTR_EXPR)
        z = GENERATE_TEST_HERE
        tl.store(Z_PTR_EXPR, z)

    to_replace = {
        'X_PTR_EXPR': make_ptr_str('X', shape_x),
        'Z_PTR_EXPR': make_ptr_str('Z', shape_z),
        'GENERATE_TEST_HERE': expr,
    }
    kernel = patch_kernel(kernel, to_replace)

    # torch result
    x = random(shape_x, dtype_str=dtype, device=device)
    y = torch.zeros(shape_z, dtype=dtype, device=device)
    z_ref = eval(expr) + y
    # triton result
    z_tri = torch.empty_like(z_ref)
    kernel[(1, )](z_tri, x, num_warps=1, SIZE=shape_x[0])
    # compare
    triton.testing.assert_almost_equal(z_ref, z_tri)


# ---------------
# test tuples
# ---------------


@triton.jit
def fn(a, b):
    return a + b, \
            a - b, \
            a * b


def test_tuples():
    device = 'cuda'

    @triton.jit
    def with_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = fn(x, y)
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    @triton.jit
    def without_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = x + y, x - y, x * y
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    x = torch.tensor([1.3], device=device, dtype=torch.float32)
    y = torch.tensor([1.9], device=device, dtype=torch.float32)
    a_tri = torch.tensor([0], device=device, dtype=torch.float32)
    b_tri = torch.tensor([0], device=device, dtype=torch.float32)
    c_tri = torch.tensor([0], device=device, dtype=torch.float32)
    for kernel in [with_fn, without_fn]:
        kernel[(1, )](x, y, a_tri, b_tri, c_tri, num_warps=1)
        a_ref, b_ref, c_ref = x + y, x - y, x * y
        assert a_tri == a_ref
        assert b_tri == b_ref
        assert c_tri == c_ref


# ---------------
# test atomics
# ---------------
@pytest.mark.parametrize("op, dtype_x_str, mode", itertools.chain.from_iterable([
    [('add', 'int32', mode), ('add', 'float16', mode), ('add', 'float32', mode), \
    ('max', 'int32', mode), ('max', 'float32', mode),\
    ('min', 'int32', mode), ('min', 'float32', mode),\
    ]
    for mode in ['all_neg', 'all_pos', 'min_neg', 'max_pos']]))
def test_atomic_rmw(op, dtype_x_str, mode, device='cuda'):
    n_programs = 5

    # triton kernel
    @triton.jit
    def kernel(X, Z):
        pid = tl.program_id(0)
        x = tl.load(X + pid)
        old = GENERATE_TEST_HERE

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.atomic_{op}(Z, x)'})
    torch_op = {'add': torch.sum, 'max': torch.max, 'min': torch.min}[op]
    max_neutral = float('-inf') if dtype_x_str in float_dtypes else torch.iinfo(dtype_x).min
    min_neutral = float('inf') if dtype_x_str in float_dtypes else torch.iinfo(dtype_x).max
    neutral = {'add': 0, 'max': max_neutral, 'min': min_neutral}[op]

    # triton result
    x_tri = random((n_programs, ), dtype_str=dtype_x_str, device=device)
    if mode == 'all_neg':
        x_tri = -torch.abs(x_tri)
    if mode == 'all_pos':
        x_tri = torch.abs(x_tri)
    if mode == 'min_neg':
        idx = torch.randint(n_programs, size=(1, )).item()
        x_tri[idx] = -torch.max(torch.abs(x_tri)) - 1
    if mode == 'max_pos':
        idx = torch.randint(n_programs, size=(1, )).item()
        x_tri[idx] = torch.max(torch.abs(x_tri)) + 1

    z_tri = torch.empty([], dtype=dtype_x, device=device)
    z_tri.fill_(neutral)
    kernel[(n_programs, )](x_tri, z_tri)
    # torch result
    z_ref = torch_op(x_tri).to(dtype_x)
    # compare
    exact = op not in ['add']
    if exact:
        assert z_ref.item() == z_tri.item()
    else:
        triton.testing.assert_almost_equal(z_ref, z_tri)


# ---------------
# test cast
# ---------------
@pytest.mark.parametrize("dtype_x, dtype_z, bitcast", [
    (dtype_x, dtype_z, False) \
                        for dtype_x in dtypes\
                        for dtype_z in dtypes
] + [ 
    ('float32', 'bfloat16', False),
    ('bfloat16', 'float32', False),
    ('float32', 'int32', True)
])
def test_cast(dtype_x, dtype_z, bitcast, device='cuda'):
    x0 = 43 if dtype_x in int_dtypes else 43.5
    x = torch.tensor([x0], dtype=cvt[dtype_x], device=device)

    # triton kernel
    @triton.jit
    def kernel(X, Z, BITCAST: tl.constexpr):
        x = tl.load(X)
        z = x.to(Z.dtype.element_ty, bitcast = BITCAST)
        tl.store(Z, z)

    # triton result
    z_tri = torch.empty((1, ), dtype=cvt[dtype_z], device=device)
    kernel[(1, )](x, z_tri, BITCAST=bitcast)
    # torch result
    if bitcast:
        import numpy as np
        z_ref = x.detach().cpu().numpy().view(getattr(np, dtype_z))
        z_ref = torch.from_numpy(z_ref).to(device)
    else:
        z_ref = x.to(z_tri.dtype)
    assert z_tri == z_ref

# ---------------
# test reduce
# ---------------
@pytest.mark.parametrize("dtype_str, shape", 
  [(dtype, shape) \
        for dtype in dtypes\
        for shape in [128, 512]])
def test_reduce1d(dtype_str, shape, device='cuda'):

    # triton kernel
    @triton.jit
    def kernel(X, Z, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        tl.store(Z, tl.sum(x, axis=0))

    x = random((shape,), dtype_str=dtype_str, device=device)
    # triton result
    z_tri = random((1,), dtype_str=dtype_str, device=device)
    kernel[(1,)](x, z_tri, BLOCK=shape)
    # torch result
    z_ref = torch.sum(x).to(dtype)
    # compare
    triton.testing.assert_almost_equal(z_tri, z_ref)


@pytest.mark.parametrize("dtype_str, shape, axis", 
  [(dtype, shape, 1) \
        for dtype in ['float32']\
        for shape in [(1, 1024)]])
def test_reduce2d(dtype_str, shape, axis, device='cuda'):
    # triton kernel
    @triton.jit
    def kernel(X, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
        range_m = tl.arange(0, BLOCK_M)
        range_n = tl.arange(0, BLOCK_N)
        x = tl.load(X + range_m[:, None]*BLOCK_N + range_n[None, :])
        z = tl.sum(x, axis=AXIS)
        tl.store(Z + range_m, z)
    # input
    x = random(shape, dtype_str=dtype_str, device=device)
    # triton result
    z_tri = torch.empty((shape[0],), dtype=dtype, device=device)
    kernel[(1,)](x, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], AXIS=axis)
    # torch result
    z_ref = torch.sum(x, axis=axis).to(dtype)
    # compare
    triton.testing.assert_almost_equal(z_tri, z_ref)

# ---------------
# test permute
# ---------------

@pytest.mark.parametrize("dtype_str, shape, perm",
  [(dtype, shape, perm) \
        for dtype in ['float32']\
        for shape in [(128, 128)]\
        for perm  in [(1, 0)]])
def test_permute(dtype_str, shape, perm, device='cuda'):

    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xn, 
               Z, stride_zm, stride_zn, 
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        tl.store(Zs, tl.load(Xs))
    # input
    x = random(shape, dtype_str=dtype_str, device=device)
    # triton result
    z_tri = torch.empty_like(x)
    pgm = kernel[(1, 1)](x, x.stride(0), x.stride(1), 
                        z_tri, z_tri.stride(1), z_tri.stride(0), 
                        BLOCK_M=shape[0], BLOCK_N=shape[1])
    # torch result
    z_ref = x.permute(*perm).contiguous()
    # compare
    triton.testing.assert_almost_equal(z_tri, z_ref)
    # parse ptx to make sure ld/st are vectorized
    ptx = pgm.asm['ptx']
    assert 'ld.global.v4' in ptx
    assert 'st.global.v4' in ptx

# ---------------
# test dot
# ---------------

@pytest.mark.parametrize("epilogue", ['none', 'trans', 'add-matrix', 'add-rows', 'add-cols'])
def test_dot(epilogue, device='cuda'):
    torch.manual_seed(0)
    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xk, 
               Y, stride_yk, stride_yn,
               Z, stride_zm, stride_zn, 
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        z = tl.dot(tl.load(Xs), tl.load(Ys))
        if ADD_MATRIX:
            z += tl.load(Zs)
        if ADD_ROWS:
            ZRs = Z + off_m * stride_zm
            z += tl.load(ZRs)[:, None]
        if ADD_COLS:
            ZCs = Z + off_n * stride_zn 
            z += tl.load(ZCs)[None, :]
        tl.store(Zs, z)
    # input
    M, N, K = 64, 64, 32
    x = random((M, K), dtype_str='float32', device=device)
    y = random((K, N), dtype_str='float32', device=device)
    # triton result
    z = random((M, N), dtype_str='float32', device=device)
    z_tri = z.clone()
    if epilogue == 'trans':
        z_tri = torch.as_strided(z_tri, (M, N), z_tri.stride()[::-1])
    pgm = kernel[(1, 1)](x, x.stride(0), x.stride(1),
                         y, y.stride(0), y.stride(1),
                         z_tri, z_tri.stride(0), z_tri.stride(1),
                         BLOCK_M=M, BLOCK_K=K, BLOCK_N=N,
                         ADD_MATRIX = epilogue=='add-matrix',
                         ADD_ROWS = epilogue=='add-rows',
                         ADD_COLS = epilogue=='add-cols')
    # torch result
    z_ref = torch.matmul(x.float(), y.float())
    if epilogue == 'add-matrix':
        z_ref += z
    if epilogue == 'add-rows':
        z_ref += z[:,0][:, None]
    if epilogue == 'add-cols':
        z_ref += z[0,:][None, :]
    z_ref = z_ref.to(torch.float16)
    # compare
    triton.testing.assert_almost_equal(z_tri, z_ref)
    # make sure ld/st are vectorized
    ptx = pgm.asm['ptx']
    assert 'ld.global.v4' in ptx
    assert 'st.global.v4' in ptx

def test_dot_without_load():
    @triton.jit
    def kernel(out):
        pid = tl.program_id(axis=0)
        a = tl.zeros((32, 32), tl.float32)
        b = tl.zeros((32, 32), tl.float32)
        c = tl.zeros((32, 32), tl.float32)
        c = tl.dot(a, b)
        pout = out + tl.arange(0, 32)[:, None]*32 + tl.arange(0, 32)[None, :]
        tl.store(pout, c)
        
    out = torch.ones((32,32), dtype=torch.float32, device="cuda")
    kernel[(1,)](out)

# ---------------
# test arange
# ---------------

@pytest.mark.parametrize("start", [0, 1, 7, 16])
def test_arange(start, device='cuda'):
    BLOCK = 128
    z_tri = torch.empty(BLOCK, dtype=torch.int32, device=device)
    @triton.jit
    def _kernel(z, BLOCK: tl.constexpr, 
                START: tl.constexpr, END: tl.constexpr):
        off = tl.arange(0, BLOCK)
        val = tl.arange(START, END)
        tl.store(z + off, val)
    _kernel[(1,)](z_tri, START=start, END=start+BLOCK, BLOCK=BLOCK)
    z_ref = torch.arange(start, BLOCK+start, dtype=torch.int32, device=device)
    triton.testing.assert_almost_equal(z_tri, z_ref)

# ---------------
# test load
# ---------------
# 'bfloat16': torch.bfloat16,
# Testing masked loads with an intermate copy to shared memory run.
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_masked_load_shared_memory(dtype, device='cuda'):
    M = 32
    N = 32
    K = 8

    in1 = torch.rand((M, K), dtype=dtype, device=device)
    in2 = torch.rand((K, N), dtype=dtype, device=device)
    out = torch.zeros((M, N), dtype=dtype, device=device)

    @triton.jit
    def _kernel(in1_ptr, in2_ptr, output_ptr,
                in_stride, in2_stride, out_stride,
                in_numel, in2_numel, out_numel,
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):

        M_offsets = tl.arange(0, M)
        N_offsets = tl.arange(0, N)
        K_offsets = tl.arange(0, K)

        in_offsets =  M_offsets[:, None] * in_stride + K_offsets[None,:]
        in2_offsets =  K_offsets[:, None] * in2_stride + N_offsets[None,:]

        # Load inputs.
        x = tl.load(in1_ptr + in_offsets, mask=in_offsets < in_numel)
        w = tl.load(in2_ptr + in2_offsets, mask=in2_offsets < in2_numel)

        # Without a dot product the memory doesn't get promoted to shared.
        o = tl.dot(x, w)

        # Store output
        output_offsets =  M_offsets[:, None] * out_stride + N_offsets[None,:]
        tl.store(output_ptr + output_offsets, o, mask=output_offsets < in2_numel)

    pgm = _kernel[(1,)](in1, in2, out,
                  in1.stride()[0],
                  in2.stride()[0],
                  out.stride()[0],
                  in1.numel(),
                  in2.numel(),
                  out.numel(),
                  M=M, N=N, K=K)

    reference_out =torch.matmul(in1, in2)
    triton.testing.allclose(out, reference_out)

@pytest.mark.parametrize("cache", ["", ".ca", ".cg"])
def test_load_cache_modifier(cache):
    src = torch.empty(128, device='cuda')
    dst = torch.empty(128, device='cuda')

    @triton.jit
    def _kernel(dst, src, CACHE: tl.constexpr):
        offsets = tl.arange(0, 128)
        x = tl.load(src+offsets, cache_modifier=CACHE)
        tl.store(dst+offsets, x)

    pgm = _kernel[(1,)](dst, src, CACHE=cache)
    ptx = pgm.asm['ptx']
    if cache == '':
        assert 'ld.global.ca' not in ptx
        assert 'ld.global.cg' not in ptx
    if cache == '.cg':
        assert 'ld.global.cg' in ptx
        assert 'ld.global.ca' not in ptx
    if cache == '.ca':
        assert 'ld.global.ca' in ptx
        assert 'ld.global.cg' not in ptx

# ---------------
# test store
# ---------------

# ---------------
# test if
# ---------------

# ---------------
# test for
# ---------------

# ---------------
# test while
# ---------------

# ---------------
# test default
# ---------------
#TODO: can't be local to test_default
@triton.jit
def _impl(value = 10):
    return value

def test_default():
    value = 5
    ret0 = torch.zeros(1, dtype=torch.int32, device='cuda')
    ret1 = torch.zeros(1, dtype=torch.int32, device='cuda')

    @triton.jit
    def _kernel(ret0, ret1, value):
        tl.store(ret0, _impl())
        tl.store(ret1, _impl(value))
    
    _kernel[(1,)](ret0, ret1, value)
    assert ret0.item() == 10
    assert ret1.item() == value

# ---------------
# test noop
#----------------
def test_noop(device='cuda'):
    @triton.jit
    def kernel(x):
        pass
    x = numpy_to_triton(numpy_random((1,), dtype_str='int32'), device=device)
    kernel[(1, )](x)
