import torch
import triton
import triton._C.libtriton.triton as _triton

def get_dram_bw(backend=None, device=None):
  ''' return DRAM bandwidth in GB/s '''
  # assert backend == CUDA
  if not backend:
    backend = _triton.runtime.backend.CUDA
  if not device:
    device = torch.cuda.current_device()
  mem_clock_khz = _triton.runtime.memory_clock_rate(backend, device)
  bus_width = _triton.runtime.global_memory_bus_width(backend, device)
  bw = mem_clock_khz * bus_width * 2 // 1024 // 1024 // 8 # In GB/s
  return bw

def get_matmul_tput(backend, device, num_ctas, num_warps):
  ''' return compute throughput in TOPS '''
  num_subcores = _triton.runtime.num_sm(backend, device) * 4 # on recent GPUs
  total_warps = num_ctas * min(num_warps, 4)
  clock_rate = _triton.runtime.clock_rate(backend, device) # in kHz
  # assume fp32 += fp16*fp16
  tput = min(num_subcores, total_warps) * clock_rate *2*4*4*4*2 # 2 4x4x4 Tensor Cores
  tput /= 1024*1024*1024
  return tput

def estimate_matmul_time(
  backend, device, num_warps,
  m, n, k, 
  BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K,
  debug=False
):
  ''' return estimated running time in ms 
        = max(compute, loading) + store '''
  num_cta_m = triton.cdiv(m, BLOCK_M)
  num_cta_n = triton.cdiv(n, BLOCK_N)
  num_cta_k = SPLIT_K
  num_ctas = num_cta_m * num_cta_n * num_cta_k
  # time to compute
  total_ops = 2*m*n*k / (1024*1024*1024) # GOPS
  tput = get_matmul_tput(backend, device, num_ctas, num_warps)
  compute_ms = total_ops / tput

  # time to load data
  num_sm  = _triton.runtime.num_sm(backend, device)
  active_cta_ratio = min(1, num_ctas/num_sm)
  dram_bw = get_dram_bw(backend, device) * active_cta_ratio # in GB/s
  l2_bw   = dram_bw * 3 * active_cta_ratio # rough estimation
  # assume 80% of (following) loads are in L2 cache
  load_a_dram = m*k*2*(1+0.2*(num_cta_n-1)) # assume dtype=float16 (size==2)
  load_a_l2   = m*k*2*0.8*(num_cta_n-1)
  load_b_dram = n*k*2*(1+0.2*(num_cta_m-1))
  load_b_l2   = n*k*2*0.8*(num_cta_m-1)
  # total
  total_dram = (load_a_dram + load_b_dram) / (1024*1024) # MB
  total_l2   = (load_a_l2   + load_b_l2)   / (1024*1024)     
  # loading time in ms
  load_ms = total_dram/dram_bw + total_l2/l2_bw

  # estimate storing time
  store_bw = dram_bw * 0.6 # :o
  store_c_dram = m*n*2*SPLIT_K / (1024*1024) # MB
  if SPLIT_K == 1:
    store_ms = store_c_dram /store_bw
  else:
    reduce_bw = store_bw/2
    store_ms = store_c_dram/reduce_bw

  total_time_ms = max(compute_ms, load_ms) + store_ms
  if debug:
    print(f'Total time: {total_time_ms}ms, compute time: {compute_ms}ms, '
          f'loading time: {load_ms}ms, store time: {store_ms}ms, '
          f'Activate CTAs: {active_cta_ratio*100}%')
  return total_time_ms