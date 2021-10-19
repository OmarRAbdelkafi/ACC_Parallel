#ifndef _GPU_EVALUATION_QAP_CUH_
#define _GPU_EVALUATION_QAP_CUH_

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <curand_kernel.h>

__host__ void h_compute_delta(int n, int* g_p, int* g_delta, int *c_a, int* c_b);
__host__ void h_update_delta(int n, int* g_p, int* g_delta, int *c_a, int* c_b, int i_retained, int j_retained);

#endif
