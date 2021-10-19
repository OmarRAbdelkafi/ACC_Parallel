#ifndef _GPU_EVALUATION_QAP_CUH_
#define _GPU_EVALUATION_QAP_CUH_

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <curand_kernel.h>

typedef int*  type_vector;
typedef int** type_matrix;

int compute_delta(int n, type_vector a, type_vector b, type_vector p, int i, int j);
int compute_delta_part(type_vector a, type_vector b, type_vector p, type_vector delta, int i, int j, int r, int s,int n);

#endif
