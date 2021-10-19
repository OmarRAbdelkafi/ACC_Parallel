#ifndef _META_CUH_
#define _META_CUH_

#include <cuda_runtime.h>
#include <cutil.h>
#include <data.h>
#include <curand.h>

typedef int*  type_vector;
typedef int** type_matrix;


__host__ void Meta_Init(int size,type_vector a,type_vector b);

__host__ void Meta_Optimize(int size,int opt,type_vector a,type_vector b);

__host__ void Meta_Display_results(FILE* fres, int size, int BKS);

__host__ void Meta_Free();

__host__ void Meta_Display_trials_results(FILE* fres, int max_runs, int BKS);


#endif
