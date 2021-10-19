#ifndef _CPU_TS_CUH_
#define _CPU_TS_CUH_

#include <cuda_runtime.h>
#include <cutil_inline.h>

typedef int*  type_vector;
typedef int** type_matrix;

extern const int infinite;
extern const int FALSE;
extern const int TRUE;

void tabu_search_seq(int n,              /* problem size */
                 type_vector a,          /* flows matrix */
                 type_vector b,          /* distance matrix */
                 type_vector best_sol,   /* best solution found */
                 int *best_cost,         /* cost of best solution */
                 int tabu_duration,      /* parameter 1 (< n^2/2) */
                 int aspiration,         /* parameter 2 (> n^2/2)*/
                 int nr_iterations,int BKS);     /* number of iterations */ 


#endif
