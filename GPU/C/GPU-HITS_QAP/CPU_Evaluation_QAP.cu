#ifndef _GPU_EVALUATION_QAP_CU_
#define _GPU_EVALUATION_QAP_CU_

#include <stdint.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>

typedef int*  type_vector;
typedef int** type_matrix;

/*--------------------------------------------------------------*/
/*       compute the cost difference if elements i and j        */
/*         are transposed in permutation (solution) p           */
/*--------------------------------------------------------------*/
int compute_delta(int n, type_vector a, type_vector b, type_vector p, int i, int j){
  int d; int k;
  d = (a[i*n+i]-a[j*n+j])*(b[p[j]*n+p[j]]-b[p[i]*n+p[i]]) + (a[i*n+j]-a[j*n+i])*(b[p[j]*n+p[i]]-b[p[i]*n+p[j]]);
  for (k = 0; k < n; k = k + 1) if (k!=i && k!=j)
    d = d + (a[k*n+i]-a[k*n+j])*(b[p[k]*n+p[j]]-b[p[k]*n+p[i]]) + (a[i*n+k]-a[j*n+k])*(b[p[j]*n+p[k]]-b[p[i]*n+p[k]]);
  return(d);
 }

/*--------------------------------------------------------------*/
/*      Idem, but the value of delta[i][j] is supposed to       */
/*    be known before the transposition of elements r and s     */
/*--------------------------------------------------------------*/
int compute_delta_part(type_vector a, type_vector b, type_vector p, type_vector delta, int i, int j, int r, int s,int n)
  {
     return ( delta[i*n+j]+(a[r*n+i]-a[r*n+j]+a[s*n+j]-a[s*n+i]) * (b[p[s]*n+p[i]]-b[p[s]*n+p[j]]+b[p[r]*n+p[j]]-b[p[r]*n+p[i]]) +
            (a[i*n+r]-a[j*n+r]+a[j*n+s]-a[i*n+s]) * (b[p[i]*n+p[s]]-b[p[j]*n+p[s]]+b[p[j]*n+p[r]]-b[p[i]*n+p[r]]) );
  }

#endif


