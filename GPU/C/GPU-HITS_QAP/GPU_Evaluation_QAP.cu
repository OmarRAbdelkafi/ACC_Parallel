#ifndef _GPU_EVALUATION_QAP_CU_
#define _GPU_EVALUATION_QAP_CU_

#include <stdint.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>


static __global__ void g_compute_delta(int n, int* g_p, int* g_delta, int* c_a, int* c_b){

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int k; int d;
  if(idx < (n*n)){
	  int i=(int)(idx / n);
	  int j=(int)(idx % n);

	  if(i<j){
		    d = (c_a[(i*n)+i]-c_a[(j*n)+j]) * (c_b[(g_p[j]*n)+g_p[j]] - c_b[(g_p[i]*n)+g_p[i]]) +
			(c_a[(i*n)+j] - c_a[(j*n)+i]) * (c_b[(g_p[j]*n)+g_p[i]] - c_b[(g_p[i]*n)+g_p[j]]);

		    for (k = 0; k < n; k = k + 1) 
		    {
				 if (k!=i && k!=j){
				      d = d + (c_a[(k*n)+i]-c_a[(k*n)+j]) * (c_b[(g_p[k]*n)+g_p[j]]-c_b[(g_p[k]*n)+g_p[i]]) +
					 (c_a[(i*n)+k]-c_a[(j*n)+k]) * (c_b[(g_p[j]*n)+g_p[k]]-c_b[(g_p[i]*n)+g_p[k]]);
				 }
		    }
		    g_delta[i*n+j] = d;
	  }
  }
}

static __global__ void g_update_delta(int n, int* g_p, int* g_delta, int* c_a, int* c_b,int i_retained,int j_retained){

  __shared__ int p_j_r;
  __shared__ int p_i_r;

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int k; int d;
  if(idx < (n*n)){
	  int i=(int)(idx / n);
	  int j=(int)(idx % n);

	  p_j_r = g_p[j_retained];
	  p_i_r = g_p[i_retained];
	  __syncthreads();

	  if(i<j){
	       if (i != i_retained && i != j_retained && j != i_retained && j != j_retained){

			 d = g_delta[i*n+j] + (c_a[i_retained*n+i]-c_a[i_retained*n+j]+c_a[j_retained*n+j]-c_a[j_retained*n+i]) *
			 (c_b[p_j_r*n+g_p[i]]-c_b[p_j_r*n+g_p[j]]+c_b[p_i_r*n+g_p[j]]-
			  c_b[p_i_r*n+g_p[i]]) + (c_a[i*n+i_retained]-c_a[j*n+i_retained]+c_a[j*n+j_retained]-
			  c_a[i*n+j_retained]) * (c_b[g_p[i]*n+p_j_r]-c_b[g_p[j]*n+p_j_r]+
			  c_b[g_p[j]*n+p_i_r]-c_b[g_p[i]*n+p_i_r]);

			 __syncthreads();
	       }
	       else{
		 	 d = (c_a[(i*n)+i]-c_a[(j*n)+j]) * (c_b[(g_p[j]*n)+g_p[j]] - c_b[(g_p[i]*n)+g_p[i]]) + (c_a[(i*n)+j] -
			      c_a[(j*n)+i]) * (c_b[(g_p[j]*n)+g_p[i]] - c_b[(g_p[i]*n)+g_p[j]]);

			 for (k = 0; k < n; k = k + 1) 
			 {
				 if (k!=i && k!=j){
					  d = d + (c_a[(k*n)+i]-c_a[(k*n)+j])*(c_b[(g_p[k]*n)+g_p[j]]-c_b[(g_p[k]*n)+g_p[i]]) +
					      (c_a[(i*n)+k]-c_a[(j*n)+k])*(c_b[(g_p[j]*n)+g_p[k]]-c_b[(g_p[i]*n)+g_p[k]]);
				 }
			 }
	       }
	       g_delta[i*n+j] = d;
	  }
  }
}

__host__ void h_compute_delta(int n, int* g_p, int* g_delta, int *c_a, int* c_b) 
{
	//kernel parameters for delta matrix
	//	- block is the size of the thread block
	//	- grid is the number of block computed by the size of the instance
	dim3 Grid((((n*n)/256)+1), 1, 1);
	dim3 Block(256,1,1);
	
	//initialize the matrix of cost of moves
	g_compute_delta<<<Grid, Block>>>(n, g_p, g_delta,c_a,c_b);
	cutilCheckMsg("tabu search: g_compute_delta() execution failed\n");	 
}

__host__ void h_update_delta(int n, int* g_p, int* g_delta, int *c_a, int* c_b, int i_retained, int j_retained) 
{
	//kernel parameters for delta matrix
	//	- block is the size of the thread block
	//	- grid is the number of block computed by the size of the instance
	dim3 Grid((((n*n)/256)+1), 1, 1);
	dim3 Block(256,1,1);
	
	//Update the matrix of move costs
	g_update_delta<<<Grid, Block>>>(n, g_p, g_delta,c_a,c_b,i_retained,j_retained);
	cutilCheckMsg("TS: g_update_delta() execution failed\n");	 
}



#endif


