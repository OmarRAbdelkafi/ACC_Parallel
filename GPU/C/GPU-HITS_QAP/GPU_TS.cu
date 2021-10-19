#ifndef _GPU_TS_CU_
#define _GPU_TS_CU_

#include <GPU_Evaluation_QAP.cuh>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <curand_kernel.h>

typedef int*  type_vector;
typedef int** type_matrix;

extern const int infinite;
extern const int FALSE;
extern const int TRUE;

double rando();

int unif(int low, int high);

void transpose(int *a, int *b);

int minim(int a, int b);

double cube(double x);

static __global__ void g_update_p(int* g_p, int i_retained, int j_retained){

	int temp = g_p[i_retained];
	g_p[i_retained] = g_p[j_retained];
	g_p[j_retained] = temp;

}


__host__ void tabu_search_parallel(int n,         /* problem size */
                 type_vector a,         	  /* flows matrix */
                 type_vector b,        	 	  /* distance matrix */
                 type_vector best_sol,  	  /* best solution found */
                 int *best_cost,        	  /* cost of best solution */
                 int tabu_duration,    	          /* parameter 1 (< n^2/2) */
                 int aspiration,        	  /* parameter 2 (> n^2/2)*/
                 int nr_iterations,int BKS,       /* number of iterations */ 
           	 int *c_a,int* c_b, int* g_delta, int* g_p)
 
{
	  type_vector p;                        /* current solution */
	  type_vector delta;                    /* store move costs */
	  type_matrix tabu_list;                /* tabu status */
	  int current_iteration;                /* current iteration */
	  int current_cost;                     /* current sol. value */
	  int i, j, k, i_retained, j_retained;  /* indices */
	  int min_delta;                        /* retained move cost */
	  int autorized;                        /* move not tabu? */
	  int aspired;                          /* move forced? */
	  int already_aspired;                  /* in case many moves forced */

	  /***************** dynamic memory allocation *******************/
	  p = (int*)calloc(n, sizeof(int));
	  delta = (int*)calloc((n*n),sizeof(int));
	  tabu_list = (int**)calloc(n,sizeof(int*));
	  for (i = 0; i < n; i = i+1) tabu_list[i] = (int*)calloc(n, sizeof(int));


	  /************** current solution initialization ****************/
	  for (i = 0; i < n; i = i + 1) p[i] = best_sol[i];

	  /********** initialization of current solution value ***********/
	  /**************** and matrix of cost of moves  *****************/
	  current_cost = 0;
	  for (i = 0; i < n; i = i + 1) for (j = 0; j < n; j = j + 1)
	   {
	    current_cost = current_cost + a[i*n+j] * b[p[i]*n+p[j]];
	   };

	  //cudaMemcpy(g_delta, delta, (n * n * sizeof(int)), cudaMemcpyHostToDevice);
	  //cutilCheckMsg("tabu cudaITSSD delta: cudaMemcpy() execution failed\n");
	  cudaMemcpy(g_p, p, (n * sizeof(int)), cudaMemcpyHostToDevice);
	  cutilCheckMsg("tabu search delta: cudaMemcpy() execution failed\n");

	  h_compute_delta(n, g_p, g_delta, c_a, c_b);

	   cudaMemcpy(delta, g_delta, (n * n * sizeof(int)), cudaMemcpyDeviceToHost);
	   cutilCheckMsg("tabu search DeviceToHost delta: cudaMemcpy() execution failed\n");
	  
	  *best_cost = current_cost;

	  /****************** tabu list initialization *******************/
	  for (i = 0; i < n; i = i + 1) for (j = 0; j < n; j = j+1)
	    tabu_list[i][j] = -(n*i + j);

	  /******************** main tabu search loop ********************/
	  for (current_iteration = 1; current_iteration <= nr_iterations && *best_cost > BKS; current_iteration++ )
	   {/** find best move (i_retained, j_retained) **/
	    i_retained = infinite;       /* in case all moves are tabu */
	    j_retained = infinite;
	    min_delta = infinite;
	    already_aspired = FALSE;
	    
	    for (i = 0; i < n-1; i = i + 1) 
	      for (j = i+1; j < n; j = j+1)
	       {autorized = (tabu_list[i][p[j]] < current_iteration) || (tabu_list[j][p[i]] < current_iteration);
		aspired = (tabu_list[i][p[j]] < current_iteration-aspiration)||(tabu_list[j][p[i]] < current_iteration-aspiration)||
		          (current_cost + delta[i*n+j] < *best_cost);                

		if ((aspired && !already_aspired) || /* first move aspired */
		   (aspired && already_aspired &&    /* many move aspired  */
		    (delta[i*n+j] < min_delta)) ||   /* => take best one   */
		   (!aspired && !already_aspired &&  /* no move aspired yet*/
		    (delta[i*n+j] < min_delta) && autorized))
		  {i_retained = i; j_retained = j;
		   min_delta = delta[i*n+j];
		   if (aspired) {already_aspired = TRUE;};};};

	    if (i_retained == infinite) printf("All moves are tabu! \n"); 
	    else 
	     {/** transpose elements in pos. i_retained and j_retained **/
	      transpose(&p[i_retained], &p[j_retained]);
	      /* update solution value*/
	      current_cost = current_cost + delta[i_retained*n+j_retained];
	      /* forbid reverse move for a random number of iterations*/
	      tabu_list[i_retained][p[j_retained]] = current_iteration + (int)(cube(rando())*tabu_duration);
	      tabu_list[j_retained][p[i_retained]] = current_iteration + (int)(cube(rando())*tabu_duration);

	      /* best solution improved ?*/
	      if (current_cost < *best_cost)
	       {*best_cost = current_cost;
		for (k = 0; k < n; k = k+1) best_sol[k] = p[k];
		//printf("Solution of value: %d found at iter. %d\n", current_cost, current_iteration);
	       };  
	   
		g_update_p<<<1, 1>>>(g_p, i_retained, j_retained);
		cutilCheckMsg("tabu search: g_update_p() execution failed\n");


	        h_update_delta(n, g_p, g_delta, c_a, c_b, i_retained, j_retained);

	        cudaMemcpy(delta, g_delta, (n * n * sizeof(int)), cudaMemcpyDeviceToHost);
	        cutilCheckMsg("TS DeviceToHost delta: cudaMemcpy() execution failed\n");
	     
	     };
	   }; 
	  /* free memory*/
	  free(p);
	  free(delta);
	  for (i=0; i < n; i = i+1) free(tabu_list[i]); free(tabu_list);
} /* parallel tabu*/

#endif


