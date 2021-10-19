#ifndef _META_CU_
#define _META_CU_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <curand_kernel.h>
#include <GPU_TS.cuh>
#include <CPU_TS.cuh>
#include <curand.h>
#include "mpi.h"
#include "data.h"

/******************************************************************************************************/
/***************************************USER DECLARATION***********************************************/
/******************************************************************************************************/

//********
//Activation
bool GPU;
bool CPU;

//********
//CPU Data
const int infinite = 2147483647;//max int
const int FALSE = 0;
const int TRUE = 1;

double  somme_sol = 0.0;
double  avg_time = 0.0;
double  cost_sol;

typedef int*  type_vector;
typedef int** type_matrix;

int max_global_iteration;
double  stop_condition;
int L;//limit of stagnacy

type_vector individual;           /* current solution (permutation) */
int fitness;                      /* current cost                   */
type_vector best_individual;      /* best solution                  */
int best_fitness;                 /* best cost                      */

int w;    /* history of search parameters       */
int step; /* step of the glover diversification */

bool Stangnancy;
int perturb; /* perturbation parameters */
int TS_iterations; /* number of Tabu Search iteration*/

clock_t iter_end,start,end;

//********
//GPU Data
int          *c_a;       /* Matrix of flows     */
int          *c_b;       /* Matrix of distances */
int          *g_delta;   /* Matrix of costs     */
int          *g_p;       /* current solution    */

/***************************************End User Declaration******************************************/

/***************************************************************************************************/
/***************************************USER FONCTION***********************************************/
/***************************************************************************************************/

/*************** L'Ecuyer random number generator ***************/
double rando()
 {
  static int x10 = 12345, x11 = 67890, x12 = 13579, /* initial value*/
             x20 = 24680, x21 = 98765, x22 = 43210; /* of seeds*/
  const int m = 2147483647; const int m2 = 2145483479;
  const int a12= 63308; const int q12=33921; const int r12=12979;
  const int a13=-183326; const int q13=11714; const int r13=2883;
  const int a21= 86098; const int q21=24919; const int r21= 7417;
  const int a23=-539608; const int q23= 3976; const int r23=2071;
  const double invm = 4.656612873077393e-10;
  int h, p12, p13, p21, p23;
  h = x10/q13; p13 = -a13*(x10-h*q13)-h*r13;
  h = x11/q12; p12 = a12*(x11-h*q12)-h*r12;
  if (p13 < 0) p13 = p13 + m; if (p12 < 0) p12 = p12 + m;
  x10 = x11; x11 = x12; x12 = p12-p13; if (x12 < 0) x12 = x12 + m;
  h = x20/q23; p23 = -a23*(x20-h*q23)-h*r23;
  h = x22/q21; p21 = a21*(x22-h*q21)-h*r21;
  if (p23 < 0) p23 = p23 + m2; if (p21 < 0) p21 = p21 + m2;
  x20 = x21; x21 = x22; x22 = p21-p23; if(x22 < 0) x22 = x22 + m2;
  if (x12 < x22) h = x12 - x22 + m; else h = x12 - x22;
  if (h == 0) return(1.0); else return(h*invm);
 }

/*********** return an integer between low and high *************/
int unif(int low, int high)
 {return low + (int)((double)(high - low + 1) * rando()) ;}

void transpose(int *a, int *b) {int temp = *a; *a = *b; *b = temp;}

int minim(int a, int b) {if (a < b) return(a); else return(b);}

double cube(double x) {return x*x*x;}


void generate_random_solution(int n, type_vector  p)
 {int i;
  for (i = 0; i < n;   i++) p[i] = i;
  for (i = 0; i < n-1; i++) transpose(&p[i], &p[unif(i, n-1)]);
 }

/***************************************End User Fonction******************************************/

__host__ void Meta_Init(int n,type_vector a,type_vector b)
{
  //********
  //Activation
  GPU = TRUE;
  CPU = FALSE;

  if(GPU) printf("GPGPU optimizer activated\n");
  if(CPU) printf("CPU optimizer activated\n");
  //********
  //GPU allocation and initialization
  printf("GPU init Data...\n");

  int dim;
  dim = n * n * sizeof(int);

  //Allocation for distance and flow matrices
  cudaMalloc((void **)&c_a, dim);
  cutilCheckMsg("Meta_init a: cudaMalloc() execution failed\n");
  cudaMemset(c_a, 0, dim);
  cutilCheckMsg("Meta_init a: cudaMemset() execution failed\n");
  cudaMemcpy(c_a, a, (n * n * sizeof(int)), cudaMemcpyHostToDevice);
  cutilCheckMsg("Meta_init a: cudaMemcpy() execution failed\n");

  cudaMalloc((void **)&c_b, dim);
  cutilCheckMsg("Meta_init b: cudaMalloc() execution failed\n");
  cudaMemset( c_b, 0, dim);
  cutilCheckMsg("Meta_init b: cudaMemset() execution failed\n");
  cudaMemcpy(c_b, b, (n * n * sizeof(int)), cudaMemcpyHostToDevice);
  cutilCheckMsg("Meta_init b: cudaMemcpy() execution failed\n");

  //Allocation for the delta matrix
  cudaMalloc((void **)&g_delta, dim);
  cutilCheckMsg("Meta_init delta: cudaMalloc() execution failed\n");
  cudaMemset( g_delta, 0, dim);
  cutilCheckMsg("Meta_init delta: cudaMemset() execution failed\n");

  dim = n * sizeof(int);

  //Allocation for the vector of solution p
  cudaMalloc((void **)&g_p, dim);
  cutilCheckMsg("Meta_init p: cudaMalloc() execution failed\n");
  cudaMemset( g_p, 0, dim);
  cutilCheckMsg("Meta_init p: cudaMemset() execution failed\n");

  //********
  //CPU allocation and initialization
  printf("CPU init Data...\n");

  TS_iterations = n*1000; /* SIZE MULTIPLY BY 1000*/
  max_global_iteration = 200;
  if(n <= 100) stop_condition = 3600;//1 h
  else stop_condition = 14400;//4 h

  cost_sol  = 0.0;

  individual = (int*)calloc(n, sizeof(int));
  best_individual = (int*)calloc(n, sizeof(int));

  best_fitness = infinite;

  L = 20;
  w = 0;
  step = 3;

  Stangnancy = FALSE;
  perturb    = n/4; /* 25% of the solution */

}


__host__ void Meta_Optimize(int n,int BKS,type_vector a,type_vector b)
{

  start = clock();

  int i, k,position,star;
  int global_iter = 0;

  generate_random_solution(n, individual);

  do{
	if(GPU)
	{
	   	tabu_search_parallel(n, a, b,                /* problem data         */
	            	 individual, &fitness,               /* tabu search results  */
	             	 8*n, n*n*5,                         /* parameters           */
	               TS_iterations,BKS,                   /* number of iterations */
			           c_a, c_b, g_delta, g_p);             /* GPU DATA*/
	}

	if(CPU)
	{
	   	tabu_search_seq(n, a, b,                    /* problem data         */
		     	 individual, &fitness,              /* tabu search results  */
		     	 8*n, n*n*5,                        /* parameters           */
		 	 TS_iterations,BKS);                /* number of iterations */
        }
        /* Amelioration */
	if(fitness < best_fitness){

		Stangnancy = FALSE;
		w=0;
		best_fitness = fitness;

        	for (k = 0; k < n; k = k+1) best_individual[k] = individual[k];

        }
	else w++;
	/*End Amelioration part */

	if (w == L){ /* stagnancy*/
		Stangnancy = TRUE;
	}

	if (Stangnancy == FALSE){
		/* re-construction of the solution using the Golver method */  /* Rule 1 */
		position = 0;
		for (star = step; star >= 1; star--){
			for (k = (star-1); k < n; k = k+step){
				individual[position] = best_individual[k];
				position++;
			}
		}
		if(step == (n-1)) step=3;//reinitialization
		else step++;
	}//End reconstruction

	else {
		Stangnancy = FALSE;
		w=0;
		for (i = 0; i < n-1; i++) transpose(&individual[i], &individual[unif(i, n-1)]);/* Rule 2 */
	}//End re-localization

    	if(w == (L/2)){
		for (i = perturb; i < n-1; i++) transpose(&individual[i], &individual[unif(i, n-1)]); /* Rule 3 */
	}

	iter_end = clock();

        global_iter++;

   }while( global_iter < max_global_iteration /*&& best_fitness > BKS*/ );
   //}while( ((double)(iter_end-start)/CLOCKS_PER_SEC) < stop_condition && best_fitness > BKS );

   end = clock();

   cost_sol = best_fitness;

}

// Display and write results
__host__ void Meta_Display_results(FILE* fres, int n, int BKS){

   printf(" Best solution value found by GPU_HITS: %f \n", cost_sol);
   printf(" Best deviation found by GPU_HITS: %f \n", 100*(cost_sol - BKS)/BKS);
   for (int i = 0; i < n; i = i+1) printf("%d- ", (best_individual[i]+1));/* +1 pour l'affichage */
   printf("\n");
   printf(" Execution time = [%.3lf] second\n", (double)(end-start)/CLOCKS_PER_SEC);

   fprintf(fres,"%.3lf\t",(double)(end-start)/CLOCKS_PER_SEC);
   //fprintf(fres,"%f\n",100*(cost_sol - BKS)/BKS);
    fprintf(fres,"%f\n",cost_sol);

    somme_sol += best_fitness;
    avg_time += (double)(end-start)/CLOCKS_PER_SEC;

}


// Frees resources
__host__ void Meta_Free(){
	//**********************
	//  ARRAYS DE-ALLOCATION
	//**********************

	cudaFree(c_a);
	cutilCheckMsg("h_HITS_Free a: cudaFree() execution failed\n");
	cudaFree(c_b);
	cutilCheckMsg("h_HITS_Free b: cudaFree() execution failed\n");
	cudaFree(g_delta);
	cutilCheckMsg("h_HITS_Free delta: cudaFree() execution failed\n");
	cudaFree(g_p);
	cutilCheckMsg("h_HITS_Free P: cudaFree() execution failed\n");

	free(individual);
	free(best_individual);
}

// Display and write global results
__host__ void Meta_Display_trials_results(FILE* fres, int max_runs, int BKS){

   printf("****Recap****\n");
   printf("Average cost: %f, average dev: %f\n", somme_sol/max_runs, 100*(somme_sol/max_runs - BKS)/BKS);
   printf("Average time: %.3lf\n", avg_time/max_runs);

   fprintf(fres,"***AVG*** \n");
   fprintf(fres,"%.3lf\t",avg_time/max_runs);
   fprintf(fres,"%f\n",100*(somme_sol/max_runs - BKS)/BKS);
   fprintf(fres,"%f\n",somme_sol/max_runs);

   somme_sol = 0.0;
   avg_time  = 0.0;

}

#endif
