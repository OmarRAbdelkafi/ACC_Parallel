/*-------------------------------------------------------------------------
   GPU based Hybrid iterative tabu search:
  - Universite de Haute-Alsace (LMIA-MIAGE)
  - Omar Abdelkafi
  ------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#include "data.h"
#include "Meta.cuh"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cutil_inline.h>



int main()
{

FILE* global_file=NULL;
char instance[30];
int max_runs, run; /* number of trials */


FILE * fres=fopen("global_results.xls","w");
fprintf(fres,"Instance: {iteration = n*1000;1 process; 200 global iteration w<20 et 10} \n");
fprintf(fres,"T_gpu_hits en secondes\tglobal_best_fitness\n");

global_file = fopen("global.txt","r");

for(int c=0;c<37;c++){//37 instances dans le fichier


  fscanf(global_file,"%s\n", instance);
  printf("%s",instance); printf("\n");
  fprintf(fres,"instance: %s\n",instance);

  /****************************** read file name and specifique data for the problem *******************************/

  int size;                      /* problem size        */
  int BKS;                       /* Best known solution */

  DATA *d;
  d = load_data(instance);

  size = d->n;
  BKS  = d->opt;
  max_runs = 10;

  /******************************************* Metaheuristic execution ********************************************/
  for(run = 0; run < max_runs; run++)
  {
	  //Initialization of CUDA
	  cudaSetDevice( cutGetMaxGflopsDeviceId() );

	  Meta_Init(size, d->a, d->b);

	  Meta_Optimize(size, BKS, d->a, d->b);

	  Meta_Display_results(fres, size, BKS);

	  Meta_Free();

  }//End trials

  Meta_Display_trials_results(fres, max_runs, BKS);

  free_data(d);
  fflush(stdin);

}//fin global file

   fclose(global_file);
   fclose(fres);
  return EXIT_SUCCESS;
 }
