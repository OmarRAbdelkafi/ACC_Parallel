/*-------------------------------------------------------------------------
   ITERATIVE TABOU SEARCH:
  - Universite de Lille (BONUS-CRISTAL)
  - Omar Abdelkafi
  ------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "data.h"
#include "Meta.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
#include "mpi.h"
//#include <cutil_inline.h>



int main(int argc, char **argv){

	/************************MPI for distribute algorithm activated**********************/
	int all, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &all);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;
	//************
	//************


	FILE* global_file=NULL;
	char instance[30];
	int max_runs, run; /* number of trials */


	FILE * fres=fopen("global_results_CPU_MPI.xls","w");
	fprintf(fres,"Instance: {iteration = n*100; 4 process; 100 global iteration} \n");
	fprintf(fres,"T_cpu_hits en secondes\tglobal_best_fitness\n");

	global_file = fopen("global_test.txt","r");

	for(int c=0;c<2;c++){//2 instances in file global_test


		fscanf(global_file,"%s\n", instance);
		if(rank==0){
			printf("%s",instance); printf("\n");
			fprintf(fres,"instance: %s\n",instance);
		}
		/* Recolte */
		char nom_fichier2[31] = "Display3/global_results_TS_";
		sprintf(&nom_fichier2[27], "%d_",c);

		/******** read file name and specifique data for the problem **********/

		int size;                      /* problem size        */
		int BKS;                       /* Best known solution */

		DATA *d;
		d = load_data(instance);

		size = d->n;
		BKS  = d->opt;
		max_runs = 20;

		/********************* Metaheuristic execution ************************/
		for(run = 0; run < max_runs; run++)
		{
			  sprintf(&nom_fichier2[29], "%d",run+1);
			  FILE * fresGTS=fopen(nom_fichier2,"w");

			  Meta_Init(size, d->a, d->b, rank, all);
			  //printf("pass init rank %d pour run %d\n",rank,run);

			  Meta_Optimize(size, BKS, d->a, d->b, fresGTS, rank, all, status);
			  //printf("pass Optim rank %d pour run %d\n",rank,run);

			  Meta_Display_results(fres, size, BKS, rank, all, run, status);
			  //printf("pass display rank %d pour run %d\n",rank,run);

			  Meta_Free(rank);
			  //printf("pass free rank %d pour run %d\n",rank,run);

			  fclose(fresGTS);
		}//End trials

		Meta_Display_trials_results(fres, max_runs, BKS, rank, all, status);
		//printf("*****pass global display rank %d *******\n",rank);

		free_data(d);
		fflush(stdin);

	}//fin global file

	fclose(global_file);
	fclose(fres);
	MPI_Finalize();

	return EXIT_SUCCESS;
 }
