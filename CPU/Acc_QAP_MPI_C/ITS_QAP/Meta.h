#ifndef _META_H_
#define _META_H_

//#include <data.h>
#include "mpi.h"

typedef int*  type_vector;
typedef int** type_matrix;


void Meta_Init(int size,type_vector a,type_vector b, int rank, int all);

void Meta_Optimize(int size,int opt,type_vector a,type_vector b, FILE* fresGTS,
				int rank, int all, MPI_Status status);

void Meta_Display_results(FILE* fres, int size, int BKS, int rank, int all, int run,
					MPI_Status status);

void Meta_Free(int rank);

void Meta_Display_trials_results(FILE* fres, int max_runs, int BKS,int rank, int all, 
						MPI_Status status);


#endif
