#ifndef _META_CPP_
#define _META_CPP_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#include "data.h"

/*****************************************************************/
/*********************USER DECLARATION***********************/
/*****************************************************************/

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
double  best_cost_sol;

typedef int*  type_vector;
typedef int** type_matrix;

int max_global_iteration;
double  stop_condition;
//int L;  //limit of stagnacy


type_vector individual;           /* current solution (permutation) */
int fitness;                      /* current cost                   */
type_vector best_individual;      /* best solution                  */
int best_fitness;                 /* best cost                      */
type_vector tmp_individual;       /* solution for send              */

//int w;     /* history of search parameters       */
int step; /* step of the glover diversification */

//bool Stangnancy;
int perturb;         /* perturbation parameters            */
int TS_iterations; /* number of Tabu Search iteration*/

int taille_pool;
int* pool_best_fitness;
int* pool_best_individuals;
bool INSERT;

float prc_remp; //pourcentage remplissage du pool

clock_t iter_end,start,end;

/*****************************End User Declaration**********************/

/*****************************************************************/
/***********************USER FONCTION*************************/
/*****************************************************************/

/*************** L'Ecuyer random number generator ***************/
double rando(){
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
int unif(int low, int high){
	return low + (int)((double)(high - low + 1) * rando()) ;
}

void transpose(int *a, int *b) {int temp = *a; *a = *b; *b = temp;}

int minim(int a, int b) {if (a < b) return(a); else return(b);}

double cube(double x) {return x*x*x;}


void generate_random_solution(int n, type_vector  p){
	int i;
	for (i = 0; i < n;   i++) p[i] = i;
	for (i = 0; i < n-1; i++) transpose(&p[i], &p[unif(i, n-1)]);
 }
 /**********************MPI FUNCTION*****************************/

 /*************** MPI: L'Ecuyer random number generator ***************/
double MPI_rando(int rank)
 {
  static int x10 = 12345+rank, x11 = 67890+rank, x12 = 13579+rank, /* initial value*/
             x20 = 24680+rank, x21 = 98765+rank, x22 = 43210+rank; /* of seeds*/
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

/*********** MPI: return an integer between low and high *************/
int MPI_unif(int low, int high,int rank)
 {return low + (int)((double)(high - low + 1) * MPI_rando(rank)) ;}

void MPI_generate_random_solution(int n, type_vector  p, int rank)
 {int i;
  for (i = 0; i < n;   i++) p[i] = i;
  for (i = 0; i < n-1; i++) transpose(&p[i], &p[MPI_unif(i, n-1,rank)]);
 }

/*-----------------------------------------------------------------------------------*/
/*       compute the cost difference if elements i and j    */
/*         are transposed in permutation (solution) p        */
/*-----------------------------------------------------------------------------------*/
int compute_delta(int n, type_vector a, type_vector b, type_vector p, int i, int j){
	int d; int k;
	d = (a[i*n+i]-a[j*n+j])*(b[p[j]*n+p[j]]-b[p[i]*n+p[i]]) +
	(a[i*n+j]-a[j*n+i])*(b[p[j]*n+p[i]]-b[p[i]*n+p[j]]);

	for (k = 0; k < n; k = k + 1){
		if (k!=i && k!=j){
			d = d + (a[k*n+i]-a[k*n+j])*(b[p[k]*n+p[j]]-b[p[k]*n+p[i]])
			+ (a[i*n+k]-a[j*n+k])*(b[p[j]*n+p[k]]-b[p[i]*n+p[k]]);
		}
	}
	return(d);
 }

/*---------------------------------------------------------------------------------------------*/
/*      Idem, but the value of delta[i][j] is supposed to              */
/*    be known before the transposition of elements r and s     */
/*---------------------------------------------------------------------------------------------*/
int compute_delta_part(type_vector a, type_vector b, type_vector p, type_vector delta,
					int i, int j, int r, int s,int n){
	return ( delta[i*n+j]+(a[r*n+i]-a[r*n+j]+a[s*n+j]-a[s*n+i]) *
	(b[p[s]*n+p[i]]-b[p[s]*n+p[j]]+b[p[r]*n+p[j]]-b[p[r]*n+p[i]]) +
	(a[i*n+r]-a[j*n+r]+a[j*n+s]-a[i*n+s]) *
	(b[p[i]*n+p[s]]-b[p[j]*n+p[s]]+b[p[j]*n+p[r]]-b[p[i]*n+p[r]])
	);
}

void tabu_search_seq(int n,              /* problem size */
                 type_vector a,                   /* flows matrix */
                 type_vector b,                   /* distance matrix */
                 type_vector best_sol,         /* best solution found */
                 int *best_cost,                  /* cost of best solution */
                 int tabu_duration,             /* parameter 1 (< n^2/2) */
                 int aspiration,                    /* parameter 2 (> n^2/2)*/
                 int nr_iterations,		   /* number of iterations */
		 int BKS, int global_iter,
		 FILE* fresGTS,
		 int rank, int all, MPI_Status status){



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
	  //for (i = 0; i < n; i = i+1) delta[i] = (int*)calloc(n, sizeof(int));
	  tabu_list = (int**)calloc(n,sizeof(int*));
	  for (i = 0; i < n; i = i+1) tabu_list[i] = (int*)calloc(n, sizeof(int));

	  /************** current solution initialization ****************/
	  for (i = 0; i < n; i = i + 1) p[i] = best_sol[i];

	  /********** initialization of current solution value ***********/
	  /**************** and matrix of cost of moves  *****************/
	  current_cost = 0;
	  for (i = 0; i < n; i = i + 1) for (j = 0; j < n; j = j + 1){
		current_cost = current_cost + a[i*n+j] * b[p[i]*n+p[j]];
		if (i < j) {delta[i*n+j] = compute_delta(n, a, b, p, i, j);};
	   };

	   *best_cost = current_cost;

	   //file 2 GTS
	   /*cost_sol = current_cost;
	   if(rank == 0) fprintf(fresGTS,"%f ", 100*(cost_sol - BKS)/BKS);*/

	  /****************** tabu list initialization *******************/
	  for (i = 0; i < n; i = i + 1){
		for (j = 0; j < n; j = j+1) tabu_list[i][j] = -(n*i + j);
	  }

	  /******************** main tabu search loop ********************/
	for (current_iteration = 1; current_iteration <= nr_iterations /*&& *best_cost > BKS*/;
		current_iteration = current_iteration + 1){
		/** find best move (i_retained, j_retained) **/
		i_retained = infinite;       /* in case all moves are tabu */
		j_retained = infinite;
		min_delta = infinite;
		already_aspired = FALSE;

		for (i = 0; i < n-1; i = i + 1){
				for (j = i+1; j < n; j = j+1){
					autorized = (tabu_list[i][p[j]] < current_iteration) ||
					(tabu_list[j][p[i]] < current_iteration);

					aspired = (tabu_list[i][p[j]] < current_iteration-aspiration) ||
					(tabu_list[j][p[i]] < current_iteration-aspiration) ||
					(current_cost + delta[i*n+j] < *best_cost);

					if ((aspired && !already_aspired) ||    /* first move aspired */
					    (aspired && already_aspired &&     /* many move aspired  */
					    (delta[i*n+j] < min_delta)) ||         /* => take best one   */
					    (!aspired && !already_aspired &&  /* no move aspired yet*/
					    (delta[i*n+j] < min_delta) && autorized))
					{
						    i_retained = i; j_retained = j;
						    min_delta = delta[i*n+j];
						    if (aspired) {already_aspired = TRUE;};
					};
				};
		}

		if (i_retained == infinite) printf("All moves are tabu! \n");
		else{
			      /** transpose elements in pos. i_retained and j_retained **/
			      transpose(&p[i_retained], &p[j_retained]);
			      /* update solution value*/
			      current_cost = current_cost + delta[i_retained*n+j_retained];
			      /* forbid reverse move for a random number of iterations*/
			      tabu_list[i_retained][p[j_retained]] = current_iteration +
			      (int)(cube(rando())*tabu_duration);

			      tabu_list[j_retained][p[i_retained]] = current_iteration +
			      (int)(cube(rando())*tabu_duration);

			      /* best solution improved ?*/
			      if (current_cost < *best_cost){
				      *best_cost = current_cost;
				      for (k = 0; k < n; k = k+1) best_sol[k] = p[k];
			       };

			       /* update matrix of the move costs*/
			       for (i = 0; i < n-1; i = i+1){
					for (j = i+1; j < n; j = j+1){
						if (i != i_retained && i != j_retained && j != i_retained && j != j_retained){
							delta[i*n+j] = compute_delta_part(a, b, p, delta, i, j, i_retained,
														j_retained,n);
						}
						else{
							delta[i*n+j] = compute_delta(n, a, b, p, i,j);
						};
					}
				}

		};
	};

} /* End tabu*/

/***************************************End User Fonction******************************************/

void Meta_Init(int n,type_vector a,type_vector b, int rank, int all){

	//********
	//Activation
	GPU = FALSE;
	CPU = TRUE;

	if(GPU && rank == 0) printf("GPGPU optimizer activated\n");
	if(CPU && rank == 0) printf("CPU optimizer activated\n");

	//********
	//CPU allocation and initialization
	if (rank == 0) printf("GPU init Data...\n");

	//DISPLAY2
	TS_iterations = n*100; /* SIZE MULTIPLY BY 100*/
	max_global_iteration = 100;

	if(n <= 100) stop_condition = 3600;//1 h
	else stop_condition = 14400;//4 h

	cost_sol  = 0.0;

	individual = (int*)calloc(n, sizeof(int));
	best_individual = (int*)calloc(n, sizeof(int));
	tmp_individual = (int*)calloc(n, sizeof(int));

	best_fitness = infinite;

	/*L = 20;
	w = 0;*/
	step = 3;

	//Stangnancy = FALSE;

	/*K_perturbation param*/
	perturb    = n/5; /* 20% of the solution */

	//MASTER
	if(rank == 0){

		/*initialisation des pools*/
		int p;
		taille_pool = 1;
		pool_best_fitness = (int*)calloc(taille_pool, sizeof(int));

		/*pool_best_individuals = (int**)calloc(taille_pool, sizeof(int));
		for(p=0;p<taille_pool;p++) pool_best_individuals[p] = (int*)calloc(n, sizeof(int));*/
		pool_best_individuals = (int*)calloc(taille_pool*n, sizeof(int));

		for(p=0;p<taille_pool;p++){
			pool_best_fitness[p] = infinite;
		}

		/*for(int p1=0;p1<taille_pool;p1++){
			for(int p2=0;p2<n;p2++){
				pool_best_individuals[p1][p2] = -1;
			}
		}*/
		for(int p1=0;p1<taille_pool*n;p1++){
				pool_best_individuals[p1] = -1;
		}

		prc_remp = 0;

	}//END MASTER


}


void Meta_Optimize(int n,int BKS,type_vector a,type_vector b, FILE* fresGTS,
				int rank, int all, MPI_Status status){

	MPI_Barrier(MPI_COMM_WORLD); //all processes need to be ready

	start = clock();

	int i,k,position,star,p,p1,p2;
	int global_iter = 0;

	MPI_generate_random_solution(n, individual, rank);

	do{

		if(CPU)
		{
			tabu_search_seq(n, a, b,                  /* problem data         */
				 individual, &fitness,            /* tabu search results  */
				 8*n, n*n*5,                      /* parameters           */
				 TS_iterations,BKS, global_iter,  /* number of iterations */
				 fresGTS, 			  /* save files           */
				 rank, all, status);
		}


		/* Amelioration */
		if(fitness < best_fitness){
			best_fitness = fitness;
			for (k = 0; k < n; k = k+1) best_individual[k] = individual[k];
		}
		/*End Amelioration part */

		   /*Recherche d'un best pour les processes*/
		   int tmpBest = best_fitness;
		   for (k = 0; k < n; k = k+1) tmp_individual[k] = best_individual[k];

		   MPI_Barrier(MPI_COMM_WORLD);

		   if(rank != 0){
			//All processes send their values to process 0
			MPI_Send(&tmpBest, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
			MPI_Send(tmp_individual, n, MPI_INT, 0, 1, MPI_COMM_WORLD);
			//printf("process rank %d send to process 0\n",rank);
		   }

		   MPI_Barrier(MPI_COMM_WORLD);

		   //MASTER
		   if(rank == 0){
		   	   int comp;
			   INSERT = FALSE;
			   p=0;
			   while(p<taille_pool && INSERT == FALSE){
				//pour ejecter les solution existante dans le pool
				if(tmpBest == pool_best_fitness[p]) tmpBest = infinite;
			   	if(tmpBest < pool_best_fitness[p]){
					INSERT = TRUE;
					prc_remp++;
					if(p != taille_pool-1){
						//décalage de la solution vers la suivante
						pool_best_fitness[p+1] = pool_best_fitness[p];
						for(p2=0;p2<n;p2++){
							pool_best_individuals[((p+1)*n)+p2] = pool_best_individuals[(p*n)+p2];
						}
					}

					//Ajout de la solution dans le pool
					pool_best_fitness[p] = tmpBest;
					for(p2=0;p2<n;p2++){
						pool_best_individuals[(p*n)+p2] = tmp_individual[p2];
					}
				}
				p++;
			   }

			   for(comp = 1; comp < all; comp++){
				//receive values by process 0
				MPI_Recv(&tmpBest, 1, MPI_INT, comp, 0, MPI_COMM_WORLD,&status);
				MPI_Recv(tmp_individual, n, MPI_INT, comp, 1, MPI_COMM_WORLD,&status);
				//printf("process rank 0 receiv from process %d\n",comp);

			   	INSERT = FALSE;
				p=0;
			   	while(p<taille_pool && INSERT == FALSE){
					//pour ejecter les solution existante dans le pool
					if(tmpBest == pool_best_fitness[p]) tmpBest = infinite;
			   		if(tmpBest < pool_best_fitness[p]){
						INSERT = TRUE;
						prc_remp++;
						if(p != taille_pool-1){

							//si p=3 donc problème
							//printf("p=%d\n",p);

							//décalage de la solution vers la suivante
							pool_best_fitness[p+1] = pool_best_fitness[p];
							for(p2=0;p2<n;p2++){
								pool_best_individuals[((p+1)*n)+p2] = 									pool_best_individuals[(p*n)+p2];
							}
						}

						//debegage pour voir si on insert une infinite
						//if(tmpBest == infinite) printf("ERROR\n");

						//Ajout de la solution dans le pool
						pool_best_fitness[p] = tmpBest;
						for(p2=0;p2<n;p2++){
							//if(tmp_individual[p2]==-1) printf("ERR_IN\n");
							//detecter les -1 dans la solution transmise

							pool_best_individuals[(p*n)+p2]=
							tmp_individual[p2];
						}
					}
					p++;
				}
			   }
			//printf("le pool est à %f pourcent\n",(prc_remp*1.0/taille_pool)*100);

/*
			//Vérification du pool
			for(p=0;p<taille_pool;p++){
				printf("indice %d = %d\n",p,pool_best_fitness[p]);
			}
			for(p1=0;p1<taille_pool;p1++){
				for(p2=0;p2<n;p2++){
					printf("%d-",pool_best_individuals[(p1*n) + p2]);
				}
				printf("\n");
			}
*/

			int random_take;

			//choix du master pour lui même
			random_take = unif(0, taille_pool-1);

			//pour éviter les solution à -1 dans le pool
			while(pool_best_fitness[random_take] == infinite){
				//printf("random mauvais = %d\n",random_take);
				random_take = unif(0, taille_pool-1);
			}
			//printf("random bon = %d\n",random_take);

			/*best_fitness = pool_best_fitness[random_take];
			for (k = 0; k < n; k = k+1){
				best_individual[k] = pool_best_individuals[(random_take*n)+k];
			}*/

			fitness = pool_best_fitness[random_take];
			for (k = 0; k < n; k = k+1){
				individual[k] = pool_best_individuals[(random_take*n)+k];
			}


			//choix du master pour les autres
			for(comp = 1; comp < all; comp++){
				random_take = unif(0, taille_pool-1);
				//pour éviter les solution à -1 dans le pool
				while(pool_best_fitness[random_take] == infinite){
					random_take = unif(0, taille_pool-1);
				}
		   		tmpBest = pool_best_fitness[random_take];
		   		for (k = 0; k < n; k = k+1){
					tmp_individual[k] = pool_best_individuals[(random_take*n)+k];
					//printf("%d-",tmp_individual[k]);
				}
				//printf("\n");
				//process 0 send to all the other process a best from the pool
				MPI_Send(&tmpBest, 1, MPI_INT, comp, 3, MPI_COMM_WORLD);
				MPI_Send(tmp_individual, n, MPI_INT, comp, 4, MPI_COMM_WORLD);
				//printf("process rank 0 send to process %d\n",comp);
			}

		   }//END MASTER

		   MPI_Barrier(MPI_COMM_WORLD);

		   if(rank != 0){
		   	//receive values from process 0
		   	MPI_Recv(&tmpBest, 1, MPI_INT, 0, 3, MPI_COMM_WORLD,&status);
			MPI_Recv(tmp_individual, n, MPI_INT, 0, 4, MPI_COMM_WORLD,&status);
			//printf("process rank %d receiv from process 0\n",rank);

			fitness = tmpBest;
			for (k = 0; k < n; k = k+1){
				individual[k] = tmp_individual[k];
			}

		   }


		   MPI_Barrier(MPI_COMM_WORLD);


		   //printf(" Best fitness in iteration %d: %d \n", global_iter, best_fitness);
		   /*printf(" Best deviation fitness in G_iteration %d: %f \n",(global_iter+1),
		   100*(cost_sol - BKS)/BKS);*/

		   /* re-construction of the solution using the Golver method */  /* Rule 1 */
		   /*position = 0;
		   for (star = step; star >= 1; star--){
		   	for (k = (star-1); k < n; k = k+step){
				individual[position] = best_individual[k];
				position++;
			}
		   }
		   if(step == (n-1)) step=3;//reinitialization
		   else step++;*/

		   /* K_premier_perturbation */
		   /*for (i = 0; i < perturb; i++) transpose(&individual[i],
						&individual[MPI_unif(i, n-1,rank)]);*/

		   /* K_random_perturbation */
		   for (i = 0; i < perturb; i++) transpose(&individual[MPI_unif(0, n-1,rank)],
						&individual[MPI_unif(0, n-1,rank)]);

		   MPI_Barrier(MPI_COMM_WORLD);
       		   iter_end = clock();

		   //file 2 GTS
		   if(rank == 0) best_cost_sol = pool_best_fitness[0];//le best ever
		   //cost_sol = fitness;

		   if(rank == 0) fprintf(fresGTS,"%f %.3lf %d %d %f\n", 100*(best_cost_sol - BKS)/BKS, ((double)(iter_end-start)/CLOCKS_PER_SEC), (global_iter+1), TS_iterations, (prc_remp*1.0/taille_pool)*100);

		   global_iter++;

		   //MPI_Barrier(MPI_COMM_WORLD);

		   //printf("pass tabu rank %d et g_iter = %d\n",rank,global_iter);

	}while( global_iter < max_global_iteration /*&& best_fitness > BKS*/ );
	//}while( ((double)(iter_end-start)/CLOCKS_PER_SEC) < stop_condition && best_fitness > BKS );

	MPI_Barrier(MPI_COMM_WORLD);

	end = clock();

	cost_sol = best_fitness;

}

// Display and write results
void Meta_Display_results(FILE* fres, int n, int BKS, int rank, int all, int run,
					MPI_Status status){

	   //filtring results
	   double tmpBest = cost_sol;
	   double Time = (double)(end-start)/CLOCKS_PER_SEC;
	   double Timetmp = Time;
	   int result = 0;
	   int j;

	   MPI_Barrier(MPI_COMM_WORLD);

	   if(rank != 0){
		//All processes send their values to process 0
		MPI_Send(&tmpBest, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&Timetmp, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		//printf("process rank %d send to process 0\n",rank);
	   }

	   MPI_Barrier(MPI_COMM_WORLD);

	   if(rank == 0){
		for(int comp = 1; comp < all; comp++){
			//receive values by process 0
			MPI_Recv(&tmpBest, 1, MPI_DOUBLE, comp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&Timetmp, 1, MPI_DOUBLE, comp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			//printf("process rank %d receiv from process 0\n",comp);
			if(tmpBest < cost_sol){
				cost_sol = tmpBest;
				result = comp;
				Time = Timetmp;
			}//Update
		}
	   }

	   MPI_Barrier(MPI_COMM_WORLD);

	   //printf("process : %d [avant brodcast pour run %d avec results = %d]\n",rank,run,result);

	   //brodcast the value "result" for all processes from process 0
	   MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
	   MPI_Barrier(MPI_COMM_WORLD);
	   //printf("process : %d [apres brodcast pour run %d avec results = %d]\n",rank,run,result);

	   if(rank == result){//Only the best process
	       printf(" Solution found by Distr-ITS: %f (process %d trial %d) \n",  100*(  cost_sol - BKS)/BKS, rank, run);
	       for (j = 0; j < n; j++) printf("%d ", (best_individual[j]+1));// +1 pour l'affichage
	       printf("\n");
	       printf(" Execution time = [%.3lf] second (process %d trial %d) \n", (double)(end-start)/CLOCKS_PER_SEC, rank, run);


	       fprintf(fres,"%.3lf\t",(double)(end-start)/CLOCKS_PER_SEC);
	       fprintf(fres,"%f\n", 100*( cost_sol  - BKS)/BKS );
	    }

	    //Update somme_sol for the final computation with process 0 which contains the best value
	    if(rank == 0){
	      somme_sol += cost_sol;
	      avg_time += Time;
	    }

	    MPI_Barrier(MPI_COMM_WORLD);

}


// Frees resources
void Meta_Free(int rank){
}

// Display and write global results
void Meta_Display_trials_results(FILE* fres, int max_runs, int BKS, int rank, int all,
						MPI_Status status){
	if(rank == 0){
		printf("****Recap****\n");

		printf("Average cost: %f, average dev: %f\n", somme_sol/max_runs,
		100*(somme_sol/max_runs - BKS)/BKS);

		printf("Average time: %.3lf\n", avg_time/max_runs);

		fprintf(fres,"***AVG*** \n");
		fprintf(fres,"%.3lf\t",avg_time/max_runs);
		fprintf(fres,"%f\n",100*(somme_sol/max_runs - BKS)/BKS);
		fprintf(fres,"%f\n",somme_sol/max_runs);
	}

	somme_sol = 0.0;
	avg_time  = 0.0;

}

#endif
