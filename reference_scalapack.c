#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include "ref_blacs.h"

#define MAX(a,b) (a > b ? a : b)

#define FLUSH_STD() {fflush(stdout);fflush(stderr);MPI_Barrier(MPI_COMM_WORLD);}


// Initialize a new BLACS context
// Pr
//   number of row distributions
// Pc
//   number of column distributions
// context
//   returned BLACS-context
// pr
//   local process row ID
// pc
//   local process column ID
void initialize_blacs_context(int Pr, int Pc, int *context, int *pr, int *pc) {
  int i_one = 1, i_negone = -1, i_zero = 0;
  
  // A zero(negative) argument *retrieves* the default context
  blacs_get_(&i_negone, &i_zero, context);
  
  // Initialize a natural ROW indexing order with `Pr` rows and `Pc` columns
  blacs_gridinit_(context, "R", &Pr, &Pc);
  
  // Figure out the local processors <row>,<column> index in the 2D distribution.
  blacs_gridinfo_(context, &Pr, &Pc, pr, pc);
}


// Count number of local elements for a given process ID on a given distribution direction
// with a given block-size
// N
//     matrix size
// NB
//     block size
// proc_id
//     processor ID along the distribution with `procs` processors
// procs
//     number of processors used for the distribution
int local_elements(int *N, int *NB, int *proc_id, int *procs) {
  int i_zero = 0;
  return numroc_(N, NB, proc_id, &i_zero, procs);
}


// Create a matrix descriptor for a given matrix with a specific block-size
// desc
//     array of length 9 with ScaLAPACK descriptions for the matrix
// N
//     matrix dimension
// NB
//     block size along both directions
// context
//     the BLACS context in which the matrix is distributed
// lld
//     local leading dimension of the matrix size (typically always equal to the number
//     of rows)
// info
//     ScaLAPACK error information
void matrix_descriptor(int *desc, int *N, int *NB, int *context, int *lld, int *info) {
  int i_zero = 0;
  descinit_(desc, N, N, NB, NB, &i_zero, &i_zero, context, lld, info);
}


// Initialize an array of doubles to a specific value.
// A
//    array to initialize
// size
//    total number of elements in the array
// val
//    value assigned to every element
void initialize_array_value(double *A, int size, double val) {
  int i;
  for ( i = 0 ; i < size ; i++ )
    A[i] = val;
}
  

// Initialize a square matrix with the identity matrix
// A
//    matrix to initialize
// N
//    matrix dimension
void initialize_identity(double *A, int N) {
  int i, j;
  double *Arow;

  for ( i = 0 ; i < N; i++) {
    // Retrieve local row
    Arow = &A[i*N];
    for ( j = 0 ; j < N; j++) {
      Arow[j] = 0.;
    }
    // Assign diagonal
    Arow[i] = 1.;
  }
}


// Print distributed matrix with local indexing
// Aloc
//    distributed matrix
// N
//    matrix size
// NB
//    matrix block size
// Pr
//    number of processors per row
// Pc
//    number of processors per column
// pr
//    local row processors
// pc
//    local column processors
void print_distributed_matrix(double *Aloc, int *N, int *NB, int *Pr, int *Pc, int *pr, int *pc) {
  int i, j, I, J;
  int i_zero = 0;

  int lld_row = local_elements(N, NB, pr, Pr);
  int lld_column = local_elements(N, NB, pc, Pc);
  FLUSH_STD();

  // Retrieve the local number of elements along the row
  for ( j = 1 ; j <= lld_column ; j++ ) {
    J = indxl2g_(&j, NB, pc, &i_zero, Pc);
    for ( i = 1 ; i <= lld_row ; i++ ) {
      I = indxl2g_(&i, NB, pr, &i_zero, Pr);
      printf(" pc/pr[%3d, %3d]  global[%3d, %3d] local[%3d, %3d] = %f\n", *pc, *pr, J-1, I-1, j-1, i-1, Aloc[(j-1)*lld_row+i-1]);
      //printf(" pc/pr[%3d, %3d]   global 2D C-index[%3d, %3d]  local 2D C-index[%3d, %3d]  local linear C-index[%3d]\n", *pc, *pr, J-1, I-1, j-1, i-1, (j-1)*lld_row+i-1);
    }
  }
  
  FLUSH_STD();
}


int main(int argc, char* argv[]) {
  // Constants!
  int i_one = 1, i_negone = -1, i_zero = 0;
  int i, j;

  // Initialize MPI
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if ( rank == 0 )
    printf("Running on %d ranks\n", size);

  // Read in the matrix size and Np X Nc grid layout
  // Pc == processors on the column
  // Pr == processors on the row
  int Pc, Pr;
  // Full matrix size
  int N, NB;

  if ( argc != 5 ) {
    printf("Requires 3 inputs:\n");
    printf("   N  : matrix size\n");
    printf("   NB : block-size\n");
    printf("   Pc : number of processors in the 2D column layout\n");
    printf("   Pr : number of processors in the 2D row layout\n");

    // Quick exist since the arguments are inconsistent
    MPI_Finalize();
    return 1;
  }

  // Read arguments
  N = atoi(argv[1]);
  NB = atoi(argv[2]);
  Pc = atoi(argv[3]);
  Pr = atoi(argv[4]);

  // Check that Pc * Pr == size
  if ( Pc * Pr != size ) {
    printf("Please correct input!\n");
    printf("   Pc * Pr != size\n");
    printf("   %d * %d != %d\n", Pc, Pr, size);

    // Quick exist since the arguments are inconsistent
    MPI_Finalize();
    return 1;
  }

  // ScaLAPACK/BLACS typically returns an info parameter to signal how it behaved!
  int info;

  // Create both a 1D descriptor and the distribution descriptor!
  int blacs_1d_context;
  // pc_1d == local processor column ID
  // pr_1d == local processor row ID
  int pr_1d, pc_1d;
  initialize_blacs_context(size, 1, &blacs_1d_context, &pr_1d, &pc_1d);
  printf("[%d] blacs_1d_context %d\n", rank, blacs_1d_context);
  FLUSH_STD();
  
  // Count number of local elements (on rows!)
  int lld_1d = local_elements(&N, &N, &pr_1d, &size);
  lld_1d = MAX(lld_1d, 1); // require at least 1 element since we are dealing with a "non" distributed matrix
  
  // Create distribution descriptor
  printf("[%d] local_elements 1d (%d, %d) lld=%d\n", rank, pr_1d, pc_1d, lld_1d);
  FLUSH_STD();

  int desc_1d[9];
  matrix_descriptor(desc_1d, &N, &N, &blacs_1d_context, &lld_1d, &info);

  
  
  // Retrieve a new context for the 2D distribution
  int blacs_context;
  // pc == local processor column ID
  // pr == local processor row ID
  int pc, pr;
  initialize_blacs_context(Pr, Pc, &blacs_context, &pr, &pc);
  printf("[%d] blacs_context %d\n", rank, blacs_context);
  FLUSH_STD();


  // Calculate how much data we need to allocate on a each processor.
  int Nr, Nc;
  Nr = local_elements(&N, &NB, &pr, &Pr);
  Nc = local_elements(&N, &NB, &pc, &Pc);

  // Leading dimension of the local array (row distributed == Nr)
  int lld = Nr;
  printf("[%d] local_elements (%d, %d) Nr=%d Nc=%d lld=%d\n", rank, pr, pc, Nr, Nc, lld);
  FLUSH_STD();

  int desc_distr[9];
  matrix_descriptor(desc_distr, &N, &NB, &blacs_context, &lld, &info);
  
  printf("[%d] Initialize matrix...\n", rank);
  FLUSH_STD();

  
  // Allocate matrices
  double *A, *A_distr;
  if ( pr_1d == 0 && pc_1d == 0 ) {
    A = (double*) malloc(N * N * sizeof(double));
    initialize_identity(A, N);
  } else {
    A = NULL;
  }

  // Allocate distributed matrix
  A_distr = (double*) malloc(Nr * Nc * sizeof(double));

  // Initialize with 2. (then we can check...)
  initialize_array_value(A_distr, Nr * Nc, 2.);

  printf("[%d] Distribute matrix...\n", rank);
  FLUSH_STD();

  // Ensure timings! Yes, FLUSH_STD also has it... but
  MPI_Barrier(MPI_COMM_WORLD);

  // Do actual distribution, and time it!
  double t0 = MPI_Wtime();
  pdgemr2d_(&N, &N, A, &i_one, &i_one, desc_1d, A_distr, &i_one, &i_one, desc_distr, &blacs_context);
  double t1 = MPI_Wtime();

  
  printf("[%d] Time of (re)-distribution: %f\n", rank, t1 - t0);

  print_distributed_matrix(A_distr, &N, &NB, &Pr, &Pc, &pr, &pc);

  // Clean-up memory
  if ( A != NULL ) free(A);
  free(A_distr);

  // End of ScaLAPACK part. Exit process grid.
  blacs_gridexit_(&blacs_context);
  blacs_gridexit_(&blacs_1d_context);

  MPI_Finalize();

  return 0;
}


