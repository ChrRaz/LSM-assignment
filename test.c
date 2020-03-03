#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <mpi.h>

// Cartesian to linear
#define to_lin(r, c, w) r * w + c

/*
  Convert a global index into a local index (irrespective of which rank has the index)
 */
int global2local(int index_global, int nblock, int size) {
  int cur_block = index_global / (size * nblock);
  return nblock * cur_block + index_global % nblock;
}

/*
  Calculate which rank has the global index residing in its local index
 */
int global2rank(int index_global, int nblock, int size) {
  return (index_global / nblock) % size;
}

/*
  Calculate total number of elements on the local rank
*/
int local_size(int N, int nblock, int rank, int size) {
  // Total number of full blocks
  int full_blocks = N / nblock;

  // Minimum number of elements on each rank
  int min_elem = (full_blocks / size) * nblock;
  
  // Overlapping blocks
  int overlap = full_blocks % size;

  if ( rank < overlap ) {
    return min_elem + nblock;
  } else if ( rank == overlap ) {
    return min_elem + N % nblock;
  } else {
    return min_elem;
  }
}

int global2rank2d(int i_global, int j_global, int nblock, int pr, int pc) {
  return to_lin(global2rank(i_global, nblock, pr),
                global2rank(j_global, nblock, pc), pc);
}

int global2local2d(int N, int i_global, int j_global, int nblock, int pr, int pc) {
  return to_lin(global2local(i_global, nblock, pr),
                global2local(j_global, nblock, pc),
                local_size(N, nblock, global2rank(j_global, nblock, pc), pc));
}

double * redistribute(int N, double *A_in, int nblock_in,
		      int nblock_out,
		      int rank, int size) {
  
  // First we calculate the new (local) matrix size
  int N_local = local_size(N, nblock_out, rank, size);

  if ( rank == 0 ) {
    printf("\nStart redistribute matrix from nb=%d to nb=%d\n", nblock_in, nblock_out);
    fflush(stdout);
  }

#ifdef DEBUG
  MPI_Barrier(MPI_COMM_WORLD);
  printf(" [%2d] local rows %d\n", rank, N_local);
  fflush(stdout);
#endif

  // Allocate memory for the new distributed matrix
  double *A_out = (double *) malloc(N * N_local * sizeof(double));

  // align ranks
  MPI_Barrier(MPI_COMM_WORLD);

  double t0 = MPI_Wtime();

  // Now all processors are ready for send/recv data
  for ( int i = 0 ; i < N ; i++ ) {

    // Figure out which ranks has and which should have the given global index
    int rank_in = global2rank(i, nblock_in, size);
    int rank_out = global2rank(i, nblock_out, size);

#ifdef DEBUG
    if ( rank == 0 ) {
      printf("row %3d  %2d -> %2d\n", i, rank_in, rank_out);
    }
#endif
    
    if ( rank_in == rank ) {
      // currently hosting rank has this row
      int idx_in = global2local(i, nblock_in, size);
      if ( rank_out == rank ) {
	
	// this rank also has the receive
	int idx_out = global2local(i, nblock_out, size);
#ifdef DEBUG
	// Copy data
	printf(" [%2d] COPY %3d (%3d)\n", rank, i, idx_in);
	fflush(stdout);
#endif
	      memcpy(&A_out[N*idx_out], &A_in[N*idx_in], N * sizeof(double));
	
      } else {
#ifdef DEBUG
	printf(" [%2d] SEND %3d (%3d) to %2d\n", rank, i, idx_in, rank_out);
	fflush(stdout);
#endif
      	MPI_Send(&A_in[N*idx_in], N, MPI_DOUBLE, rank_out, i, MPI_COMM_WORLD);
      }
      
    } else if ( rank_out == rank ) {
      
      // We already know that we *have* to post a recieve
      int idx_out = global2local(i, nblock_out, size);
#ifdef DEBUG
      printf(" [%2d] RECV %3d (%3d) from %2d\n", rank, i, idx_out, rank_in);
      fflush(stdout);
#endif
      MPI_Recv(&A_out[N*idx_out], N, MPI_DOUBLE, rank_in, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // Final timing
  double t1 = MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);

  if ( rank == 0 ) {
    printf("Done redistributing matrix from nb=%d to nb=%d\n", nblock_in, nblock_out);
    fflush(stdout);
  }

  // Print timings, min/max
  double t = t1 - t0;
  MPI_Reduce(&t, &t0, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t, &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if ( rank == 0 ) {
    printf("Time min/max  %12.8f / %12.8f ms\n", 1000 * t0, 1000 * t1);
    fflush(stdout);
  }

  return A_out;
}

int main(int argc, char const *argv[])
{
    int N = 13, pr = 3, pc = 2, nb = 3;

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            // printf(" %d", global2rank(i, j, nb, pr, pc));
            printf(" %d", to_lin(global2rank(i, nb, pr), global2rank(j, nb, pc), pc));
        }

        printf(" | ");

        for(int j = 0; j < N; j++) {
            printf(" %d", global2rank2d(i, j, nb, pr, pc));
        }
        puts("");
    }

    puts("");

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            // printf(" %2d", global2local(N, global2rank(i, j, nb, pr, pc), i, j, nb, pr, pc));
            printf(" %2d", to_lin(global2local(i, nb, pr), global2local(j, nb, pc), local_size(N, nb, global2rank(j, nb, pc), pc)));
        }

        printf(" | ");

        for(int j = 0; j < N; j++) {
            printf(" %2d", global2local2d(N, i, j, nb, pr, pc));
        }
        puts("");
    }

    puts("");

    for(int i = 0; i < pr*pc; i++)
        printf(" %dx%d\n", local_size(N, nb, i/pc, pr), local_size(N, nb, i%pc, pc));
      
    
    return 0;
}
