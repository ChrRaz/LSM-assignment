#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

#include <mpi.h>

// Cartesian to linear
#define to_lin(r, c, w) r * w + c

#define MIN(a,b) ((a) < (b) ? (a) : (b))

#ifndef redis
#define redis redistribute3
#endif

//#define rprintf(format, ...) printf("[%d] " format, rank, __VA_ARGS__)

/*
  Convert a global index into a local index (irrespective of which rank has the
  index)
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

  if (rank < overlap) {
    return min_elem + nblock;
  } else if (rank == overlap) {
    return min_elem + N % nblock;
  } else {
    return min_elem;
  }
}

int global2rank2d(int i_global, int j_global, int nblock, int pr, int pc) {
  return to_lin(global2rank(i_global, nblock, pr),
                global2rank(j_global, nblock, pc), pc);
}

int global2local2d(int N, int i_global, int j_global, int nblock, int pr,
                   int pc) {
  return to_lin(global2local(i_global, nblock, pr),
                global2local(j_global, nblock, pc),
                local_size(N, nblock, global2rank(j_global, nblock, pc), pc));
}

double *redistribute1(int N, double *A_in, int nblock_in, int nblock_out,
                      int rank, int pr, int pc) {

  // Count the number of messages sent
  int ncomms = 0;

  // First we calculate the new (local) matrix size
  int rows_local = local_size(N, nblock_out, rank/pc, pr);
  int cols_local = local_size(N, nblock_out, rank%pc, pc);

  if (rank == 0) {
    printf("\nStart redistribute matrix from nb=%d to nb=%d\n", nblock_in,
           nblock_out);
    fflush(stdout);
  }

#ifdef DEBUG
  MPI_Barrier(MPI_COMM_WORLD);
  printf(" [%2d] local rows %d\n", rank, rows_local);
  printf(" [%2d] local cols %d\n", rank, cols_local);
  fflush(stdout);
#endif

  // Allocate memory for the new distributed matrix
  double *A_out = (double *)malloc(rows_local * cols_local * sizeof(double));

  // align ranks
  MPI_Barrier(MPI_COMM_WORLD);

  double t0 = MPI_Wtime();

  // Now all processors are ready for send/recv data
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      // Figure out which ranks has and which should have the given global index
      int rank_in = global2rank2d(i, j, nblock_in, pr, pc);
      int rank_out = global2rank2d(i, j, nblock_out, pr, pc);

#ifdef DEBUG
      if (rank == 0) {
        printf("cell (%3d,%3d)  %2d -> %2d\n", i, j, rank_in, rank_out);
      }
#endif

      if (rank_in == rank) {
        // currently hosting rank has this row
        int idx_in = global2local2d(N, i, j, nblock_in, pr, pc);
        if (rank_out == rank) {
          // this rank also has the receive
          int idx_out = global2local2d(N, i, j, nblock_out, pr, pc);

#ifdef DEBUG
          // Copy data
          printf(" [%2d] COPY %3d,%3d (%3d)\n", rank, i, j, idx_in);
          fflush(stdout);
#endif

          // memcpy(&A_out[N * idx_out], &A_in[N * idx_in], sizeof(double));
          A_out[idx_out] = A_in[idx_in];
        } else {

#ifdef DEBUG
          printf(" [%2d] SEND %3d,%3d (%3d) to %2d\n", rank, i, j, idx_in,
                 rank_out);
          fflush(stdout);
#endif

          MPI_Send(&A_in[idx_in], 1, MPI_DOUBLE, rank_out, 0, MPI_COMM_WORLD);
        }
      } else if (rank_out == rank) {
        // We already know that we *have* to post a recieve
        int idx_out = global2local2d(N, i, j, nblock_out, pr, pc);

#ifdef DEBUG
        printf(" [%2d] RECV %3d,%3d (%3d) from %2d\n", rank, i, j, idx_out, rank_in);
        fflush(stdout);
#endif

        MPI_Recv(&A_out[idx_out], 1, MPI_DOUBLE, rank_in, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }

#ifdef DEBUGSLEEP
      usleep(250000);
      MPI_Barrier(MPI_COMM_WORLD);
#endif

      ncomms++;
    }
  }

  // Final timing
  double t1 = MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    printf("Done redistributing matrix from nb=%d to nb=%d\n", nblock_in,
           nblock_out);
    fflush(stdout);
  }

  // Print timings, min/max
  double t = t1 - t0;
  MPI_Reduce(&t, &t0, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t, &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("#transactions: %d\n", ncomms);
    printf("Time min/max  %12.8f / %12.8f ms\n", 1000 * t0, 1000 * t1);
    fflush(stdout);
  }

  return A_out;
}

double *redistribute2(int N, double *A_in, int nblock_in, int nblock_out,
                      int rank, int pr, int pc) {

  // Count the number of messages sent
  int ncomms = 0;

  int rows_local_old = local_size(N, nblock_in, rank/pc, pr);
  int cols_local_old = local_size(N, nblock_in, rank%pc, pc);

  // First we calculate the new (local) matrix size
  int rows_local = local_size(N, nblock_out, rank/pc, pr);
  int cols_local = local_size(N, nblock_out, rank%pc, pc);

  if (rank == 0) {
    printf("\nStart redistribute matrix from nb=%d to nb=%d\n", nblock_in,
           nblock_out);
    fflush(stdout);
  }

#ifdef DEBUG
  MPI_Barrier(MPI_COMM_WORLD);
  printf(" [%2d] local rows %d\n", rank, rows_local);
  printf(" [%2d] local cols %d\n", rank, cols_local);
  fflush(stdout);
#endif

  // Allocate memory for the new distributed matrix
  double *A_out = (double *)malloc(rows_local * cols_local * sizeof(double));

  // align ranks
  MPI_Barrier(MPI_COMM_WORLD);

  double t0 = MPI_Wtime();

  // Now all processors are ready for send/recv data
  for (int i = 0; i < N;) {
    for (int j = 0; j < N;) {
      // Figure out which ranks has and which should have the given global index
      int rank_in = global2rank2d(i, j, nblock_in, pr, pc);
      int rank_out = global2rank2d(i, j, nblock_out, pr, pc);

      // Figure out how wide the current block is
      int next_div = MIN((j / nblock_in + 1) * nblock_in, (j / nblock_out + 1) * nblock_out);
      int width = MIN(N - j, next_div - j);

#ifdef DEBUG
      if (rank == 0) {
        printf("cell (%3d,%3d)  %2d -> %2d\n", i, j, rank_in, rank_out);
      }
#endif

      if (rank_in == rank) {
        // currently hosting rank has this row
        int idx_in = global2local2d(N, i, j, nblock_in, pr, pc);
        if (rank_out == rank) {
          // this rank also has the receive
          int idx_out = global2local2d(N, i, j, nblock_out, pr, pc);

#ifdef DEBUG
          // Copy data
          printf(" [%2d] COPY %3d,%3d (%3d)\n", rank, i, j, idx_in);
          fflush(stdout);
#endif

          assert(idx_in + width <= cols_local_old * rows_local_old);
          assert(idx_out + width <= cols_local * rows_local);

          memcpy(&A_out[idx_out], &A_in[idx_in], width * sizeof(double));
        } else {

#ifdef DEBUG
          printf(" [%2d] SEND %3d,%3d (%3d) to %2d\n", rank, i, j, idx_in,
                 rank_out);
          fflush(stdout);
#endif

          assert(idx_in + width <= cols_local_old * rows_local_old);
          MPI_Send(&A_in[idx_in], width, MPI_DOUBLE, rank_out, 0, MPI_COMM_WORLD);
        }
      } else if (rank_out == rank) {
        // We already know that we *have* to post a recieve
        int idx_out = global2local2d(N, i, j, nblock_out, pr, pc);

#ifdef DEBUG
        printf(" [%2d] RECV %3d,%3d (%3d) from %2d\n", rank, i, j, idx_out, rank_in);
        fflush(stdout);
#endif

        assert(idx_out + width <= cols_local * rows_local);
        MPI_Recv(&A_out[idx_out], width, MPI_DOUBLE, rank_in, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }

#ifdef DEBUGSLEEP
      usleep(250000);
      MPI_Barrier(MPI_COMM_WORLD);
#endif

      j += width;
      ncomms++;
    }
    i++;
  }

  // Final timing
  double t1 = MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    printf("Done redistributing matrix from nb=%d to nb=%d\n", nblock_in,
           nblock_out);
    fflush(stdout);
  }

  // Print timings, min/max
  double t = t1 - t0;
  MPI_Reduce(&t, &t0, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t, &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("#transactions: %d\n", ncomms);    
    printf("Time min/max  %12.8f / %12.8f ms\n", 1000 * t0, 1000 * t1);
    fflush(stdout);
  }

  return A_out;
}

double *redistribute3(int N, double *A_in, int nblock_in, int nblock_out,
                      int rank, int pr, int pc) {

  // Count the number of messages sent
  int ncomms = 0;

  int rows_local_old = local_size(N, nblock_in, rank/pc, pr);
  int cols_local_old = local_size(N, nblock_in, rank%pc, pc);

  // First we calculate the new (local) matrix size
  int rows_local = local_size(N, nblock_out, rank/pc, pr);
  int cols_local = local_size(N, nblock_out, rank%pc, pc);

  if (rank == 0) {
    printf("\nStart redistribute matrix from nb=%d to nb=%d\n", nblock_in,
           nblock_out);
    fflush(stdout);
  }

#ifdef DEBUG
  MPI_Barrier(MPI_COMM_WORLD);
  printf(" [%2d] local rows %d\n", rank, rows_local);
  printf(" [%2d] local cols %d\n", rank, cols_local);
  fflush(stdout);
#endif

  // Allocate memory for the new distributed matrix
  double *A_out = (double *)malloc(rows_local * cols_local * sizeof(double));

  MPI_Request *reqs = malloc(sizeof(MPI_Request) * (cols_local_old + cols_local));
  MPI_Status *stats = malloc(sizeof(MPI_Status) * (cols_local_old + cols_local));

  // align ranks
  MPI_Barrier(MPI_COMM_WORLD);

  double t0 = MPI_Wtime();

  // Now all processors are ready for send/recv data
  for (int i = 0; i < N;) {
    int req_count = 0;
    for (int j = 0; j < N;) {
      // Figure out which ranks has and which should have the given global index
      int rank_in = global2rank2d(i, j, nblock_in, pr, pc);
      int rank_out = global2rank2d(i, j, nblock_out, pr, pc);

      // Figure out how wide the current block is
      int next_div = MIN((j / nblock_in + 1) * nblock_in, (j / nblock_out + 1) * nblock_out);
      int width = MIN(N - j, next_div - j);      

#ifdef DEBUG
      if (rank == 0) {
        printf("cell (%3d,%3d)  %2d -> %2d\n", i, j, rank_in, rank_out);
      }
#endif

      if (rank_in == rank) {
        // currently hosting rank has this row
        int idx_in = global2local2d(N, i, j, nblock_in, pr, pc);
        if (rank_out == rank) {
          // this rank also has the receive
          int idx_out = global2local2d(N, i, j, nblock_out, pr, pc);

#ifdef DEBUG
          // Copy data
          printf(" [%2d] COPY %3d,%3d (%3d)\n", rank, i, j, idx_in);
          fflush(stdout);
#endif

          assert(idx_in + width <= cols_local_old * rows_local_old);
          assert(idx_out + width <= cols_local * rows_local);
          
          memcpy(&A_out[idx_out], &A_in[idx_in], width * sizeof(double));
        } else {

#ifdef DEBUG
          printf(" [%2d] SEND %3d,%3d (%3d) to %2d\n", rank, i, j, idx_in,
                 rank_out);
          fflush(stdout);
#endif

          assert(idx_in + width <= cols_local_old * rows_local_old);
          MPI_Isend(&A_in[idx_in], width, MPI_DOUBLE, rank_out, 0, MPI_COMM_WORLD,&reqs[req_count]);
          req_count++;
        }
      } else if (rank_out == rank) {
        // We already know that we *have* to post a recieve
        int idx_out = global2local2d(N, i, j, nblock_out, pr, pc);

#ifdef DEBUG
        printf(" [%2d] RECV %3d,%3d (%3d) from %2d\n", rank, i, j, idx_out, rank_in);
        fflush(stdout);
#endif

        assert(idx_out + width <= cols_local * rows_local);
        MPI_Irecv(&A_out[idx_out], width, MPI_DOUBLE, rank_in, 0, MPI_COMM_WORLD,&reqs[req_count]);
        req_count++;
      }

#ifdef DEBUGSLEEP
      usleep(250000);
      MPI_Barrier(MPI_COMM_WORLD);
#endif

      j += width;
      ncomms++;
    }
    MPI_Waitall(req_count,reqs,stats);
    i++;
  }

  free(reqs);
  free(stats);

  // Final timing
  double t1 = MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    printf("Done redistributing matrix from nb=%d to nb=%d\n", nblock_in,
           nblock_out);
    fflush(stdout);
  }

  // Print timings, min/max
  double t = t1 - t0;
  MPI_Reduce(&t, &t0, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t, &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("#transactions: %d\n", ncomms);    
    printf("Time min/max  %12.8f / %12.8f ms\n", 1000 * t0, 1000 * t1);
    fflush(stdout);
  }

  return A_out;
}

int main(int argc, char *argv[]) {
  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Query size and rank
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // No buffering of stdout
  setbuf(stdout, NULL);

  if (argc < 6) {
    printf("Requires 5 inputs:\n");
    printf("   N   : matrix size\n");
    printf("   Pc  : number of processors in the 2D column layout\n");
    printf("   Pr  : number of processors in the 2D row layout\n");
    printf("   NB1 : block-size 1\n");
    printf("   NB2 : block-size 2\n");

    // Quick exist since the arguments are inconsistent
    MPI_Finalize();
    return 1;
  }

  // Read arguments
  int N = atoi(argv[1]);
  int pc = atoi(argv[2]);
  int pr = atoi(argv[3]);
  int nblock_1 = atoi(argv[4]);
  int nblock_2 = atoi(argv[5]);

  // Check that Pc * Pr == size
  if (pr * pc != size) {
    printf("Please correct input!\n");
    printf("   Pc * Pr != size\n");
    printf("   %d * %d != %d\n", pr, pc, size);

    // Quick exist since the arguments are inconsistent
    MPI_Finalize();
    return 1;
  }

  if (rank == 0) {
    printf("Arguments:\n");
    printf("  N = %d\n", N);
    printf("  nblock_1 = %d\n", nblock_1);
    printf("  nblock_2 = %d\n", nblock_2);
    printf("MPI processors: %d\n", size);
  }

  // Initialize a matrix with *random* data
  double *A_dense = NULL;
  if (rank == 0) {
    // allocate and initialize
    A_dense = (double *)malloc(N * N * sizeof(double));

    // Consecutive numbers (linear matrix index)
    for (int i = 0; i < N * N; i++) {
      A_dense[i] = i;
    }

    printf("Done initializing matrix on root node\n");
    fflush(stdout);
  }

  // Distribute the first (dense) matrix
  double *A_1 = redis(N, A_dense, N, nblock_1, rank, pr, pc);

  // Redistribute to the next level
  double *A_2 = redis(N, A_1, nblock_1, nblock_2, rank, pr, pc);

  // if (rank == 0) {
  //   for ( int i = 0 ; i < N * N ; i++ ) {
  //     printf(" %f%s", A_dense[i], i % N == N - 1 ? "\n" : "");
  //   }
  //   fflush(stdout);
  // }

  // Clean-up memory
  free(A_1);

  // Redistribute to the local one again to check we have done it correctly
  double *A_final = redis(N, A_2, nblock_2, N, rank, pr, pc);
  // Clean-up memory
  free(A_2);

  if (rank == 0) {
    for (int i = 0; i < N * N; i++) {
      if (fabs(A_dense[i] - A_final[i]) > 0.1) {
        printf("Error on index %3d   %5.1f %5.1f\n", i, A_dense[i], A_final[i]);
      }
    }
  }

  // Clean-up memory!
  if (A_final != NULL)
    free(A_final);
  if (A_dense != NULL)
    free(A_dense);

  MPI_Finalize();
}
