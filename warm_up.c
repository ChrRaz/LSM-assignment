#include <mpi.h>

void warm_up() {
  static const int N = 500;
  double send[N];

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Post a send-recv between all nodes
  for ( int rank_send = 0; rank_send < size ; rank_send++ ) {
    if ( rank == rank_send ) {
      for ( int i = 0; i < N ; i++ ) {
	send[i] = i;
      }

      // Do send
      for ( int rank_recv = 0 ; rank_recv < size ; rank_recv++ ) {
	if ( rank_send != rank_recv ) {
	  MPI_Send(send, N, MPI_DOUBLE, rank_recv, rank_send, MPI_COMM_WORLD);
	}
      }
    } else {
      MPI_Recv(send, N, MPI_DOUBLE, rank_send, rank_send, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}
