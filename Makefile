#
# Makefile for the matrix redistribution project
#
MPICC = mpicc
MPIFC = mpifort
CFLAGS = -O2
FFLAGS = -O2
TARGETS = reference_scalapack project

CC = $(MPICC)

# Libraries that are required in the project
# We are linking against ScaLAPACK which depends on BLAS+LAPACK.
# Hence we need both BLAS+LAPACK+ScaLAPACK!
LAPACKBLAS_PATH = /zhome/0e/2/36189/teaching/02616/2019/libraries/lib
LAPACKBLAS_LIBS = -L$(LAPACKBLAS_PATH) -llapack -lblas
SCALAPACK_PATH = /zhome/0e/2/36189/teaching/02616/2019/libraries/lib
SCALAPACK_LIBS = -L$(SCALAPACK_PATH) -lscalapack -llapack -lblas

# Gather the full list of libraries for the linker line
# Since we are linking against a Fortran library we also
# need to link against the gfortran library.
LIBS = $(SCALAPACK_LIBS) $(LAPACKBLAS_LIBS) -lgfortran

all: $(TARGETS)

# Application
project: CPPFLAGS += -Dredis=redistribute2
#project: CPPFLAGS += -DDEBUG
project: project.o
	$(MPICC) -o $@ $^

reference_scalapack: reference_scalapack.o
	$(MPICC) -o $@ $< $(LIBS)

# Object file
reference_scalapack.o: reference_scalapack.c
	$(MPICC) -c $(CFLAGS) $<

clean:
	$(RM) $(TARGETS) $(TARGETS:%=%.o)
