CFLAGS= -O3 -Wall -Wextra -g
EXEC= mpi_omp
OBJ= mmio.o cg.o
LIB= -lm -fopenmp
MPI_C = mpicc

all: $(OBJ)
	$(MPI_C) $(CFLAGS)  $(OBJ) -o $(EXEC) $(LIB)

cg.o: cg.c
	$(MPI_C) -c $(CFLAGS) $< $(LIB)

mmio.o: ../lib/mmio.c ../lib/mmio.h
	$(CC) -c $(CFLAGS) ../lib/mmio.c $(LIB)

.PHONY: clean

clean :
	rm -rf $(EXEC) *~ *.o
