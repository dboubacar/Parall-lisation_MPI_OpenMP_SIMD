all :
	cd mpi && $(MAKE)
	cd sequentiel && $(MAKE)
	cd mpi_omp && $(MAKE)
	cd omp && $(MAKE)

clean:
	cd mpi && $(MAKE) clean
	cd sequentiel && $(MAKE) clean
	cd mpi_omp && $(MAKE) clean
	cd omp && $(MAKE) clean

.PHONY: all clean install
