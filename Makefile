DESTDIR = /usr/local
all :
	cd mpi && $(MAKE)
	cd Sequentiel && $(MAKE)
	cd mpi_omp && $(MAKE)
	cd omp && $(MAKE)

clean:
	cd mpi && $(MAKE) clean
	cd Sequentiel && $(MAKE) clean
	cd mpi_omp && $(MAKE) clean
	cd omp && $(MAKE) clean
	
.PHONY: all clean install
