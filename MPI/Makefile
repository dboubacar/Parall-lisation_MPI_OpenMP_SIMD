CC= gcc
CFLAGS= -O3 -Wall -Wextra -g
EXEC= cg_seq
OBJ= mmio.o cg.o
LIB= -lm

CMD_EXEC=mpirun -n 32 -hostfile hostfile --map-by node $(EXEC)

all: $(OBJ)
	$(CC) $(CFLAGS)  $(OBJ) -o $(EXEC) $(LIB)


cg.o: cg.c
	$(CC) -c $(CFLAGS) $< $(LIB)

mmio.o: mmio.c
%.o: %.c
	$(CC) -c $(CFLAGS) mmio.c cg.c $(LIB)

.PHONY: clean

clean :
	rm -rf $(EXEC) *~ *.o
