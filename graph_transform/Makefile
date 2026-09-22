CC=nvcc

FLAG=-O3

UTILS_DIR=/a100-data/wyh/boost_1_87_0/

all: transform reorder_gr

transform: transform.cu csr_graph.cu
	$(CC) $^ -o $@ $(FLAG) -I$(UTILS_DIR)

reorder_gr: reorder_gr.cu csr_graph.cu
	$(CC) $^ -o $@ $(FLAG) -I$(UTILS_DIR)

.PHONY:

again:
	make clean
	make

clean:
	rm transform reorder_gr