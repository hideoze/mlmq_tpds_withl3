CC=nvcc

LFLAGS=-lcusparse

INCLUDE_DIR=../core/include
LIB_DIR=../build
UTILS_DIR=/a100-data/wyh/boost_1_87_0/

# Other queues
FLAG= -O3 -m64 -gencode=arch=compute_80,code=sm_80 -rdc=true -lcuda -lcudart -w
# For heap queue
#FLAG=-O3 -m64 -Xptxas -dlcm=cg -rdc=true -gencode=arch=compute_86,code=sm_86 -lcuda -lcudart

all: main

main: main.cu csr_graph.cu sssp_run.cu
	$(CC) $^ -o $@ $(FLAG) -I$(INCLUDE_DIR) -I$(UTILS_DIR) -L$(LIB_DIR) $(LFLAGS)

.PHONY:

again:
	make clean
	make --quiet -j4

clean:
	rm main