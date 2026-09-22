
NVCC ?= nvcc
ARCH ?= 80
TARGET ?= sssp
RUN_LOOP ?= 8

all: $(TARGET)

$(TARGET): kernel.cu support.cu csr_graph.cu wl.h common.h csr_graph.h support.h ../../SSSP/benchmark.h Makefile
	$(NVCC) kernel.cu support.cu csr_graph.cu -O3 -DRUN_LOOP=$(RUN_LOOP) -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) -rdc=true -lcudadevrt -Xptxas -O3 -Xptxas -v -lcuda -lcudart -o $@
#nvcc $^ -O3 -gencode=arch=compute_86,code=sm_86 -rdc=true -lcudadevrt -Xptxas -O3  -Xptxas -v -lcuda -lcudart -o $@


clean:
	rm -f sssp *.o

again:
	make clean
	make
