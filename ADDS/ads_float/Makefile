
all: sssp  

sssp: kernel.cu support.cu csr_graph.cu 
	nvcc $^ -O3 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86  -rdc=true -lcudadevrt -Xptxas -O3  -Xptxas -v -lcuda -lcudart -o $@

clean:
	rm -f sssp *.o

again:
	make clean
	make
