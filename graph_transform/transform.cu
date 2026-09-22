#include "csr_graph.h"
#include "utils.h"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: ./transform [input file (.mtx)] [output file (.gr)]\n");
        return -1;
    }

    char *in_file_name = argv[1];
    char *out_file_name = argv[2];

    int m, n;
    int nnz;
    int *csrRowPtr;
    int *csrColIdx;
    VALUE_TYPE *csrVal;

    int error = read_mtx_t<VALUE_TYPE>(in_file_name, &m, &n, &nnz, &csrRowPtr, &csrColIdx, &csrVal);
    if (error)
    {
        printf("Read error occurs!\n");
        exit(-1);
    }

    writeToGR(m, nnz, csrRowPtr, csrColIdx, csrVal, out_file_name);
    
    return 0;

}