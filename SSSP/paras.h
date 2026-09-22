#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void print_usage(int argc, char *argv[])
{
    printf("Usage ./sssp -i [graph.gr] -n [nGPU] [-s sources] [-d delta]\n");
}

int parse_args(int argc, char *argv[], char * &input_name, int &n_gpu,
               int &delta_override, int &num_sources)
{
    int ch;

    int input_flag = 0;
    n_gpu = 1;
    delta_override = -1;
    num_sources = 1;

    while ((ch = getopt(argc, argv, "i:n:d:s:")) != -1)
    {
        switch (ch)
        {
            case 'i':
                printf("%s\n", optarg);
                input_flag = 1;
                input_name = optarg;
                break;
            case 'n':
                n_gpu = atoi(optarg);
                if (n_gpu < 1 || n_gpu > 8)
                {
                    printf("n_gpu must be in [1, 8], got %d\n", n_gpu);
                    return -1;
                }
                break;
            case 'd':
                delta_override = atoi(optarg);
                break;
            case 's':
                num_sources = atoi(optarg);
                if (num_sources < 1)
                {
                    printf("sources must be >= 1, got %d\n", num_sources);
                    return -1;
                }
                break;
        }
    }

    if (!input_flag) 
    {
        print_usage(argc, argv);
        return -1;
    }

    return 0;
}
