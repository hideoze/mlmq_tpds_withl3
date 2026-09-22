#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void print_usage(int argc, char *argv[])
{
    printf("Usage ./sssp -i [graph.gr]\n");
}

int parse_args(int argc, char *argv[], char * &input_name)
{
    int ch;

    int input_flag = 0;

    while ((ch = getopt(argc, argv, "i:n:d:")) != -1)
    {
        switch (ch)
        {
            case 'i':
                printf("%s\n", optarg);
                input_flag = 1;
                input_name = optarg;
                break;
            case 'n':
                if (atoi(optarg) != 1) return -1;
                break;
            case 'd':
                if (atoi(optarg) <= 0) return -1;
                setenv("BENCH_DELTA", optarg, 1);
                break;
            default:
                return -1;
        }
    }

    if (!input_flag) 
    {
        print_usage(argc, argv);
        return -1;
    }

    return 0;
}
