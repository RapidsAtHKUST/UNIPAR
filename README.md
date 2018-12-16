# UNIPAR
UNItig construction in PARallel for de novo assembly with CPUs and GPUs

## Pre-requisites:

CUDA 3.5+ to support atomic features

GCC 4.9+ with support of C11 features

*tbb library* used for parallel sort and scan on CPU

## Build UNIPAR:


## Input file:

UNIPAR currently takes a single fastq or fasta as input. 

## Output files:

Miminizer based partitioning files [*intermediate file*]

De Bruijn subgraph files [optional]: 
Users can choose to output constructed De Bruijn graph if they only needed the raw graph instead of the unitigs
The number of subgraph files is a user defined parameter, and set to 512 by default

unitig files [*this is the output results of UNIPAR*]: 
The total number of unitig files equals to the total number of processors run with UNIPAR. Unitigs in all the files contributes to the final results.


