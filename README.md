# UNIPAR
UNItig construction in PARallel for de novo assembly with CPUs and GPUs

## Pre-requisites:

CUDA 3.5+ to support atomic features

GCC 4.9+ with support of C11 features

*tbb library* used for parallel sort and scan on CPU

## Build UNIPAR:

## Run UNIPAR:
### Run on a single machine:

./unipar -i &lt;input file&gt; -r &lt;read length&gt; -k &lt;kmer length&gt; -n &lt;number of partitions&gt; -c &lt;number of CPUs&gt; -g &lt;number of GPUs&gt; -d &lt;intermediate file directory&gt; -o &lt;unitig output directory&gt; -t &lt;cutoff threshold&gt;
  
### Run with multi-process:
mpirun -np 6 ./unipar [parameter options]

### A simple example:

./unipar -i ./example/ecoli.fa -r 72 -k 27

mpirun -np 6 ./unipar -i ./example/ecoli.fa -r 72 -k 27

### Parameter options:
**-i** [*STRING*]: input file, either a fasta or a fastq file

**-r** [*INT*]: read length, the first r number of base pairs in a read will be taken

**-k** [*INT*]: kmer length, no longer than the  read length

**-n** [*INT*]: [Optional] number of partitions, set to be 512 by default

**-c** [*INT*]: [Optional] number of CPUs to run, either 0 or 1, set to be 1 by default

**-g** [*INT*]: [Optional] number of GPUs to run, either set to be 0 or the number of GPUs detected with UNIPAR, set to be the number of GPUs detected by default

**-d** [*STRING*]: [Optional] intermediate output directory for partitions, set to be ./partitions by default

**-o** [*STRING*]: [Optional] unitig output directory, set to be the current directory by default

**-t** [*INT*]: [Optional] the cutoff threshold for the number of kmer coverage, set to be 1 by default


## Output files:

Miminizer based partitioning files [*intermediate file*]

De Bruijn subgraph files [optional]: 
Users can choose to output constructed De Bruijn graph if they only needed the raw graph instead of the unitigs
The number of subgraph files is a user defined parameter, and set to 512 by default
Output of subgraph files is turned off by default

unitig files [*this is the output results of UNIPAR*]: 
The total number of unitig files equals to the total number of processors run with UNIPAR. Unitigs in all the files contributes to the final results.


