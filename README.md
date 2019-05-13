# UNIPAR
**UNI**tig construction in **PAR**allel for de novo assembly

UNIPAR is a fast assembly tool that use De Bruijn graph based algorithms to assemble short sequencing reads to long unitigs. It uses both CPUs and GPUs to run in parallel, and scales to multiple computer nodes in a cluster.

## Pre-requisites:

CUDA 5 or later, with GPU compute capability 3.5 or higher

GCC 4.9 or later

MPI library

*tbb library* used for parallel sort and scan on CPUs

## Build UNIPAR:
git clone https://github.com/ShuangQiuac/UNIPAR

cd &lt;PATH_TO_UNIPAR&gt;

mkdir build

cd build

cmake ..

make

## Run UNIPAR:
### Run on a single machine:

./unipar -i &lt;input file&gt; -r &lt;read length&gt; -k &lt;kmer length&gt; -n &lt;number of partitions&gt; -c &lt;number of CPUs&gt; -g &lt;number of GPUs&gt; -d &lt;intermediate file directory&gt; -o &lt;unitig output directory&gt; -t &lt;cutoff threshold&gt;
  
### Run with multiple processes:
mpirun -np &lt;number of processes&gt; [host options] ./unipar [parameter options]

### A simple example:

./unipar -i &lt;PATH_TO_UNIPAR&gt;/example/test.fa -r 36 -k 27

mpirun -np 2 ./unipar -i &lt;PATH_TO_UNIPAR&gt;/example/test.fa -r 36 -k 27

### Parameter options:
**-i** [*STRING*]: input file, either a fasta or a fastq file

**-r** [*INT*]: read length, the first r number of base pairs in a read will be taken

**-k** [*INT*]: kmer length, less than or equal to the read length, suggestted to be an odd number

**-n** [*INT*]: [Optional] number of partitions, set to be 512 by default, suggestted to be a number of power of 2

**-c** [*INT*]: [Optional] number of CPUs to run, either 0 or 1, set to 1 by default

**-g** [*INT*]: [Optional] number of GPUs to run, either set to 0 or the number of GPUs detected with UNIPAR, set to the number of GPUs detected by default

**-d** [*STRING*]: [Optional] intermediate output directory for partitions, set to ./partitions by default

**-o** [*STRING*]: [Optional] unitig output directory, set to the current directory by default

**-t** [*INT*]: [Optional] the cutoff threshold for kmer coverage, set to 1 by default


## Output files:

Miminizer based partitioning files [*intermediate file*]

De Bruijn subgraph files [*optional*]: 
Users can choose to output constructed De Bruijn graph if they only needed the raw graph instead of the unitigs.
The number of subgraph files is a user defined parameter, and set to 512 by default.
Output of subgraph files is turned off by default.

**Unitig files** [*these are output results of UNIPAR*]: 
The total number of unitig files equals to the total number of processors run with UNIPAR. 

Format: contig_&lt;processor id&gt;_&lt;process id&gt;.fa

Unitigs in all the files contributes to the final results.

## Tested Datasets:
**Ecoli** on SRA (SRR001665) https://www.ncbi.nlm.nih.gov/sra/?term=SRR001665
**Human Chr14** on GAGE: http://gage.cbcb.umd.edu/data/Hg_chr14/ 
**Bumbblebee** on GAGE http://gage.cbcb.umd.edu/data/Rhodobacter_sphaeroides/
**Whole Human Genomes** on SRA (SRX016231) https://www.ncbi.nlm.nih.gov/sra?term=SRX016231

## Tested Machine Configuration:
**GPU** Nvidia K80, Nvidia P40
**Total number of GPUs** Upto 24 (2\*12 K40)
**Total number of Computer Nodes** Upto 6
