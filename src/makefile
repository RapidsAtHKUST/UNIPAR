CUDA_FLAG := -Xcompiler -fopenmp -Xptxas -dlcm=cg -G --gpu-architecture=sm_35
CFLAG := -g -lpthread -std=c++11
GCCFLAG := -g -lpthread
#GCC := /home/sqiuac/usr/local/bin/gcc
#G++ := /home/sqiuac/usr/local/bin/g++
GCC := ~/.linuxbrew/bin/gcc
G++ := ~/.linuxbrew/bin/g++
MPI_FLAG := -I/home/sqiuac/usr/local/include -L/home/sqiuac/usr/local/lib
#MPICC := /home/sqiuac/usr/local/bin/mpicc

OBJECTS := share.o pio.o msp.o gpu_msp.o bitkmer.o hash.o dbgraph.o gpu_dbgraph.o malloc.o tbb.o distribute.o all2all.o \
io.o preprocess.o gpu_preprocess.o comm.o gpu_share.o gpu_comm.o contig.o gpu_contig.o main.o 


graph: $(OBJECTS)
	    nvcc $(CFLAG) $(CUDA_FLAG) $(MPI_FLAG) -ltbb -lmpi $(OBJECTS) -o graph 

pio.o: pio.c
	    nvcc $(CFLAG) $(CUDA_FLAG) -c pio.c -o pio.o

main.o: main.c
	    nvcc $(CFLAG) $(CUDA_FLAG) -c main.c -o main.o 
	     
msp.o: msp.c
	    nvcc $(CFLAG) $(CUDA_FLAG) -c msp.c -o msp.o
	        
gpu_msp.o: msp.cu
		nvcc $(CFLAG) $(CUDA_FLAG) -c msp.cu -o gpu_msp.o	        
	        
bitkmer.o: bitkmer.c
		nvcc $(CFLAG) $(CUDA_FLAG) -O3 -c bitkmer.c -o bitkmer.o     
	        
dbgraph.o: dbgraph.c 
	    nvcc $(CFLAG) $(CUDA_FLAG) -c dbgraph.c -o dbgraph.o

gpu_dbgraph.o: dbgraph.cu
		nvcc $(CFLAG) $(CUDA_FLAG) -c dbgraph.cu -o gpu_dbgraph.o

hash.o: hash.c
#		$(GCC) $(GCCFLAG) -c hash.c -o hash.o  
		nvcc $(CFLAG) -c hash.c -o hash.o  
		
share.o: share.c
		nvcc $(CFLAG) -c share.c -o share.o  

gpu_share.o: share.cu
		nvcc $(CFLAG) $(CUDA_FLAG) -c share.cu -o gpu_share.o

tbb.o: tbbsort.cpp #only compiles with g++!
		$(G++) $(CFLAG) -fopenmp -I ~/usr/local/include/ -L ~/usr/local/lib -ltbb -c tbbsort.cpp -o tbb.o
		
malloc.o: malloc.c
		nvcc $(CFLAG) -c malloc.c -o malloc.o
		
distribute.o: distribute.c
		nvcc $(CUDA_FLAG) $(CFLAG) $(MPI_FLAG) -c distribute.c -o distribute.o
		
preprocess.o: preprocess.c
		nvcc $(CFLAG) $(CUDA_FLAG) -c preprocess.c -o preprocess.o
		
gpu_preprocess.o: preprocess.cu
		nvcc $(CFLAG) $(CUDA_FLAG) -c preprocess.cu -o gpu_preprocess.o

all2all.o: all2all.cu
		nvcc $(CFLAG) $(CUDA_FLAG) -c all2all.cu -o all2all.o
		
io.o: io.c
		nvcc $(CFLAG) $(CUDA_FLAG) -c io.c -o io.o
		
comm.o: comm.c
		nvcc $(CFLAG) $(CUDA_FLAG) -c comm.c -o comm.o
		
gpu_comm.o: comm.cu
		nvcc $(CFLAG) $(CUDA_FLAG) -c comm.cu -o gpu_comm.o
		
contig.o: contig.c
		nvcc $(CFLAG) $(CUDA_FLAG) -c contig.c -o contig.o
		
gpu_contig.o: contig.cu
		nvcc $(CFLAG) $(CUDA_FLAG) -c contig.cu -o gpu_contig.o
					
clean: 
	        rm graph $(OBJECTS)
