/*
 * msp.c
 *
 *  Created on: 2015-3-5
 *      Author: qiushuang
 */

#include <omp.h>
#include <math.h>
#include "../include/dbgraph.h"
#include "../include/io.h"
#include "../include/msp.h"
#include "../include/hash.h"
#include "../include/bitkmer.h"

extern ull * max_msp_malloc;
static char rev_table[20] = {'T', '0', 'G', '0', '0', '0', 'C', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', 'A'};
extern int table[256];
extern ull * read_size;
int * dflag;
int * queue;

#define read_ptr (shared_reads + index) // pointer to read of each thread in shared memory

int prd = -1;
int rd = 0;
int rdy = -1;
int wrt = 0;

uint
get_partition_id_from_string (char * min_substring, int p, int num_of_partitions)
{
	ull id = 0;
	int i;
	for (i = 0; i < p; i++)
	{
		id *= 3;
		id += table[min_substring[p-1-i]];
	}
	return (id % num_of_partitions);
}

void cpu_msp_workflow (char * read_buffer[], uch * d_msp_ptr[], char ** rbufs[], uint * rnums[], int p, int k, int num_of_partitions, int nstreams)
{
	evaltime_t start, end;
	evaltime_t cpus, cpue;
	uint num_of_reads = 0;
	float cpu_qtime = 0;
	float cpu_comtime = 0;
	gettimeofday (&cpus, NULL);
	while (rd < nstreams)
	{
	gettimeofday (&start, NULL);
	int q = atomic_increase (&rd, 1);
	while (q > prd) {}
	gettimeofday (&end, NULL);
	cpu_qtime += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	while (q - wrt > 5) {} // wait if producing is much faster than writing

	d_msp_ptr[q] = (uch *) malloc (sizeof(uch) * max_msp_malloc[q]); // malloc buffer size to store msp info
//	printf ("&&&&&&&&&&&&&&&&queue id: %d, prd = %d, rd = %d&&&&&&&&&&&&&&&&&\n", q, prd, rd);
	CHECK_PTR_RETURN (d_msp_ptr[q], "malloc msp ptr %d error!!!!!!!!!!!\n", q);

	gettimeofday (&start, NULL);
	num_of_reads += compute_msp (q, p, k, num_of_partitions, read_buffer, read_size[q], d_msp_ptr[q], rbufs, rnums);
	gettimeofday (&end, NULL);
	cpu_comtime += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;

	dflag[q] = 0;
	int wrt_id = atomic_increase (&rdy, 1);
	queue[wrt_id + 1] = q;
	}
	gettimeofday (&cpue, NULL);
	print_exec_time (cpus, cpue, "~~~~~~~~~~~~~~~~~overall msp compute time on cpu: \n");
	printf ("~~~~~~~~~~~~~~CPU MSP compute time %f\n", cpu_comtime);
	printf ("~~~~~~~~~~~~~~CPU MSP queuing time %f\n", cpu_qtime);
	printf ("~~~~~~~~~~~~~~number of reads on CPU: %d\n", num_of_reads);
}

void output_msp_control (char * read_buffer[], uch * d_msp_ptr[], char ** rbufs[], uint * rnums[], int p, int k, int num_of_partitions, int nstreams, int world_rank)
{
	evaltime_t start, end;
	evaltime_t cpus, cpue;
	float cpu_qtime = 0;
	float output_time = 0;

	gettimeofday (&cpus, NULL);
	while (wrt < nstreams)
	{
		gettimeofday (&start, NULL);
		int q = atomic_increase (&wrt, 1);
		while (q > rdy) {}
//		printf ("&&&&&&&&&&&&&&&& Writing MSP queue id: %d, rdy = %d, wrt = %d&&&&&&&&&&&&&&&&&\n", q, rdy, wrt);
		gettimeofday (&end, NULL);
		cpu_qtime += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
		int rdid = queue[q];
/*		if (rdid != q)
		{
			printf ("warning ::: rd id != wrt id!!!: %d, %d\n", rdid, q);
			while (1) {}
		}*/
		gettimeofday (&start, NULL);
		output_msp_cpu (d_msp_ptr[rdid], rbufs, rnums, k, num_of_partitions, q, world_rank);
		gettimeofday (&end, NULL);
		output_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
		free (d_msp_ptr[rdid]);
		free (read_buffer[rdid]);
	}
	gettimeofday (&cpue, NULL);
	printf ("~~~~~~~~~~~~~~~~~output queuing time on CPU: %f\n", cpu_qtime);
	printf ("~~~~~~~~~~~~~~~~~CPU output superkmer time: %f\n", output_time);
	print_exec_time (cpus, cpue, "~~~~~~~~~~~~~~~~~overall output superkmer partitions time on cpu: \n");
}

void *
cpu_partition_workflow (void * arg)
{
	cpu_msp_arg * harg = (cpu_msp_arg *) arg;
	cpu_msp_workflow (harg->read_buffer, harg->d_msp_ptr, harg->rbufs, harg->rnums, harg->p, harg->k, harg->num_of_partitions, harg->nstreams);
	return ((void *)0);
}

void *
output_msp_workflow (void * arg)
{
	output_msp_arg * harg = (output_msp_arg *) arg;
	output_msp_control (harg->read_buffer, harg->d_msp_ptr, harg->rbufs, harg->rnums, harg->p, harg->k, harg->num_of_partitions, harg->nstreams, harg->world_rank);
	return ((void *) 0);
}
