/*
 * io.h
 *
 *  Created on: 2015-5-27
 *      Author: qiushuang
 */

#ifndef IO_H_
#define IO_H_

#include <stdio.h>
#include <string.h>
#include "utility.h"
#include "msp.h"
#include "dbgraph.h"
#include "graph.h"
#include "comm.h"

/********** Set this BUF_SIZE properly to let the nstreams be an even number!!! **********/
#define BUF_SIZE (1073741824/8) //(1073741824/4) //1G 1073741824, 12G 12884901888, 4G 4294967296
#define MAX_READ_LENGTH 150
#define CODE_BUF_SIZE BUF_SIZE //(1073741824/2) //2G 2147483648, 3G 3221225472, 1G 1073741824
/* note! CODE_BUF_SIZE should be sufficient in corresponding to BUF_SIZE */

//#define MAX_MSP_MALLOC (2147483648*2) //2G*2

#define LINE (1024*1024)
#define PATH_MAX_LEN 200

//#define USE_DISK_IO

#define WAIT 5 // used for pipelining

typedef struct msp_stats
{
	ull spk_num;
	ull spk_size;
} msp_stats_t;


#ifdef __cplusplus
extern "C"
{
uint parse_data (char ** read_buffer, read_buf_t * reads, ull read_size, int turn, char *** rbufs, uint ** rnums);
int init_mpi_input (char * file_name, int rlen, int world_size, int world_rank);
void init_lookup_table (void);
double estimate_num_reads_from_input (char * filename, int read_length);
int init_msp_output (char * file_dir, int num_of_partitions, int world_rank);
void init_msp_meta (int num_of_partitions, double read_ratio, int read_length, int k);
void set_length_range (int, int);
size_t read_file (char * read_buf, int world_rank);
size_t mpi_read_file (char * read_buf, int world_size, int world_rank);
void * cpu_partition_workflow (void * arg);
void * output_msp_workflow (void * arg);
int finalize_msp_output (int);
void finalize_msp_meta (int k, int num_of_partitions, offset_t * max_kmers, offset_t * max_spks, offset_t * max_spksizes, int world_size, int world_rank);
int finalize_input (void);
int finalize_mpi_input (void);

msp_stats_t get_spk_stats (int pid, int world_size, char * msp_dir, FILE ** mspinput);
uint load_superkmers (FILE ** mspinput, rid_t * ridarr, uch * lenarr, uint * indices, seq_t * read_buffer, uint total_num_of_spk, int k, int world_size);

int init_code (seq_t *);
int finalize_code (void);

int init_input (char * filename, int rlen, int world_size, int world_rank);
void return_dbgraph_hashtab (dbtable_t * tbs, subgraph_t * subgraph, node_t * tab, voff_t size, voff_t elem_hashed, int pid); // defined in io.c
uint gather_sorted_dbgraph (dbgraph_t * graph, dbtable_t * tbs, subgraph_t * subgraph, uint num_of_kmers, int pid, int start_pid, int np_node); // defined in io.c


int init_output (char *);
FILE ** malloc_msp_input (void);

uint compute_msp (int t, int p, int k, int num_of_partitions, char * read_buffer[], ull read_size, uch * d_msp_ptr, char ** rbufs[], uint * rnums[]);
void output_msp_cpu (uch * msp_arr, char ** rbufs[], uint * rnums[], int k, int num_of_partitions, int wrt_id, int world_rank);
uint write_graph (dbgraph_t *, int);
int finalize_msp_input (FILE ** mspinput, int world_size);
int finalize_output (void);
uint get_superkmers (char **, spkmer_t *, int, FILE **);


void usage (void);
int get_opt (int argc, char * const argv[], char * input, int * r, int * k, int * p, int * n, int * c, int * g, \
		char * dir, char * out, int * t, float * f, int * m);
}
#endif

void usage (void);
int get_opt (int argc, char * const argv[], char * input, int * r, int * k, int * p, int * n, int * c, int * g, \
		char * dir, char * out, int * t, float * f, int * m);
void return_dbgraph_hashtab (dbtable_t * tbs, subgraph_t * subgraph, node_t * tab, voff_t size, voff_t elem_hashed, int pid);
uint gather_sorted_dbgraph (dbgraph_t * graph, dbtable_t * tbs, subgraph_t * subgraph, uint num_of_kmers, int pid, int start_pid, int np_node);
uint compute_msp (int t, int p, int k, int num_of_partitions, char * read_buffer[], ull read_size, uch * d_msp_ptr, char ** rbufs[], uint * rnums[]);
void output_msp_cpu (uch * msp_arr, char ** rbufs[], uint * rnums[], int k, int num_of_partitions, int wrt_id, int world_rank);

#endif /* IO_H_ */
