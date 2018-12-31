/*
 * msp.h
 *
 *  Created on: 2015-7-7
 *      Author: qiushuang
 */

#ifndef MSP_H_
#define MSP_H_

#include "utility.h"
#include "dbgraph.h"
//#include "io.h"

#define MAX_NUM_THREADS 64
#ifndef THREADS_MSP_COMPUTE
#define THREADS_MSP_COMPUTE MAX_NUM_THREADS
#endif

#define HIST_BLOCK_SIZE 512
#define HIST_TOTAL_THREADS (HIST_BLOCK_SIZE * THREADS_PER_BLOCK)

#ifdef LONG_MINSTR
#define PSTR_UNIT_LENGTH 2
#else
#define PSTR_UNIT_LENGTH 1
#endif

#define PSTR_UNIT_BYTES 4

#define NUM_OF_RANGES 5
#define AVE_NUM_SPK 4

/*
typedef struct bound
{
	uint up;
	uint low;
} bound_t;*/

#ifdef NUM_OF_RANGES
typedef uint bound_t;

typedef struct length_range
{
	bound_t l1;
	bound_t l2;
	bound_t l3;
	bound_t l4;
	bound_t l5;
} length_range_t;
#endif

typedef struct read_buf
{
	seq_t * buf;
	offset_t offset[THREADS_MSP_COMPUTE]; // offsets for encoded reads
	uint num[THREADS_MSP_COMPUTE];
} read_buf_t;

typedef struct
{
	char ** read_buffer;
	uch ** d_msp_ptr;
//	ull read_size;
	char *** rbufs;
	uint ** rnums;
	int p;
	int k;
	int num_of_partitions;
	int read_length;
	int nstreams;
	int world_size;
	int world_rank;
} cpu_msp_arg;

typedef struct
{
	char ** read_buffer;
	uch ** d_msp_ptr;
	char *** rbufs;
	uint ** rnums;
	int p;
	int k;
	int num_of_partitions;
	int read_length;
	int nstreams;
	int world_size;
	int world_rank;
} output_msp_arg;

typedef struct
{
	char ** read_buffer;
	uch ** h_msp_ptr;
	read_buf_t * reads;
	uch * d_msp_ptr;
	seq_t * d_reads;
	char *** rbufs;
	uint ** rnums;
	int p;
	int k;
	int num_of_partitions;
	int read_length;
	int nstreams;
	int did;
	int world_size;
	int world_rank;
} gpu_msp_arg;


/* SoA for msp of all reads in buffer */
typedef struct msp
{
	uch * nums;
	uch * poses; // start positions of partitioned superkmer on the read
	msp_id_t * ids; // msp id of the corresponding superkmer
} msp_t;


/* meta information for both msp output buffer and superkmers */
typedef struct msp_meta
{
	uint size; // number of superkmers
	uint offset; // array offset: number of superkmers in current msp buffer -- for output control
	uint num_kmers;
	offset_t spksize;
	offset_t spkoffset;
	rid_t * idarr; // read id array pointer
	uch * lenarr; // length array of superkmers
	seq_t * spkbuf;
} msp_meta_t;


typedef void (* thread_function) (kmer_t *, int);

#endif /* MSP_H_ */
