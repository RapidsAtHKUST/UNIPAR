/*
 * dbgraph.h
 *
 *  Created on: 2015-5-27
 *      Author: qiushuang
 */

/*
 * This file contains macro definitions used in constructing the de bruijn graph.
 * Input file: sequenced short reads, usually fasta or fastq; two paired files should be taken if paired end reads are in split files.
 * Read length should be predefined in the macro READ_LENGTH and should be at least 32 (AND SHOULD NOT BE LONGER THAN 121 + K).
 */

#ifndef DBGRAPH_H_
#define DBGRAPH_H_

#include "utility.h"

#define NUM_OF_PARTITIONS 9990

#define NUM_OF_CPUS 1
#define NUM_OF_DEVICES 4

#define NUM_OF_MSP_CPUS 1
#define NUM_OF_MSP_DEVICES 4 
#define NUM_OF_HASH_CPUS 1
#define NUM_OF_HASH_DEVICES 4

//#define READ_LENGTH 36 // read length should be no longer than 121 + k, where k is the kmer length !!!
#define DOUBLE_STRAND 1
#define CUTOFF_N 5
#define CUTOFF 1
#define LONG_KMER // comment this when short reads are used
#define LONG_MINSTR // comment this when short kmers are preferred

#define CHAR_BITS 8

/* sequence storage type */
#define SEQ_BIT_LENGTH 8
typedef uch seq_t;

/* read id type */
#define RID_SIZE 4
typedef uint rid_t;

/* msp id type */
#define MSP_ID_SIZE 4
typedef int msp_id_t;


/* long kmer: length > 32bp; short kmer: length < 32bp */
typedef uint unit_kmer_t;

#ifdef LONG_KMER
typedef struct MY_ALIGN(16) kmer
{
	unit_kmer_t x, y, z, w; // 128 bit for maximum 59bp length kmer
} kmer_t;
#else
typedef struct MY_ALIGN(8) kmer
{
	unsigned int x, y; // 64 bit for maximum 27bp length kmer, due to kmer stream process
} kmer_t;
#endif

#define KMER_UNIT_BITS 32 // 32bit per element in kmer structure
#define KMER_UNIT_CHAR_LENGTH 16 // number of characters stored in a unit of kmer structure
#define KMER_UNIT_BYTES 4 // 4 bytes per element in kmer structure

#ifdef LONG_KMER
#define KMER_BIT_LENGTH 128
#define KMER_CHAR_LENGTH 64
#define KMER_UNIT_LENGTH 4
#else
#define KMER_BIT_LENGTH 64 // maximum total number of bits in a kmer
#define KMER_CHAR_LENGTH 32 // maximum number of characters stored in a kmer
#define KMER_UNIT_LENGTH 2 // number of units to store a kmer, e.g., 2 of x, y or 4 of x, y, z, w
#endif

/* minimum p substring type */
#ifdef LONG_MINSTR
typedef ull minstr_t;
#define MINSTR_BIT_LENGTH 64
#define STR_MARSK 18446744073709551615
#else
typedef uint minstr_t;
#define MINSTR_BIT_LENGTH 32
#define STR_MARSK 4294967295
#endif

typedef uch edge_type;
#define EDGE_DIC_SIZE 8


typedef struct spkmer
{
	uint * indices;
	rid_t * ridarr; // read id array pointer
	uch * lenarr; // length array of superkmers, where the most significant bit is used as a mark for hashing
	seq_t * spks; // superkmer buffer pointer
	ull spksize; // superkmer size
	uint num; // number of superkmers
	uint numkmer; // number of kmers corresponding to the superkmers
} spkmer_t;

typedef struct node
{
	kmer_t kmer; // put this field in the first
	ull edge; //multiplicity of edges for both kmer and its reverse, be careful if the multiplicity exceeds 255!!!
//	edge_type edge[EDGE_DIC_SIZE];
//	rid_t rid; // read id
	uint occupied; // atomic operation flag
} node_t;

typedef node_t entry_t;

typedef struct dbgraph
{
	uint size;
//	uint num;
	node_t * nodes;
} dbgraph_t;

typedef struct graph
{
	ull offset;
	uint size;
	uint * vids;
	node_t * nodes;
} graph_t;

typedef struct dstructs
{
	kmer_t * jkmers;
	kmer_t * jkmers_sorted;
	kmer_t * lkmers;
	kmer_t * lkmers_sorted;
	ull * jedges;
	ull * jedges_sorted;
	ull * ledges;
	ull * ledges_sorted;
	int * ngbs[EDGE_DIC_SIZE];
	int * pres;
	int * posts;
	void * dtemp;
	int * vtemp;
	size_t temp_size;
	size_t vtemp_size;
	uint * pre_offsets;
	uint * post_offsets;
	uint * ng_offsets[EDGE_DIC_SIZE];
	uch * node_flag;
	int * fwlen;
	int * bwlen;
	int * jid;
	int * num_partitions;
	int * partition_list;
	int * id2index;
	uint * node_size;
	uint * send_offsets;
	uint * receive_offsets;
	uint * id_offsets;
} dstructs_t;

typedef struct hstructs
{
	kmer_t * jkmers;
	kmer_t * lkmers;
	ull * jedges;
	ull * ledges;
	int * ngbs[EDGE_DIC_SIZE];
	int * pres;
	int * posts;
} hstructs_t;

#endif /* DBGRAPH_H_ */
