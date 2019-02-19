/*
 * hash.h
 *
 *  Created on: 2015-7-9
 *      Author: qiushuang
 */

#ifndef HASH_H_
#define HASH_H_


#include "utility.h"
#include "io.h"
#include "dbgraph.h"
#include "msp.h"
#include "comm.h"

//#define CHECK_EDGE_BOUND
//#define CHECK_HASH_TABLE

#define ONE32   0xFFFFFFFFUL
#define LL(v)   (v##ULL)
#define ONE64   LL(0xFFFFFFFFFFFFFFFF)

#define T32(x)  ((x) & ONE32)
#define T64(x)  ((x) & ONE64)

#define ROTL32(v, n)   \
	(T32((v) << ((n)&0x1F)) | T32((v) >> (32 - ((n)&0x1F))))

#define ROTL64(v, n)   \
	(T64((v) << ((n)&0x3F)) | T64((v) >> (64 - ((n)&0x3F))))

#define ELEM_FACTOR 0.4 // 0.25?0.4? be careful about this factor

#define DEFAULT_SEED 3735928559

#define EMPTY // empty entry slot
#define MAX_TABLE_SIZE 4294967295 // 2^32-1

typedef uint hashsize_t;
typedef uint hashval_t;
typedef node_t entry_t; // define entry type here

typedef bool (* test_empty_function) (entry_t * entry);
typedef bool (* is_equal_function) (entry_t * entry1, entry_t * entry2);

typedef struct hashtab
{
	uint size_prime_index;
	uint size;
	uint num;
	entry_t * entries;
	uint collisions; // number of collisions in hashing
	uint searches; // number of searches
	test_empty_function test_empty; // test empty entry slot function
	is_equal_function is_equal;
} hashtab_t;

typedef struct hashstat
{
	entry_t * d_entries;
	ull overall_size;
} hashstat_t;

typedef struct
{
	uint * indices;
	rid_t * ridarr; // read id array pointer
	uch * lenarr; // length array of superkmers, where the most significant bit is used as a mark for hashing
	seq_t * spks; // superkmer buffer pointer
	ull spksize; // superkmer size
	uint num; // number of superkmers
	uint numkmer; // number of kmers corresponding to the superkmers
	int k;
	int tabid;
	ull * search;
	ull * collision;
} hash_arg;

typedef struct
{
	char * file_dir; // file directory to output hash tables
	spkmer_t * superkmers;
	dbgraph_t * graph;
	subgraph_t * subgraph;
	dbtable_t * tbs;
	int * varr;
	int tabid;
	int k;
	int p;
	int world_size;
	int world_rank;
	int total_num_partitions;
} cpu_hash_arg;

typedef struct
{
	char * file_dir; // file directory to output hash tables
	spkmer_t * spkmers;
	hashstat_t * stat;
	uint * indices;
	dbgraph_t * graph;
	subgraph_t * subgraph;
	dbtable_t * tbs;
	uint * node_hist;
	int * varr;
	int tabid;
	int k;
	int p;
	int world_size;
	int world_rank;
	int total_num_partitions;
} gpu_hash_arg;


#ifdef __cplusplus
extern "C"
{
uint higher_prime_index (unsigned long n);
void adjust_hashtab (dbgraph_t * stat, uint elem_size);
int atomic_increase (int *, int);

void * cpu_dbgraph_workflow (void * arg);
void init_hashtab_size (hashstat_t * stat, uint num_of_elems);

void init_hashtab_size (hashstat_t * stat, uint num_of_elems);
void create_hashtab (hashsize_t num_of_elems, uint elem_size, dbgraph_t * stat, uint tabid);
void destroy_hashtab (dbgraph_t * graph);
bool find_and_update2_hashtab_with_hash (hashval_t hashval, kmer_t * kmer, edge_type edge, edge_type redge, rid_t rid, int tabid, ull * searches, ull * collisions);
bool find_and_update_hashtab_with_hash (hashval_t hashval, kmer_t * kmer, edge_type edge, rid_t rid, int tabid, ull * searches, ull * collisions);

ull atomic_and (ull * address, ull value);
bool atomic_set_value (int*, int, int);
hashval_t hashtab_mod_cpu (hashval_t hash, uint size_prime_index);
hashval_t hashtab_mod_m2_cpu (hashval_t hash, uint size_prime_index);
bool is_equal_kmer_cpu (kmer_t * t_entry, kmer_t * kmer);
}
#endif

uint higher_prime_index (unsigned long n);
bool find_and_update2_hashtab_with_hash (hashval_t hashval, kmer_t * kmer, edge_type edge, edge_type redge, rid_t rid, int tabid, ull * searches, ull * collisions);
bool find_and_update_hashtab_with_hash (hashval_t hashval, kmer_t * kmer, edge_type edge, rid_t rid, int tabid, ull * searches, ull * collisions);
int atomic_increase (int *, int);
void create_hashtab (hashsize_t num_of_elems, uint elem_size, dbgraph_t * stat, uint tabid);
void destroy_hashtab (dbgraph_t * graph);
bool is_equal_kmer_cpu (kmer_t * t_entry, kmer_t * kmer);
hashval_t hashtab_mod_cpu (hashval_t hash, uint size_prime_index);
hashval_t hashtab_mod_m2_cpu (hashval_t hash, uint size_prime_index);
ull atomic_and (ull * address, ull value);
uint atomic_and_int (uint *, uint);
uint atomic_or_int (uint *, uint);
bool atomic_set_value (int*, int, int);

#endif /* HASH_H_ */
