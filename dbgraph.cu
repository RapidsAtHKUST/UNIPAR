/*
 * dbgraph.c
 *
 *  Created on: 2015-7-12
 *      Author: qiushuang
 */

#include <omp.h>
#include <pthread.h>

#include "include/io.h"
#include "include/distribute.h"
#include "include/hash.h"
#include "include/dbgraph.h"
#include "include/msp.h"
#include "bitkmer.cuh"
#include "hash.cuh"
#include "dbgraph.cuh"

//#define NUM_OF_READS 1024 // number of reads loaded into shared memory, and maximum number of reads: 2048 for Kepler
#define LOAD_WIDTH_PER_THREAD 4 // 32 bit word width in terms of chars
#define THREADS_PER_BLOCK 512 // 1D threads in 1D block
#define MAX_NUM_BLOCKS 4096
#define MAX_NUM_THREADS_PER_BLOCK 1024
#define TOTAL_THREADS (MAX_NUM_BLOCKS * THREADS_PER_BLOCK)

float write_graph_time = 0;
float input_spk_time = 0;
float hash_time = 0;
float create_time = 0;
float destroy_time = 0;
extern float inmemory_time;

ull count[4] = {0, 0, 0, 0};
ull kmer_count[4] = {0, 0, 0, 0};
//ull cpu_count[2] = {0, 0};

/* consume and serve queue: spkmers[NUM_OF_PARTITIONS]
 * cns points to the partition yet to be consumed; srv points to the partition having been served.
 * cns and srv are shared by cpu side calling threads and gpu side calling threads
 */
extern int cns;
extern int srv;

extern uint max_num_kmers;
extern uint max_num_spks;
extern uint max_spksize;

extern int read_length;

//extern __constant__ thread_function shift_dictionary[];
extern __shared__ seq_t shared_spks[];
__shared__ uint nhist;

__device__ bool
test_empty_node (entry_t * entry)
{
	ull * node_ptr = (ull *) (((node_t *)entry)->edge);
	if (*node_ptr == 0) return 1;
	return 0;
}


__global__ void compute_index_base (uch * lens, uint * indices, int num_of_spks, int k, int read_length)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	int w, r;
	w = (num_of_spks + TOTAL_THREADS - 1) / TOTAL_THREADS;

	for (r = 0; r < w; r++)
	{
	if (gid + r * TOTAL_THREADS > num_of_spks)
		return;
	if (gid + r * TOTAL_THREADS == 0)
		indices[gid + r * TOTAL_THREADS] = 0;
	else
	{
		uint len = lens[gid + r * TOTAL_THREADS - 1] & 0x7f;
		indices[gid + r * TOTAL_THREADS] = (len + k) + 3;
		if (len + k - 1 == read_length)
			indices[gid + r * TOTAL_THREADS] -= 1;
		indices[gid + r * TOTAL_THREADS] /= 4;
	}
	}
}

#define max_size(spkmers, x, max, num_of_partitions) {	\
	int i;							\
	max = (spkmers)[0].(x);			\
	for (i = 1; i < num_of_partitions; i++)	{	\
		if (max < (spkmers)[i].(x))				\
			max = (spkmers)[i].(x);				\
	}											\
}

extern "C"
{
__host__ inline uint max_kmers (spkmer_t * spkmers, int num_of_partitions)
{
	uint max = spkmers[0].numkmer;
	uint i;
	for (i = 1; i < num_of_partitions; i++)
	{
		if (spkmers[i].numkmer > max)
			max = spkmers[i].numkmer;
	}
	return max;
}

void gpu_hash_subgraph_workflow (int total_num_partitions, char * file_dir, spkmer_t * spkmer, uint * indices, hashstat_t * stat, dbgraph_t * graph, \
		subgraph_t * subgraph, dbtable_t * tbs, int tabid, int k, int p, int world_size, int world_rank)
{
	cudaSetDevice (tabid);

	evaltime_t start, end;
	evaltime_t gpus, gpue;

	float gpu_qtime = 0;
	float gpu_hashtime = 0;

	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int np_node; // this is the real number of partitions in this compute node
	if (world_rank == world_size - 1)
		np_node = total_num_partitions - (world_rank) * np_per_node;
	else
		np_node = np_per_node;

	msp_id_t start_id = np_per_node * world_rank;

	uint max_nkmers = 0;
	gettimeofday (&gpus, NULL);
	while (cns < np_node)
	{
		gettimeofday (&start, NULL);
		uint q = atomic_increase (&cns, 1);
		while (q > srv) {}
		gettimeofday (&end, NULL);
		gpu_qtime += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;

		spkmer_t * spkmers = &spkmer[q];
		if (spkmers->num == 0)
		{
			printf ("WARNING: !!!!!!!!!!!number of superkmers is 0!!!!!!!!!!!!!!!\n");
			continue;
		}

		ull * searches = NULL;
		ull * collisions = NULL;
		uint num_of_spks = spkmers->num;
		uint num_of_kmers = spkmers->numkmer;
		kmer_count[tabid] += num_of_kmers;
		ull spksize = spkmers->spksize;
//		printf ("@@@@@@@@@@@@@@@queue id: %d @@@@@@@@@@@@@@@@@@@\n", q);

		uint shared_size = THREADS_PER_BLOCK * ((read_length + 3) / 4 + 1);

		if (max_nkmers < num_of_kmers)
		{
			max_nkmers = num_of_kmers;
		}

		ull malloc_size = sizeof(uint) * (num_of_spks + 1) + (sizeof(rid_t) + sizeof(uch)) * num_of_spks + sizeof(seq_t) * spksize + (sizeof(rid_t) - (sizeof(uch) * num_of_spks + sizeof(seq_t) * spksize) % sizeof(rid_t));
		CUDA_CHECK_RETURN (cudaMemcpy (indices, spkmers->indices, malloc_size, cudaMemcpyHostToDevice));
		rid_t * ridarr = (rid_t *)(indices + num_of_spks + 1);
		uch * lenarr = (uch *)(ridarr + num_of_spks);
		uch * spks = (uch *)indices + sizeof(uint) * (num_of_spks + 1) + (sizeof(rid_t) + sizeof(uch)) * num_of_spks;

		uint num_of_blocks = (num_of_spks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		uint block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;

		/* Hash kmers */
		gettimeofday (&start, NULL);
		graph->size = adjust_hashtab (num_of_kmers, stat, sizeof(node_t));
		CUDA_CHECK_RETURN (cudaFuncSetCacheConfig (hash_kmers, cudaFuncCachePreferL1));
		hash_kmers <<<block_size, THREADS_PER_BLOCK, shared_size>>> (num_of_spks, indices, lenarr, ridarr, spks, k, read_length, searches, collisions);

//		printf ("graph size in partition %d: %u\n", q, graph->size);
		CUDA_CHECK_RETURN (cudaMemcpy (graph->nodes, stat->d_entries, sizeof(entry_t) * graph->size, cudaMemcpyDeviceToHost));
		gettimeofday (&end, NULL);
		gpu_hashtime += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;

#ifdef USE_DISK_IO
		write_hashtab (file_dir, graph->nodes, graph->size, num_of_kmers, q+start_id, total_num_partitions, tabid);
		/*test for edges */
		count[tabid] += write_graph (graph, k);
#else
		count[tabid] += gather_sorted_dbgraph (graph, tbs, subgraph, num_of_kmers, q, start_id, np_node);
#endif

		free (spkmers->indices);


	}
	gettimeofday (&gpue, NULL);
	print_exec_time (gpus, gpue, "WORLD RANK: %d: ~~~~~~~~~~~~~overall hashing time on gpu %d: \n", world_rank, tabid);
	printf ("WORLD RANK: %d: ~~~~~~~~~~~~~gpu %d hashing time only: %f\n", world_rank, tabid, gpu_hashtime);
	printf ("WORLD RANK: %d: ~~~~~~~~~~~~~gpu %d quequing time: %f\n", world_rank, tabid, gpu_qtime);
	printf ("WORLD RANK: %d: ~~~~~~~~~~~~~gpu %d: maximum number of kmers: %u\n", world_rank, tabid, max_nkmers);
	printf ("WORLD RANK: %d: ~~~~~~~~~~~~~gpu %d: number of vertices counted: %u\n", world_rank, tabid, count[tabid]);
	printf ("WORLD RANK: %d: ~~~~~~~~~~~~~gpu %d: total number of kmers counted: %u\n", world_rank, tabid, kmer_count[tabid]);
}

void *
gpu_dbgraph_workflow (void * arg)
{
	gpu_hash_arg * garg = (gpu_hash_arg *) arg;
	gpu_hash_subgraph_workflow (garg->total_num_partitions, garg->file_dir, garg->spkmers, garg->indices, garg->stat, garg->graph, \
			garg->subgraph, garg->tbs, garg->tabid, garg->k, garg->p, garg->world_size, garg->world_rank);

	return ((void *)0);
}

void init_gpu_workflow (uint * indices[], hashstat_t * stat, dbgraph_t * graph, int num_of_hash_devices)
{
	int j;
	ull malloc_size = sizeof(uint) * (max_num_spks + 1) + (sizeof(rid_t) + sizeof(uch)) * max_num_spks + sizeof(seq_t) * max_spksize + \
			(sizeof(rid_t) - (sizeof(uch) * max_num_spks + sizeof(seq_t) * max_spksize) % sizeof(rid_t));
	for (j = 0; j < num_of_hash_devices; j++)
	{
		cudaSetDevice (j);
		create_hashtab (max_num_kmers, sizeof(node_t), &stat[j]);
		CUDA_CHECK_RETURN (cudaMallocHost (&graph[j].nodes, stat[j].overall_size * sizeof(node_t)));
		graph[j].size = stat[j].overall_size;
		CUDA_CHECK_RETURN (cudaMalloc (&indices[j], malloc_size));
	}
}

void finalize_gpu_workflow (uint * indices[], hashstat_t * stat, dbgraph_t * graph, int num_of_hash_devices)
{
	int j;
	for (j = 0; j < num_of_hash_devices; j++)
	{
		destroy_hashtab (&graph[j], stat[j]);
		cudaFree (indices[j]);
		cudaFreeHost (graph[j].nodes);
	}

}


void
construct_dbgraph_hetero (int k, int p, char * filename, char * msp_dir, char * hash_dir, int total_num_partitions, int num_of_hash_devices, int num_of_hash_cpus, \
		subgraph_t * subgraph, dbtable_t * tbs, int world_size, int world_rank)
{
	spkmer_t spkmers[NUM_OF_PARTITIONS];
	dbgraph_t graph[NUM_OF_HASH_DEVICES];


	// device hashing structures
	uint * indices[NUM_OF_HASH_DEVICES];
	hashstat_t stat[NUM_OF_HASH_DEVICES];
	gpu_hash_arg harg[NUM_OF_HASH_DEVICES];
	pthread_t thread[NUM_OF_HASH_DEVICES];
	int turn;

	// host hashing structures
	dbgraph_t cpu_graph[NUM_OF_HASH_CPUS];
	cpu_hash_arg cpu_harg[NUM_OF_HASH_CPUS];
	pthread_t cpu_thread[NUM_OF_HASH_CPUS];
	int cpu_turn;


	/*  validation and time measurement */
//	ull total_num_kmers = 0;
	float input_spk_time = 0;
	evaltime_t start, end;
	evaltime_t overs, overe;

	gettimeofday (&start, NULL);
	init_output (filename);

	int flag = 0;

	init_gpu_workflow (indices, stat, graph, num_of_hash_devices);
	if (num_of_hash_devices == 0)
	{
		init_hashtab_size (stat, max_num_kmers);
	}
	/* The last parameter hash_flag is set to 1 if hashtable based graph traversal is conducted. */

	gettimeofday (&end, NULL);
//	print_exec_time (start, end, "init hashing time: ");

	int np_per_node;
	int np_node; // this is the real number of partitions in this compute node
	get_np_node (&np_per_node, &np_node, total_num_partitions, world_size, world_rank);

	msp_id_t start_id = np_per_node * world_rank;
	FILE ** mspinput = (FILE **) malloc (sizeof(FILE*) * world_size);
	CHECK_PTR_RETURN (mspinput, "malloc msp patition file pointers error!\n");
	gettimeofday (&overs, NULL);
	int i;
	for (i = 0; i < np_node; i++)
	{
		while (srv - cns > WAIT) {} //waiting when serving is much faster than consuming
		gettimeofday (&start, NULL);
		spkmer_t privspk;
		uint * hspk_ptr;
		msp_stats_t stats;

		 // input initialization in get_superkmers
		stats = get_spk_stats (i+start_id, world_size, msp_dir, mspinput);
		if (stats.spk_num == 0)
		{
			spkmers[i].num = 0;
			continue;
		}
		privspk.num = stats.spk_num;
		privspk.spksize = stats.spk_size;

		ull malloc_size = sizeof(uint) * (privspk.num + 1) + (sizeof(rid_t) + sizeof(uch)) * privspk.num + sizeof(seq_t) * privspk.spksize \
				+ (sizeof(rid_t) - (sizeof(uch) * privspk.num + sizeof(seq_t) * privspk.spksize) % sizeof(rid_t));
		hspk_ptr = (uint *) malloc (malloc_size);
		CHECK_PTR_RETURN (hspk_ptr, "malloc superkmer buffer error\n");

		privspk.indices = hspk_ptr;
		privspk.ridarr = (rid_t *)(hspk_ptr + privspk.num + 1);
		privspk.lenarr = (uch *)(privspk.ridarr + privspk.num);
		privspk.spks = (uch *)hspk_ptr + sizeof(uint) * (privspk.num + 1) + (sizeof(rid_t) + sizeof(uch)) * privspk.num;

		privspk.numkmer = load_superkmers (mspinput, privspk.ridarr, privspk.lenarr, privspk.indices, privspk.spks, privspk.num, k, world_size);

		finalize_msp_input (mspinput, world_size); // finalize msp input for superkmers
		memcpy (&spkmers[i], &privspk, sizeof(spkmer_t));
		gettimeofday (&end, NULL);
		input_spk_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;

		atomic_increase (&srv, 1);

		if (flag == 0)
		{
			// cpu calling thread setting for cpu-hashing
			for (cpu_turn = 0; cpu_turn < num_of_hash_cpus; cpu_turn++)
			{
				cpu_harg[cpu_turn].total_num_partitions = total_num_partitions;
				cpu_harg[cpu_turn].graph = &cpu_graph[cpu_turn];
				cpu_harg[cpu_turn].superkmers = &spkmers[0];
				cpu_harg[cpu_turn].tabid = cpu_turn;
				cpu_harg[cpu_turn].k = k;
				cpu_harg[cpu_turn].p = p;
				cpu_harg[cpu_turn].file_dir = hash_dir;
				cpu_harg[cpu_turn].subgraph = subgraph;
				cpu_harg[cpu_turn].tbs = tbs;
				cpu_harg[cpu_turn].world_size = world_size;
				cpu_harg[cpu_turn].world_rank = world_rank;
				if (pthread_create (&cpu_thread[cpu_turn], NULL, cpu_dbgraph_workflow, &cpu_harg[cpu_turn]) != 0)
				{
					printf ("create thread for hashing kmers %d error!\n", cpu_turn);
				}
			}
			// gpu calling threads setting for gpu hashing and graph traversal
			for (turn = 0; turn < num_of_hash_devices; turn++)
			{
				harg[turn].total_num_partitions = total_num_partitions;
				harg[turn].indices = indices[turn];
				harg[turn].stat = &stat[turn];
				harg[turn].graph = &graph[turn];
				harg[turn].spkmers = &spkmers[0];
				harg[turn].tabid = turn;
				harg[turn].k = k;
				harg[turn].p = p;
				harg[turn].file_dir = hash_dir;
				harg[turn].subgraph = subgraph;
				harg[turn].tbs = tbs;
				harg[turn].world_size = world_size;
				harg[turn].world_rank = world_rank;
				if (pthread_create (&thread[turn], NULL, gpu_dbgraph_workflow, &harg[turn]) != 0)
				{
					printf ("create thread for hashing kmers %d error!\n", turn);
				}
			}
			flag = 1;
		}

	}
	for (cpu_turn = 0; cpu_turn < num_of_hash_cpus; cpu_turn++)
	{
		if (pthread_join (cpu_thread[cpu_turn], NULL) != 0)
		{
			printf ("Join thread on hash table %d failure!\n", cpu_turn);
		}
	}
	for (turn = 0; turn < num_of_hash_devices; turn++)
	{
		if (pthread_join (thread[turn], NULL) != 0)
		{
			printf ("Join thread on hashing with GPU %d failure!\n", turn);
		}
	}
	gettimeofday (&overe, NULL);
	inmemory_time += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;
	print_exec_time (overs, overe, "***********************Overall hashing kmers time: \n");
	printf ("Unipar until graph construction in-memory processing time: %f\n", inmemory_time);

	free (mspinput);
	finalize_gpu_workflow (indices, stat, graph, num_of_hash_devices);
}


}
