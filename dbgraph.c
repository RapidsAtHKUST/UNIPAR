/*
 * dbgraph.c
 *
 *  Created on: 2015-7-12
 *      Author: qiushuang
 */

#include <omp.h>
#include <pthread.h>
#include "include/io.h"
#include "include/dbgraph.h"
#include "include/msp.h"
#include "include/hash.h"
#include "include/bitkmer.h"

#define THREADS_PER_TABLE MAX_NUM_THREADS

extern long cpu_threads;
extern thread_function shift_dictionary[];
//extern __shared__ seq_t shared_spks[];
//extern uint max_num_kmers;
//extern uint max_num_spks;
//extern uint max_spksize;
ull search[NUM_OF_PARTITIONS];
ull collision[NUM_OF_PARTITIONS];


//static evaltime_t inss, inse;
//static ull instime[THREADS_PER_TABLE];
//extern ull cpu_count[2];
ull cpu_count[NUM_OF_HASH_DEVICES] = {0, 0, 0, 0};
ull cpu_kmer_count[NUM_OF_HASH_DEVICES] = {0, 0, 0, 0};

extern uint max_num_spks;
extern uint  max_spksize;
extern uint max_num_kmers;
extern int read_length;

int cns = 0; // read and write by cpu/gpu calling threads
int srv = -1; // read by main thread and cpu/gpu calling threads, write only by main thread

#define get_reverse_edge(edge, kmer) {	\
	edge = (kmer.x >> (KMER_UNIT_BITS - 2)) ^ 0x3; }

void hash_kmers2 (uint num, uint * indices_all, uch * lenarr_all, rid_t * ridarr_all, seq_t * spks_all, const int k, const int tabid, ull * searches, ull * collisions)
{
	omp_set_num_threads (cpu_threads);
#pragma omp parallel
	{
	int threads_per_table = omp_get_num_threads ();
	int spknum_th = (num + threads_per_table - 1) / threads_per_table;
	if (spknum_th <= threads_per_table)
		spknum_th = num/threads_per_table;
	int thid = omp_get_thread_num ();
	uint * indices = indices_all + spknum_th * thid;
	uch * lenarr = lenarr_all + spknum_th * thid;
	rid_t * ridarr = ridarr_all + spknum_th * thid;
	seq_t * spks = spks_all;
	uint tnum = num;
	if (thid == threads_per_table - 1)
	{
		spknum_th = (int)tnum - spknum_th * thid;
//		printf ("DDDDDEBUG: %u\n", spknum_th);
	}

//	evaltime_t inss, inse;
//	ull insth = 0;
	int r;
	uint index;
	uch len;
	uch mark;
	kmer_t kmer[2];
	seq_t * spk_ptr;

	uint seed = DEFAULT_SEED; // set it as a fixed number or input argument
	hashval_t hash[2]; // hash[0] is hash value for kmer, hash[1] is hash value for reverse kmer
	rid_t rid;
	int flag;
	edge_type edge[2];  // edge[0] is edge for kmer, edge[1] is redge for reverse kmer
	unit_kmer_t * ptr;
	int table_index;


	int i;

	for (r = 0; r < spknum_th; r++)
	{

		mark = lenarr[r] & 0x80; // mark whether the most significant bit is 0 or 1
		len = lenarr[r] & 0x7f; // do not need to add k, for its utilization
		rid = ridarr[r];

		index = indices[r];
		spk_ptr = spks + index; // point to the superkmer

		/* initialize kmer and its reverse */
#ifdef LONG_KMER
		kmer[0].x = kmer[0].y = kmer[0].z = kmer[0].w = 0;
		kmer[1].x = kmer[1].y = kmer[1].z = kmer[1].w = 0;
#else
		kmer[0].x = 0, kmer[0].y = 0;
		kmer[1].x = 0, kmer[1].y = 0;
#endif
		i = 0;

		/* Get first kmer from superkmer and insert/update into hashtable */
		if (mark)
		{
			get_first_kmer_cpu (&kmer[0], spk_ptr, k + 1);
			get_reverse_edge (edge[1], kmer[0]); // get redge for reverse if mark == 1
			kmer_32bit_left_shift (&kmer[0], 2);
			i++;
		}
		else
			get_first_kmer_cpu (&kmer[0], spk_ptr, k);

		edge[0] = (*(spk_ptr + (k + i) / 4) >> (6 - ((k + i) % 4) * 2)) & 0x3;
		i++;

		/* get reverse complementary of the kmer */
		get_reverse (&kmer[0], &kmer[1]);
#ifdef LONG_KMER
		table_index = (128 - k * 2) / 32;
		shift_dictionary[table_index] (&kmer[1], 128 - k * 2);
#else
		table_index = (64 - k * 2) / 32;
		shift_dictionary[table_index] (&kmer[1], 64 - k * 2);
#endif

		hash[0] = murmur_hash3_32 ((uint *)&kmer[0], seed);
		hash[1] = murmur_hash3_32 ((uint *)&kmer[1], seed);
		if (hash[0] == hash[1])
		{
			int ret = compare_2kmers_cpu (&kmer[0], &kmer[1]);
			if (ret >= 0)
				flag = 0;
			else flag = 1;
		}
		else if (hash[0] < hash[1]) flag = 0;
		else flag = 1;

		if ( mark )
		{
//			gettimeofday (&inss, NULL);
			if (find_and_update2_hashtab_with_hash (hash[flag], &kmer[flag], edge[flag], edge[(1 + flag) % 2] + 4, rid, tabid, searches, collisions) == false)
			{
				printf ("cpu*******hash kmer error: cannot find space or element!\n");
			}
//			gettimeofday (&inse, NULL);
//			insth += ((inse.tv_sec * 1000000 + inse.tv_usec) - (inss.tv_sec * 1000000 + inss.tv_usec));
		}
		else
		{
//			gettimeofday (&inss, NULL);
			if ( find_and_update_hashtab_with_hash (hash[flag], &kmer[flag], edge[0] + flag * 4, rid, tabid, searches, collisions) == false )
			{
				printf ("cpu*******hash kmer error: cannot find space or element!\n");
			}
//			gettimeofday (&inse, NULL);
//			insth += ((inse.tv_sec * 1000000 + inse.tv_usec) - (inss.tv_sec * 1000000 + inss.tv_usec));
		}

		ptr = (unit_kmer_t *)&kmer[0] + (k * 2) / KMER_UNIT_BITS;
		for (; i < len - 1; i++)
		{
			/* Get redge from previous kmer */
			get_reverse_edge (edge[1], kmer[0]);

			/* Get next kmer by using its edge */
			kmer_32bit_left_shift (&kmer[0], 2);
			*ptr |= ((unit_kmer_t)edge[0]) << (KMER_UNIT_BITS - (k * 2) % KMER_UNIT_BITS);

			edge[0] = (*(spk_ptr + (k + i) / 4) >> (6 - ((k + i) % 4) * 2)) & 0x3;

			/* get reverse complementary of the kmer */
			get_reverse (&kmer[0], &kmer[1]);
#ifdef LONG_KMER
			table_index = (128 - k * 2) / 32;
			shift_dictionary[table_index] (&kmer[1], 128 - k * 2);
#else
			table_index = (64 - k * 2) / 32;
			shift_dictionary[table_index] (&kmer[1], 64 - k * 2);
#endif

			hash[0] = murmur_hash3_32 ((uint *)&kmer[0], seed);
			hash[1] = murmur_hash3_32 ((uint *)&kmer[1], seed);
			if (hash[0] == hash[1])
			{
				int ret = compare_2kmers_cpu (&kmer[0], &kmer[1]);
				if (ret >= 0)
					flag = 0;
				else flag = 1;
			}
			else if (hash[0] < hash[1]) flag = 0;
			else flag = 1;

//			gettimeofday (&inss, NULL);
			if( find_and_update2_hashtab_with_hash (hash[flag], &kmer[flag], edge[flag], edge[(1 + flag) % 2] + 4, rid, tabid, searches, collisions) == false )
			{
				printf ("cpu*******hash kmer error: cannot find space or element!\n");
			}
//			gettimeofday (&inse, NULL);
//			insth += ((inse.tv_sec * 1000000 + inse.tv_usec) - (inss.tv_sec * 1000000 + inss.tv_usec));
		}

		if (i > len - 1) continue; // in this case, len == 0: the number of kmers is 1


			get_reverse_edge (edge[1], kmer[0]);

			/* Get next kmer by using its edge */
			kmer_32bit_left_shift (&kmer[0], 2);
			*ptr |= ((unit_kmer_t)edge[0]) << (KMER_UNIT_BITS - (k * 2) % KMER_UNIT_BITS);
			/* get reverse complementary of the kmer */
			get_reverse (&kmer[0], &kmer[1]);
#ifdef LONG_KMER
			table_index = (128 - k * 2) / 32;
			shift_dictionary[table_index] (&kmer[1], 128 - k * 2);
#else
			table_index = (64 - k * 2) / 32;
			shift_dictionary[table_index] (&kmer[1], 64 - k * 2);
#endif

			hash[0] = murmur_hash3_32 ((uint *)&kmer[0], seed);
			hash[1] = murmur_hash3_32 ((uint *)&kmer[1], seed);
			if (hash[0] == hash[1])
			{
				int ret = compare_2kmers_cpu (&kmer[0], &kmer[1]);
				if (ret >= 0)
					flag = 0;
				else flag = 1;
			}
			else if (hash[0] < hash[1]) flag = 0;
			else flag = 1;

		if (len + k - 1 == read_length)
		{
//			gettimeofday (&inss, NULL);
			if ( find_and_update_hashtab_with_hash (hash[flag], &kmer[flag], edge[1] + ((1 + flag) % 2) * 4, rid, tabid, searches, collisions) == false )
			{
				printf ("cpu*******hash kmer error: cannot find space or element!\n");
			}
//			gettimeofday (&inse, NULL);
//			insth += ((inse.tv_sec * 1000000 + inse.tv_usec) - (inss.tv_sec * 1000000 + inss.tv_usec));
		}
		else
		{
			edge[0] = (*(spk_ptr + (k + i) / 4) >> (6 - ((k + i) % 4) * 2)) & 0x3;
//			gettimeofday (&inss, NULL);
			if( find_and_update2_hashtab_with_hash (hash[flag], &kmer[flag], edge[flag], edge[(1 + flag) % 2] + 4, rid, tabid, searches, collisions) == false )
			{
				printf ("cpu*******hash kmer error: cannot find space or element!\n");
			}
//			gettimeofday (&inse, NULL);
//			insth += ((inse.tv_sec * 1000000 + inse.tv_usec) - (inss.tv_sec * 1000000 + inss.tv_usec));
		}
	}
//	instime[thid] += insth;
	}

}

inline uint max_kmers (spkmer_t * spkmers, int num_of_partitions)
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


void cpu_hash_subgraph_workflow (int total_num_partitions, char * file_dir, spkmer_t * spkmer, dbgraph_t * graph, \
		subgraph_t * subgraph, dbtable_t * tbs, int tabid, int k, int p, int world_size, int world_rank)
{
	float write_graph_time = 0;
	float hash_time = 0;
	float create_time = 0;
	float visit_nodes_time = 0;
	float cpu_qtime = 0;
	float cpu_stime = 0;

	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int np_node; // this is the real number of partitions in this compute node
	if (world_rank == world_size - 1)
		np_node = total_num_partitions - (world_rank) * np_per_node;
	else
		np_node = np_per_node;

	msp_id_t start_id = np_per_node * world_rank;

	uint max_nkmers = 0;
	evaltime_t start, end;
	evaltime_t cpus, cpue;
	gettimeofday (&cpus, NULL);
	while (cns < np_node)
	{
		gettimeofday (&start, NULL);
		uint q = atomic_increase (&cns, 1);
		while (q > srv) {}
//		printf ("################queue id: %d ################\n", q);
		gettimeofday (&end, NULL);
		cpu_qtime += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;

		spkmer_t * spkmers = &spkmer[q];
		if (spkmers->num == 0)
		{
			printf ("!!!!!!!!!!!number of superkmers is 0!!!!!!!!!!!!!!!\n");
			continue;
		}

		uint num_of_spks = spkmers->num;
		uint num_of_kmers = spkmers->numkmer;
		ull spksize = spkmers->spksize;
		cpu_kmer_count[tabid] += num_of_kmers;
		ull countn = 0;
		int j;

		if (max_nkmers < num_of_kmers)
		{
			max_nkmers = num_of_kmers;
		}
		uint * indices = spkmers->indices;
		rid_t * ridarr = (rid_t *)(indices + num_of_spks + 1);
		uch * lenarr = (uch *)(ridarr + num_of_spks);
		uch * spks = (uch *)indices + sizeof(uint) * (num_of_spks + 1) + (sizeof(rid_t) + sizeof(uch)) * num_of_spks;

		/* Hash kmers */
		gettimeofday (&start, NULL);
		create_hashtab (num_of_kmers, sizeof(node_t), graph, tabid);
		gettimeofday (&end, NULL);
		create_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;

		gettimeofday (&start, NULL);
		hash_kmers2 (num_of_spks, indices, lenarr, ridarr, spks, k, tabid, &search[0], &collision[0]);
		gettimeofday (&end, NULL);
		hash_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	#ifdef USE_DISK_IO
		// ************* write hashtab to disk for new traversal ***********
		write_hashtab (file_dir, graph->nodes, graph->size, num_of_kmers, q+start_id, total_num_partitions, tabid);
		countn = write_graph (graph, k);
	#else
		countn = gather_sorted_dbgraph (graph, tbs, subgraph, num_of_kmers, q, start_id, np_node);
	#endif
		gettimeofday (&start, NULL);

		cpu_count[tabid] += countn;
	//	printf ("countn = %lu, count = %lu, partition %d\n", countn, cpu_count[tabid], q);
		gettimeofday (&end, NULL);
		write_graph_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;

		free (spkmers->indices);
		destroy_hashtab (graph);
	}
	gettimeofday (&cpue, NULL);
	print_exec_time (cpus, cpue, "~~~~~~~~~~~~~~~~~overall hashing time on cpu: ");
	printf ("~~~~~~~~~~~~~~~~cpu queuing time: %f\n", cpu_qtime);
	printf ("~~~~~~~~~~~~~~~~Test hashing performance - hashing time only: %f\n", hash_time);
	printf ("~~~~~~~~~~~~~~~~CPU hashing time only: %f\n", hash_time + create_time + write_graph_time);
	printf ("^^^^^^^^^^^^^^^^ output hash table time: %f\n", write_graph_time);
	printf ("~~~~~~~~~~~~~~~~ CPU: max number of kmers: %u", max_nkmers);
	printf ("~~~~~~~~~~~~~~~~ CPU: number of vertices counted: %lu\n", cpu_count[tabid]);
	printf ("~~~~~~~~~~~~~~~~ CPU: total number of kmers counted: %lu\n", cpu_kmer_count[tabid]);
}

void *
cpu_dbgraph_workflow (void * arg)
{
	cpu_hash_arg * harg = (cpu_hash_arg *) arg;
	cpu_hash_subgraph_workflow (harg->total_num_partitions, harg->file_dir, harg->superkmers, harg->graph, \
			harg->subgraph, harg->tbs, harg->tabid, harg->k, harg->p, harg->world_size, harg->world_rank);

	return ((void *)0);
}

