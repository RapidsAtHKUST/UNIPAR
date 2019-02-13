/*
 * hash.c
 *
 *  Created on: 2015-7-9
 *      Author: qiushuang
 */


/*
 * This hash table construction method is based on the gcc libiberty hash.
 * Hash function used in searching is murmur hashing, favoring its value distribution and collision properties.
 */

#ifndef MSP_CUH
#define MSP_CUH

#include "../include/hash.h"
#include "hash.cuh"

//#define NUM_OF_READS 1024 // number of reads loaded into shared memory, and maximum number of reads: 2048 for Kepler
#define LOAD_WIDTH_PER_THREAD 4 // 32 bit word width in terms of chars
#define THREADS_PER_BLOCK 512 // 1D threads in 1D block
#define MAX_NUM_BLOCKS 4096
#define MAX_NUM_THREADS_PER_BLOCK 1024
#define TOTAL_THREADS (MAX_NUM_BLOCKS * THREADS_PER_BLOCK)

extern float elem_factor;
//__constant__ static int read_length_d = READ_LENGTH;

#ifdef LITTLE_ENDIAN
__constant__ static const ull addtable[8] = { 0x1, 0x100, 0x10000, 0x1000000, 0x100000000, 0x10000000000, 0x1000000000000, 0x100000000000000 };
#else
__constant__ static const ull addtable[8] = { 0x100000000000000, 0x1000000000000, 0x10000000000, 0x100000000, 0x1000000, 0x10000, 0x100, 0x1 };
#endif

__constant__ static uint size_prime_index;
__constant__ static uint table_size;
__constant__ static entry_t * table_entries;


#define hashtab_size(hashtab) { table_size; }
//#define increase_num_of_elems(hashtab) { hashtab->num++; }


// ********* create hash table on gpu, called on host *************
__host__ static void
create_hashtab (hashsize_t num_of_elems, uint elem_size, hashstat_t * stat)
{
	uint h_size_prime_index;
	hashsize_t h_table_size;
	h_size_prime_index = higher_prime_index (num_of_elems * elem_factor);
	h_table_size = hprime_tab[h_size_prime_index];
	entry_t * d_entries;

	CUDA_CHECK_RETURN (cudaMemcpyToSymbol (size_prime_index, &h_size_prime_index, sizeof(uint)));
	CUDA_CHECK_RETURN (cudaMemcpyToSymbol (table_size, &h_table_size, sizeof(hashsize_t)));

	/* Be careful: hashtable size may exceed GPU memory size! */
	CUDA_CHECK_RETURN (cudaMalloc (&d_entries, (ull) h_table_size * elem_size));
#ifdef CHECK_HASH_TABLE
	printf ("initiated number of entries in hash table: %u\nhash table size: %ld\n", h_table_size, (ull) h_table_size * elem_size);
#endif
	CUDA_CHECK_RETURN (cudaMemcpyToSymbol (table_entries, &d_entries, sizeof(entry_t *)));

	stat->d_entries = d_entries;
	stat->overall_size = h_table_size;
}

__host__ static hashsize_t
adjust_hashtab (uint num_of_elems, hashstat_t * stat, uint elem_size)
{
	uint h_size_prime_index = higher_prime_index (num_of_elems * elem_factor);
	hashsize_t h_table_size = hprime_tab[h_size_prime_index];
	CUDA_CHECK_RETURN (cudaMemcpyToSymbol (size_prime_index, &h_size_prime_index, sizeof(uint)));
	CUDA_CHECK_RETURN (cudaMemcpyToSymbol (table_size, &h_table_size, sizeof(hashsize_t)));
//	printf ("hashtable size %lu, h_table_size = %lu\n", stat->overall_size, h_table_size);
	CUDA_CHECK_RETURN (cudaMemset (stat->d_entries, 0, stat->overall_size * elem_size));
	return h_table_size;
}

__host__ static void
destroy_hashtab (dbgraph_t * graph, hashstat_t stat)
{
	cudaFree (stat.d_entries);
}

__device__ static inline void
update_node_as_elem2_in_hashtab (entry_t * node, edge_type edge, edge_type redge, rid_t rid)
{
#ifdef CHECK_EDGE_BOUND
	if ((((ull)(node->edge) >> edge * 8) & 0xff) < 255)
#endif
		atomicAdd ((ull *)(&(node->edge)), addtable[edge]);
#ifdef CHECK_EDGE_BOUND
	if ((((ull)(node->edge) >> redge * 8) & 0xff) < 255)
#endif
		atomicAdd ((ull *)(&(node->edge)), addtable[redge]);
//	atomicMax (&(node->rid), rid);
}

__device__ static void
update_node_as_elem_in_hashtab (entry_t * node, edge_type edge, rid_t rid)
{
#ifdef CHECK_EDGE_BOUND
	if ((((ull)(node->edge) >> edge * 8) & 0xff) < 255)
#endif
		atomicAdd ((ull *)(&(node->edge)), addtable[edge]);
//	atomicMax (&(node->rid), rid);
}

/* TEST OCCUPIED: if the slot is not occupied, return 1; else return 0 */
__device__ static inline bool
test_occupied (entry_t * entry)
{
	if (atomicCAS (&((node_t *)entry)->occupied, 0, 1) == 0)
		return 1;
	return 0;
}

__device__ static bool
find_and_update2_hashtab_with_hash (hashval_t hashval, kmer_t * kmer, edge_type edge, edge_type redge, rid_t rid, ull * searches, ull * collisions)
{
	  hashval_t index, hash2;
	  entry_t * entry;
	  hashsize_t size = table_size;
	  entry_t * entries = table_entries;
	  int value = -1;
	  uint i;

//	  atomicAdd (searches, 1);
	  index = hashtab_mod (hashval, size_prime_index);

	  /* debug only: */
	  if (index >= size)
	  {
		  return false;
	  }
	  entry = (entry_t *)(entries) + index;
	  value = atomicCAS (&(entry->occupied), 0, 1);
	  if ( value == 0 )
	  {
    	  atomicExch (&(entry->kmer.x), kmer->x);
    	  atomicExch (&(entry->kmer.y), kmer->y);
		#ifdef LONG_KMER
    	  atomicExch (&(entry->kmer.z), kmer->z);
    	  atomicExch (&(entry->kmer.w), kmer->w);
		#endif
    	  atomicExch (&(entry->occupied), 2);
	  }

      if (entry->occupied == 0)
      	  return false;
	  while (entry->occupied != 2) {}

	  if ( (is_equal_kmer)(&entry->kmer, kmer) )
	  {
		  update_node_as_elem2_in_hashtab (entry, edge, redge, rid);
		  return true;
	  }

	  hash2 = hashtab_mod_m2 (hashval, size_prime_index);
	  for (i = 0; i < size; i++)
	  {
//		  atomicAdd (collisions, 1);
	      index += hash2;
	      if (index >= size)
	    	  index -= size;

	      entry = (entry_t *)(entries) + index;
	      value = atomicCAS (&(entry->occupied), 0, 1);
	      if ( value == 0 )
	      {
	    	  atomicExch (&(entry->kmer.x), kmer->x);
	    	  atomicExch (&(entry->kmer.y), kmer->y);
			#ifdef LONG_KMER
	    	  atomicExch (&(entry->kmer.z), kmer->z);
	    	  atomicExch (&(entry->kmer.w), kmer->w);
			#endif
	    	  atomicExch (&(entry->occupied), 2);
	      }
	      if (entry->occupied == 0)
	      	  return false;
		  while (entry->occupied != 2) {}

		  if ( (is_equal_kmer)(&entry->kmer, kmer) )
		  {
			  update_node_as_elem2_in_hashtab (entry, edge, redge, rid);
			  return true;
		  }

	   }

	  return false;
}

__device__ static bool
find_and_update_hashtab_with_hash (hashval_t hashval, kmer_t * kmer, edge_type edge, rid_t rid, ull * searches, ull * collisions)
{
	  hashval_t index, hash2;
	  entry_t * entry;
	  hashsize_t size = table_size;
	  entry_t * entries = table_entries;
	  int value = -1;
	  uint i;

//	  atomicAdd (searches, 1);
	  index = hashtab_mod (hashval, size_prime_index);

	  /* debug only: */
	  if (index >= size)
	  {
		  return false;
	  }

	  entry = (entry_t *)(entries) + index;
	  value = atomicCAS (&(entry->occupied), 0, 1);
	  if ( value == 0 )
	  {
    	  atomicExch (&(entry->kmer.x), kmer->x);
    	  atomicExch (&(entry->kmer.y), kmer->y);
		#ifdef LONG_KMER
    	  atomicExch (&(entry->kmer.z), kmer->z);
    	  atomicExch (&(entry->kmer.w), kmer->w);
		#endif
    	  atomicExch (&(entry->occupied), 2);
	  }
      if (entry->occupied == 0)
      	  return false;
	  while (entry->occupied != 2) {}

	  if ( (is_equal_kmer)(&entry->kmer, kmer) )
	  {
		  update_node_as_elem_in_hashtab (entry, edge, rid);
		  return true;
	  }

	  hash2 = hashtab_mod_m2 (hashval, size_prime_index);
	  for (i = 0; i < size; i++)
	  {
//		  atomicAdd (collisions, 1);
	      index += hash2;
	      if (index >= size)
	    	  index -= size;

	      entry = (entry_t *)(entries) + index;
	      value = atomicCAS (&(entry->occupied), 0, 1);
	      if ( value == 0 )
	      {
	    	  atomicExch (&(entry->kmer.x), kmer->x);
	    	  atomicExch (&(entry->kmer.y), kmer->y);
			#ifdef LONG_KMER
	    	  atomicExch (&(entry->kmer.z), kmer->z);
	    	  atomicExch (&(entry->kmer.w), kmer->w);
			#endif
	    	  atomicExch (&(entry->occupied), 2);
	      }
	      if (entry->occupied == 0)
	      	  return false;
	      while (entry->occupied != 2) {}

	      if ( (is_equal_kmer)(&entry->kmer, kmer) )
	      {
			 update_node_as_elem_in_hashtab (entry, edge, rid);
			 return true;
	      }

	   }

	  return false;
}


extern __shared__ seq_t shared_spks[];

__global__ static void hash_kmers (uint num, uint * __restrict__ indices, uch * __restrict__ lenarr, const rid_t * __restrict__ ridarr, seq_t * __restrict__ spks, int k, int read_length, ull * searches, ull * collisions)
{
	const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint tid = threadIdx.x;

	int w, r;
	w = (num + TOTAL_THREADS - 1) / TOTAL_THREADS;

	for (r = 0; r < w; r++)
	{

	if (r * TOTAL_THREADS + blockIdx.x * blockDim.x >= num)
		return;
	ull start = indices[r * TOTAL_THREADS + blockIdx.x * blockDim.x];
	ull increase = MIN (THREADS_PER_BLOCK, num - (r * TOTAL_THREADS + blockIdx.x * blockDim.x)); // number of elements in this load
	ull end = indices[r * TOTAL_THREADS + blockIdx.x * blockDim.x + increase];
	ull load_size = end - start; // size of superkmers of this load
	int load_width_per_thread = sizeof(uint);
	uint size_per_load = THREADS_PER_BLOCK * load_width_per_thread;
	seq_t * sptr = spks + start;
//	ull bound = MIN(load_size + load_width_per_thread, THREADS_PER_BLOCK * ((read_length + 3) / 4 + 1));
	ull bound = load_size + load_width_per_thread - load_size % load_width_per_thread;

	/*** load superkmers to shared memory for a block of threads ***/
	int t;
	for (t = 0; t < (load_size + size_per_load - 1) / size_per_load; t++)
	{
		if ((tid + 1) * load_width_per_thread + t * size_per_load < bound)
		{
//			*((uint *) (shared_spks + tid * load_width_per_thread + t * size_per_load)) = *((uint *) (sptr + tid * load_width_per_thread + t * size_per_load));
			int tt;
#pragma unroll
			for (tt = 0; tid * load_width_per_thread + t * size_per_load + tt < load_size; tt++)
			{
				*(shared_spks + tid * load_width_per_thread + t * size_per_load + tt) = *(sptr + tid * load_width_per_thread + t * size_per_load + tt);
			}
		}
	}
	__syncthreads();

		if (gid + r * TOTAL_THREADS >= num)
			return;
		uint index;
	//	uint index0;
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
		mark = lenarr[gid + r * TOTAL_THREADS] & 0x80; // mark whether the most significant bit is 0 or 1
		len = lenarr[gid + r * TOTAL_THREADS] & 0x7f; // do not need to add k, for its utilization
		rid = ridarr[gid + r * TOTAL_THREADS];

		index = indices[gid + r * TOTAL_THREADS] - start;
		spk_ptr = shared_spks + index; // point to the superkmer

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
			get_first_kmer (&kmer[0], spk_ptr, k + 1);
			get_reverse_edge (edge[1], kmer[0]); // get redge for reverse if mark == 1
			kmer_32bit_left_shift (&kmer[0], 2);
			i++;
		}
		else
			get_first_kmer (&kmer[0], spk_ptr, k);

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
			int ret = compare_2kmers (&kmer[0], &kmer[1]);
			if (ret >= 0)
				flag = 0;
			else flag = 1;
		}
		else if (hash[0] < hash[1]) flag = 0;
		else flag = 1;
		if ( mark )
		{
			if (find_and_update2_hashtab_with_hash (hash[flag], &kmer[flag], edge[flag], edge[(1 + flag) % 2] + 4, rid, searches + blockIdx.x, collisions + blockIdx.x) == false)
			{
				printf ("hash kmer error: cannot find space or element!\n");
			}
		}
		else
		{
			if ( find_and_update_hashtab_with_hash (hash[flag], &kmer[flag], edge[0] + flag * 4, rid, searches + blockIdx.x, collisions + blockIdx.x) == false )
			{
				printf ("hash kmer error: cannot find space or element!\n");
			}
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
				int ret = compare_2kmers (&kmer[0], &kmer[1]);
				if (ret >= 0)
					flag = 0;
				else flag = 1;
			}
			else if (hash[0] < hash[1]) flag = 0;
			else flag = 1;

			if( find_and_update2_hashtab_with_hash (hash[flag], &kmer[flag], edge[flag], edge[(1 + flag) % 2] + 4, rid, searches + blockIdx.x, collisions + blockIdx.x) == false )
			{
				printf ("hash kmer error: cannot find space or element!\n");
			}
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
				int ret = compare_2kmers (&kmer[0], &kmer[1]);
				if (ret >= 0)
					flag = 0;
				else flag = 1;
			}
			else if (hash[0] <= hash[1]) flag = 0;
			else flag = 1;

		if (len + k - 1 == read_length)
		{
			if ( find_and_update_hashtab_with_hash (hash[flag], &kmer[flag], edge[1] + ((1 + flag) % 2) * 4, rid, searches + blockIdx.x, collisions + blockIdx.x) == false )
			{
				printf ("hash kmer error: cannot find space or element!\n");
			}
		}
		else
		{
			edge[0] = (*(spk_ptr + (k + i) / 4) >> (6 - ((k + i) % 4) * 2)) & 0x3;

			if( find_and_update2_hashtab_with_hash (hash[flag], &kmer[flag], edge[flag], edge[(1 + flag) % 2] + 4, rid, searches + blockIdx.x, collisions + blockIdx.x) == false )
			{
				printf ("hash kmer error: cannot find space or element!\n");
			}
		}

		__syncthreads();
	}

}

#endif /* MSP_CUH */
