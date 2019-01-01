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

#include <stdatomic.h>
#include <pthread.h>
#include "include/hash.h"

#ifdef LITTLE_ENDIAN
static const ull addtable[8] = { 0x1, 0x100, 0x10000, 0x1000000, 4294967296, 1099511627776, 281474976710656, 72057594037927936 };
#else
static const ull addtable[8] = { 0x100000000000000, 0x1000000000000, 0x10000000000, 0x100000000, 0x1000000, 0x10000, 0x100, 0x1 };
#endif

static evaltime_t start;
static evaltime_t end;
static float mem_time = 0;

static uint size_prime_index[2];
static uint table_size[2];
static entry_t * table_entries[2];

extern float elem_factor;

static const hashval_t hprime_tab[30] = {
            7,
           13,
           31,
           61,
          127,
          251,
          509,
         1021,
         2039,
         4093,
         8191,
        16381,
        32749,
        65521,
       131071,
       262139,
       524287,
      1048573,
      2097143,
      4194301,
      8388593,
     16777213,
     33554393,
     67108859,
    134217689,
    268435399,
    536870909,
   1073741789,
   2147483647,
  /* Avoid "decimal constant so large it is unsigned" for 4294967291.  */
   0xfffffffb
};


/* The following function returns an index into the above table of the
   nearest prime number which is greater than N, and near a power of two. */

int
atomic_increase (int * address, int value)
{
	int q = atomic_fetch_add (address, 1);
	return q;
}

ull atomic_and (ull * address, ull value)
{
	ull ret = atomic_fetch_and (address, value);
	return ret;
}

bool
atomic_set_value (int * occupied, int new, int old)
{
	int assumed = old;
	bool value;
	while (*occupied == old)
	{
		value = atomic_compare_exchange_strong (occupied, &assumed, new);
	}
	return value;
}

static bool
atomic_test_occupied (int * occupied)
{
	int assumed = 0;
	bool value;
	while (*occupied == 0)
	{
		value = atomic_compare_exchange_strong (occupied, &assumed, 1);
		if (value == true)
		{
			return false;
		}
//		printf ("failed!\n");
	}
	return true;
}

/* The following function returns an index into the above table of the
   nearest prime number which is greater than N, and near a power of two. */
uint
higher_prime_index (unsigned long n)
{
  unsigned int low = 0;
  unsigned int high = sizeof(hprime_tab) / sizeof(hashval_t);

  while (low != high)
    {
      unsigned int mid = low + (high - low) / 2;
      if (n > hprime_tab[mid])
    	  low = mid + 1;
      else
    	  high = mid;
    }

  /* If we've run out of primes, abort.  */
  if (n > hprime_tab[low])
    {
      printf ("Cannot find prime bigger than %lu\n", n);
    }

  return low;
}

/*??? More work on this function! */
static hashval_t
hashtab_mod_1 (hashval_t x, hashval_t y)
{
  /* The multiplicative inverses computed above are for 32-bit types, and
     requires that we be able to compute a highpart multiply.  */
  /* Otherwise just use the native division routines.  */
  return x % y;
}

/* Compute the primary hash for HASH given HASHTAB's current size.  */
hashval_t
hashtab_mod_cpu (hashval_t hash, uint size_prime_index)
{
  return hashtab_mod_1 (hash, hprime_tab[size_prime_index]);
}

/* Compute the secondary hash for HASH given HASHTAB's current size.  */
hashval_t
hashtab_mod_m2_cpu (hashval_t hash, uint size_prime_index)
{
  return 1 + hashtab_mod_1 (hash, hprime_tab[size_prime_index] - 2);
}

bool
is_equal_kmer_cpu (kmer_t * t_entry, kmer_t * kmer)
{
	if (t_entry->x != kmer->x)
		return 0;
	if (t_entry->y != kmer->y)
		return 0;
#ifdef LONG_KMER
	if (t_entry->z != kmer->z)
		return 0;
	if (t_entry->w != kmer->w)
		return 0;
#endif
	return 1;
}


#define hashtab_size(hashtab) { table_size; }
//#define increase_num_of_elems(hashtab) { hashtab->num++; }

void init_hashtab_size (hashstat_t * stat, uint num_of_elems)
{
	uint h_size_prime_index;
	hashsize_t h_table_size;
	h_size_prime_index = higher_prime_index (num_of_elems * elem_factor);
	h_table_size = hprime_tab[h_size_prime_index];

	stat->overall_size = h_table_size;
}

void
adjust_hashtab (dbgraph_t * stat, uint elem_size)
{
	gettimeofday (&start, NULL);
	memset (stat->nodes, 0, stat->size * elem_size);
	gettimeofday (&end, NULL);
	mem_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	printf ("mem time: %f\n", mem_time);
}

void
create_hashtab (hashsize_t num_of_elems, uint elem_size, dbgraph_t * stat, uint tabid)
{
	size_prime_index[tabid] = higher_prime_index (num_of_elems * elem_factor);
	table_size[tabid] = hprime_tab[size_prime_index[tabid]];

	table_entries[tabid] = (entry_t *) malloc ((ull) table_size[tabid]*elem_size);
	CHECK_PTR_RETURN (table_entries[tabid], "error in malloc table %d!\n", tabid);
	memset (table_entries[tabid], 0, (ull) table_size[tabid]*elem_size);

	stat->nodes = table_entries[tabid];
	stat->size = table_size[tabid];
	gettimeofday (&start, NULL);
#ifdef	CHECK_HASH_TABLE
	printf ("initiated number of entries in hash table: %u\nmalloc hash table size %lu\n", stat->size, (ull) table_size[tabid], elem_size);
#endif
//	memset (stat->nodes, 0, stat->size * elem_size);
	uint i;
	gettimeofday (&end, NULL);
	mem_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
//	printf ("mem time: %f\ntable size: %u\n", mem_time, stat->size);
}

void
destroy_hashtab (dbgraph_t * graph)
{
	free (graph->nodes);
}

static inline void
update_node_as_elem2_in_hashtab (entry_t * node, edge_type edge, edge_type redge, rid_t rid)
{
#ifdef CHECK_EDGE_BOUND
	if ((((ull)(node->edge) >> edge * 8) & 0xff) < 255)
#endif
		atomic_fetch_add_explicit ((ull *)(&(node->edge)), addtable[edge], memory_order_relaxed);
#ifdef CHECK_EDGE_BOUND
	if ((((ull)(node->edge) >> redge * 8) & 0xff) < 255)
#endif
		atomic_fetch_add_explicit ((ull *)(&(node->edge)), addtable[redge], memory_order_relaxed);
//	atomic_max (&(node->rid), rid);
}

static void
update_node_as_elem_in_hashtab (entry_t * node, edge_type edge, rid_t rid)
{
#ifdef CHECK_EDGE_BOUND
	if ((((ull)(node->edge) >> edge * 8) & 0xff) < 255)
#endif
		atomic_fetch_add_explicit ((ull *)(&(node->edge)), addtable[edge], memory_order_relaxed);
//	atomic_max (&(node->rid), rid);
}

bool
find_and_update2_hashtab_with_hash (hashval_t hashval, kmer_t * kmer, edge_type edge, edge_type redge, rid_t rid, int tabid, ull * searches, ull * collisions)
{
	  hashval_t index, hash2;
	  entry_t * entry;
	  hashsize_t size = table_size[tabid];
	  entry_t * entries = table_entries[tabid];
	  bool value;
	  uint i;
//	  int assumed = 0;

//	  atomic_fetch_add_explicit (searches, 1, memory_order_relaxed);
	  index = hashtab_mod_cpu (hashval, size_prime_index[tabid]);

	  /* debug only: */
	  if (index >= size)
	  {
		  printf ("index %u is larger than size %u\n", index, size);
		  return false;
	  }
	  entry = (entry_t *)(entries) + index;
	  value = atomic_test_occupied (&(entry->occupied));
	  if ( value == false )
	  {
		  atomic_exchange (&(entry->kmer.x), kmer->x);
		  atomic_exchange (&(entry->kmer.y), kmer->y);
		#ifdef LONG_KMER
		  atomic_exchange (&(entry->kmer.z), kmer->z);
		  atomic_exchange (&(entry->kmer.w), kmer->w);
		#endif
		  atomic_exchange (&(entry->occupied), 2);
	  }

	  if (entry->occupied == 0)
      	  return false;
	  while (entry->occupied != 2) {}

	  if ( (is_equal_kmer_cpu)(&entry->kmer, kmer) )
	  {
		  update_node_as_elem2_in_hashtab (entry, edge, redge, rid);
		  return true;
	  }

	  hash2 = hashtab_mod_m2_cpu (hashval, size_prime_index[tabid]);
	  for (i = 0; i < size; i++)
	  {
//		  atomic_fetch_add_explicit (collisions, 1, memory_order_relaxed);
	      index += hash2;
	      if (index >= size)
	    	  index -= size;

	      entry = (entry_t *)(entries) + index;
	      value = atomic_test_occupied (&(entry->occupied));
	      if ( value == false )
	      {
	    	  atomic_exchange (&(entry->kmer.x), kmer->x);
	    	  atomic_exchange (&(entry->kmer.y), kmer->y);
			#ifdef LONG_KMER
	    	  atomic_exchange (&(entry->kmer.z), kmer->z);
	    	  atomic_exchange (&(entry->kmer.w), kmer->w);
			#endif
	    	  atomic_exchange (&(entry->occupied), 2);
	      }
	      if (entry->occupied == 0)
	      	  return false;
		  while (entry->occupied != 2) {}

		  if ( (is_equal_kmer_cpu)(&entry->kmer, kmer) )
		  {
			  update_node_as_elem2_in_hashtab (entry, edge, redge, rid);
			  return true;
		  }

	   }

	  return false;
}

bool
find_and_update_hashtab_with_hash (hashval_t hashval, kmer_t * kmer, edge_type edge, rid_t rid, int tabid, ull * searches, ull * collisions)
{
	  hashval_t index, hash2;
	  entry_t * entry;
	  hashsize_t size = table_size[tabid];
	  entry_t * entries = table_entries[tabid];
	  bool value;
	  uint i;
//	  int assumed = 0;

//	  atomic_fetch_add_explicit (searches, 1, memory_order_relaxed);
	  index = hashtab_mod_cpu (hashval, size_prime_index[tabid]);

	  /* debug only: */
	  if (index >= size)
	  {
		  printf ("index %u is larger than size %u\n", index, size);
		  return false;
	  }

	  entry = (entry_t *)(entries) + index;
	  value = atomic_test_occupied (&(entry->occupied));
	  if ( value == false )
	  {
    	  atomic_exchange (&(entry->kmer.x), kmer->x);
    	  atomic_exchange (&(entry->kmer.y), kmer->y);
		#ifdef LONG_KMER
    	  atomic_exchange (&(entry->kmer.z), kmer->z);
    	  atomic_exchange (&(entry->kmer.w), kmer->w);
		#endif
    	  atomic_exchange (&(entry->occupied), 2);
	  }
	  if (entry->occupied == 0)
     	  return false;
	  while (entry->occupied != 2) {}

	  if ( (is_equal_kmer_cpu)(&entry->kmer, kmer) )
	  {
		  update_node_as_elem_in_hashtab (entry, edge, rid);
		  return true;
	  }

	  hash2 = hashtab_mod_m2_cpu (hashval, size_prime_index[tabid]);
	  for (i = 0; i < size; i++)
	  {
//		  atomic_fetch_add_explicit (collisions, 1, memory_order_relaxed);
	      index += hash2;
	      if (index >= size)
	    	  index -= size;

	      entry = (entry_t *)(entries) + index;
	      value = atomic_test_occupied (&(entry->occupied));
	      if ( value == false )
	      {
	    	  atomic_exchange (&(entry->kmer.x), kmer->x);
	    	  atomic_exchange (&(entry->kmer.y), kmer->y);
			#ifdef LONG_KMER
	    	  atomic_exchange (&(entry->kmer.z), kmer->z);
	    	  atomic_exchange (&(entry->kmer.w), kmer->w);
			#endif
	    	  atomic_exchange (&(entry->occupied), 2);
	      }
	      if (entry->occupied == 0)
	      	  return false;
	      while (entry->occupied != 2) {}

	      if ( (is_equal_kmer_cpu)(&entry->kmer, kmer) )
	      {
			 update_node_as_elem_in_hashtab (entry, edge, rid);
			 return true;
	      }

	   }

	  return false;
}
