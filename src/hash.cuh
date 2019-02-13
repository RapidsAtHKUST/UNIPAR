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

#ifndef HASH_CUH
#define HASH_CUH

#include "../include/hash.h"

__constant__ static const hashval_t prime_tab[30] = {
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

/*??? More work on this function! */
__device__ static inline hashval_t
hashtab_mod_1 (hashval_t x, hashval_t y)
{
  /* The multiplicative inverses computed above are for 32-bit types, and
     requires that we be able to compute a highpart multiply.  */
  /* Otherwise just use the native division routines.  */
  return x % y;
}

/* Compute the primary hash for HASH given HASHTAB's current size.  */
__device__ static inline hashval_t
hashtab_mod (hashval_t hash, uint size_prime_index)
{
  return hashtab_mod_1 (hash, prime_tab[size_prime_index]);
}

/* Compute the secondary hash for HASH given HASHTAB's current size.  */
__device__ static inline hashval_t
hashtab_mod_m2 (hashval_t hash, uint size_prime_index)
{
  return 1 + hashtab_mod_1 (hash, prime_tab[size_prime_index] - 2);
}

__device__ static inline bool
is_equal_kmer (kmer_t * t_entry, kmer_t * kmer)
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



__device__ static uint
murmur_hash2 (uint * kmer_ptr, uint seed)
{
	const uint m = 0x5bd1e995;
	const int r = 24;

	uint h = seed ^ (KMER_UNIT_LENGTH * KMER_UNIT_BYTES);
	uint k;

	int i;
	for (i = 0; i < KMER_UNIT_LENGTH; i++)
	{
		k = *kmer_ptr++;

		k *= m;
		k ^= k >> r;
		k *= m;

		h *= m;
		h ^= k;
	}

	h ^= h >> 13;
	h *= m;
	h ^= h >> 15;

	return h;
}

__device__ static uint
murmur_hash3_32 (uint * kmer_ptr, uint seed)
{
	  int i;

	  uint h = seed;
	  uint k;

	  uint c1 = 0xcc9e2d51;
	  uint c2 = 0x1b873593;

	  for (i = 0; i < KMER_UNIT_LENGTH; i++)
	  {
		  k = *kmer_ptr++;
		  k *= c1;
		  k = ROTL32(k,15);
		  k *= c2;
		  h ^= k;
		  h = ROTL32(h,13);
		  h = h*5+0xe6546b64;
	  }

	  h ^= (KMER_UNIT_LENGTH * KMER_UNIT_BYTES);

	  h ^= h >> 16;
	  h *= 0x85ebca6b;
	  h ^= h >> 13;
	  h *= 0xc2b2ae35;
	  h ^= h >> 16;

	  return h;

}

#endif /* HASH_CUH */
