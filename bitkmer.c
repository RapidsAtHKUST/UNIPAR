/*
 * bitkmer.c
 *
 *  Created on: 2015-12-23
 *      Author: qiushuang
 */
#include "include/utility.h"
#include "include/dbgraph.h"
#include "include/msp.h"
#include "include/hash.h"
#include "include/bitkmer.h"

#ifdef LONG_KMER
void kmer_32bit_left_shift (kmer_t * kmer, int n)
{
	if (n == 0) return;
	kmer->x <<= n;
	kmer->x |= kmer->y >> (32 - n);
	kmer->y <<= n;
	kmer->y |= kmer->z >> (32 - n);
	kmer->z <<= n;
	kmer->z |= kmer->w >> (32 - n);
	kmer->w <<= n;
}

void kmer_64bit_left_shift (kmer_t * kmer, int n)
{
	if (n == 0) return;
	kmer->x = (kmer->y << (n - 32)) | (kmer->z >> (64 - n));
	kmer->y = (kmer->z << (n - 32)) | (kmer->w >> (64 - n));
	kmer->z = (kmer->w << (n - 32));
	kmer->w = 0;
}

void kmer_96bit_left_shift (kmer_t * kmer, int n)
{
	if (n == 0) return;
	kmer->x = (kmer->z << (n - 64)) | (kmer->w >> (96 - n));
	kmer->y = (kmer->w << (n - 64));
	kmer->z = kmer->w = 0;
}

void kmer_128bit_left_shift (kmer_t * kmer, int n)
{
	if (n == 0) return;
	kmer->x = (kmer->w << (n - 96));
	kmer->y = kmer->z = kmer->w = 0;
}

#else

void kmer_32bit_left_shift (kmer_t * kmer, int n)
{
	if (n == 0) return;
	kmer->x <<= n;
	kmer->x |= kmer->y >> (32 - n);
	kmer->y <<= n;
}

void kmer_64bit_left_shift (kmer_t * kmer, int n)
{
	if (n == 0) return;
	kmer->x = (kmer->y << (n - 32));
	kmer->y = 0;
}

#endif

#ifdef LONG_KMER
thread_function shift_dictionary[] = {kmer_32bit_left_shift, kmer_64bit_left_shift, kmer_96bit_left_shift, kmer_128bit_left_shift};
#else
thread_function shift_dictionary[] = {kmer_32bit_left_shift, kmer_64bit_left_shift};
#endif

/* This function is to compute the reverse complement of a kmer */
void get_reverse (kmer_t * kmer, kmer_t * reverse)
{
	/* bit reversal for every 32 bit unsigned int number */
	uint temp;
#ifdef LONG_KMER
	temp = (kmer->w) ^ 0xffffffff;
//	temp = (((temp & 0xaaaaaaaa) >> 1) | ((temp & 0x55555555) << 1));
	temp = (((temp & 0xcccccccc) >> 2) | ((temp & 0x33333333) << 2));
	temp = (((temp & 0xf0f0f0f0) >> 4) | ((temp & 0x0f0f0f0f) << 4));
	temp = (((temp & 0xff00ff00) >> 8) | ((temp & 0x00ff00ff) << 8));
	temp = (temp >> 16) | (temp << 16);
	reverse->x = temp;
	temp = (kmer->z) ^ 0xffffffff;
//	temp = (((temp & 0xaaaaaaaa) >> 1) | ((temp & 0x55555555) << 1));
	temp = (((temp & 0xcccccccc) >> 2) | ((temp & 0x33333333) << 2));
	temp = (((temp & 0xf0f0f0f0) >> 4) | ((temp & 0x0f0f0f0f) << 4));
	temp = (((temp & 0xff00ff00) >> 8) | ((temp & 0x00ff00ff) << 8));
	temp = (temp >> 16) | (temp << 16);
	reverse->y = temp;
	temp = (kmer->y) ^ 0xffffffff;
//	temp = (((temp & 0xaaaaaaaa) >> 1) | ((temp & 0x55555555) << 1));
	temp = (((temp & 0xcccccccc) >> 2) | ((temp & 0x33333333) << 2));
	temp = (((temp & 0xf0f0f0f0) >> 4) | ((temp & 0x0f0f0f0f) << 4));
	temp = (((temp & 0xff00ff00) >> 8) | ((temp & 0x00ff00ff) << 8));
	temp = (temp >> 16) | (temp << 16);
	reverse->z = temp;
	temp = (kmer->x) ^ 0xffffffff;
//	temp = (((temp & 0xaaaaaaaa) >> 1) | ((temp & 0x55555555) << 1));
	temp = (((temp & 0xcccccccc) >> 2) | ((temp & 0x33333333) << 2));
	temp = (((temp & 0xf0f0f0f0) >> 4) | ((temp & 0x0f0f0f0f) << 4));
	temp = (((temp & 0xff00ff00) >> 8) | ((temp & 0x00ff00ff) << 8));
	temp = (temp >> 16) | (temp << 16);
	reverse->w = temp;
#else
	temp = (kmer->y) ^ 0xffffffff;
//	temp = (((temp & 0xaaaaaaaa) >> 1) | ((temp & 0x55555555) << 1));
	temp = (((temp & 0xcccccccc) >> 2) | ((temp & 0x33333333) << 2));
	temp = (((temp & 0xf0f0f0f0) >> 4) | ((temp & 0x0f0f0f0f) << 4));
	temp = (((temp & 0xff00ff00) >> 8) | ((temp & 0x00ff00ff) << 8));
	temp = (temp >> 16) | (temp << 16);
	reverse->x = temp;
	temp = (kmer->x) ^ 0xffffffff;
//	temp = (((temp & 0xaaaaaaaa) >> 1) | ((temp & 0x55555555) << 1));
	temp = (((temp & 0xcccccccc) >> 2) | ((temp & 0x33333333) << 2));
	temp = (((temp & 0xf0f0f0f0) >> 4) | ((temp & 0x0f0f0f0f) << 4));
	temp = (((temp & 0xff00ff00) >> 8) | ((temp & 0x00ff00ff) << 8));
	temp = (temp >> 16) | (temp << 16);
	reverse->y = temp;
#endif

}

void get_reverse_kmer (kmer_t * kmer, kmer_t * reverse, int k)
{
	get_reverse (kmer, reverse);
	int table_index;
#ifdef LONG_KMER
	table_index = (128 - k * 2) / 32;
	shift_dictionary[table_index] (reverse, 128 - k * 2);
#else
	table_index = (64 - k * 2) / 32;
	shift_dictionary[table_index] (reverse, 64 - k * 2);
#endif
}


void get_first_pstr (unit_kmer_t * kmer, minstr_t * pstr, int p)
{
	*pstr = 0;
#ifdef LONG_MINSTR
	int i;
	for (i = 0; i < (p * 2 + (KMER_UNIT_BITS - 1)) / KMER_UNIT_BITS - 1; i++)
	{
//		*pstr |= *(kmer + (p * 2) / KMER_UNIT_BITS);
		*pstr |= *kmer;
		*pstr <<= KMER_UNIT_BITS;
		kmer++;
	}
#endif
	*pstr |= (*kmer >> ((KMER_UNIT_BITS - (p * 2) % KMER_UNIT_BITS)) % KMER_UNIT_BITS) << ((KMER_UNIT_BITS - (p * 2) % KMER_UNIT_BITS) % KMER_UNIT_BITS);
#ifdef LONG_MINSTR
	if ((p << 1) <= KMER_UNIT_BITS)
		*pstr <<= KMER_UNIT_BITS;
#endif
}

void right_shift_pstr (unit_kmer_t * kmer_cur, minstr_t * pstr, int p, int j)
{
	kmer_cur += ((j + p - 1) * 2) / KMER_UNIT_BITS; // test this with LONG_KMER
	*pstr <<= 2;
	*pstr |= ((minstr_t)((*kmer_cur << (((j + p - 1) * 2) % KMER_UNIT_BITS) ) & 0xc0000000) << (MINSTR_BIT_LENGTH - KMER_UNIT_BITS)) >> ((p * 2 - 2) % MINSTR_BIT_LENGTH);
//	*pstr |= ( ( *kmer_cur << ( ((j + p - 1) * 2) % KMER_UNIT_BITS ) ) & 0xc0000000 ) >> ((p * 2 - 2) % KMER_UNIT_BITS);
}

void
get_first_kmer_cpu (kmer_t * kmer, seq_t * spk, int k)
{
	unit_kmer_t * ptr = (unit_kmer_t *) kmer;
	int i;

//	for (i = 0; i < (2 * k + KMER_UNIT_BITS - 1) / KMER_UNIT_BITS - 1; i++)
	for (i = 0; i < 2 * k / KMER_UNIT_BITS; i++)
	{
		*ptr |= (unit_kmer_t) (*spk++);
		*ptr <<= SEQ_BIT_LENGTH;
		*ptr |= (unit_kmer_t) (*spk++);
		*ptr <<= SEQ_BIT_LENGTH;
		*ptr |= (unit_kmer_t) (*spk++);
		*ptr <<= SEQ_BIT_LENGTH;
		*ptr |= (unit_kmer_t) (*spk++);
		ptr++;
	}
	for (i = 0; i < ((2 * k) % KMER_UNIT_BITS + SEQ_BIT_LENGTH - 1) / SEQ_BIT_LENGTH; i++)
	{
		*ptr <<= SEQ_BIT_LENGTH;
		*ptr |= (unit_kmer_t) (*spk++);
	}
	*ptr >>= (SEQ_BIT_LENGTH - (2 * k) % SEQ_BIT_LENGTH) % SEQ_BIT_LENGTH;
	*ptr <<= KMER_UNIT_BITS - i * SEQ_BIT_LENGTH + (SEQ_BIT_LENGTH - (2 * k) % SEQ_BIT_LENGTH) % SEQ_BIT_LENGTH;
}

int get_pid_from_kmer (kmer_t * kmer, int k, int p, int num_of_partitions)
{
	minstr_t minpstr = 0, rminpstr = 0;
	minstr_t curr = 0;
	minstr_t pstr = 0, rpstr = 0;

	kmer_t node[2];

	node[0].x = kmer->x;
	node[0].y = kmer->y;
#ifdef LONG_KMER
	node[0].z = kmer->z;
	node[0].w = kmer->w;
#endif

	get_reverse (&node[0], &node[1]);
	int table_index;
#ifdef LONG_KMER
	table_index = (128 - k * 2) / KMER_UNIT_BITS;
	shift_dictionary[table_index] (&node[1], 128 - k * 2);
#else
	table_index = (64 - k * 2) / KMER_UNIT_BITS;
	shift_dictionary[table_index] (&node[1], 64 - k * 2);
#endif

	/* get first minimum p-substring */
	get_first_pstr ((unit_kmer_t *)&node[0], &pstr, p);
	get_first_pstr ((unit_kmer_t *)&node[1], &rpstr, p);
	minpstr = pstr;
	rminpstr = rpstr;

	int j;
	for (j = 1; j < k - p + 1; j++)
	{
		right_shift_pstr ((unit_kmer_t *)&node[0], &pstr, p, j);
		right_shift_pstr ((unit_kmer_t *)&node[1], &rpstr, p, j);
		if (pstr < minpstr) minpstr = pstr;
		if (rpstr < rminpstr) rminpstr = rpstr;
	}
	curr = rminpstr < minpstr ? rminpstr : minpstr;
	msp_id_t mspid = get_partition_id_cpu (curr, p, num_of_partitions);

	return mspid;
}

uint
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

uint
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

msp_id_t get_partition_id_cpu (minstr_t minstr, int p, int num_of_partitions)
{
	ull id = 0;
#ifdef LONG_MINSTR
	minstr >>= 64 - p * 2;
#else
	minstr >>= 32 - p * 2;
#endif
	int i;
	for (i = 0; i < p; i++)
	{
		id *= 3;
		id += minstr & 0x00000003;
 		minstr >>= 2;
	}
	return (id % num_of_partitions);
}

int compare_2kmers_cpu (kmer_t * kmer1, kmer_t * kmer2)
{
	if (kmer1->x < kmer2->x)
	{
		return 1;
	}
	else if (kmer1->x > kmer2->x)
	{
		return -1;
	}
	else if (kmer1->y < kmer2->y)
	{
		return 1;
	}
	else if (kmer1->y > kmer2->y)
	{
		return -1;
	}
	else
#ifdef LONG_KMER
		if (kmer1->z < kmer2->z)
		{
			return 1;
		}
		else if (kmer1->z > kmer2->z)
		{
			return -1;
		}
		else if (kmer1->w < kmer2->w)
		{
			return 1;
		}
		else if (kmer1->w > kmer2->w)
		{
			return -1;
		}
		else
#endif
	return 0;
}
