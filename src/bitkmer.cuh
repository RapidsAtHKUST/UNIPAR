/*
 * bitkmer.cuh
 *
 *  Created on: 2015-12-23
 *      Author: qiushuang
 */

#ifndef BITKMER_CUH_
#define BITKMER_CUH_

#include "../include/msp.h"

#ifdef LONG_KMER
__device__ static void kmer_32bit_left_shift (kmer_t * kmer, int n)
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

__device__ static void kmer_64bit_left_shift (kmer_t * kmer, int n)
{
	if (n == 0) return;
	kmer->x = (kmer->y << (n - 32)) | (kmer->z >> (64 - n));
	kmer->y = (kmer->z << (n - 32)) | (kmer->w >> (64 - n));
	kmer->z = (kmer->w << (n - 32));
	kmer->w = 0;
}

__device__ static void kmer_96bit_left_shift (kmer_t * kmer, int n)
{
	if (n == 0) return;
	kmer->x = (kmer->z << (n - 64)) | (kmer->w >> (96 - n));
	kmer->y = (kmer->w << (n - 64));
	kmer->z = kmer->w = 0;
}

__device__ static void kmer_128bit_left_shift (kmer_t * kmer, int n)
{
	if (n == 0) return;
	kmer->x = (kmer->w << (n - 96));
	kmer->y = kmer->z = kmer->w = 0;
}

#else

__device__ static void kmer_32bit_left_shift (kmer_t * kmer, int n)
{
	if (n == 0) return;
	kmer->x <<= n;
	kmer->x |= kmer->y >> (32 - n);
	kmer->y <<= n;
}

__device__ static void kmer_64bit_left_shift (kmer_t * kmer, int n)
{
	if (n == 0) return;
	kmer->x = (kmer->y << (n - 32));
	kmer->y = 0;
}

#endif

#ifdef LONG_KMER
__constant__ static thread_function shift_dictionary[] = {kmer_32bit_left_shift, kmer_64bit_left_shift, kmer_96bit_left_shift, kmer_128bit_left_shift};
#else
__constant__ static thread_function shift_dictionary[] = {kmer_32bit_left_shift, kmer_64bit_left_shift};
#endif

/* This function is to compute the reverse complement of a kmer */
__device__ static void get_reverse (kmer_t * kmer, kmer_t * reverse)
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

__device__ static void get_reverse_kmer (kmer_t * kmer, kmer_t * reverse, int k)
{
	get_reverse (kmer, reverse);
	int table_index;
#ifdef LONG_KMER
	table_index = (128 - k * 2) / KMER_UNIT_BITS;
	shift_dictionary[table_index] (reverse, 128 - k * 2);
#else
	table_index = (64 - k * 2) / KMER_UNIT_BITS;
	shift_dictionary[table_index] (reverse, 64 - k * 2);
#endif
}

__device__ static void get_first_pstr (unit_kmer_t * kmer, minstr_t * pstr, int p)
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

__device__ static void right_shift_pstr (unit_kmer_t * kmer_cur, minstr_t * pstr, int p, int j)
{
	kmer_cur += ((j + p - 1) * 2) / KMER_UNIT_BITS; // test this with LONG_KMER
	*pstr <<= 2;
	*pstr |= ((minstr_t)((*kmer_cur << (((j + p - 1) * 2) % KMER_UNIT_BITS) ) & 0xc0000000) << (MINSTR_BIT_LENGTH - KMER_UNIT_BITS)) >> ((p * 2 - 2) % MINSTR_BIT_LENGTH);
//	*pstr |= ( ( *kmer_cur << ( ((j + p - 1) * 2) % KMER_UNIT_BITS ) ) & 0xc0000000 ) >> ((p * 2 - 2) % KMER_UNIT_BITS);
}


#define get_reverse_edge(edge, kmer) {	\
	edge = ((kmer.x >> (KMER_UNIT_BITS - 2)) ^ 0x3); }

__device__ static void
get_first_kmer (kmer_t * kmer, seq_t * spk, int k)
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

/* get minimum substring partition id with p-minimum-substring and modulo of number of partitions*/
__device__ static msp_id_t
get_partition_id (minstr_t minstr, int p, int num_of_partitions)
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


/* compare two kmers: if the first kmer is greater than the second one, return -1;
 * if the first kmer is smaller than the second one, return 1; if the two kmers are equal, return 0.
 */
__device__ static inline int
compare_2kmers (kmer_t * kmer1, kmer_t * kmer2)
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


#endif /* BITKMER_CUH_ */
