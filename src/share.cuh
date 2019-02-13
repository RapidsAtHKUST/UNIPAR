/*
 * share.cuh
 *
 *  Created on: 2017-8-23
 *      Author: qiushuang
 */

#ifndef SHARE_CUH
#define SHARE_CUH

#ifdef LITTLE_ENDIAN
__constant__ static const ull masktable[8] = { 0xff, 0xff00, 0xff0000, 0xff000000, 0xff00000000, 0xff0000000000, 0xff000000000000, 0xff00000000000000 };
#else
__constant__ static const ull masktable[8] = { 0xff00000000000000, 0xff000000000000, 0xff0000000000, 0xff00000000, 0xff000000, 0xff0000, 0xff00, 0xff };
#endif

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





#endif /* SHARE_CUH */
