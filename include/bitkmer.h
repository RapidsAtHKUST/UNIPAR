/*
 * bitkmer.h
 *
 *  Created on: 2018-12-29
 *      Author: qiushuang
 */

#ifndef BITKMER_H_
#define BITKMER_H_

void kmer_32bit_left_shift (kmer_t * kmer, int n);
void kmer_64bit_left_shift (kmer_t * kmer, int n);
#ifdef LONG_KMER
void kmer_96bit_left_shift (kmer_t * kmer, int n);
void kmer_128bit_left_shift (kmer_t * kmer, int n);
#endif

void get_reverse (kmer_t * kmer, kmer_t * reverse);
void get_reverse_kmer (kmer_t * kmer, kmer_t * reverse, int k);
void get_first_pstr (unit_kmer_t * kmer, minstr_t * pstr, int p);
void right_shift_pstr (unit_kmer_t * kmer_cur, minstr_t * pstr, int p, int j);
void get_first_kmer_cpu (kmer_t * kmer, seq_t * spk, int k);
int get_pid_from_kmer (kmer_t * kmer, int k, int p, int num_of_partitions);
uint murmur_hash2 (uint * kmer_ptr, uint seed);
uint murmur_hash3_32 (uint * kmer_ptr, uint seed);
msp_id_t get_partition_id_cpu (minstr_t minstr, int p, int num_of_partitions);
int compare_2kmers_cpu (kmer_t * kmer1, kmer_t * kmer2);

#endif /* BITKMER_H_ */
