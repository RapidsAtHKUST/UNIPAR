/*
 * share.cu
 *
 *  Created on: 2017-9-10
 *      Author: qiushuang
 */

#include "include/dbgraph.h"
#include "include/comm.h"
#include "include/graph.h"

extern voff_t max_ss;
extern uint gmax_lsize;
extern uint gmax_jsize;
extern int cutoff;
void * buffer[NUM_OF_DEVICES];
static ull size = 0;

extern "C"
{
void init_write_buffer (int num_of_devices)
{
	size = (gmax_jsize * EDGE_DIC_SIZE * sizeof(vid_t)) > (gmax_lsize * sizeof(vid_t) * 2)? \
			(gmax_jsize * EDGE_DIC_SIZE * sizeof(vid_t)) : (gmax_lsize * sizeof(vid_t) * 2);
	int i;
	for (i=0; i<num_of_devices; i++)
	{
		buffer[i] = (void *) malloc (size);
		CHECK_PTR_RETURN (buffer[i], "malloc write buffer for gpu %d error!\n", i);
	}
}

void finalize_write_buffer (int num_of_devices)
{
	int i;
	for (i=0; i<num_of_devices; i++)
	{
		free (buffer[i]);
	}
}

void output_vertices_gpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did, d_jvs_t * djs, d_lvs_t * dls, subgraph_t * subgraph)
{
	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int np_per_node = (total_num_partitions + world_size - 1)/world_size;

	int start_partition_id = np_per_node*world_rank;
	msp_id_t partition_id = pid - start_partition_id;

	djs[pid - start_partition_id].size = jsize;
	int t;
	for (t=0; t<EDGE_DIC_SIZE; t++)
	{
		djs[partition_id].nbs[t] = (vid_t *)malloc(sizeof(vid_t) * jsize);
		CHECK_PTR_RETURN (djs[partition_id].nbs[t], "malloc djs[%d].nbs[%d] array error!\n", pid, t);
		CUDA_CHECK_RETURN(cudaMemcpy (djs[partition_id].nbs[t], dbm->djs.nbs[t], sizeof(vid_t) * jsize, cudaMemcpyDeviceToHost));
	}

	dls[partition_id].esize = lsize;
	dls[partition_id].asize = 0;
	dls[partition_id].id = NULL;
	dls[partition_id].posts = (vid_t*) malloc (sizeof(vid_t) * lsize);
	CHECK_PTR_RETURN (dls[partition_id].posts, "malloc posts for partition %d error!\n", pid);
	dls[partition_id].pres = (vid_t *) malloc (sizeof(vid_t) * lsize);
	CHECK_PTR_RETURN (dls[partition_id].pres, "malloc pres for partition %d error!\n", pid);
	CUDA_CHECK_RETURN(cudaMemcpy (dls[partition_id].posts, dbm->dls.posts, sizeof(vid_t) * lsize, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy (dls[partition_id].pres, dbm->dls.pres, sizeof(vid_t) * lsize, cudaMemcpyDeviceToHost));

	(subgraph->subgraphs)[partition_id].id = pid;
	(subgraph->subgraphs)[partition_id].size = lsize;
}

void write_kmers_edges_gpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did)
{
	FILE * file;
	FILE * kfile;
	FILE * efile;
	char filename[FILENAME_LENGTH];
	char fname[FILENAME_LENGTH];
	char efname[FILENAME_LENGTH];

	sprintf (filename, "%s/ledge%d_%d", mst->file_dir, pid, total_num_partitions);
	sprintf (fname, "%s/jkmer%d_%d", mst->file_dir, pid, total_num_partitions);
	sprintf (efname, "%s/jedge%d_%d", mst->file_dir, pid, total_num_partitions);

	if ((file = fopen (filename, "w")) == NULL)
	{
		printf ("OPEN subgraph %d linear subgraph kmer file for device %d error\n", pid, did);
	}
	if ((kfile = fopen (fname, "w")) == NULL)
	{
		printf ("OPEN subgraph %d junction kmer file for device %d error\n", pid, did);
		exit(0);
	}
	if ((efile = fopen (efname, "w")) == NULL)
	{
		printf ("OPEN subgraph %d junction edge file for device %d error\n", pid, did);
		exit(0);
	}

	void * buf = buffer[did];
	size_t cpy_size = 0;
	CUDA_CHECK_RETURN (cudaMemcpy ((char*)buf + cpy_size, dbm->dls.pre_edges, sizeof(edge_type) * lsize, cudaMemcpyDeviceToHost));
	cpy_size += sizeof(edge_type) * lsize;
	CUDA_CHECK_RETURN (cudaMemcpy ((char*)buf + cpy_size, dbm->dls.post_edges, sizeof(edge_type) * lsize, cudaMemcpyDeviceToHost));
	cpy_size += sizeof(edge_type) * lsize;
	if (cpy_size > size)
	{
		printf ("initiated write buffer error in size! real size = %lu, size = %lu\n", cpy_size, size);
		exit(0);
	}
	fwrite (buf, 1, cpy_size, file);

	CUDA_CHECK_RETURN (cudaMemcpy ((char*)buf, dbm->djs.kmers, sizeof(kmer_t) * jsize, cudaMemcpyDeviceToHost));
	fwrite (buf, sizeof(kmer_t), jsize, kfile);

	CUDA_CHECK_RETURN (cudaMemcpy ((char*)buf, dbm->djs.edges, sizeof(ull) * jsize, cudaMemcpyDeviceToHost));
	fwrite (buf, sizeof(ull), jsize, efile);

	fclose (file);
	fclose (kfile);
	fclose (efile);
}

void write_junctions_gpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did)
{
	FILE * file;
	FILE * kfile;
	FILE * efile;
	char filename[FILENAME_LENGTH];
	char fname[FILENAME_LENGTH];
	char efname[FILENAME_LENGTH];
	sprintf (filename, "%s/jv%d_%d", mst->file_dir, pid, total_num_partitions);
	sprintf (fname, "%s/jkmer%d_%d", mst->file_dir, pid, total_num_partitions);
	sprintf (efname, "%s/jedge%d_%d", mst->file_dir, pid, total_num_partitions);

	if ((file = fopen (filename, "w")) == NULL)
	{
		printf ("OPEN subgraph %d junction subgraph file for device %d error\n", pid, did);
		exit(0);
	}
	if ((kfile = fopen (fname, "w")) == NULL)
	{
		printf ("OPEN subgraph %d junction kmer file for device %d error\n", pid, did);
		exit(0);
	}
	if ((efile = fopen (efname, "w")) == NULL)
	{
		printf ("OPEN subgraph %d junction edge file for device %d error\n", pid, did);
		exit(0);
	}

	vid_t * nbs[EDGE_DIC_SIZE];
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		nbs[i] = dbm->djs.nbs[i];
	}

	ull cpy_size = 0;
	void * buf = buffer[did];
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		CUDA_CHECK_RETURN(cudaMemcpy ((char*)buf + cpy_size, nbs[i], sizeof(vid_t) * jsize, cudaMemcpyDeviceToHost));
		cpy_size += sizeof(vid_t) * jsize;
	}
	if (cpy_size > size)
	{
		printf ("initiated write buffer error in size! real size = %lu, size = %lu\n", cpy_size, size);
		exit(0);
	}
	fwrite (&jsize, sizeof(voff_t), 1, file);
	fwrite (buf, 1, cpy_size, file);

	if (sizeof(kmer_t) * jsize > size)
	{
		printf ("initiated write buffer error in size! real size = %lu, size = %lu\n", sizeof(kmer_t) * jsize, size);
		exit(0);
	}
	CUDA_CHECK_RETURN (cudaMemcpy ((char*)buf, dbm->djs.kmers, sizeof(kmer_t) * jsize, cudaMemcpyDeviceToHost));
	fwrite (buf, sizeof(kmer_t), jsize, kfile);

	CUDA_CHECK_RETURN (cudaMemcpy ((char*)buf, dbm->djs.edges, sizeof(ull) * jsize, cudaMemcpyDeviceToHost));
	fwrite (buf, sizeof(ull), jsize, efile);

	fclose (file);
	fclose (kfile);
	fclose (efile);
}

void write_linear_vertices_gpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did)
{
	FILE * file;
	FILE * kfile;
	char filename[FILENAME_LENGTH];
	char fname[FILENAME_LENGTH];
	sprintf (filename, "%s/lv%d_%d", mst->file_dir, pid, total_num_partitions);
	sprintf (fname, "%s/ledge%d_%d", mst->file_dir, pid, total_num_partitions);

	if ((file = fopen (filename, "w")) == NULL)
	{
		printf ("Open subgraph %d linear subgraph file for device %d error!\n", pid, did);
	}
	if ((kfile = fopen (fname, "w")) == NULL)
	{
		printf ("OPEN subgraph %d linear subgraph edge file for device %d error\n", pid, did);
	}

	vid_t * posts = dbm->dls.posts;
	vid_t * pres = dbm->dls.pres;
	ull cpy_size = 0;
	void * buf = buffer[did];
	CUDA_CHECK_RETURN(cudaMemcpy ((char*)buf + cpy_size, posts, sizeof(vid_t) * lsize, cudaMemcpyDeviceToHost));
	cpy_size += sizeof(vid_t) * lsize;
	CUDA_CHECK_RETURN(cudaMemcpy ((char*)buf + cpy_size, pres, sizeof(vid_t) * lsize, cudaMemcpyDeviceToHost));
	cpy_size += sizeof(vid_t) * lsize;

	if (cpy_size > size)
	{
		printf ("initiated write buffer error in size! real size = %lu, size = %lu\n", cpy_size, size);
		exit(0);
	}
	fwrite (&lsize, sizeof(voff_t), 1, file);
	fwrite (buf, 1, cpy_size, file);

	cpy_size = 0;
	CUDA_CHECK_RETURN (cudaMemcpy ((char*)buf + cpy_size, dbm->dls.pre_edges, sizeof(edge_type) * lsize, cudaMemcpyDeviceToHost));
	cpy_size += sizeof(edge_type) * lsize;
	CUDA_CHECK_RETURN (cudaMemcpy ((char*)buf + cpy_size, dbm->dls.post_edges, sizeof(edge_type) * lsize, cudaMemcpyDeviceToHost));
	cpy_size += sizeof(edge_type) * lsize;
	if (cpy_size > size)
	{
		printf ("initiated write buffer error in size! real size = %lu, size = %lu\n", cpy_size, size);
		exit(0);
	}
	fwrite (buf, 1, cpy_size, kfile);

	fclose (file);
	fclose (kfile);
}


void write_ids_gpu (dbmeta_t * dbm, master_t * mst, uint total_num_partitions, int did)
{
	FILE * file;
	char filename[100];
	sprintf (filename, "%s/ids_%d", mst->file_dir, total_num_partitions);
	if ((file = fopen(filename, "w")) == NULL)
	{
		printf ("open id file error!\n");
	}
	vid_t * vs = (vid_t *) malloc (sizeof(vid_t) * max_ss);
	CHECK_PTR_RETURN (vs, "malloc write_ids_gpu buffer error!\n");
	voff_t * index_offset = mst->index_offset[did];
	uint i;
	uint count = 0;
	for (i=0; i<total_num_partitions; i++)
	{
		voff_t size = index_offset[i+1] - index_offset[i];
		CUDA_CHECK_RETURN (cudaMemcpy(vs, dbm->sorted_vids + index_offset[i], sizeof(vid_t) * size, cudaMemcpyDeviceToHost));
		int j;
		for (j=0; j<size; j++)
		{
			if (vs[j] == 0)
			{
				count++;
				continue;
			}
			fprintf (file, "%u\n", vs[j]);
		}
	}
	free (vs);
	fclose (file);
	printf ("number of vertices filtered: %u\n", count);
}

void write_contigs_gpu (meta_t * dm, master_t * mst, int did, voff_t max_num, size_t max_total_len, int k)
{
	FILE * file;
	char filename[FILENAME_LENGTH];
	int world_rank = mst->world_rank;
	sprintf (filename, "%s/contig%d_%d", mst->file_dir, did, world_rank);

	if ((file = fopen (filename, "w")) == NULL)
	{
		printf ("Open contig file error with CPU %d!\n", did);
	}

	uint count_empty=0;
	size_t * ulens = (size_t *) malloc (sizeof(size_t) * max_num);
	CHECK_PTR_RETURN (ulens, "malloc ulens buffer for writing contigs with device %d error!\n", did);
	char * unitigs = (char *) malloc (sizeof(char) * max_total_len);
	CHECK_PTR_RETURN (unitigs, "malloc unitig buffer for writing contigs with device %d error!\n", did);
	voff_t * local_offs = (voff_t *) malloc (sizeof(voff_t) * max_num);
	CHECK_PTR_RETURN (local_offs, "malloc local offsets for junction neighbors with device %d error\n", did);
	int i;
	int num_of_partitions = mst->num_partitions[did+1] - mst->num_partitions[did];
	size_t size_offset = 0;
	for (i=0; i<num_of_partitions; i++)
	{
		voff_t j;
		voff_t num_unitigs = mst->jnb_index_offset[did][i+1] - mst->jnb_index_offset[did][i];
		CUDA_CHECK_RETURN (cudaMemcpy(ulens, dm->junct.ulens + mst->jnb_index_offset[did][i], sizeof(size_t) * num_unitigs, cudaMemcpyDeviceToHost));
		voff_t total_lens = ulens[num_unitigs-1];
		CUDA_CHECK_RETURN (cudaMemcpy(unitigs, dm->junct.unitigs + size_offset, sizeof(char) * total_lens, cudaMemcpyDeviceToHost));
		voff_t num_js = mst->jindex_offset[did][i+1] - mst->jindex_offset[did][i];
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset + i];
		if (num_js != mst->jid_offset[pid])
		{
			printf ("Error checked in number of junctions!\n");
		    exit(0);
		}
		CUDA_CHECK_RETURN (cudaMemcpy(local_offs, dm->junct.offs + mst->jindex_offset[did][i] + i, sizeof(voff_t) * (num_js+1), cudaMemcpyDeviceToHost));
		size_t * local_ulens = ulens;
		char * ptr = unitigs;
		for (j=0; j<num_js; j++)
		{
			voff_t nb_off = local_offs[j];
			voff_t num_nbs = local_offs[j+1] - local_offs[j];
			int num;
			for (num=0; num<num_nbs; num++)
			{
				size_t len = (nb_off+num)==0 ? local_ulens[nb_off+num] : (local_ulens[nb_off+num]-local_ulens[nb_off+num-1]);
				if (len==0)
					continue;
				vid_t contig_id = nb_off+num;
				if (ptr[0] == '\0' | ptr[k] == '\0')
				{
//					printf("Error in unitigs!\n");
				}
				if (len>k+1 && ptr[k+1] == '\0')
				{
					count_empty++;
				}
				fprintf (file, ">%lu length %lu cvg_%d_tip_0\n", contig_id, len, cutoff);
				size_t l;
				for (l=0; l<len; l++)
					fprintf(file, "%c", ptr[l]);
				fprintf (file, "\n");
				ptr = ptr + len;
			}
		}
		size_offset += total_lens;
	}

	printf ("!!!!!!!!!!!!!total number of dropped contigs: %u\n", count_empty);
	free (ulens);
	free (unitigs);
	fclose(file);
}
}
