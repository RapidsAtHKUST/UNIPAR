/*
 * graph.h
 *
 *  Created on: 2018-9-24
 *      Author: qiushuang
 */

#ifndef GRAPH_H_
#define GRAPH_H_

typedef uint vid_t; // vertex id
typedef uint voff_t; // path offset, index offset

typedef struct d_jvertices
{
	vid_t * id;
	vid_t * nbs[EDGE_DIC_SIZE];
	kmer_t * kmers;
	ull * edges;
	uint size;
	voff_t * csr_offs; // csr offset array for all processors in a computer node, for contig gather
	vid_t * csr_nbs; // csr neighbor array for all processors in a computer node, for contig gather
	voff_t * csr_offs_offs; // prefix sum of sizes of csr offset arrays of subgraphs, for contig gather
	voff_t * csr_nbs_offs; // prefix sum of sizes of csr neighbor arrays of subgraphs, for contig gather
} d_jvs_t; // junction vertex structure, one such meta structure one computer node

typedef struct d_lvertices
{
	vid_t * id; // set to null in this implementation
	vid_t * pres;
	vid_t * posts;
	kmer_t * kmers;
	edge_type * pre_edges;
	edge_type * post_edges;
	voff_t esize;
	voff_t asize; // set to zero in this implementation
} d_lvs_t; // linear vertex structure, one such meta structure one computer node

typedef struct vertex
{
	vid_t nbs[EDGE_DIC_SIZE];
	kmer_t kmer;
	ull edge;
	vid_t vid; // used in assigning ids of vertices
} vertex_t;

typedef struct vertices
{
	kmer_t * kmer; // put this field in the first
	ull * edge; //multiplicity of edges for both kmer and its reverse, be careful if the multiplicity exceeds 255!!!
	vid_t * vid;
} vertices_t; // this vertices_t is not the same as defined in dbgraph.h of hash table construction, but a converted to pointer of arrays version

/*edge: bidirected */
typedef struct edge
{
	vid_t * pre; // edge of the reverse of each linear vertex
	vid_t * post; // edge of each linear vertex
	voff_t * fwd; // length of each forward path
	voff_t * bwd; // length of each backward path (forward path of the reverse)
	vid_t * fjid; //junction id for forward path
	vid_t * bjid; //junction id for backward path
	kmer_t * kmers; //kmer values of linear vertices
	edge_type * pre_edges; // edge character of the reverse of each linear vertex
	edge_type * post_edges; // edge character of each linear vertex
} edge_t; // linear vertex info for graph traversal

typedef struct csr
{
	voff_t * offs; // offset array of neighbors of junctions
	vid_t * nbs; // neighbor array of junctions
	kmer_t * kmers; // kmer values of junctions
	ull * edges; // edges of junctions
	size_t * ulens; // length of unitigs for each junction branch
	size_t * unitig_offs; // unitig offsets of partitions in a processor
	char * unitigs; // unitigs with two junction end points
//	uint size;
} csr_t; // for each procesor, junction vertex info for graph traversal

typedef struct subsize
{
	uint size; // size of a subgraph partition
	msp_id_t id; // partition id
} subsize_t;

typedef struct subgraph
{
	subsize_t * subgraphs; // subgraphs resides in a computer node
	vid_t * num_jnbs; // number of neighbors of junctions for each subgraph
	int num_of_subs; // total number of subgraphs in a computer node
	size_t total_graph_size; // total number of vertices of subgraphs in this computer node
} subgraph_t;

#endif /* GRAPH_H_ */
