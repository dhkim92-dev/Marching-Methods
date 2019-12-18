
#define ESP 0.000001f
const sampler_t sp = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE|CLK_FILTER_NEAREST;

__constant int4 edge_types[] = {
	(int4)(0,0,0,0),  (int4)(0,0,2,0),	(int4)(0,2,0,0),  (int4)(2,0,0,0),
	(int4)(-1,1,1,0), (int4)(1,-1,1,0),	(int4)(1,1,-1,0), (int4)(1,1,1,0)
};


//(offset, edge_type)
__constant char2 MT_TABLE[][4] = {
	{(char2)(0, 0),  (char2)(0, 0),  (char2)(0, 0),  (char2)(0, 0)},
	{(char2)(0, 4),  (char2)(0, 1),  (char2)(0, 7),  (char2)(0, 0)},
	{(char2)(1, 6),  (char2)(4, 3),  (char2)(0, 7),  (char2)(0, 0)},
	{(char2)(0, 1),  (char2)(1, 6),  (char2)(0, 4),  (char2)(4, 3)},
	{(char2)(0, 1),  (char2)(4, 5),  (char2)(1, 6),  (char2)(0, 0)},
	{(char2)(1, 6),  (char2)(0, 7),  (char2)(4, 5),  (char2)(0, 4)},
	{(char2)(0, 1),  (char2)(4, 5),  (char2)(0, 7),  (char2)(4, 3)},
	{(char2)(4, 5),  (char2)(4, 3),  (char2)(0, 4),  (char2)(0, 0)},
	{(char2)(4, 5),  (char2)(0, 4),  (char2)(4, 3),  (char2)(0, 0)},
	{(char2)(0, 7),  (char2)(4, 3),  (char2)(0, 1),  (char2)(4, 5)},
	{(char2)(1, 6),  (char2)(4, 5),  (char2)(0, 7),  (char2)(0, 4)},
	{(char2)(1, 6),  (char2)(4, 5),  (char2)(0, 1),  (char2)(0, 0)},
	{(char2)(0, 1),  (char2)(0, 4),  (char2)(1, 6),  (char2)(4, 3)},
	{(char2)(4, 3),  (char2)(1, 6),  (char2)(0, 7),  (char2)(0, 0)},
	{(char2)(0, 1),  (char2)(0, 4),  (char2)(0, 7),  (char2)(0, 0)},
	{(char2)(0, 0),  (char2)(0, 0),  (char2)(0, 0),  (char2)(0, 0)}
};

__constant uint4 Permutation[6] = {
	(uint4)(0,1,2,3),(uint4)(0,2,1,3),(uint4)(1,0,2,3), 
	(uint4)(1,2,0,3),(uint4)(2,0,1,3),(uint4)(2,1,0,3),
};

__inline int flatten_bcc(int4 x, int4 dim)
{
	return ((x.z*dim.y+x.y)*dim.x+x.x-((1+dim.y)*(x.z&0x01)*dim.x));
}

__inline int4 idx2bcc(int idx, int4 dim)
{
	int4 bcc = (int4)(0,0,0,idx&0x01);
	idx >>= 1;
	bcc.x = idx%dim.x;
	idx /= dim.x;
	bcc.y = idx%dim.y;
	bcc.z = idx/dim.y;
	bcc.xyz = bcc.xyz*2+bcc.w;
	bcc.w = 0;
	return bcc;
}

__kernel void EdgeTest(__global int* EdgeType, __read_only image3d_t vol, float isovalue, int4 dim)
{
	int id = get_global_id(0); // BCC Lattice ID
	int4 bcc = idx2bcc(id, dim); 
	int o = (read_imagef(vol, sp, bcc).x+ESP)<isovalue;
	EdgeType[id*7+0] = (o != ((read_imagef(vol, sp, bcc+(int4)(0,0,2,0)).x+ESP)<isovalue));
	EdgeType[id*7+1] = (o != ((read_imagef(vol, sp, bcc+(int4)(0,2,0,0)).x+ESP)<isovalue));
	EdgeType[id*7+2] = (o != ((read_imagef(vol, sp, bcc+(int4)(2,0,0,0)).x+ESP)<isovalue));
	EdgeType[id*7+3] = (o != ((read_imagef(vol, sp, bcc+(int4)(-1,1,1,0)).x+ESP)<isovalue));
	EdgeType[id*7+4] = (o != ((read_imagef(vol, sp, bcc+(int4)(1,-1,1,0)).x+ESP)<isovalue));
	EdgeType[id*7+5] = (o != ((read_imagef(vol, sp, bcc+(int4)(1,1,-1,0)).x+ESP)<isovalue));
	EdgeType[id*7+6] = (o != ((read_imagef(vol, sp, bcc+(int4)(1,1,1,0)).x+ESP)<isovalue));
}

__kernel void VCompact(__global int4* vtx, __global int* EdgeType, __global int* EdgeCounter)
{
	int idx = get_global_id(0);
	if(EdgeType[idx]==1) vtx[EdgeCounter[idx]-1].x = idx;
}

__kernel void GenVertex(__global float4* vtx, __read_only image3d_t vol, __global int4* ivtx, float isovalue,  int4 dim, int nr_vtx)
{
	int idx = get_global_id(0);
	if(idx>=nr_vtx) return;
	int vidx = ivtx[idx].x;
	int4 edge_type = edge_types[vidx%7+1];

	int4 bcc = idx2bcc(vidx/7, dim);
	float v1 = read_imagef(vol, sp, bcc).x;
	float v2 = read_imagef(vol, sp, bcc+edge_type).x;
	vtx[idx] = convert_float4(bcc)+convert_float4(edge_type)*(isovalue-v1)/(v2-v1);
}

#define ctype(x) x.s0|x.s1|x.s2|x.s3
__kernel void CellTest(__global int* TriCounter, __global int* CellType, __read_only image3d_t vol, float isovalue, int4 dim)
{
	int id = get_global_id(0);
	int4 bcc = idx2bcc(id, dim);

	int8 val = (int8)(	((read_imagef(vol,sp,bcc).x+ESP) < isovalue), 					  //0
						((read_imagef(vol,sp,bcc+(int4)(1,1,1,0)).x+ESP)  < isovalue)<<1, //1
						((read_imagef(vol,sp,bcc+(int4)(0,0,2,0)).x+ESP) < isovalue)<<2,  //2
						((read_imagef(vol,sp,bcc+(int4)(0,2,0,0)).x+ESP) < isovalue)<<2,  //3
						((read_imagef(vol,sp,bcc+(int4)(2,0,0,0)).x+ESP) < isovalue)<<2,  //4
						((read_imagef(vol,sp,bcc+(int4)(-1,1,1,0)).x+ESP) < isovalue)<<3, //5
						((read_imagef(vol,sp,bcc+(int4)(1,-1,1,0)).x+ESP) < isovalue)<<3, //6
						((read_imagef(vol,sp,bcc+(int4)(1,1,-1,0)).x+ESP) < isovalue)<<3);//7

	int3 ct1 = (int3)(ctype(val.s0125), ctype(val.s0135), ctype(val.s0126));
	int3 ct2 = (int3)(ctype(val.s0137), ctype(val.s0146), ctype(val.s0147));
	vstore3(ct1, id*2, CellType);
	vstore3(ct2, id*2+1, CellType);

	int tag[] = {0,1,2,1,0};
	int3 tc1 = popcount(ct1);
	int3 tc2 = popcount(ct2);
	tc1 = (int3)(tag[tc1.x], tag[tc1.y], tag[tc1.z]);
	tc2 = (int3)(tag[tc2.x], tag[tc2.y], tag[tc2.z]);
	vstore3(tc1, id*2, TriCounter);
	vstore3(tc2, id*2+1, TriCounter);
}

__kernel void ICompact(__global int4* Index, __global int* TriCounter, __global int* CellType)
{
	int id = get_global_id(0);
	int tag[] = {0,1,2,1,0};
	int cell_type = CellType[id];
	int nTet = tag[popcount(cell_type)];

	if(nTet>0) {
		int tidx = TriCounter[id]-nTet;
		Index[tidx].xyz = (int3)(id, 0, cell_type);
		if(nTet==2)	Index[tidx+1].xyz = (int3)(id, 1, cell_type);
	}
}

__inline int get_vtx_idx(int tri_type, int bid, uint4 pm, int4 bcc, int4 dim)
{
	char2 eid = MT_TABLE[tri_type][bid];
	int8 v1 = shuffle((int8)(edge_types[eid.x], edge_types[eid.y]),(uint8)(pm, pm+4));
	int idx = ((flatten_bcc(bcc+v1.s0123, dim)*7) + ((v1.s4*3+v1.s5*2+v1.s6)>>1)+((v1.s4&0x01)<<2))-1;
	return idx;
}

__kernel void GenIndex(__global int4* idx, __read_only image3d_t vol, __global int* edgeTable, int4 dim, float isovalue, int nr_idx)
{
	int id = get_global_id(0);
	if(id>=nr_idx) return;
	int3 bid = idx[id].s012;
	int4 bcc = idx2bcc(bid.s0/6, dim);
	
	int4 data;
	uint4 pm = Permutation[bid.s0%6];
	data.s0 = get_vtx_idx(bid.s2, bid.s1*3, pm, bcc, dim);
	data.s1 = get_vtx_idx(bid.s2, bid.s1+1, pm, bcc, dim);
	data.s2 = get_vtx_idx(bid.s2, 2-bid.s1, pm, bcc, dim);

	data.s0 = edgeTable[data.s0] - 1;
	data.s1 = edgeTable[data.s1] - 1;
	data.s2 = edgeTable[data.s2] - 1;

	idx[id] = shuffle((int4)(data.s012,0), pm);
}

__inline float4 lattice_normal(__read_only image3d_t vol, int4 bcc)
{
	float dooo = read_imagef(vol, sp, bcc+(int4)( 1, 1, 1, 0)).x-read_imagef(vol, sp, bcc+(int4)(-1,-1,-1, 0)).x;
	float doom = read_imagef(vol, sp, bcc+(int4)( 1, 1,-1, 0)).x-read_imagef(vol, sp, bcc+(int4)(-1,-1, 1, 0)).x;
	float domo = read_imagef(vol, sp, bcc+(int4)( 1,-1, 1, 0)).x-read_imagef(vol, sp, bcc+(int4)(-1, 1,-1, 0)).x;
	float dmoo = read_imagef(vol, sp, bcc+(int4)(-1, 1, 1, 0)).x-read_imagef(vol, sp, bcc+(int4)( 1,-1,-1, 0)).x;
	return normalize((float4)(dooo+doom+domo-dmoo, dooo+doom-domo+dmoo, dooo-doom+domo+dmoo, 0));
}

__kernel void GenNormal(__global float4* N,  __read_only image3d_t vol, __global float4* vtx, int4 dim, int nr_vtx)
{
	int i = get_global_id(0);
	if(i>=nr_vtx) return;
	float4 v = vtx[i];
	float4 base = trunc(v);
	int4 var = -(v!=base);
	int4 mo = convert_int4(base)&0x01;
	int isbcc = (mo.x+mo.y+mo.z)%3 != 0;
	int isAxis = (var.x+var.y+var.z) != 3;

	int4 p1;
	int4 p2;
	float t;
	if(isAxis) {
		p1 = convert_int4(base)-isbcc*var;
		p2 = p1 + 2*var;
		t = distance(v.xyz,convert_float3(p1.xyz))/2.f;
	} else {		
		mo = (int4)(mo.y==mo.z, mo.x==mo.z, mo.x==mo.y, 1);
		p1 = convert_int4(base)+mo;
		p2 = 1-mo*2+p1;
		t = (v.x-p1.x) / (p2.x-p1.x);
	}

	float4 n1 = lattice_normal(vol, p1);
	float4 n2 = lattice_normal(vol, p2);

	N[i] = normalize(mix(n1, n2, t));
}
