

int scan1(int data, __local int* buf);

int scan1(int data, __local int* buf)
{
	int lid = get_local_id(0);
	int lsz = get_local_size(0);

	buf[lid] = 0;
	lid += lsz;
	buf[lid] = data;

	for(int offset=1; offset<lsz; offset<<=1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		int t = buf[lid] + buf[lid-offset];
		barrier(CLK_LOCAL_MEM_FENCE);
		buf[lid] = t;
	}
	return buf[lid];
}

__kernel void scan4(__global int4* dst, __global int4* src, __global int* groupsum, int limit)
{
	int id = get_global_id(0);
	if(id >= limit) return;
	
	int4 data = src[id];

	__local int buf[128];
	data.y += data.x;
	data.z += data.y;
	data.w += data.z;

	int val = scan1(data.w, buf);
	dst[id] = data + (int4)(val-data.w);

	if(id==0) groupsum[0] = 0;
	if(get_local_id(0) == get_local_size(0)-1)
		groupsum[get_group_id(0)+1] = val;
}

__kernel void uniformUpdate(__global int4* dst, __global int* groupsum)
{
	int id = get_global_id(0);
	int gid = get_group_id(0);

	if(gid!=0) {
		int4 data = dst[id];
		data += groupsum[gid];
		dst[id] = data;
	}
}

__kernel void scan_ed(__global int* dst, __global int* src, __local int* buf)
{
	int id = get_global_id(0);
	int data = src[id];
	dst[id] = scan1(data, buf);
}