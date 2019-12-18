__kernel void genVol(write_only image3d_t vol)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int mx = get_global_size(0)/2;
	int my = get_global_size(1)/2;
	int mz = get_global_size(2)/2;

	float dist = distance((float4)(x,y,z,1), (float4)(mx,my,mz,1))-mx/3.f;


	write_imagef(vol, (int4)(x,y,z,1), (float4)(dist,0,0,1));
}

__inline float R(float x, float y, float z, float alpha, float fm)
{
	float r = 1.f-sinpi(z*0.5f) + alpha*(1.f+cospi(2.f*fm*cospi(sqrt(x*x+y*y)*0.5f)));
	return r / (2.f*(1.f+alpha));
}

__kernel void genML(write_only image3d_t vol, float alpha, float fm)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);

	int nx = get_global_size(0);
	int ny = get_global_size(1);
	int nz = get_global_size(2);

	float x = (float)(i)/(float)(nx-1)*2.f-1.f;
	float y = (float)(j)/(float)(ny-1)*2.f-1.f;
	float z = (float)(k)/(float)(nz-1)*2.f-1.f;

	float r = R(x,y,z,alpha,fm);
	write_imagef(vol, (int4)(i,j,k,0), (float4)(r,0,0,1));
}

//-4 < x,y < 4, -7 < z < 9
__kernel void genPeaks(write_only image3d_t vol)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);

	int nx = get_global_size(0);
	int ny = get_global_size(1);
	int nz = get_global_size(2);

	float x = (float)(i)/(float)(nx-1)*8.f-4.f;
	float y = (float)(j)/(float)(ny-1)*8.f-4.f;
	float z = (float)(k)/(float)(nz-1)*16.f-7.f;

	float f = 3*(1-x)*(1-x)*exp(-x*x-(y+1)*(y+1))
		-(2*x - x*x*x - y*y*y*y*y)*exp(-x*x-y*y)
		-exp(-(x+1)*(x+1)-y*y)/3.0f - z;

	write_imagef(vol, (int4)(i,j,k,0), (float4)(f, 0, 0, 1));
}

// -8 < x < 8, -4 < y < 4, -4 < z < 4
// 2 : 1 : 1
__kernel void genGenus3(write_only image3d_t vol)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);

	int nx = get_global_size(0);
	int ny = get_global_size(1);
	int nz = get_global_size(2);

	float x = (float)(i)/(float)(nx-1)*16.f-8.f;
	float y = (float)(j)/(float)(ny-1)*8.f-4.f;
	float z = (float)(k)/(float)(nz-1)*8.f-4.f;

	float f = (1 - (x/6.f) * (x/6.f) - (y/3.5f)*(y/3.5f)) *
		((x-3.9f)*(x-3.9f) + y*y - 1.44) *
		(x*x + y*y - 1.44) *
		((x+3.9f)*(x+3.9f) + y*y - 1.44) - 256*z*z;

	write_imagef(vol, (int4)(i,j,k,0), (float4)(f, 0, 0, 1));
}