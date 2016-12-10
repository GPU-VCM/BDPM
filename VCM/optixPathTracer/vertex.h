#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
using namespace optix;
struct SubPathState
{
	float3 origin;
	float3 direction;
	float3 throughput;
	uint  pathlen : 30;
	uint  isLgtFinite : 1;
	uint  specPath : 1;

	float allMIS;
	float bdptMIS;
	float elseMIS;
};


template<bool tFromLight>
struct PathVertex
{
	float3 isxPoint;
	float3 throughput;
	uint  pathlen;


	//BSDF<tFromLight> bsdf;

	float allMIS;
	float bdptMIS;
	float elseMIS;

	const float3 &GetPosition() const
	{
		return isxPoint;
	}
};

typedef PathVertex<true>  LightVertex;