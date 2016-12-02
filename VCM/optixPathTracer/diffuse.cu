#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optix_world.h>
#include "PerRayData.h"
#include "bsdf.h"

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_closestHit, prd, rtPayload, );
rtDeclareVariable(PerRayData_occlusion, prd_occlusion, rtPayload, );
rtDeclareVariable(BaseMaterial,          mat, , );

using namespace optix;

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, v1, attribute v1, ); 
rtDeclareVariable(float3, v2, attribute v2, ); 
rtDeclareVariable(float3, v3, attribute v3, );

RT_PROGRAM void closest_hit_diffuse()
{
	prd.mat = mat;
	prd.dist = t_hit;
	prd.p[0] = v1;
	prd.p[1] = v2;
	prd.p[2] = v3;
	prd.normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
}

RT_PROGRAM void any_hit_occlusion()
{
	prd_occlusion.occluded = true;
	rtTerminateRay();
}