#include <optix_world.h>
#include "PerRayData.h"

rtDeclareVariable(PerRayData_closestHit, prd, rtPayload, );
rtDeclareVariable(PerRayData_occlusion, prd_occlusion, rtPayload, );

RT_PROGRAM void miss()
{
	prd.dist = -1.0f;
	prd.normal = make_float3(0, 0, 0);
}

RT_PROGRAM void miss_occlusion()
{
	prd_occlusion.occluded = false;
}