#pragma once

#include "commonStructs.h"

struct PerRayData_closestHit
{
	float dist;
	float3 normal, p[3];
	BaseMaterial mat;
};

struct PerRayData_occlusion
{
	bool occluded;
};