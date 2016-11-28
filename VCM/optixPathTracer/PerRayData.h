#pragma once

#include "commonStructs.h"

struct PerRayData_closestHit
{
	float dist;
	float3 normal, p[3];
	TriangleMaterial mat;
};

struct PerRayData_occlusion
{
	bool occluded;
};