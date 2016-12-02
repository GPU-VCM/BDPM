#pragma once

#include <optixu/optixu_vector_types.h>
#include "Frame.h"

//#define USE_CALLABLE_PROGRAM
//#define USE_CALLABLE_PROGRAM
#ifdef USE_CALLABLE_PROGRAM
#define CALLABLE RT_CALLABLE_PROGRAM
#else
#define CALLABLE __device__ __forceinline__
#endif

#define RT_FUNCTION __forceinline__ __device__ 

using namespace optix;

struct SceneSphere
{
    // Center of the scene's bounding sphere
    float3 mSceneCenter;
    // Radius of the scene's bounding sphere
    float mSceneRadius;
    // 1.f / (mSceneRadius^2)
    float mInvSceneRadiusSqr;
};

class Light
{
public:
    __host__ __device__ __forceinline__ void SetupAreaLight(
        const float3 &v0,
        const float3 &v1,
        const float3 &v2,
		const float3 &emission)
    {
        corner = v0;
        e1 = v1 - v0;
        e2 = v2 - v0;
		intensity = emission;

        const float3 normal = cross(e1, e2);
        const float len    = length(normal);
        invArea     = 2.f / len;
        frame.SetFromZ(normal);
    }

	float3 corner, e1, e2;
    Frame frame;
    float invArea;

	float3 intensity;
};

class BaseMaterial
{
public:
    float3 diffusePart;

    bool isEmitter;
    
    float3 phongPart;
    float exponent;

    float3 mirror;

    float ior;

	__host__ void Reset()
    {
		diffusePart = phongPart = mirror = make_float3(0, 0, 0);
		exponent = 1.f;
		ior = -1.f;
		isEmitter = false;
    }
};