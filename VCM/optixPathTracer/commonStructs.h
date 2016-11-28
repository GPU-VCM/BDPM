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
        const float3 &aP0,
        const float3 &aP1,
        const float3 &aP2,
		const float3 &emission)
    {
        p0 = aP0;
        e1 = aP1 - aP0;
        e2 = aP2 - aP0;
		mIntensity = emission;

        const float3 normal = cross(e1, e2);
        const float len    = length(normal);
        mInvArea     = 2.f / len;
        mFrame.SetFromZ(normal);
		IsDelta = false;
		IsFinite = true;
		lightType = 0;
    }

	__host__ __device__ __forceinline__ void SetupBackgroundLight(
		const float3 &emission)
    {
		mIntensity = emission;
		IsDelta = IsFinite = false;
		lightType = 1;
    }

	__host__ __device__ __forceinline__ void SetupDirectionalLight(
		const float3 &direction,
		const float3 &emission)
    {
		p0 = direction;
		mFrame.SetFromZ(p0);
		mIntensity = emission;
		IsDelta = true;
		IsFinite = false;
		lightType = 2;
    }

	// Area light/Directional light attributes...

    float3 p0, e1, e2;
    Frame mFrame;
    float mInvArea;

	// Common attributes...

	float3 mIntensity;
	bool IsDelta, IsFinite;
	int lightType; // 0 = AreaLight, 1 = Background, 2 = Directional
};

class TriangleMaterial
{
public:
	// diffuse is simply added to the others
    float3 mDiffuseReflectance;

	// Is this triangle an emitter?
    bool isEmitter;
    
	// Phong is simply added to the others
    float3 mPhongReflectance;
    float mPhongExponent;

    // mirror can be either simply added, or mixed using Fresnel term
    // this is governed by mIOR, if it is >= 0, fresnel is used, otherwise
    // it is not
    float3 mMirrorReflectance;

    // When mIOR >= 0, we also transmit (just clear glass)
    float mIOR;

	__host__ void Reset()
    {
        mDiffuseReflectance = mPhongReflectance = mMirrorReflectance = make_float3(0, 0, 0);
        mPhongExponent      = 1.f;
        mIOR = -1.f;
		isEmitter = false;
    }
};