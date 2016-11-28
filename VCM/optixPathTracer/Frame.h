#pragma once

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optix_world.h>
#include <optixu/optixu_vector_types.h>

using namespace optix;

class Frame
{
public:
	float3 m[3];

	__device__ __host__ __forceinline__ void SetFromZ(const float3 &z)
    {
        const float3 tmpZ = m[2] = normalize(z);
        const float3 tmpX = (abs(tmpZ.x) > 0.99f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
        m[1] = normalize(cross(tmpZ, tmpX));
        m[0] = cross(m[1], tmpZ);
    }

    __device__ __host__ __forceinline__ const float3 ToWorld(const float3 &a) const
    {
        return m[0] * a.x + m[1] * a.y + m[2] * a.z;
    }

    __device__ __host__ __forceinline__ const float3 ToLocal(const float3 &a) const
    {
        return make_float3(dot(a, m[0]), dot(a, m[1]), dot(a, m[2]));
    }

    __device__ __forceinline__ const float3 &Binormal() const { return m[0]; }
    __device__ __forceinline__ const float3 &Tangent () const { return m[1]; }
    __device__ __forceinline__ const float3 &Normal  () const { return m[2]; }
};