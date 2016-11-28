#pragma once

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optix_world.h>
#include <optixu/optixu_vector_types.h>
#include "utils.h"

using namespace optix;

class Camera
{
public:
	__host__ __device__ __forceinline__ Matrix4x4 Perspective(
        float aFov,
        float aNear,
        float aFar,
		float aspect)
    {
        // Camera points towards -z.  0 < near < far.
        // Matrix maps z range [-near, -far] to [-1, 1], after homogeneous division.
        float f = 1.f / (std::tan(aFov * PI_F / 360.0f));
        float d = 1.f / (aNear - aFar);

        Matrix4x4 r;
        r[0] = f/aspect;    r[1] = 0.0f; r[2] = 0.0f;                r[3] = 0.0f;
        r[4] = 0.0f; r[5] = -f;   r[6] = 0.0f;                r[7] = 0.0f;
        r[8] = 0.0f; r[9] = 0.0f; r[10] = (aNear + aFar) * d; r[11] = 2.0f * aNear * aFar * d;
        r[12] = 0.0f; r[13] = 0.0f; r[14] = -1.0f;            r[15] = 0.0f;

        return r;
    }

    __host__ __device__ __forceinline__ void Setup(
        const float3 &aPosition,
        const float3 &aForward,
        const float3 &aUp,
        const float2 &aResolution,
        float        aHorizontalFOV)
    {
        const float3 forward = normalize(aForward);
        const float3 up      = normalize(cross(aUp, -forward));
        const float3 left    = -cross(-forward, up);

        mPosition   = aPosition;
        mForward    = forward;
        mResolution = aResolution;

        const float3 pos = make_float3(
            dot(up, aPosition),
            dot(left, aPosition),
            dot(-forward, aPosition));

        Matrix4x4 worldToCamera = Matrix4x4::identity();
        worldToCamera.setRow(0, make_float4(up,       -pos.x));
        worldToCamera.setRow(1, make_float4(left,     -pos.y));
        worldToCamera.setRow(2, make_float4(-forward, -pos.z));
		
        const Matrix4x4 perspective = Perspective(aHorizontalFOV, 0.1f, 10000.f, aResolution.x / aResolution.y);
        const Matrix4x4 worldToNScreen = perspective * worldToCamera;
        const Matrix4x4 nscreenToWorld = worldToNScreen.inverse();

        mWorldToRaster  =
            Matrix4x4::scale(make_float3(aResolution.x * 0.5f, aResolution.y * 0.5f, 0)) *
            Matrix4x4::translate(make_float3(1.f, 1.f, 0)) * worldToNScreen;

        mRasterToWorld  = nscreenToWorld *
            Matrix4x4::translate(make_float3(-1.f, -1.f, 0)) *
            Matrix4x4::scale(make_float3(2.f / aResolution.x, 2.f / aResolution.y, 0));

        const float tanHalfAngle = std::tan(aHorizontalFOV * PI_F / 360.f);
        mImagePlaneDist = aResolution.x / (2.f * tanHalfAngle);
    }

    __host__ __device__ __forceinline__ int RasterToIndex(const float2 &aPixelCoords) const
    {
        return int(floorf(aPixelCoords.x) + floorf(aPixelCoords.y) * mResolution.x);
    }

    __host__ __device__ __forceinline__ float2 IndexToRaster(const int &aPixelIndex) const
    {
        const float y = floorf(aPixelIndex / mResolution.x);
        const float x = float(aPixelIndex) - y * mResolution.x;
        return make_float2(x, y);
    }

    __host__ __device__ __forceinline__ float3 RasterToWorld(const float2 &aRasterXY) const
    {
        const float4 result = mRasterToWorld * make_float4(aRasterXY.x, aRasterXY.y, 0, 1);
		return make_float3(result.x / result.w, result.y / result.w, result.z / result.w);
    }

    __host__ __device__ __forceinline__ float2 WorldToRaster(const float3 &aWorldPos) const
    {
        const float4 temp = mWorldToRaster * make_float4(aWorldPos, 1);
        return make_float2(temp.x / temp.w, temp.y / temp.w);
    }

    // returns false when raster position is outside screen space
    __host__ __device__ __forceinline__ bool CheckRaster(const float2 &aRasterPos) const
    {
        return aRasterPos.x >= 0 && aRasterPos.y >= 0 &&
            aRasterPos.x < mResolution.x && aRasterPos.y < mResolution.y;
    }

    __host__ __device__ __forceinline__ void GenerateRay(const float2 &aRasterXY, float3 &org, float3 &dir) const
    {
        const float3 worldRaster = RasterToWorld(aRasterXY);

        org  = mPosition;
        dir  = normalize(worldRaster - mPosition);
    }

public:
    float3 mPosition;
    float3 mForward;
    float2 mResolution;
    Matrix4x4 mRasterToWorld;
    Matrix4x4 mWorldToRaster;
    float mImagePlaneDist;
};