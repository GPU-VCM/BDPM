#pragma once

#include "PerRayData.h"
#include "Frame.h"
#include "utils.h"

struct ComponentProbabilities
{
    float diffProb;
    float phongProb;
    float reflProb;
    float refrProb;
};

enum Events
{
    kNONE        = 0,
    kDiffuse     = 1,
    kPhong       = 2,
    kReflect     = 4,
    kRefract     = 8,
    kSpecular    = (kReflect  | kRefract),
    kNonSpecular = (kDiffuse  | kPhong),
    kAll         = (kSpecular | kNonSpecular)
};

template<bool FixIsLight>
class BSDF
{
public:
	bool isValid;
	BaseMaterial mat; 
    Frame frame;      
    float3 localDir;  
    bool  isDelta;    
    ComponentProbabilities probabilities;
    float continuationProb;  
    float fresnelCoeff;      

	__device__ __forceinline__ float CosThetaFix() const { return localDir.z; }
	__device__ __forceinline__ bool IsValid() const { return isValid; }

	__device__ __forceinline__ void Setup(const optix::Ray &ray, const PerRayData_closestHit &isx)
	{
		frame.SetFromZ(isx.normal);
		localDir = frame.ToLocal(-ray.direction);
        isValid = (abs(localDir.z) > EPS_COSINE);
		if (isValid)
		{
			mat = isx.mat;
			GetComponentProbabilities(isx.mat);
			isDelta = (probabilities.diffProb == 0) && (probabilities.phongProb == 0);
		}
	}
	__device__ __forceinline__ float Luminance(const float3 &RGB) const
	{
		return 0.212671f * RGB.x +
			   0.715160f * RGB.y +
			   0.072169f * RGB.z;
	}

	__device__ __forceinline__ float AlbedoDiffuse(const BaseMaterial& material) const
    {
		return Luminance(material.diffusePart);
    }

	__device__ __forceinline__ float AlbedoPhong(const BaseMaterial& material) const
    {
		return Luminance(material.phongPart);
    }

	__device__ __forceinline__ float AlbedoReflect(const BaseMaterial& material) const
    {
		return Luminance(material.mirror);
    }

	__device__ __forceinline__ float AlbedoRefract(const BaseMaterial& material) const
    {
		return material.ior > 0.f ? 1.f : 0.f;
    }

	__device__ __forceinline__ void GetComponentProbabilities(const BaseMaterial &material)
    {
		fresnelCoeff = FresnelDielectric(localDir.z, material.ior);

		const float albedoDiffuse = AlbedoDiffuse(material);
		const float albedoPhong = AlbedoPhong(material);
		const float albedoReflect = fresnelCoeff         * AlbedoReflect(material);
		const float albedoRefract = (1.f - fresnelCoeff) * AlbedoRefract(material);

        const float totalAlbedo = albedoDiffuse + albedoPhong + albedoReflect + albedoRefract;

        if (totalAlbedo < 1e-9f)
        {
            probabilities.diffProb  = 0.f;
            probabilities.phongProb = 0.f;
            probabilities.reflProb  = 0.f;
            probabilities.refrProb  = 0.f;
            continuationProb = 0.f;
        }
        else
        {
            probabilities.diffProb  = albedoDiffuse / totalAlbedo;
            probabilities.phongProb = albedoPhong   / totalAlbedo;
            probabilities.reflProb  = albedoReflect / totalAlbedo;
            probabilities.refrProb  = albedoRefract / totalAlbedo;
            
            continuationProb =
				maximum(material.diffusePart +
						material.phongPart +
						fresnelCoeff * material.mirror) +
			    (1.f - fresnelCoeff);

            continuationProb = min(1.f, max(0.f, continuationProb));
        }
    }
	__device__ __forceinline__ float3 ReflectLocal(const float3& vec) const
	{
		return make_float3(-vec.x, -vec.y, vec.z);
	}

	__device__ __forceinline__ float PowerCosHemispherePdfW(
		const float3  &nor,
		const float3  &dir,
		const float  power) const
	{
		const float cosTheta = max(0.f, dot(nor, dir));

		return (power + 1.f) * pow(cosTheta, power) * (INV_PI_F * 0.5f);
	}

	__device__ __forceinline__ float3 SampleDiffuse(
		const BaseMaterial &material,
        const float2    &rnd2,
        float3          &localdirNew,
        float           &pdf) const
    {
        if (localDir.z < EPS_COSINE)
            return make_float3(0.0f, 0.0f, 0.0f);

        float unweightedPdfW;
		localdirNew = SampleCosHemisphereW(rnd2, &unweightedPdfW);
        pdf += unweightedPdfW * probabilities.diffProb;

        return material.diffusePart * INV_PI_F;
    }

    __device__ __forceinline__ float3 EvaluateDiffuse(
		const BaseMaterial &material,
        const float3    &localdirNew,
        float          *dirPdf = NULL,
        float          *revPdf = NULL) const
    {
        if(probabilities.diffProb == 0)
            return make_float3(0.0f, 0.0f, 0.0f);

		if (localDir.z < EPS_COSINE || localdirNew.z < EPS_COSINE)
            return make_float3(0.0f, 0.0f, 0.0f);

		if (dirPdf)
			*dirPdf += probabilities.diffProb * max(0.f, localdirNew.z * INV_PI_F);

		if (revPdf)
			*revPdf += probabilities.diffProb * max(0.f, localdirNew.z * INV_PI_F);

        return material.diffusePart * INV_PI_F;
    }

	__device__ __forceinline__ float3 EvaluatePhong(
		const BaseMaterial &material,
		const float3    &localGenDirection,
		float          *dirPdf = NULL,
		float          *revPdf = NULL) const
	{
		if (probabilities.phongProb == 0)
			return make_float3(0.0f, 0.0f, 0.0f);

		if (localDir.z < EPS_COSINE || localGenDirection.z < EPS_COSINE)
			return make_float3(0.0f, 0.0f, 0.0f);
		const float3 reflLocalDirIn = ReflectLocal(localDir);
		const float dot_R_Wi = dot(reflLocalDirIn, localGenDirection);

		if (dot_R_Wi <= EPS_PHONG)
			return make_float3(0.0f, 0.0f, 0.0f);

		if (dirPdf || revPdf)
		{
			// the sampling is symmetric
			const float pdfW = probabilities.phongProb *
				PowerCosHemispherePdfW(reflLocalDirIn, localGenDirection, material.exponent);

			if (dirPdf)
				*dirPdf  += pdfW;

			if (revPdf)
				*revPdf += pdfW;
		}

		const float3 rho = material.phongPart *
			(material.exponent + 2.f) * 0.5f * INV_PI_F;

		return rho * pow(dot_R_Wi, material.exponent);
	}
    __device__ __forceinline__ float3 Evaluate(
        const float3 &worldSpaceDirection,
        float       &cosThetaNew,
        float       *dirPdf = NULL,
        float       *revPdf = NULL) const
    {
        float3 result = make_float3(0.0f, 0.0f, 0.0f);

        if (dirPdf)  *dirPdf = 0;
        if (revPdf) *revPdf = 0;

        const float3 localDirGen = frame.ToLocal(worldSpaceDirection);

        if (localDirGen.z * localDir.z < 0)
            return result;

        cosThetaNew = abs(localDirGen.z);

        result += EvaluateDiffuse(mat, localDirGen, dirPdf, revPdf);
        result += EvaluatePhong(mat, localDirGen, dirPdf, revPdf);

        return result;
    }

	__device__ __forceinline__ void PdfDiffuse(
        const BaseMaterial &material,
        const float3    &localGenDirection,
        float          *dirPdf = NULL,
        float          *revPdf = NULL) const
    {
        if (probabilities.diffProb == 0)
            return;

        if (dirPdf)
            *dirPdf  += probabilities.diffProb *
            max(0.f, localGenDirection.z * INV_PI_F);

        if (revPdf)
            *revPdf += probabilities.diffProb *
            max(0.f, localDir.z * INV_PI_F);
    }

	__device__ __forceinline__ void PdfPhong(
        const BaseMaterial &material,
        const float3    &localGenDirection,
        float          *dirPdf = NULL,
        float          *revPdf = NULL) const
    {
        if (probabilities.phongProb == 0)
            return;

        const float3 reflLocalDirIn = ReflectLocal(localDir);
        const float dot_R_Wi = dot(reflLocalDirIn, localGenDirection);

        if(dot_R_Wi <= EPS_PHONG)
            return;

        if(dirPdf || revPdf)
        {
            const float pdfW = PowerCosHemispherePdfW(reflLocalDirIn, localGenDirection,
                material.exponent) * probabilities.phongProb;

            if(dirPdf)
                *dirPdf  += pdfW;

            if(revPdf)
                *revPdf += pdfW;
        }
    }

	__device__ __forceinline__ float3 SamplePowerCosHemisphereW(
		const float2  &rnd21,
		const float  power,
		float        *pdf) const
	{
		const float term1 = 2.f * PI_F * rnd21.x;
		const float term2 = powf(rnd21.y, 1.f / (power + 1.f));
		const float term3 = sqrtf(1.f - term2 * term2);

		if (pdf)
		{
			*pdf = (power + 1.f) * powf(term2, power) * (0.5f * INV_PI_F);
		}

		return make_float3(
			cosf(term1) * term3,
			sinf(term1) * term3,
			term2);
	}

	__device__ __forceinline__ float3 SamplePhong(
        const BaseMaterial &material,
        const float2    &rnd21,
        float3          &localDirection,
        float          &pdf) const
    {
        localDirection = SamplePowerCosHemisphereW(rnd21, material.exponent, NULL);
        const float3 reflLocalDirFixed = ReflectLocal(localDir);
        {
            Frame frame;
            frame.SetFromZ(reflLocalDirFixed);
            localDirection = frame.ToWorld(localDirection);
        }

        const float dot_R_Wi = dot(reflLocalDirFixed, localDirection);

        if(dot_R_Wi <= EPS_PHONG)
            return make_float3(0.0f, 0.0f, 0.0f);

        PdfPhong(material, localDirection, &pdf);

        const float3 rho = material.phongPart *
            (material.exponent + 2.f) * 0.5f * INV_PI_F;

        return rho * pow(dot_R_Wi, material.exponent);
    }

    __device__ __forceinline__ float3 SampleReflect(
        const BaseMaterial &material,
        const float2    &rnd21,
        float3          &localDirection,
        float          &pdf) const
    {
        localDirection = ReflectLocal(localDir);

        pdf += probabilities.reflProb;
        return fresnelCoeff * material.mirror /
            abs(localDirection.z);
    }

    __device__ __forceinline__ float3 SampleRefract(
        const BaseMaterial &material,
        const float2    &rnd21,
        float3          &localDirection,
        float          &pdf) const
    {
        if(material.ior < 0)
            return make_float3(0.0f, 0.0f, 0.0f);

        float cosI = localDir.z;

        float cosT;
        float etaIncOverEtaTrans;

        if(cosI < 0.f) 
        {
            etaIncOverEtaTrans = material.ior;
            cosI = -cosI;
            cosT = 1.f;
        }
        else
        {
            etaIncOverEtaTrans = 1.f / material.ior;
            cosT = -1.f;
        }

        const float sinI2 = 1.f - cosI * cosI;
        const float sinT2 = (etaIncOverEtaTrans * etaIncOverEtaTrans) * sinI2;

        if(sinT2 < 1.f) 
        {
            cosT *= sqrtf(max(0.f, 1.f - sinT2));

            localDirection = make_float3(
                -etaIncOverEtaTrans * localDir.x,
                -etaIncOverEtaTrans * localDir.y,
                cosT);

            pdf += probabilities.refrProb;

            const float refractCoeff = 1.f - fresnelCoeff;
            if(!FixIsLight)
                return make_float3(refractCoeff * (etaIncOverEtaTrans * etaIncOverEtaTrans) / abs(cosT));
            else
                return make_float3(refractCoeff / abs(cosT));
        }

        pdf += 0.f;
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    __device__ __forceinline__ float3 Sample(
        const float3 &rnd31,
        float3       &worldSpaceDirection,
        float       &pdf,
        float       &cosThetaNew,
        uint        *matType = NULL) const
    {
        uint sampledEvent;

        if (rnd31.z < probabilities.diffProb)
            sampledEvent = kDiffuse;
        else if (rnd31.z < probabilities.diffProb + probabilities.phongProb)
            sampledEvent = kPhong;
        else if (rnd31.z < probabilities.diffProb + probabilities.phongProb + probabilities.reflProb)
            sampledEvent = kReflect;
        else
            sampledEvent = kRefract;

        if (matType)
            *matType = sampledEvent;

        pdf = 0;
        float3 result = make_float3(0.0f, 0.0f, 0.0f);
        float3 localDirGen;

		const float2 rndSample = make_float2(rnd31.x, rnd31.y);

        if (sampledEvent == kDiffuse)
        {
            result += SampleDiffuse(mat, rndSample, localDirGen, pdf);
            
            if (isZero(result))
				return make_float3(0.0f, 0.0f, 0.0f);
            
            result += EvaluatePhong(mat, localDirGen, &pdf);
        }
        else if (sampledEvent == kPhong)
        {
            result += SamplePhong(mat, rndSample, localDirGen, pdf);
            
            if (isZero(result))
                return make_float3(0.0f, 0.0f, 0.0f);
            
            result += EvaluateDiffuse(mat, localDirGen, &pdf);
        }
        else if (sampledEvent == kReflect)
        {
            result += SampleReflect(mat, rndSample, localDirGen, pdf);

            if (isZero(result))
                return make_float3(0.0f, 0.0f, 0.0f);
        }
        else
        {
            result += SampleRefract(mat, rndSample, localDirGen, pdf);
            if (isZero(result))
                return make_float3(0.0f, 0.0f, 0.0f);
        }

        cosThetaNew   = abs(localDirGen.z);
        if (cosThetaNew < EPS_COSINE)
            return make_float3(0.0f, 0.0f, 0.0f);

        worldSpaceDirection = frame.ToWorld(localDirGen);
        return result;
    }

    __device__ __forceinline__ float Pdf(
        const float3 &worldSpaceDirection,
        const bool ifRevPdf = false) const
    {
        const float3 localDirGen = frame.ToLocal(worldSpaceDirection);

        if (localDirGen.z * localDir.z < 0)
            return 0;

        float directPdfW  = 0;
        float reversePdfW = 0;

        PdfDiffuse(mat, localDirGen, &directPdfW, &reversePdfW);
        PdfPhong(mat, localDirGen, &directPdfW, &reversePdfW);

        return ifRevPdf ? reversePdfW : directPdfW;
    }
};

struct SubPathState
{
    float3 origin;        
    float3 direction;     
    float3 throughput;    
    uint  pathlen    : 30;
	uint  isLgtFinite : 1;
    uint  specPath  :  1; 

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

  
    BSDF<tFromLight> bsdf;

    float allMIS; 
    float bdptMIS;
    float elseMIS;

    const float3 &GetPosition() const
    {
        return isxPoint;
    }
};

typedef PathVertex<true>  LightVertex;
typedef BSDF<false>       CameraBSDF;
typedef BSDF<true>        LightBSDF;