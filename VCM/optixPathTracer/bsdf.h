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
	TriangleMaterial mat;    //!< Id of scene material, < 0 ~ invalid
    Frame mFrame;            //!< Local frame of reference
    float3 mLocalDirFix;      //!< Incoming (fixed) direction, in local
    bool  mIsDelta;          //!< True when material is purely specular
    ComponentProbabilities mProbabilities; //!< Sampling probabilities
    float mContinuationProb; //!< Russian roulette probability
    float mReflectCoeff;     //!< Fresnel reflection coefficient (for glass)

	__device__ __forceinline__ float CosThetaFix() const { return mLocalDirFix.z; }
	__device__ __forceinline__ bool IsValid() const { return isValid; }

	__device__ __forceinline__ void Setup(const optix::Ray &aRay, const PerRayData_closestHit &aIsect)
	{
		mFrame.SetFromZ(aIsect.normal);
		mLocalDirFix = mFrame.ToLocal(-aRay.direction);

		 // reject rays that are too parallel with tangent plane
        isValid = (abs(mLocalDirFix.z) > EPS_COSINE);
		if (isValid)
		{
			mat = aIsect.mat;
			GetComponentProbabilities(aIsect.mat);
			mIsDelta = (mProbabilities.diffProb == 0) && (mProbabilities.phongProb == 0);
		}
	}

	// sRGB luminance
	__device__ __forceinline__ float Luminance(const float3 &aRGB) const
	{
		return 0.212671f * aRGB.x +
			   0.715160f * aRGB.y +
			   0.072169f * aRGB.z;
	}

	__device__ __forceinline__ float AlbedoDiffuse(const TriangleMaterial& aMaterial) const
    {
        return Luminance(aMaterial.mDiffuseReflectance);
    }

    __device__ __forceinline__ float AlbedoPhong(const TriangleMaterial& aMaterial) const
    {
        return Luminance(aMaterial.mPhongReflectance);
    }

    __device__ __forceinline__ float AlbedoReflect(const TriangleMaterial& aMaterial) const
    {
        return Luminance(aMaterial.mMirrorReflectance);
    }

    __device__ __forceinline__ float AlbedoRefract(const TriangleMaterial& aMaterial) const
    {
		return aMaterial.mIOR > 0.f ? 1.f : 0.f;
    }

	__device__ __forceinline__ void GetComponentProbabilities(const TriangleMaterial &aMaterial)
    {
        mReflectCoeff = FresnelDielectric(mLocalDirFix.z, aMaterial.mIOR);

        const float albedoDiffuse = AlbedoDiffuse(aMaterial);
        const float albedoPhong   = AlbedoPhong(aMaterial);
        const float albedoReflect = mReflectCoeff         * AlbedoReflect(aMaterial);
        const float albedoRefract = (1.f - mReflectCoeff) * AlbedoRefract(aMaterial);

        const float totalAlbedo = albedoDiffuse + albedoPhong + albedoReflect + albedoRefract;

        if (totalAlbedo < 1e-9f)
        {
            mProbabilities.diffProb  = 0.f;
            mProbabilities.phongProb = 0.f;
            mProbabilities.reflProb  = 0.f;
            mProbabilities.refrProb  = 0.f;
            mContinuationProb = 0.f;
        }
        else
        {
            mProbabilities.diffProb  = albedoDiffuse / totalAlbedo;
            mProbabilities.phongProb = albedoPhong   / totalAlbedo;
            mProbabilities.reflProb  = albedoReflect / totalAlbedo;
            mProbabilities.refrProb  = albedoRefract / totalAlbedo;
            // The continuation probability is max component from reflectance.
            // That way the weight of sample will never rise.
            // Luminance is another very valid option.
            mContinuationProb =
                maximum(aMaterial.mDiffuseReflectance +
						aMaterial.mPhongReflectance +
					    mReflectCoeff * aMaterial.mMirrorReflectance) +
			    (1.f - mReflectCoeff);

            mContinuationProb = min(1.f, max(0.f, mContinuationProb));
        }
    }

	// reflect vector through (0,0,1)
	__device__ __forceinline__ float3 ReflectLocal(const float3& aVector) const
	{
		return make_float3(-aVector.x, -aVector.y, aVector.z);
	}

	__device__ __forceinline__ float PowerCosHemispherePdfW(
		const float3  &aNormal,
		const float3  &aDirection,
		const float  aPower) const
	{
		const float cosTheta = max(0.f, dot(aNormal, aDirection));

		return (aPower + 1.f) * pow(cosTheta, aPower) * (INV_PI_F * 0.5f);
	}

	////////////////////////////////////////////////////////////////////////////
    // Sampling methods
    // All sampling methods take material, 2 random numbers [0-1[,
    // and return BSDF factor, generated direction in local coordinates, and PDF
    ////////////////////////////////////////////////////////////////////////////

	__device__ __forceinline__ float3 SampleDiffuse(
        const TriangleMaterial &aMaterial,
        const float2    &aRndTuple,
        float3          &oLocalDirGen,
        float           &oPdfW) const
    {
        if (mLocalDirFix.z < EPS_COSINE)
            return make_float3(0.0f, 0.0f, 0.0f);

        float unweightedPdfW;
        oLocalDirGen = SampleCosHemisphereW(aRndTuple, &unweightedPdfW);
        oPdfW += unweightedPdfW * mProbabilities.diffProb;

        return aMaterial.mDiffuseReflectance * INV_PI_F;
    }

	
	////////////////////////////////////////////////////////////////////////////
    // Evaluation methods
    ////////////////////////////////////////////////////////////////////////////

    __device__ __forceinline__ float3 EvaluateDiffuse(
        const TriangleMaterial &aMaterial,
        const float3    &aLocalDirGen,
        float          *oDirectPdfW = NULL,
        float          *oReversePdfW = NULL) const
    {
        if(mProbabilities.diffProb == 0)
            return make_float3(0.0f, 0.0f, 0.0f);

        if(mLocalDirFix.z < EPS_COSINE || aLocalDirGen.z < EPS_COSINE)
            return make_float3(0.0f, 0.0f, 0.0f);

        if(oDirectPdfW)
            *oDirectPdfW += mProbabilities.diffProb * max(0.f, aLocalDirGen.z * INV_PI_F);

        if(oReversePdfW)
            *oReversePdfW += mProbabilities.diffProb * max(0.f, mLocalDirFix.z * INV_PI_F);

        return aMaterial.mDiffuseReflectance * INV_PI_F;
    }

	__device__ __forceinline__ float3 EvaluatePhong(
		const TriangleMaterial &aMaterial,
		const float3    &aLocalDirGen,
		float          *oDirectPdfW = NULL,
		float          *oReversePdfW = NULL) const
	{
		if (mProbabilities.phongProb == 0)
			return make_float3(0.0f, 0.0f, 0.0f);

		if (mLocalDirFix.z < EPS_COSINE || aLocalDirGen.z < EPS_COSINE)
			return make_float3(0.0f, 0.0f, 0.0f);

		// assumes this is never called when rejectShadingCos(oLocalDirGen.z) is true
		const float3 reflLocalDirIn = ReflectLocal(mLocalDirFix);
		const float dot_R_Wi = dot(reflLocalDirIn, aLocalDirGen);

		if (dot_R_Wi <= EPS_PHONG)
			return make_float3(0.0f, 0.0f, 0.0f);

		if (oDirectPdfW || oReversePdfW)
		{
			// the sampling is symmetric
			const float pdfW = mProbabilities.phongProb *
				PowerCosHemispherePdfW(reflLocalDirIn, aLocalDirGen, aMaterial.mPhongExponent);

			if (oDirectPdfW)
				*oDirectPdfW  += pdfW;

			if (oReversePdfW)
				*oReversePdfW += pdfW;
		}

		const float3 rho = aMaterial.mPhongReflectance *
			(aMaterial.mPhongExponent + 2.f) * 0.5f * INV_PI_F;

		return rho * pow(dot_R_Wi, aMaterial.mPhongExponent);
	}

	/* \brief Given a direction, evaluates BSDF
     *
     * Returns value of BSDF, as well as cosine for the
     * aWorldDirGen direction.
     * Can return probability (w.r.t. solid angle W),
     * of having sampled aWorldDirGen given mLocalDirFix (oDirectPdfW),
     * and of having sampled mLocalDirFix given aWorldDirGen (oReversePdfW).
     *
     */
    __device__ __forceinline__ float3 Evaluate(
        const float3 &aWorldDirGen,
        float       &oCosThetaGen,
        float       *oDirectPdfW = NULL,
        float       *oReversePdfW = NULL) const
    {
        float3 result = make_float3(0.0f, 0.0f, 0.0f);

        if (oDirectPdfW)  *oDirectPdfW = 0;
        if (oReversePdfW) *oReversePdfW = 0;

        const float3 localDirGen = mFrame.ToLocal(aWorldDirGen);

        if (localDirGen.z * mLocalDirFix.z < 0)
            return result;

        oCosThetaGen = abs(localDirGen.z);

        result += EvaluateDiffuse(mat, localDirGen, oDirectPdfW, oReversePdfW);
        result += EvaluatePhong(mat, localDirGen, oDirectPdfW, oReversePdfW);

        return result;
    }

	__device__ __forceinline__ void PdfDiffuse(
        const TriangleMaterial &aMaterial,
        const float3    &aLocalDirGen,
        float          *oDirectPdfW = NULL,
        float          *oReversePdfW = NULL) const
    {
        if (mProbabilities.diffProb == 0)
            return;

        if (oDirectPdfW)
            *oDirectPdfW  += mProbabilities.diffProb *
            max(0.f, aLocalDirGen.z * INV_PI_F);

        if (oReversePdfW)
            *oReversePdfW += mProbabilities.diffProb *
            max(0.f, mLocalDirFix.z * INV_PI_F);
    }

	__device__ __forceinline__ void PdfPhong(
        const TriangleMaterial &aMaterial,
        const float3    &aLocalDirGen,
        float          *oDirectPdfW = NULL,
        float          *oReversePdfW = NULL) const
    {
        if (mProbabilities.phongProb == 0)
            return;

        // assumes this is never called when rejectShadingCos(oLocalDirGen.z) is true
        const float3 reflLocalDirIn = ReflectLocal(mLocalDirFix);
        const float dot_R_Wi = dot(reflLocalDirIn, aLocalDirGen);

        if(dot_R_Wi <= EPS_PHONG)
            return;

        if(oDirectPdfW || oReversePdfW)
        {
            // the sampling is symmetric
            const float pdfW = PowerCosHemispherePdfW(reflLocalDirIn, aLocalDirGen,
                aMaterial.mPhongExponent) * mProbabilities.phongProb;

            if(oDirectPdfW)
                *oDirectPdfW  += pdfW;

            if(oReversePdfW)
                *oReversePdfW += pdfW;
        }
    }

	__device__ __forceinline__ float3 SamplePowerCosHemisphereW(
		const float2  &aSamples,
		const float  aPower,
		float        *oPdfW) const
	{
		const float term1 = 2.f * PI_F * aSamples.x;
		const float term2 = powf(aSamples.y, 1.f / (aPower + 1.f));
		const float term3 = sqrtf(1.f - term2 * term2);

		if (oPdfW)
		{
			*oPdfW = (aPower + 1.f) * powf(term2, aPower) * (0.5f * INV_PI_F);
		}

		return make_float3(
			cosf(term1) * term3,
			sinf(term1) * term3,
			term2);
	}

	__device__ __forceinline__ float3 SamplePhong(
        const TriangleMaterial &aMaterial,
        const float2    &aRndTuple,
        float3          &oLocalDirGen,
        float          &oPdfW) const
    {
        oLocalDirGen = SamplePowerCosHemisphereW(aRndTuple, aMaterial.mPhongExponent, NULL);

        // Due to numeric issues in MIS, we actually need to compute all pdfs
        // exactly the same way all the time!!!
        const float3 reflLocalDirFixed = ReflectLocal(mLocalDirFix);
        {
            Frame frame;
            frame.SetFromZ(reflLocalDirFixed);
            oLocalDirGen = frame.ToWorld(oLocalDirGen);
        }

        const float dot_R_Wi = dot(reflLocalDirFixed, oLocalDirGen);

        if(dot_R_Wi <= EPS_PHONG)
            return make_float3(0.0f, 0.0f, 0.0f);

        PdfPhong(aMaterial, oLocalDirGen, &oPdfW);

        const float3 rho = aMaterial.mPhongReflectance *
            (aMaterial.mPhongExponent + 2.f) * 0.5f * INV_PI_F;

        return rho * pow(dot_R_Wi, aMaterial.mPhongExponent);
    }

    __device__ __forceinline__ float3 SampleReflect(
        const TriangleMaterial &aMaterial,
        const float2    &aRndTuple,
        float3          &oLocalDirGen,
        float          &oPdfW) const
    {
        oLocalDirGen = ReflectLocal(mLocalDirFix);

        oPdfW += mProbabilities.reflProb;
        // BSDF is multiplied (outside) by cosine (oLocalDirGen.z),
        // for mirror this shouldn't be done, so we pre-divide here instead
        return mReflectCoeff * aMaterial.mMirrorReflectance /
            abs(oLocalDirGen.z);
    }

    __device__ __forceinline__ float3 SampleRefract(
        const TriangleMaterial &aMaterial,
        const float2    &aRndTuple,
        float3          &oLocalDirGen,
        float          &oPdfW) const
    {
        if(aMaterial.mIOR < 0)
            return make_float3(0.0f, 0.0f, 0.0f);

        float cosI = mLocalDirFix.z;

        float cosT;
        float etaIncOverEtaTrans;

        if(cosI < 0.f) // hit from inside
        {
            etaIncOverEtaTrans = aMaterial.mIOR;
            cosI = -cosI;
            cosT = 1.f;
        }
        else
        {
            etaIncOverEtaTrans = 1.f / aMaterial.mIOR;
            cosT = -1.f;
        }

        const float sinI2 = 1.f - cosI * cosI;
        const float sinT2 = (etaIncOverEtaTrans * etaIncOverEtaTrans) * sinI2;

        if(sinT2 < 1.f) // no total internal reflection
        {
            cosT *= sqrtf(max(0.f, 1.f - sinT2));

            oLocalDirGen = make_float3(
                -etaIncOverEtaTrans * mLocalDirFix.x,
                -etaIncOverEtaTrans * mLocalDirFix.y,
                cosT);

            oPdfW += mProbabilities.refrProb;

            const float refractCoeff = 1.f - mReflectCoeff;
            // only camera paths are multiplied by this factor, and etas
            // are swapped because radiance flows in the opposite direction
            if(!FixIsLight)
                return make_float3(refractCoeff * (etaIncOverEtaTrans * etaIncOverEtaTrans) / abs(cosT));
            else
                return make_float3(refractCoeff / abs(cosT));
        }
        //else total internal reflection, do nothing

        oPdfW += 0.f;
        return make_float3(0.0f, 0.0f, 0.0f);
    }

	/* \brief Given 3 random numbers, samples new direction from BSDF.
     *
     * Uses z component of random triplet to pick BSDF component from
     * which it will sample direction. If non-specular component is chosen,
     * it will also evaluate the other (non-specular) BSDF components.
     * Return BSDF factor for given direction, as well as PDF choosing that direction.
     * Can return event which has been sampled.
     * If result is Vec3f(0,0,0), then the sample should be discarded.
     */
    __device__ __forceinline__ float3 Sample(
        const float3 &aRndTriplet,
        float3       &oWorldDirGen,
        float       &oPdfW,
        float       &oCosThetaGen,
        uint        *oSampledEvent = NULL) const
    {
        uint sampledEvent;

        if (aRndTriplet.z < mProbabilities.diffProb)
            sampledEvent = kDiffuse;
        else if (aRndTriplet.z < mProbabilities.diffProb + mProbabilities.phongProb)
            sampledEvent = kPhong;
        else if (aRndTriplet.z < mProbabilities.diffProb + mProbabilities.phongProb + mProbabilities.reflProb)
            sampledEvent = kReflect;
        else
            sampledEvent = kRefract;

        if (oSampledEvent)
            *oSampledEvent = sampledEvent;

        oPdfW = 0;
        float3 result = make_float3(0.0f, 0.0f, 0.0f);
        float3 localDirGen;

		const float2 rndSample = make_float2(aRndTriplet.x, aRndTriplet.y);

        if (sampledEvent == kDiffuse)
        {
            result += SampleDiffuse(mat, rndSample, localDirGen, oPdfW);
            
            if (isZero(result))
				return make_float3(0.0f, 0.0f, 0.0f);
            
            result += EvaluatePhong(mat, localDirGen, &oPdfW);
        }
        else if (sampledEvent == kPhong)
        {
            result += SamplePhong(mat, rndSample, localDirGen, oPdfW);
            
            if (isZero(result))
                return make_float3(0.0f, 0.0f, 0.0f);
            
            result += EvaluateDiffuse(mat, localDirGen, &oPdfW);
        }
        else if (sampledEvent == kReflect)
        {
            result += SampleReflect(mat, rndSample, localDirGen, oPdfW);

            if (isZero(result))
                return make_float3(0.0f, 0.0f, 0.0f);
        }
        else
        {
            result += SampleRefract(mat, rndSample, localDirGen, oPdfW);
            if (isZero(result))
                return make_float3(0.0f, 0.0f, 0.0f);
        }

        oCosThetaGen   = abs(localDirGen.z);
        if (oCosThetaGen < EPS_COSINE)
            return make_float3(0.0f, 0.0f, 0.0f);

        oWorldDirGen = mFrame.ToWorld(localDirGen);
        return result;
    }

    /* \brief Given a direction, evaluates Pdf
     *
     * By default returns PDF with which would be aWorldDirGen
     * generated from mLocalDirFix. When aEvalRevPdf == true,
     * it provides PDF for the reverse direction.
     */
    __device__ __forceinline__ float Pdf(
        const float3 &aWorldDirGen,
        const bool aEvalRevPdf = false) const
    {
        const float3 localDirGen = mFrame.ToLocal(aWorldDirGen);

        if (localDirGen.z * mLocalDirFix.z < 0)
            return 0;

        float directPdfW  = 0;
        float reversePdfW = 0;

        PdfDiffuse(mat, localDirGen, &directPdfW, &reversePdfW);
        PdfPhong(mat, localDirGen, &directPdfW, &reversePdfW);

        return aEvalRevPdf ? reversePdfW : directPdfW;
    }
};

// The sole point of this structure is to make carrying around the ray baggage easier.
struct SubPathState
{
    float3 mOrigin;             // Path origin
    float3 mDirection;          // Where to go next
    float3 mThroughput;         // Path throughput
    uint  mPathLength    : 30; // Number of path segments, including this
    uint  mIsFiniteLight :  1; // Just generate by finite light
    uint  mSpecularPath  :  1; // All scattering events so far were specular

    float dVCM; // MIS quantity used for vertex connection and merging
    float dVC;  // MIS quantity used for vertex connection
    float dVM;  // MIS quantity used for vertex merging
};

// Path vertex, used for merging and connection
template<bool tFromLight>
struct PathVertex
{
    float3 mHitpoint;   // Position of the vertex
    float3 mThroughput; // Path throughput (including emission)
    uint  mPathLength; // Number of segments between source and vertex

    // Stores all required local information, including incoming direction.
    BSDF<tFromLight> mBsdf;

    float dVCM; // MIS quantity used for vertex connection and merging
    float dVC;  // MIS quantity used for vertex connection
    float dVM;  // MIS quantity used for vertex merging

    // Used by HashGrid
    const float3 &GetPosition() const
    {
        return mHitpoint;
    }
};

typedef PathVertex<true>  LightVertex;
typedef BSDF<false>       CameraBSDF;
typedef BSDF<true>        LightBSDF;