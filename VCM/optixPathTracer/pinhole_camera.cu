#include <optix_world.h>
#include "commonStructs.h"
#include "random.h"
#include "BSDF.h"
#include "Camera.h"

using namespace optix;

//#define STRATIFIED

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>              temp_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );
rtDeclareVariable(unsigned int,  shadow_ray_type, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtBuffer<Light>			 lightBuffer;
rtDeclareVariable(float,         aRadiusFactor, ,);
rtDeclareVariable(float,	     aRadiusAlpha, ,);
//rtDeclareVariable(SceneSphere,   mSceneSphere, , );
rtDeclareVariable(Camera,        camera, , );
rtDeclareVariable(float,         vfov, ,);
//rtDeclareVariable(int,  backLightIx, , );
rtDeclareVariable(int,  mMaxPathLength, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

#ifdef USE_CALLABLE_PROGRAM
rtCallableProgram(float3, IlluminateBackground,
		(const Light *const light,
		 const float		   mInvSceneRadiusSqr,
         const float3       &aReceivingPosition,
         const float2       &aRndTuple,
         float3             &oDirectionToLight,
         float              &oDistance,
         float              &oDirectPdfW,
         float              *oEmissionPdfW,
         float              *oCosAtLight));

rtCallableProgram(float3, EmitBackground,
	    (const Light *const light,
         const float3      mSceneCenter,
		 const float       mSceneRadius,
		 const float		  mInvSceneRadiusSqr,
         const float2      &aDirRndTuple,
         const float2      &aPosRndTuple,
         float3            &oPosition,
         float3            &oDirection,
         float             &oEmissionPdfW,
         float             *oDirectPdfA,
         float             *oCosThetaLight));

rtCallableProgram(float3, GetRadianceBackground,
		(const Light *const light,
		 const float		   mInvSceneRadiusSqr,
		 const float3       &aRayDirection,
		 const float3       &aHitPoint,
		 float             *oDirectPdfA,
		 float             *oEmissionPdfW));

rtCallableProgram(float3, IlluminateAreaLight,
		(const Light *const light,
         const float3       &aReceivingPosition,
         const float2       &aRndTuple,
         float3             &oDirectionToLight,
         float              &oDistance,
         float              &oDirectPdfW,
         float              *oEmissionPdfW,
         float              *oCosAtLight));

rtCallableProgram(float3, EmitAreaLight,
	    (const Light *const light,
         const float2      &aDirRndTuple,
         const float2      &aPosRndTuple,
         float3            &oPosition,
         float3            &oDirection,
         float             &oEmissionPdfW,
         float             *oDirectPdfA,
         float             *oCosThetaLight));

rtCallableProgram(float3, GetRadianceAreaLight,
		(const Light *const light,
		 const float3      &aRayDirection,
		 const float3      &aHitPoint,
		 float             *oDirectPdfA,
		 float             *oEmissionPdfW));

rtCallableProgram(float3, IlluminateDirectional,
		(const Light *const light,
		 const float		   mInvSceneRadiusSqr,
         float3             &oDirectionToLight,
         float              &oDistance,
         float              &oDirectPdfW,
         float              *oEmissionPdfW,
         float              *oCosAtLight));

rtCallableProgram(float3, EmitDirectional,
	    (const Light *const light,
         const float3      mSceneCenter,
		 const float       mSceneRadius,
		 const float		  mInvSceneRadiusSqr,
         const float2      &aPosRndTuple,
         float3            &oPosition,
         float3            &oDirection,
         float             &oEmissionPdfW,
         float             *oDirectPdfA,
         float             *oCosThetaLight));

rtCallableProgram(float3, GetRadianceDirectional, ());
#else
#include "AreaLight.h"
#include "BackgroundLight.h"
#include "DirectionalLight.h"
#endif

#ifdef STRATIFIED
rtBuffer<float2> strat_buffer;
#endif

// Samples light emission
__device__ __forceinline__ void GenerateLightSample(SubPathState &oLightState, const float &mMisVcWeightFactor, unsigned int &seed)
{
    // We sample lights uniformly
	const int lightCount = lightBuffer.size();
    const float lightPickProb = 1.0f / lightCount;

	const int lightIx = (int)(rnd(seed) * lightCount);
    const float2 rndDirSamples = make_float2(rnd(seed), rnd(seed));
    const float2 rndPosSamples = make_float2(rnd(seed), rnd(seed));

	float emissionPdfW, directPdfW, cosLight;
	const Light &light = lightBuffer[lightIx];
	oLightState.mThroughput = EmitAreaLight(&light, rndDirSamples, rndPosSamples,
		oLightState.mOrigin, oLightState.mDirection,
		emissionPdfW, &directPdfW, &cosLight);

    emissionPdfW *= lightPickProb;
    directPdfW   *= lightPickProb;

    oLightState.mThroughput    /= emissionPdfW;
    oLightState.mPathLength    = 1;
    oLightState.mIsFiniteLight = 1;

    // Light sub-path MIS quantities. Implements [tech. rep. (31)-(33)] partially.
    // The evaluation is completed after tracing the emission ray in the light sub-path loop.
    // Delta lights are handled as well [tech. rep. (48)-(50)].
    {
        oLightState.dVCM = Mis(directPdfW / emissionPdfW);
      
         const float usedCosLight = cosLight;
         oLightState.dVC = Mis(usedCosLight / emissionPdfW);
    
        oLightState.dVM = oLightState.dVC * mMisVcWeightFactor;
    }
}

// Computes contribution of light sample to camera by splatting is onto the
// framebuffer. Multiplies by throughput (obviously, as nothing is returned).

__device__ __forceinline__ void ConnectToCamera(
    const SubPathState &aLightState,
    const float3       &aHitpoint,
    const LightBSDF    &aBsdf,
	const float mLightSubPathCount,
	const float mMisVcWeightFactor)
{
	float3 directionToCamera = camera.mPosition - aHitpoint;

    // Check point is in front of camera...

    if (dot(W, -directionToCamera) <= 0.f)
        return;

    // Check it projects to the screen (and where)
    const float2 imagePos = camera.WorldToRaster(aHitpoint);
    if (!camera.CheckRaster(imagePos))
        return;

    // Compute distance and normalize direction to camera...

    const float distEye2 = dot(directionToCamera, directionToCamera);
    const float distance = sqrtf(distEye2);
    directionToCamera   /= distance;

    // Get the BSDF
    float cosToCamera, bsdfDirPdfW, bsdfRevPdfW;
    const float3 bsdfFactor = aBsdf.Evaluate(
        directionToCamera, cosToCamera, &bsdfDirPdfW, &bsdfRevPdfW);

    if (isZero(bsdfFactor))
        return;

    bsdfRevPdfW *= aBsdf.mContinuationProb;

    // Compute pdf conversion factor from image plane area to surface area

	const float cosAtCamera = dot(camera.mForward, -directionToCamera);
    const float imagePointToCameraDist = camera.mImagePlaneDist / cosAtCamera;
	const float aspect = camera.mResolution.y / camera.mResolution.x;
	const float imageToSolidAngleFactor = aspect * aspect * (imagePointToCameraDist * imagePointToCameraDist) / cosAtCamera;
    const float imageToSurfaceFactor = imageToSolidAngleFactor * abs(cosToCamera) / (distance * distance);

    // We put the virtual image plane at such a distance from the camera origin
    // that the pixel area is one and thus the image plane sampling pdf is 1.
    // The area pdf of aHitpoint as sampled from the camera is then equal to
    // the conversion factor from image plane area density to surface area density

    const float cameraPdfA = imageToSurfaceFactor;

    // Partial light sub-path weight [tech. rep. (46)]. Note the division by
    // mLightPathCount, which is the number of samples this technique uses.
    // This division also appears a few lines below in the framebuffer accumulation.

    const float wLight = Mis(cameraPdfA / mLightSubPathCount) * (
        /*mMisVmWeightFactor + */aLightState.dVCM + aLightState.dVC * Mis(bsdfRevPdfW));

    // Partial eye sub-path weight is 0 [tech. rep. (47)]

    // Full path MIS weight [tech. rep. (37)]. No MIS for traditional light tracing.
    const float misWeight = 1.f / (wLight + 1.f);   // place 1.f for light tracing...

    const float surfaceToImageFactor = 1.f / imageToSurfaceFactor;

    // We divide the contribution by surfaceToImageFactor to convert the (already
    // divided) pdf from surface area to image plane area, w.r.t. which the
    // pixel integral is actually defined. We also divide by the number of samples
    // this technique makes, which is equal to the number of light sub-paths

    const float3 contrib = misWeight * aLightState.mThroughput * bsdfFactor /
        (mLightSubPathCount * surfaceToImageFactor);

    if (!isZero(contrib))
    {
		PerRayData_occlusion prd_occlusion;
		const optix::Ray shadow_ray = optix::make_Ray(aHitpoint, directionToCamera, shadow_ray_type, scene_epsilon, distance);
		rtTrace(top_object, shadow_ray, prd_occlusion);
        if (!prd_occlusion.occluded)
        {
			const uint2 ups = make_uint2(imagePos.x, imagePos.y);
			float *const base_output = (float *)&temp_buffer[make_uint2(0, 0)];
			const uint offset = (ups.x + ups.y * temp_buffer.size().x) << 2;
			atomicAdd(base_output + offset,     contrib.x);
			atomicAdd(base_output + offset + 1, contrib.y);
			atomicAdd(base_output + offset + 2, contrib.z);
        }
    }
}

// Samples a scattering direction camera/light sample according to BSDF.
// Returns false for termination

template<bool tLightSample>
__device__ __forceinline__ bool SampleScattering(
    const BSDF<tLightSample> &aBsdf,
    const float3             &aHitPoint,
    SubPathState             &aoState,
	const float mMisVcWeightFactor,
	uint &seed)
{
    // x,y for direction, z for component. No rescaling happens

    const float3 rndTriplet = make_float3(rnd(seed), rnd(seed), rnd(seed));
    float bsdfDirPdfW, cosThetaOut;
    uint  sampledEvent;

    const float3 bsdfFactor = aBsdf.Sample(rndTriplet, aoState.mDirection,
        bsdfDirPdfW, cosThetaOut, &sampledEvent);

    if (isZero(bsdfFactor))
        return false;

    // If we sampled specular event, then the reverse probability
    // cannot be evaluated, but we know it is exactly the same as
    // forward probability, so just set it. If non-specular event happened,
    // we evaluate the pdf
    float bsdfRevPdfW = bsdfDirPdfW;
    if ((sampledEvent & kSpecular) == 0)
        bsdfRevPdfW = aBsdf.Pdf(aoState.mDirection, true);

    // Russian roulette
    const float contProb = aBsdf.mContinuationProb;
    if (rnd(seed) > contProb)
        return false;

    bsdfDirPdfW *= contProb;
    bsdfRevPdfW *= contProb;

    // Sub-path MIS quantities for the next vertex. Only partial - the
    // evaluation is completed when the actual hit point is known,
    // i.e. after tracing the ray, in the sub-path loop.

    if (sampledEvent & kSpecular)
    {
        // Specular scattering case [tech. rep. (53)-(55)] (partially, as noted above)
        aoState.dVCM = 0.f;
        //aoState.dVC *= Mis(cosThetaOut / bsdfDirPdfW) * Mis(bsdfRevPdfW);
        //aoState.dVM *= Mis(cosThetaOut / bsdfDirPdfW) * Mis(bsdfRevPdfW);
        //assert(bsdfDirPdfW == bsdfRevPdfW);
        aoState.dVC *= Mis(cosThetaOut);
        aoState.dVM *= Mis(cosThetaOut);

        aoState.mSpecularPath &= 1;
    }
    else
    {
        // Implements [tech. rep. (34)-(36)] (partially, as noted above)
        aoState.dVC = Mis(cosThetaOut / bsdfDirPdfW) * (
            aoState.dVC * Mis(bsdfRevPdfW) +
            aoState.dVCM /*+ mMisVmWeightFactor*/);

        aoState.dVM = Mis(cosThetaOut / bsdfDirPdfW) * (
            aoState.dVM * Mis(bsdfRevPdfW) +
            aoState.dVCM * mMisVcWeightFactor + 1.f);

        aoState.dVCM = Mis(1.f / bsdfDirPdfW);

        aoState.mSpecularPath &= 0;
    }

    aoState.mOrigin  = aHitPoint;
    aoState.mThroughput *= bsdfFactor * (cosThetaOut / bsdfDirPdfW);
        
    return true;
}

RT_PROGRAM void clear_temp()
{
	temp_buffer[launch_index] = make_float4(0, 0, 0, 0);
}

RT_PROGRAM void cumulate_samples()
{
	if (frame_number > 1)
	{
		const float a = 1.0f / (float)frame_number;
		const float b = ((float)frame_number - 1.0f) * a;
		output_buffer[launch_index] = a * temp_buffer[launch_index] + b * output_buffer[launch_index];
	}
	else
	{
		output_buffer[launch_index] = temp_buffer[launch_index];
	}
}

// Returns the radiance of a light source when hit by a random ray,
// multiplied by MIS weight. Can be used for both Background and Area lights.
//
// For Background lights:
//    Has to be called BEFORE updating the MIS quantities.
//    Value of aHitpoint is irrelevant (passing Vec3f(0))
//
// For Area lights:
//    Has to be called AFTER updating the MIS quantities.

__device__ __forceinline__ float3 GetLightRadiance(
	const int			 lightType,
	const Light *const aLight,
    const SubPathState   &aCameraState,
    const float3         &aHitpoint,
    const float3         &aRayDirection)
{
    // We sample lights uniformly

	const int lightCount = lightBuffer.size();
    const float lightPickProb = 1.0f / lightCount;

    float directPdfA, emissionPdfW;
	float3 radiance;
    
	radiance = GetRadianceAreaLight(aLight, aRayDirection, aHitpoint, &directPdfA, &emissionPdfW);

    if (isZero(radiance))
        return make_float3(0);

    // If we see light source directly from camera, no weighting is required

    if (aCameraState.mPathLength == 1)
        return radiance;

    directPdfA   *= lightPickProb;
    emissionPdfW *= lightPickProb;

    // Partial eye sub-path MIS weight [tech. rep. (43)].
    // If the last hit was specular, then dVCM == 0.

    const float wCamera = Mis(directPdfA) * aCameraState.dVCM +
        Mis(emissionPdfW) * aCameraState.dVC;

    // Partial light sub-path weight is 0 [tech. rep. (42)].

    // Full path MIS weight [tech. rep. (37)].
    const float misWeight = 1.f / (1.f + wCamera);
        
    return misWeight * radiance;
}

// Generates new camera sample given a pixel index

__device__ __forceinline__ void GenerateCameraSample(
	const uint2 xy,
	const float mLightSubPathCount,
    SubPathState &oCameraState,
	uint &seed)
{
	const float2 jitter = make_float2(rnd(seed), rnd(seed));
	float3 org, dir;
	camera.GenerateRay((make_float2(xy) + jitter), org, dir);
	
    // Compute pdf conversion factor from area on image plane to solid angle on ray

    const float cosAtCamera = dot(camera.mForward, dir);
    const float imagePointToCameraDist = camera.mImagePlaneDist / cosAtCamera;
	const float aspect = camera.mResolution.y / camera.mResolution.x;
    const float imageToSolidAngleFactor = aspect * aspect * (imagePointToCameraDist * imagePointToCameraDist) / cosAtCamera;

    // We put the virtual image plane at such a distance from the camera origin
    // that the pixel area is one and thus the image plane sampling pdf is 1.
    // The solid angle ray pdf is then equal to the conversion factor from
    // image plane area density to ray solid angle density

    const float cameraPdfW = imageToSolidAngleFactor;

    oCameraState.mOrigin       = org;
    oCameraState.mDirection    = dir;
    oCameraState.mThroughput   = make_float3(1.0f);

    oCameraState.mPathLength   = 1;
    oCameraState.mSpecularPath = 1;

    // Eye sub-path MIS quantities. Implements [tech. rep. (31)-(33)] partially.
    // The evaluation is completed after tracing the camera ray in the eye sub-path loop.

    oCameraState.dVCM = Mis(mLightSubPathCount / cameraPdfW);
    oCameraState.dVC  = 0;
    oCameraState.dVM  = 0;
}

// Connects camera vertex to randomly chosen light point.
// Returns emitted radiance multiplied by path MIS weight.
// Has to be called AFTER updating the MIS quantities.

__device__ __forceinline__ float3 DirectIllumination(
    const SubPathState     &aCameraState,
    const float3           &aHitpoint,
    const CameraBSDF       &aBsdf,
	uint				   &seed)
{
    // We sample lights uniformly

    const int   lightCount    = lightBuffer.size();
    const float lightPickProb = 1.0f / lightCount;

    const int lightIx = (int)(rnd(seed) * lightCount);
	const float2 rndPosSamples = make_float2(rnd(seed), rnd(seed));

	float3 directionToLight, radiance;
    float directPdfW, emissionPdfW, cosAtLight, distance;

	const Light &light = lightBuffer[lightIx];
	radiance = IlluminateAreaLight(&light, aHitpoint,
				rndPosSamples, directionToLight, distance, directPdfW,
				&emissionPdfW, &cosAtLight);

    // If radiance == 0, other values are undefined, so have to early exit
    if (isZero(radiance))
        return make_float3(0);

    float bsdfDirPdfW, bsdfRevPdfW, cosToLight;
    const float3 bsdfFactor = aBsdf.Evaluate(
        directionToLight, cosToLight, &bsdfDirPdfW, &bsdfRevPdfW);

    if (isZero(bsdfFactor))
        return make_float3(0);

    const float continuationProbability = aBsdf.mContinuationProb;
        
    // If the light is delta light, we can never hit it
    // by BSDF sampling, so the probability of this path is 0
    bsdfDirPdfW *= continuationProbability;

    bsdfRevPdfW *= continuationProbability;

    // Partial light sub-path MIS weight [tech. rep. (44)].
    // Note that wLight is a ratio of area pdfs. But since both are on the
    // light source, their distance^2 and cosine terms cancel out.
    // Therefore we can write wLight as a ratio of solid angle pdfs,
    // both expressed w.r.t. the same shading point.

    const float wLight = Mis(bsdfDirPdfW / (lightPickProb * directPdfW));

    // Partial eye sub-path MIS weight [tech. rep. (45)].
    //
    // In front of the sum in the parenthesis we have Mis(ratio), where
    //    ratio = emissionPdfA / directPdfA,
    // with emissionPdfA being the product of the pdfs for choosing the
    // point on the light source and sampling the outgoing direction.
    // What we are given by the light source instead are emissionPdfW
    // and directPdfW. Converting to area pdfs and plugging into ratio:
    //    emissionPdfA = emissionPdfW * cosToLight / dist^2
    //    directPdfA   = directPdfW * cosAtLight / dist^2
    //    ratio = (emissionPdfW * cosToLight / dist^2) / (directPdfW * cosAtLight / dist^2)
    //    ratio = (emissionPdfW * cosToLight) / (directPdfW * cosAtLight)
    //
    // Also note that both emissionPdfW and directPdfW should be
    // multiplied by lightPickProb, so it cancels out.

    const float wCamera = Mis(emissionPdfW * cosToLight / (directPdfW * cosAtLight)) * (
        aCameraState.dVCM + aCameraState.dVC * Mis(bsdfRevPdfW));

    // Full path MIS weight [tech. rep. (37)]
    const float misWeight = 1.f / (wLight + 1.f + wCamera);

    const float3 contrib =
        (misWeight * cosToLight / (lightPickProb * directPdfW)) * (radiance * bsdfFactor);

    if (isZero(contrib))
        return make_float3(0.0f);

	PerRayData_occlusion prd_occlusion;
	const optix::Ray shadow_ray = optix::make_Ray(aHitpoint, directionToLight, shadow_ray_type, scene_epsilon, distance);
	rtTrace(top_object, shadow_ray, prd_occlusion);
    if (prd_occlusion.occluded)
		return make_float3(0.0f);

    return contrib;
}

// Connects an eye and a light vertex. Result multiplied by MIS weight, but
// not multiplied by vertex throughputs. Has to be called AFTER updating MIS
// constants. 'direction' is FROM eye TO light vertex.
__device__ __forceinline__ float3 ConnectVertices(
    const LightVertex      &aLightVertex,
    const CameraBSDF       &aCameraBsdf,
    const float3           &aCameraHitpoint,
    const SubPathState     &aCameraState)
{
    // Get the connection
    float3 direction   = aLightVertex.mHitpoint - aCameraHitpoint;
    const float dist2 = dot(direction, direction);
    float  distance   = sqrtf(dist2);
    direction        /= distance;

    // Evaluate BSDF at camera vertex
    float cosCamera, cameraBsdfDirPdfW, cameraBsdfRevPdfW;
    const float3 cameraBsdfFactor = aCameraBsdf.Evaluate(
        direction, cosCamera, &cameraBsdfDirPdfW,
        &cameraBsdfRevPdfW);

    if (isZero(cameraBsdfFactor))
        return make_float3(0.0f);

    // Camera continuation probability (for Russian roulette)
    const float cameraCont = aCameraBsdf.mContinuationProb;
    cameraBsdfDirPdfW *= cameraCont;
    cameraBsdfRevPdfW *= cameraCont;

    // Evaluate BSDF at light vertex
    float cosLight, lightBsdfDirPdfW, lightBsdfRevPdfW;
    const float3 lightBsdfFactor = aLightVertex.mBsdf.Evaluate(
        -direction, cosLight, &lightBsdfDirPdfW,
        &lightBsdfRevPdfW);

    if (isZero(lightBsdfFactor))
        return make_float3(0);

    // Light continuation probability (for Russian roulette)
    const float lightCont = aLightVertex.mBsdf.mContinuationProb;
    lightBsdfDirPdfW *= lightCont;
    lightBsdfRevPdfW *= lightCont;

    // Compute geometry term
    const float geometryTerm = cosLight * cosCamera / dist2;
    if (geometryTerm < 0)
        return make_float3(0);

    // Convert pdfs to area pdf
    const float cameraBsdfDirPdfA = PdfWtoA(cameraBsdfDirPdfW, distance, cosLight);
    const float lightBsdfDirPdfA  = PdfWtoA(lightBsdfDirPdfW,  distance, cosCamera);

    // Partial light sub-path MIS weight [tech. rep. (40)]
    const float wLight = Mis(cameraBsdfDirPdfA) * (
        /*mMisVmWeightFactor + */aLightVertex.dVCM + aLightVertex.dVC * Mis(lightBsdfRevPdfW));

    // Partial eye sub-path MIS weight [tech. rep. (41)]
    const float wCamera = Mis(lightBsdfDirPdfA) * (
        /*mMisVmWeightFactor + */aCameraState.dVCM + aCameraState.dVC * Mis(cameraBsdfRevPdfW));

    // Full path MIS weight [tech. rep. (37)]
    const float misWeight = 1.f / (wLight + 1.f + wCamera);

    const float3 contrib = (misWeight * geometryTerm) * cameraBsdfFactor * lightBsdfFactor;

    if (isZero(contrib))
		return make_float3(0.0f);

	PerRayData_occlusion prd_occlusion;
	const optix::Ray shadow_ray = optix::make_Ray(aCameraHitpoint, direction, shadow_ray_type, scene_epsilon, distance);
	rtTrace(top_object, shadow_ray, prd_occlusion);
    if (prd_occlusion.occluded)
		return make_float3(0.0f);

    return contrib;
}

#define FAST_CONNECTION
#define CONNECT_VERTEXES

RT_PROGRAM void pinhole_camera()
{
#ifdef CONNECT_VERTEXES
	LightVertex lvertexes[10]; // N.B.: MAXIMUM PATH LENGTH
#endif
	SubPathState lightState, cameraState;

	size_t2 screen = temp_buffer.size();
	unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame_number);

	const float mLightSubPathCount = (float)(launch_dim.x * launch_dim.y);
	const float mBaseRadius = aRadiusFactor * (9.5f / 2.0f) * sqrtf(2.0f);
	
	// Setup our radius, 1st iteration has aIteration == 0, thus offset
    float radius = mBaseRadius / powf(float(frame_number), 0.5f * (1 - aRadiusAlpha));
    
	// Purely for numeric stability
    radius = max(radius, 1e-7f);
    const float radiusSqr = radius * radius;

	const float etaVCM = (PI_F * radiusSqr) * mLightSubPathCount;
	const float mMisVcWeightFactor = Mis(1.0f / etaVCM);

	const int spp = 1;
	for (int k = 0; k < spp; ++k)
	{
		GenerateLightSample(lightState, mMisVcWeightFactor, seed);

		//////////////////////////////////////////////////////////////////////////
        // Trace light path

		int numLightVertex = 0;
        for (;; ++lightState.mPathLength)
        {
			const optix::Ray ray = optix::make_Ray(lightState.mOrigin, lightState.mDirection, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
			PerRayData_closestHit prd;
			rtTrace(top_object, ray, prd);
			if (prd.dist == -1.0f)
				break;

			LightBSDF bsdf;
			bsdf.Setup(ray, prd);
			if (!bsdf.IsValid())
				break;

			const float3 hitPoint = ray.origin + ray.direction * prd.dist;

            // Update the MIS quantities before storing them at the vertex.
            // These updates follow the initialization in GenerateLightSample() or
            // SampleScattering(), and together implement equations [tech. rep. (31)-(33)]
            // or [tech. rep. (34)-(36)], respectively.
            {
                // Infinite lights use MIS handled via solid angle integration,
                // so do not divide by the distance for such lights [tech. rep. Section 5.1]

                if (lightState.mPathLength > 1 || lightState.mIsFiniteLight == 1)
                    lightState.dVCM *= Mis(prd.dist * prd.dist);

				const float den = Mis(abs(bsdf.CosThetaFix()));
                lightState.dVCM /= den;
                lightState.dVC  /= den;
                lightState.dVM  /= den;
            }

            // Store vertex, unless BSDF is purely specular, which prevents
            // vertex connections and merging
            if (!bsdf.mIsDelta)
            {
#ifdef CONNECT_VERTEXES
				lvertexes[numLightVertex].mHitpoint   = hitPoint;
                lvertexes[numLightVertex].mThroughput = lightState.mThroughput;
                lvertexes[numLightVertex].mPathLength = lightState.mPathLength;
                lvertexes[numLightVertex].mBsdf       = bsdf;

                lvertexes[numLightVertex].dVCM = lightState.dVCM;
                lvertexes[numLightVertex].dVC  = lightState.dVC;
                lvertexes[numLightVertex++].dVM  = lightState.dVM;
#endif
	            // Connect to camera, unless BSDF is purely specular

                ConnectToCamera(lightState, hitPoint, bsdf, mLightSubPathCount, mMisVcWeightFactor);
            }

            // Terminate if the path would become too long after scattering
            if (lightState.mPathLength + 2 > mMaxPathLength)
                break;

            // Continue random walk
            if (!SampleScattering(bsdf, hitPoint, lightState, mMisVcWeightFactor, seed))
                break;
        }

		//////////////////////////////////////////////////////////////////////////
        // Generate camera paths
        //////////////////////////////////////////////////////////////////////////

        GenerateCameraSample(launch_index, mLightSubPathCount, cameraState, seed);
        float3 color = make_float3(0);

        //////////////////////////////////////////////////////////////////////
        // Trace camera path

        for (;; ++cameraState.mPathLength)
        {
            // Offset ray origin instead of setting tmin due to numeric
            // issues in ray-sphere intersection. The isect.dist has to be
            // extended by this EPS_RAY after hit point is determined

			const optix::Ray ray = optix::make_Ray(cameraState.mOrigin, cameraState.mDirection, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
			PerRayData_closestHit prd;
			rtTrace(top_object, ray, prd);

            // Get radiance from environment

			if (prd.dist == -1.0f)
			{
				break;
			}

            CameraBSDF bsdf;
			bsdf.Setup(ray, prd);
			if (!bsdf.IsValid())
				break;

			const float3 hitPoint = ray.origin + ray.direction * prd.dist;

            // Update the MIS quantities, following the initialization in
            // GenerateLightSample() or SampleScattering(). Implement equations
            // [tech. rep. (31)-(33)] or [tech. rep. (34)-(36)], respectively.
            {
                cameraState.dVCM *= Mis(prd.dist * prd.dist);
				const float den = Mis(abs(bsdf.CosThetaFix()));
                cameraState.dVCM /= den;
                cameraState.dVC  /= den;
                cameraState.dVM  /= den;
            }

            // Light source has been hit; terminate afterwards, since
            // our light sources do not have reflective properties
		
#ifdef USE_CALLABLE_PROGRAM
			Light light;
			light.SetupAreaLight(prd.p[0], prd.p[1], prd.p[2], prd.mat.mDiffuseReflectance);
			const float3 lrd = GetLightRadiance(0, &light, cameraState, hitPoint, ray.direction);

			if (prd.mat.isEmitter)
			{
                color += cameraState.mThroughput * lrd;
                break;
            }
#else
            if (prd.mat.isEmitter)
			{
				Light light;
				light.SetupAreaLight(prd.p[0], prd.p[1], prd.p[2], prd.mat.mDiffuseReflectance);
				const float3 lrd = GetLightRadiance(0, &light, cameraState, hitPoint, ray.direction);
                color += cameraState.mThroughput * lrd;
                break;
            }
#endif
            // Terminate if eye sub-path is too long for connections or merging
            if (cameraState.mPathLength >= mMaxPathLength)
                break;

            ////////////////////////////////////////////////////////////////
            // Vertex connection: Connect to a light source

            if (!bsdf.mIsDelta)
            {
                color += cameraState.mThroughput *
                    DirectIllumination(cameraState, hitPoint, bsdf, seed);

#ifdef CONNECT_VERTEXES
				////////////////////////////////////////////////////////////////
				// Vertex connection: Connect to light vertices

                // For VC, each light sub-path is assigned to a particular eye
                // sub-path, as in traditional BPT. It is also possible to
                // connect to vertices from any light path, but MIS should
                // be revisited.

#ifndef FAST_CONNECTION
                for (int i = 0; i < numLightVertex; ++i)
#endif
                {
#ifdef FAST_CONNECTION
					const int i = (int)(rnd(seed) * numLightVertex);
#endif
                    const LightVertex &lightVertex = lvertexes[i];

                    // Light vertices are stored in increasing path length
                    // order; once we go above the max path length, we can
                    // skip the rest
                    if (lightVertex.mPathLength + 1 +
                        cameraState.mPathLength > mMaxPathLength)
                        break;

                    color += cameraState.mThroughput * lightVertex.mThroughput *
                        ConnectVertices(lightVertex, bsdf, hitPoint, cameraState);
                }
#endif
            }

            if (!SampleScattering(bsdf, hitPoint, cameraState, mMisVcWeightFactor, seed))
                break;
        }

		float *const base_output = (float *)&temp_buffer[make_uint2(0, 0)];
		const uint offset = (launch_index.x + launch_index.y * temp_buffer.size().x) << 2;
		atomicAdd(base_output + offset,     color.x);
		atomicAdd(base_output + offset + 1, color.y);
		atomicAdd(base_output + offset + 2, color.z);
	}
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
	temp_buffer[launch_index] = make_float4(bad_color, 0.0f);
}