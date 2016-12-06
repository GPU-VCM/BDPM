#include <optix_world.h>
#include "commonStructs.h"
#include "random.h"
#include "BSDF.h"
#include "Camera.h"
#include "EventTimer.h"

using namespace optix;

#define STRATIFIED

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

rtDeclareVariable(Camera,        camera, , );
rtDeclareVariable(float,         vfov, ,);

rtDeclareVariable(int,  mMaxPathLength, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

rtBuffer<int, 1> raysTracedThisFrame; // should allow for a buffer with room for one integer

//#define USE_CALLABLE_PROGRAM

#ifdef USE_CALLABLE_PROGRAM
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

#else
#include "AreaLight.h"
#endif

#ifdef STRATIFIED
rtBuffer<float2> strat_buffer;
#endif

// Samples light emission
__device__ __forceinline__ void SampleLight(SubPathState &lightSubPath, const float &mMisVcWeightFactor, unsigned int &seed)
{
	int nlights = lightBuffer.size();
	float lightPickProb = 1.0f / nlights;

	int lightIdx = (int)(rnd(seed) * nlights);
	//sampling position on light and direction from diffuse light
    float2 rndDirSamples = make_float2(rnd(seed), rnd(seed));
    float2 rndPosSamples = make_float2(rnd(seed), rnd(seed));

	float lightPdf, dirPdf, cosLight;
	Light &light = lightBuffer[lightIdx];
	//rtPrintf("lightIX: %d\n",lightIx);
	lightSubPath.throughput = EmitAreaLight(&light, rndDirSamples, rndPosSamples,
		lightSubPath.origin, lightSubPath.direction,
		lightPdf, &dirPdf, &cosLight);

	//over all lights; simplicity, start with 1 light
    lightPdf *= lightPickProb;
    dirPdf   *= lightPickProb;

	lightSubPath.throughput /= lightPdf;
	lightSubPath.pathlen = 1;

	//accumulating mis for light subpath; look: Recursive MIS with BDPT on GPU
    {
		lightSubPath.allMIS = Mis(dirPdf / lightPdf);
      
         const float usedCosLight = cosLight;
		 lightSubPath.bdptMIS = Mis(usedCosLight / lightPdf);
    
		 lightSubPath.elseMIS = lightSubPath.bdptMIS * mMisVcWeightFactor;
    }
}

__device__ __forceinline__ void JoinWithCamera(
    const SubPathState &aLightState,
    const float3       &aHitpoint,
    const LightBSDF    &aBsdf,
	const float mLightSubPathCount,
	const float mMisVcWeightFactor)
{
	float3 directionToCamera = camera.mPosition - aHitpoint;

	// check if its in front
    if (dot(W, -directionToCamera) <= 0.f)
        return;

    const float2 imagePos = camera.WorldToRaster(aHitpoint);
    if (!camera.CheckRaster(imagePos))
        return;

    //pixel space
    const float distEye2 = dot(directionToCamera, directionToCamera);
    const float distance = sqrtf(distEye2);
    directionToCamera   /= distance;

    // calculate brdf
    float cosToCamera, bsdfDirPdfW, bsdfRevPdfW;
    const float3 bsdfFactor = aBsdf.Evaluate(
        directionToCamera, cosToCamera, &bsdfDirPdfW, &bsdfRevPdfW);

    if (isZero(bsdfFactor))
        return;

    bsdfRevPdfW *= aBsdf.continuationProb;

	//image/pixel space calculation
	const float cosAtCamera = dot(camera.mForward, -directionToCamera);
    const float imagePointToCameraDist = camera.mImagePlaneDist / cosAtCamera;
	const float aspect = camera.mResolution.y / camera.mResolution.x;
	const float imageToSolidAngleFactor = aspect * aspect * (imagePointToCameraDist * imagePointToCameraDist) / cosAtCamera;
    const float imageToSurfaceFactor = imageToSolidAngleFactor * abs(cosToCamera) / (distance * distance);

    const float cameraPdfA = imageToSurfaceFactor;

    const float wLight = Mis(cameraPdfA / mLightSubPathCount) * (
        aLightState.allMIS + aLightState.bdptMIS * Mis(bsdfRevPdfW));

    const float misWeight = 1.f / (wLight + 1.f);  

    const float surfaceToImageFactor = 1.f / imageToSurfaceFactor;

    const float3 contrib = misWeight * aLightState.throughput * bsdfFactor /
        (mLightSubPathCount * surfaceToImageFactor);

    if (!isZero(contrib))
    {
		PerRayData_occlusion prd_occlusion;
		//trace occlusion ray
		const optix::Ray shadow_ray = optix::make_Ray(aHitpoint, directionToCamera, shadow_ray_type, scene_epsilon, distance);
		rtTrace(top_object, shadow_ray, prd_occlusion);
        if (!prd_occlusion.occluded)
        {
			const uint2 ups = make_uint2(imagePos.x, imagePos.y);
			float *const base_output = (float *)&temp_buffer[make_uint2(0, 0)];
			const uint offset = (ups.x + ups.y * temp_buffer.size().x) << 2;
			//add color into temp buffer
			atomicAdd(base_output + offset,     contrib.x);
			atomicAdd(base_output + offset + 1, contrib.y);
			atomicAdd(base_output + offset + 2, contrib.z);
        }
    }
}

//following similar structure as smallvcm
template<bool tLightSample>
__device__ __forceinline__ bool SampleScattering(
    const BSDF<tLightSample> &aBsdf,
    const float3             &aHitPoint,
    SubPathState             &aoState,
	const float mMisVcWeightFactor,
	uint &seed)
{
    const float3 rndTriplet = make_float3(rnd(seed), rnd(seed), rnd(seed));
    float bsdfDirPdfW, cosThetaOut;
    uint  sampledEvent;

    const float3 bsdfFactor = aBsdf.Sample(rndTriplet, aoState.direction,
        bsdfDirPdfW, cosThetaOut, &sampledEvent);

    if (isZero(bsdfFactor))
        return false;

    float bsdfRevPdfW = bsdfDirPdfW;
    if ((sampledEvent & kSpecular) == 0)
        bsdfRevPdfW = aBsdf.Pdf(aoState.direction, true);

    // exit strategy
    const float contProb = aBsdf.continuationProb;
    if (rnd(seed) > contProb)
        return false;

    bsdfDirPdfW *= contProb;
    bsdfRevPdfW *= contProb;


	//for spec path, scatter and calculate mis
    if (sampledEvent & kSpecular)
    {
        aoState.allMIS = 0.f;
        aoState.bdptMIS *= Mis(cosThetaOut);
        aoState.elseMIS *= Mis(cosThetaOut);

        aoState.specPath &= 1;
    }
    else
    {
        aoState.bdptMIS = Mis(cosThetaOut / bsdfDirPdfW) * (
            aoState.bdptMIS * Mis(bsdfRevPdfW) +
            aoState.allMIS);

        aoState.elseMIS = Mis(cosThetaOut / bsdfDirPdfW) * (
            aoState.elseMIS * Mis(bsdfRevPdfW) +
            aoState.allMIS * mMisVcWeightFactor + 1.f);

        aoState.allMIS = Mis(1.f / bsdfDirPdfW);

        aoState.specPath &= 0;
    }

    aoState.origin  = aHitPoint;
    aoState.throughput *= bsdfFactor * (cosThetaOut / bsdfDirPdfW);
        
    return true;
}

RT_PROGRAM void clear_temp()
{
	temp_buffer[launch_index] = make_float4(0, 0, 0, 0);
}

RT_PROGRAM void cumulate_samples()
{
	// move temp buffer to output (frame) buffer. to avoid race conditions in calculations for temp buffer
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

__device__ __forceinline__ float3 getRadiance(
	const int			 lightType,
	const Light *const aLight,
    const SubPathState   &aCameraState,
    const float3         &aHitpoint,
    const float3         &aRayDirection)
{

	const int lightCount = lightBuffer.size();
    const float lightPickProb = 1.0f / lightCount;

    float directPdfA, emissionPdfW;
	float3 radiance;
    
	//sample area light
	radiance = GetRadianceAreaLight(aLight, aRayDirection, aHitpoint, &directPdfA, &emissionPdfW);

    if (isZero(radiance))
        return make_float3(0);

	// exit if camera hits light directly.
    if (aCameraState.pathlen == 1)
        return radiance;

    directPdfA   *= lightPickProb;
    emissionPdfW *= lightPickProb;

    const float wCamera = Mis(directPdfA) * aCameraState.allMIS +
        Mis(emissionPdfW) * aCameraState.bdptMIS;

    const float misWeight = 1.f / (1.f + wCamera);
        
    return misWeight * radiance;
}

__device__ __forceinline__ void GenerateCameraSample(
	const uint2 xy,
	const float mLightSubPathCount,
    SubPathState &oCameraState,
	uint &seed)
{
	const float2 jitter = make_float2(rnd(seed), rnd(seed));
	float3 org, dir;
	camera.GenerateRay((make_float2(xy) + jitter), org, dir);

    const float cosAtCamera = dot(camera.mForward, dir);
    const float imagePointToCameraDist = camera.mImagePlaneDist / cosAtCamera;
	const float aspect = camera.mResolution.y / camera.mResolution.x;
    const float imageToSolidAngleFactor = aspect * aspect * (imagePointToCameraDist * imagePointToCameraDist) / cosAtCamera;
	//in effect to convert pdf to solid angle pdf at image space

    const float cameraPdfW = imageToSolidAngleFactor;

    oCameraState.origin       = org;
    oCameraState.direction    = dir;
    oCameraState.throughput   = make_float3(1.0f);

    oCameraState.pathlen   = 1;
    oCameraState.specPath = 1;

    oCameraState.allMIS = Mis(mLightSubPathCount / cameraPdfW);
    oCameraState.bdptMIS  = 0;
    oCameraState.elseMIS  = 0;
}

__device__ __forceinline__ float3 DirectLighting(
    const SubPathState     &aCameraState,
    const float3           &aHitpoint,
    const CameraBSDF       &aBsdf,
	uint				   &seed)
{
	clock_t start_time = clock();
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

	//no point if radiance is zero. means hit no object or no point going further
    if (isZero(radiance))
        return make_float3(0);

    float bsdfDirPdfW, bsdfRevPdfW, cosToLight;
    const float3 bsdfFactor = aBsdf.Evaluate(
        directionToLight, cosToLight, &bsdfDirPdfW, &bsdfRevPdfW);

    if (isZero(bsdfFactor))
        return make_float3(0);

    const float continuationProbability = aBsdf.continuationProb;
        
    bsdfDirPdfW *= continuationProbability;

    bsdfRevPdfW *= continuationProbability;

    const float wLight = Mis(bsdfDirPdfW / (lightPickProb * directPdfW));

    const float wCamera = Mis(emissionPdfW * cosToLight / (directPdfW * cosAtLight)) * (
        aCameraState.allMIS + aCameraState.bdptMIS * Mis(bsdfRevPdfW));

	//same as getting light randiance
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
	clock_t stop_time = clock();
	int time = (int)(stop_time - start_time);
	//rtPrintf("time in JoinVertices %fms\n", time / 1038000.f);
	//rtPrintf("contrib: %f, %f, %f\n", contrib.x, contrib.y, contrib.z);
    return contrib;
}

__device__ __forceinline__ float3 JoinVertices(
    const LightVertex      &aLightVertex,
    const CameraBSDF       &aCameraBsdf,
    const float3           &aCameraHitpoint,
    const SubPathState     &aCameraState)
{
	clock_t start_time = clock();
	//get the vertices to be joined
    float3 direction   = aLightVertex.isxPoint - aCameraHitpoint;
    const float dist2 = dot(direction, direction);
    float  distance   = sqrtf(dist2);
    direction        /= distance;

    //brdf at cam
    float cosCamera, cameraBsdfDirPdfW, cameraBsdfRevPdfW;
    const float3 cameraBsdfFactor = aCameraBsdf.Evaluate(
        direction, cosCamera, &cameraBsdfDirPdfW,
        &cameraBsdfRevPdfW);

    if (isZero(cameraBsdfFactor))
        return make_float3(0.0f);

    //exit strategy
    const float cameraCont = aCameraBsdf.continuationProb;
    cameraBsdfDirPdfW *= cameraCont;
    cameraBsdfRevPdfW *= cameraCont;

    //brdf at light
    float cosLight, lightBsdfDirPdfW, lightBsdfRevPdfW;
    const float3 lightBsdfFactor = aLightVertex.bsdf.Evaluate(
        -direction, cosLight, &lightBsdfDirPdfW,
        &lightBsdfRevPdfW);

    if (isZero(lightBsdfFactor))
        return make_float3(0);

    const float lightCont = aLightVertex.bsdf.continuationProb;
    lightBsdfDirPdfW *= lightCont;
    lightBsdfRevPdfW *= lightCont;

    const float geometryTerm = cosLight * cosCamera / dist2;
    if (geometryTerm < 0)
        return make_float3(0);

    // Convert pdfs to area pdf
    const float cameraBsdfDirPdfA = PdfWtoA(cameraBsdfDirPdfW, distance, cosLight);
    const float lightBsdfDirPdfA  = PdfWtoA(lightBsdfDirPdfW,  distance, cosCamera);

    const float wLight = Mis(cameraBsdfDirPdfA) * (
        aLightVertex.allMIS + aLightVertex.bdptMIS * Mis(lightBsdfRevPdfW));

    const float wCamera = Mis(lightBsdfDirPdfA) * (
       aCameraState.allMIS + aCameraState.bdptMIS * Mis(cameraBsdfRevPdfW));

    const float misWeight = 1.f / (wLight + 1.f + wCamera);

    const float3 contrib = (misWeight * geometryTerm) * cameraBsdfFactor * lightBsdfFactor;

    if (isZero(contrib))
		return make_float3(0.0f);

	PerRayData_occlusion prd_occlusion;
	const optix::Ray shadow_ray = optix::make_Ray(aCameraHitpoint, direction, shadow_ray_type, scene_epsilon, distance);
	rtTrace(top_object, shadow_ray, prd_occlusion);
    if (prd_occlusion.occluded)
		return make_float3(0.0f);
	clock_t stop_time = clock();
	int time = (int)(stop_time - start_time);
	//rtPrintf("time in JoinVertices %fms\n", time / 1038000.f);
    return contrib;
}

#define FAST_CONNECTION
#define CONNECT_VERTEXES

RT_PROGRAM void pinhole_camera()
{
	//rtPrintf("here in pinhole camera");
#ifdef CONNECT_VERTEXES
	LightVertex lvertexes[10]; 
#endif
	clock_t start_time = clock();
	SubPathState lightState, cameraState;

	size_t2 screen = temp_buffer.size();
	unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame_number);

	const float mLightSubPathCount = (float)(launch_dim.x * launch_dim.y);
	const float mBaseRadius = aRadiusFactor * (9.5f / 2.0f) * sqrtf(2.0f);
    float radius = mBaseRadius / powf(float(frame_number), 0.5f * (1 - aRadiusAlpha));
    
	// Apparently for numeric stability
    radius = max(radius, 1e-7f);
    const float radiusSqr = radius * radius;

	const float etaVCM = (PI_F * radiusSqr) * mLightSubPathCount;
	const float mMisVcWeightFactor = Mis(1.0f / etaVCM);

	const int spp = 1;
	//for samples per pixel
	for (int k = 0; k < spp; ++k)
	{
		SampleLight(lightState, mMisVcWeightFactor, seed);

		int numLightVertex = 0;
		for (;; ++lightState.pathlen)
		{
			const optix::Ray ray = optix::make_Ray(lightState.origin, lightState.direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
			PerRayData_closestHit prd;
			//trace light rays and save radiance
			atomicAdd(&raysTracedThisFrame[0], 1);
			rtTrace(top_object, ray, prd);
			if (prd.dist == -1.0f)
				break;

			LightBSDF bsdf;
			bsdf.Setup(ray, prd);
			if (!bsdf.IsValid())
				break;

			const float3 hitPoint = ray.origin + ray.direction * prd.dist;
			{

				if (lightState.pathlen > 1 || lightState.isLgtFinite == 1)
					lightState.allMIS *= Mis(prd.dist * prd.dist);

				const float den = Mis(abs(bsdf.CosThetaFix()));
				lightState.allMIS /= den;
				lightState.bdptMIS /= den;
				lightState.elseMIS /= den;
			}

			if (!bsdf.isDelta)
			{
#ifdef CONNECT_VERTEXES
				lvertexes[numLightVertex].isxPoint   = hitPoint;
				lvertexes[numLightVertex].throughput = lightState.throughput;
				lvertexes[numLightVertex].pathlen = lightState.pathlen;
				lvertexes[numLightVertex].bsdf       = bsdf;

				lvertexes[numLightVertex].allMIS = lightState.allMIS;
				lvertexes[numLightVertex].bdptMIS  = lightState.bdptMIS;
				lvertexes[numLightVertex++].elseMIS  = lightState.elseMIS;
#endif

				JoinWithCamera(lightState, hitPoint, bsdf, mLightSubPathCount, mMisVcWeightFactor);
			}
			if (lightState.pathlen + 2 > mMaxPathLength)
				break;
			if (!SampleScattering(bsdf, hitPoint, lightState, mMisVcWeightFactor, seed))
				break;
		}


		GenerateCameraSample(launch_index, mLightSubPathCount, cameraState, seed);
		float3 color = make_float3(0);

		//trace eye/camera rays and save radiance to add with light subpaths
		for (;; ++cameraState.pathlen)
		{
			const optix::Ray ray = optix::make_Ray(cameraState.origin, cameraState.direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
			PerRayData_closestHit prd;
			rtTrace(top_object, ray, prd);

			if (prd.dist == -1.0f)
			{
				break;
			}

			CameraBSDF bsdf;
			bsdf.Setup(ray, prd);
			if (!bsdf.IsValid())
				break;

			const float3 hitPoint = ray.origin + ray.direction * prd.dist;
			{
				cameraState.allMIS *= Mis(prd.dist * prd.dist);
				const float den = Mis(abs(bsdf.CosThetaFix()));
				cameraState.allMIS /= den;
				cameraState.bdptMIS /= den;
				cameraState.elseMIS /= den;
			}

#ifdef USE_CALLABLE_PROGRAM
			Light light;
			light.SetupAreaLight(prd.p[0], prd.p[1], prd.p[2], prd.mat.diffusePart);
			const float3 lrd = getRadiance(0, &light, cameraState, hitPoint, ray.direction);

			if (prd.mat.isEmitter)
			{
				color += cameraState.throughput * lrd;
				break;
			}
#else
			if (prd.mat.isEmitter)
			{
				Light light;
				//calculate light radiance if hit with camera direct; maybe remove the setup part from here
				light.SetupAreaLight(prd.p[0], prd.p[1], prd.p[2], prd.mat.diffusePart);
				const float3 lrd = getRadiance(0, &light, cameraState, hitPoint, ray.direction);
				color += cameraState.throughput * lrd;

				break;
			}
#endif
			if (cameraState.pathlen >= mMaxPathLength)
				break;
			if (!bsdf.isDelta)
			{
				color += cameraState.throughput *
					DirectLighting(cameraState, hitPoint, bsdf, seed);


#ifdef CONNECT_VERTEXES


#ifndef FAST_CONNECTION
				for (int i = 0; i < numLightVertex; ++i)
#endif
				{
#ifdef FAST_CONNECTION
					const int i = (int)(rnd(seed) * numLightVertex);
#endif
					const LightVertex &lightVertex = lvertexes[i];

					if (lightVertex.pathlen + 1 +
						cameraState.pathlen > mMaxPathLength)
						break;

					color += cameraState.throughput * lightVertex.throughput *
						JoinVertices(lightVertex, bsdf, hitPoint, cameraState);
					//rtPrintf("camera thruput: %f\n", cameraState.throughput);
					//rtPrintf("light thruput: %f\n", lightVertex.throughput);
				}
#endif
			}

			if (!SampleScattering(bsdf, hitPoint, cameraState, mMisVcWeightFactor, seed))
				break;
		}
		//rtPrintf("color: %f, %f, %f\n", color.x, color.y, color.z);
		//color = make_float3(10.4f, 0.f, 0.f);
		float *const base_output = (float *)&temp_buffer[make_uint2(0, 0)];
		const uint offset = (launch_index.x + launch_index.y * temp_buffer.size().x) << 2;
		atomicAdd(base_output + offset, color.x);
		atomicAdd(base_output + offset + 1, color.y);
		atomicAdd(base_output + offset + 2, color.z);
	}
	clock_t stop_time = clock();
	float *const base_output = (float *)&temp_buffer[make_uint2(0, 0)];
	const uint offset = (launch_index.x + launch_index.y * temp_buffer.size().x) << 2;

	int *const temp = (int*)&temp_buffer[make_uint2(0, 0)];

	int time = (int)(stop_time - start_time);
	/*atomicMin(temp + offset, 0);
	atomicMin(temp + offset + 1, 0);
	atomicMin(temp + offset + 2, 0);
	atomicAdd(base_output + offset,		time / 10380000.f);
	atomicAdd(base_output + offset + 1, time / 10380000.f);
	atomicAdd(base_output + offset + 2, time / 10380000.f);*/
	//rtPrintf("time in pinhole_camera %fms\n", time / 1038000.f);
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
	temp_buffer[launch_index] = make_float4(bad_color, 0.0f);
}