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
		 const float3      &rayDir,
		 const float3      &isxPt,
		 float             *oDirectPdfA,
		 float             *oEmissionPdfW));

#else
#include "AreaLight.h"
#endif

#ifdef STRATIFIED
rtBuffer<float2> strat_buffer;
#endif

// Samples light emission
__device__ __forceinline__ void SampleLight(SubPathState &lightSubPath, const float &MisScale, unsigned int &seed)
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

	//accumulating   for light subpath; look: Recursive   with BDPT on GPU
    {
		lightSubPath.allMIS = dirPdf / lightPdf;
      
         const float usedCosLight = cosLight;
		 lightSubPath.bdptMIS = usedCosLight / lightPdf;
    
		 lightSubPath.elseMIS = lightSubPath.bdptMIS * MisScale;
    }
}

__device__ __forceinline__ void JoinWithCamera(
    const SubPathState &aLightState,
    const float3       &isxPt,
    const LightBSDF    &bxdf,
	const float totalLightPaths,
	const float MisScale)
{
	float3 directionToCamera = camera.mPosition - isxPt;

	// check if its in front
    if (dot(W, -directionToCamera) <= 0.f)
        return;

    const float2 imagePos = camera.WorldToRaster(isxPt);
    if (!camera.CheckRaster(imagePos))
        return;

    //pixel space
    const float distEye2 = dot(directionToCamera, directionToCamera);
    const float distance = sqrtf(distEye2);
    directionToCamera   /= distance;

    // calculate brdf
    float cosToCamera, bxdfDirectPdf, bsdfRevPdfW;
    const float3 bsdfFactor = bxdf.Evaluate(
        directionToCamera, cosToCamera, &bxdfDirectPdf, &bsdfRevPdfW);

    if (isZero(bsdfFactor))
        return;

    bsdfRevPdfW *= bxdf.continuationProb;

	//image/pixel space calculation
	const float cosAtCamera = dot(camera.mForward, -directionToCamera);
    const float imagePointToCameraDist = camera.mImagePlaneDist / cosAtCamera;
	const float aspect = camera.mResolution.y / camera.mResolution.x;
	const float imageToSolidAngleFactor = aspect * aspect * (imagePointToCameraDist * imagePointToCameraDist) / cosAtCamera;
    const float imageToSurfaceFactor = imageToSolidAngleFactor * abs(cosToCamera) / (distance * distance);

    const float cameraPdfA = imageToSurfaceFactor;

    const float wLight = cameraPdfA / totalLightPaths * (
        aLightState.allMIS + aLightState.bdptMIS * bsdfRevPdfW);

    const float misWeight = 1.f / (wLight + 1.f);  

    const float surfaceToImageFactor = 1.f / imageToSurfaceFactor;

    const float3 contrib = misWeight * aLightState.throughput * bsdfFactor /
        (totalLightPaths * surfaceToImageFactor);

    if (!isZero(contrib))
    {
		PerRayData_occlusion prd_occlusion;
		//trace occlusion ray
		const optix::Ray shadow_ray = optix::make_Ray(isxPt, directionToCamera, shadow_ray_type, scene_epsilon, distance);
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
    const BSDF<tLightSample> &bxdf,
    const float3             &isxPt,
    SubPathState             &sState,
	const float MisScale,
	uint &seed)
{
    const float3 rndTriplet = make_float3(rnd(seed), rnd(seed), rnd(seed));
    float bxdfDirectPdf, cosThetaOut;
    uint  sampledEvent;

    const float3 bsdfFactor = bxdf.Sample(rndTriplet, sState.direction,
        bxdfDirectPdf, cosThetaOut, &sampledEvent);

    if (isZero(bsdfFactor))
        return false;

    float bsdfRevPdfW = bxdfDirectPdf;
    if ((sampledEvent & kSpecular) == 0)
        bsdfRevPdfW = bxdf.Pdf(sState.direction, true);

    // exit strategy
    const float contProb = bxdf.continuationProb;
    if (rnd(seed) > contProb)
        return false;

    bxdfDirectPdf *= contProb;
    bsdfRevPdfW *= contProb;


	//for spec path, scatter and calculate  
    if (sampledEvent & kSpecular)
    {
        sState.allMIS = 0.f;
        sState.bdptMIS *= cosThetaOut;
        sState.elseMIS *= cosThetaOut;

        sState.specPath &= 1;
    }
    else
    {
        sState.bdptMIS = cosThetaOut / bxdfDirectPdf * (
            sState.bdptMIS * (bsdfRevPdfW) +
            sState.allMIS);

        sState.elseMIS =  (cosThetaOut / bxdfDirectPdf) * (
            sState.elseMIS *  (bsdfRevPdfW) +
            sState.allMIS * MisScale + 1.f);

        sState.allMIS =  (1.f / bxdfDirectPdf);

        sState.specPath &= 0;
    }

    sState.origin  = isxPt;
    sState.throughput *= bsdfFactor * (cosThetaOut / bxdfDirectPdf);
        
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
	const Light *const light,
    const SubPathState   &eyeSubPath,
    const float3         &isxPt,
    const float3         &rayDir)
{

	const int lightCount = lightBuffer.size();
    const float lightPickProb = 1.0f / lightCount;

    float directPdfA, emissionPdfW;
	float3 radiance;
    
	//sample area light
	radiance = GetRadianceAreaLight(light, rayDir, isxPt, &directPdfA, &emissionPdfW);

    if (isZero(radiance))
        return make_float3(0);

	// exit if camera hits light directly.
    if (eyeSubPath.pathlen == 1)
        return radiance;

    directPdfA   *= lightPickProb;
    emissionPdfW *= lightPickProb;

    const float wCamera =  (directPdfA) * eyeSubPath.allMIS +
         (emissionPdfW) * eyeSubPath.bdptMIS;

    const float misWeight = 1.f / (1.f + wCamera);
        
    return misWeight * radiance;
}

__device__ __forceinline__ void GenerateCameraSample(
	const uint2 xy,
	const float totalLightPaths,
    SubPathState &eyeState,
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

    eyeState.origin       = org;
    eyeState.direction    = dir;
    eyeState.throughput   = make_float3(1.0f);

    eyeState.pathlen   = 1;
    eyeState.specPath = 1;

    eyeState.allMIS =  (totalLightPaths / cameraPdfW);
    eyeState.bdptMIS  = 0;
    eyeState.elseMIS  = 0;
}

__device__ __forceinline__ float3 DirectLighting(
    const SubPathState     &eyeSubPath,
    const float3           &isxPt,
    const CameraBSDF       &bxdf,
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
	radiance = IlluminateAreaLight(&light, isxPt,
				rndPosSamples, directionToLight, distance, directPdfW,
				&emissionPdfW, &cosAtLight);

	//no point if radiance is zero. means hit no object or no point going further
    if (isZero(radiance))
        return make_float3(0);

    float bxdfDirectPdf, bsdfRevPdfW, cosToLight;
    const float3 bsdfFactor = bxdf.Evaluate(
        directionToLight, cosToLight, &bxdfDirectPdf, &bsdfRevPdfW);

    if (isZero(bsdfFactor))
        return make_float3(0);

    const float continuationProbability = bxdf.continuationProb;
        
    bxdfDirectPdf *= continuationProbability;

    bsdfRevPdfW *= continuationProbability;

    const float wLight =  (bxdfDirectPdf / (lightPickProb * directPdfW));

    const float wCamera =  (emissionPdfW * cosToLight / (directPdfW * cosAtLight)) * (
        eyeSubPath.allMIS + eyeSubPath.bdptMIS *  (bsdfRevPdfW));

	//same as getting light randiance
    const float misWeight = 1.f / (wLight + 1.f + wCamera);

    const float3 contrib =
        (misWeight * cosToLight / (lightPickProb * directPdfW)) * (radiance * bsdfFactor);

    if (isZero(contrib))
        return make_float3(0.0f);

	PerRayData_occlusion prd_occlusion;
	const optix::Ray shadow_ray = optix::make_Ray(isxPt, directionToLight, shadow_ray_type, scene_epsilon, distance);
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
    const CameraBSDF       &bxdfEye,
    const float3           &eyeIsxPt,
    const SubPathState     &eyeSubPath)
{
	clock_t start_time = clock();
	//get the vertices to be joined
    float3 direction   = aLightVertex.isxPoint - eyeIsxPt;
    const float dist2 = dot(direction, direction);
    float  distance   = sqrtf(dist2);
    direction        /= distance;

    //brdf at cam
    float cosCamera, bxdfDirectEye, cameraBsdfRevPdfW;
    const float3 cameraBsdfFactor = bxdfEye.Evaluate(
        direction, cosCamera, &bxdfDirectEye,
        &cameraBsdfRevPdfW);

    if (isZero(cameraBsdfFactor))
        return make_float3(0.0f);

    //exit strategy
    const float cameraCont = bxdfEye.continuationProb;
    bxdfDirectEye *= cameraCont;
    cameraBsdfRevPdfW *= cameraCont;

    //brdf at light
    float cosLight, bxdfDirectLight, bxdfReverseLight;
    const float3 lightBsdfFactor = aLightVertex.bsdf.Evaluate(
        -direction, cosLight, &bxdfDirectLight,
        &bxdfReverseLight);

    if (isZero(lightBsdfFactor))
        return make_float3(0);

    const float lightCont = aLightVertex.bsdf.continuationProb;
    bxdfDirectLight *= lightCont;
    bxdfReverseLight *= lightCont;

    const float geometryTerm = cosLight * cosCamera / dist2;
    if (geometryTerm < 0)
        return make_float3(0);

    // Convert pdfs to area pdf
    const float cameraBsdfDirPdfA = PdfWtoA(bxdfDirectEye, distance, cosLight);
    const float lightBsdfDirPdfA  = PdfWtoA(bxdfDirectLight,  distance, cosCamera);

    const float wLight =  (cameraBsdfDirPdfA) * (
        aLightVertex.allMIS + aLightVertex.bdptMIS *  (bxdfReverseLight));

    const float wCamera =  (lightBsdfDirPdfA) * (
       eyeSubPath.allMIS + eyeSubPath.bdptMIS *  (cameraBsdfRevPdfW));

    const float misWeight = 1.f / (wLight + 1.f + wCamera);

    const float3 contrib = (misWeight * geometryTerm) * cameraBsdfFactor * lightBsdfFactor;

    if (isZero(contrib))
		return make_float3(0.0f);

	PerRayData_occlusion prd_occlusion;
	const optix::Ray shadow_ray = optix::make_Ray(eyeIsxPt, direction, shadow_ray_type, scene_epsilon, distance);
	rtTrace(top_object, shadow_ray, prd_occlusion);
    if (prd_occlusion.occluded)
		return make_float3(0.0f);
	clock_t stop_time = clock();
	int time = (int)(stop_time - start_time);
	//rtPrintf("time in JoinVertices %fms\n", time / 1038000.f);
    return contrib;
}

//#define FAST_CONNECTION
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

	const float totalLightPaths = (float)(launch_dim.x * launch_dim.y);
	const float mBaseRadius = aRadiusFactor * (9.5f / 2.0f) * sqrtf(2.0f);
    float radius = mBaseRadius / powf(float(frame_number), 0.5f * (1 - aRadiusAlpha));
    
	// Apparently for numeric stability
    radius = max(radius, 1e-7f);
    const float radiusSqr = radius * radius;

	const float etaVCM = (PI_F * radiusSqr) * totalLightPaths;
	const float MisScale =  (1.0f / etaVCM);

	const int spp = 1;
	//for samples per pixel
	for (int k = 0; k < spp; ++k)
	{
		SampleLight(lightState, MisScale, seed);

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
					lightState.allMIS *=  (prd.dist * prd.dist);

				const float den =  (abs(bsdf.CosThetaFix()));
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

				JoinWithCamera(lightState, hitPoint, bsdf, totalLightPaths, MisScale);
			}
			if (lightState.pathlen + 2 > mMaxPathLength)
				break;
			if (!SampleScattering(bsdf, hitPoint, lightState, MisScale, seed))
				break;
		}


		GenerateCameraSample(launch_index, totalLightPaths, cameraState, seed);
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
				cameraState.allMIS *=  (prd.dist * prd.dist);
				const float den =  (abs(bsdf.CosThetaFix()));
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

			if (!SampleScattering(bsdf, hitPoint, cameraState, MisScale, seed))
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
	float scale = 0.1f;
	atomicAdd(base_output + offset,		time / (1038000.f * scale));
	atomicAdd(base_output + offset + 1, time / (1038000.f * scale));
	atomicAdd(base_output + offset + 2, time / (1038000.f * scale));*/
	//rtPrintf("time in pinhole_camera %fms\n", time / 1038000.f);
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
	temp_buffer[launch_index] = make_float4(bad_color, 0.0f);
}