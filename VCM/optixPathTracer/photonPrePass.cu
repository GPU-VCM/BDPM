/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optixu/optixu_math_namespace.h>
#include "optixPathTracer.h"
#include "random.h"
#include <stdio.h>
#include <time.h>

using namespace optix;

struct PerRayData_pathtrace
{
    float3 result;
    float3 radiance;
    float3 attenuation;
    float3 origin;
    float3 direction;
    unsigned int seed;
    int depth;
    int countEmitted;
    int done;

	float tValue;
	int isSpecular;
};

struct PerRayData_pathtrace_shadow
{
    bool inShadow;
};

// Scene wide variables
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );



//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  sqrt_num_samples, , );
rtDeclareVariable(unsigned int,  rr_begin_depth, , );
rtDeclareVariable(unsigned int,  pathtrace_ray_type, , );
rtDeclareVariable(unsigned int,  pathtrace_shadow_ray_type, , );
rtDeclareVariable(int,  row, , );
rtDeclareVariable(int,  maxDepth, , );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<ParallelogramLight>     lights;

rtBuffer<Photon>	photonBuffer;
rtBuffer<int>	isHitBuffer;

#define DECRESE_FACTOR 0.3

RT_PROGRAM void pathtrace_camera()
{
	/*clock_t start_time = clock();*/

    size_t2 screen = output_buffer.size();
	//int offset = frame_number * screen.x * screen.y * 5;
	//printf("%d\n", offset);
	unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame_number);
    //float2 u1u2 = make_float2(launch_index) / make_float2(screen);
	float2 u1u2;
	u1u2.x = rnd(seed);
	u1u2.y = rnd(seed);
	//printf("%d\n", frame_number);

    float3 result = make_float3(0.0f);
	int index = screen.x*launch_index.y+launch_index.x;

	//float r = sqrt(1.0f - u1u2.x * u1u2.x);
	//float phi = 2 * M_PI *u1u2.y;
	//float3 dir = make_float3(cos(phi) * r, -u1u2.x, sin(phi) * r);
	//printf("%f %f %f\n", dir.x, dir.y, dir.z);
	//float r = sqrt(1.0f - u1u2.x * u1u2.x);
	float phi = 2 * M_PI * u1u2.y;
	float theta = M_PI * 0.5f * u1u2.x;
	float r = sin(theta);
	float3 dir = make_float3(r * cos(phi), -cos(theta), r * sin(phi));
	//printf("%f %f %f\n", dir.x, dir.y, dir.z);
	
	int xx = frame_number / row;
	int yy = frame_number % row;
	float d = 1.f / (row - 1);

	ParallelogramLight light = lights[0];
	const float z1 = rnd(seed);
	const float z2 = rnd(seed);
	const float3 light_pos = light.corner + light.v1 * d * xx + light.v2 * d * yy;
	//const float3 light_pos = light.corner + light.v1 * 0.5f + light.v2 * 0.5f;
	//printf("%d %f %f\n",row,  d * xx, d * yy);
	//printf("%f %f %f\n", light_pos.x, light_pos.y, light_pos.z);

    float3 ray_origin = light_pos;
    float3 ray_direction = dir;
	//printf("%d\n", maxDepth);

	float3 firstRay_direction = ray_direction;
	//bool firstIntersection = false;
	float t;
    // Initialze per-ray data
    PerRayData_pathtrace prd;
    prd.result = make_float3(0.f);
    prd.attenuation = make_float3(1.f);
	prd.radiance = make_float3(0.45f, 0.45f, 0.15f);
    prd.countEmitted = true;
    prd.done = false;
	prd.seed = seed;
    prd.depth = 0;
	prd.isSpecular = 0;

	for (int i = 0; i < maxDepth; i++)
		isHitBuffer[maxDepth * index + i] = 0;
    // Each iteration is a segment of the ray path.  The closest hit will
    // return new segments to be traced here.
	//ray_origin = light.corner + light.v1 * 0.5f + light.v2 * 0.5f;
	//ray_direction = make_float3(0.0f, -1.0f, 0.0f);
    for(;;)
    {
		if (prd.depth >= maxDepth)
			break;
		prd.isSpecular = 0;
		ray_direction = normalize(ray_direction);
        Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
        rtTrace(top_object, ray, prd);

        if(prd.done)
        {
            // We have hit the background or a luminaire
            prd.result += prd.radiance * prd.attenuation;
            break;
        }

		//if (!firstIntersection)
		//{
		//	firstIntersection = true;
		//	t = prd.tValue;
		//}
        // Russian roulette termination 
        if(prd.depth >= rr_begin_depth)
        {
            float pcont = fmaxf(prd.attenuation);
            if(rnd(prd.seed) >= pcont)
                break;
            prd.attenuation /= pcont;
        }

		prd.result += prd.radiance * prd.attenuation;
		if (!prd.isSpecular)
			isHitBuffer[maxDepth * index + prd.depth] = 1;

		// Be careful of calculating the indices!
		photonBuffer[maxDepth * index + prd.depth].position = ray.origin + prd.tValue * ray.direction;
		photonBuffer[maxDepth * index + prd.depth].color = prd.result;
        prd.depth++;
        

        // Update ray data for the next path segment
        ray_origin = prd.origin;
        ray_direction = prd.direction;
    }

    result += prd.result;
	seed = prd.seed;
    //
    // Update the output buffer
    //
    float3 pixel_color = result;

	/*clock_t stop_time = clock();*/
	
	//int time = (int)(stop_time - start_time);
	//double tt = 950000000;
	//printf("%f\n", (double)time/ tt);
  //  if (frame_number > 1)
  //  {
  //      float a = 1.0f / (float)frame_number;
  //      float3 old_color = make_float3(output_buffer[launch_index]);
  //      output_buffer[launch_index] = make_float4( lerp( old_color, pixel_color, a ), 1.0f );
		////output_buffer[launch_index] = make_float4(0.5f, 0.0f, 0.0f, 1.0f);
		////photonBuffer[index * 2 + 1] = lerp( old_color, pixel_color, a );
		////printf("IN\n");
  //  }
  //  else
  //  {
  //      output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
		////photonBuffer[index * 2 + 1] = make_float3(output_buffer[launch_index]);
  //  }
	//printf("%f %f %f %f\n", t, firstRay_direction.x, firstRay_direction.y, firstRay_direction.z);
	//photonBuffer[index * 2] = eye + t * firstRay_direction;
	//photonBuffer[index * 2 + 1] = make_float3(output_buffer[launch_index]);
	//photonBuffer[index * 2 + 1] = make_float3(0.5f, 0.0f, 0.0f);
	//printf("%f %f %f\n", pixel_color.x, pixel_color.y, pixel_color.z);
	//output_buffer[launch_index] += make_float4(0.01f, 0.0f, 0.0f, 1.0f);
}


//-----------------------------------------------------------------------------
//
//  Emissive surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        emission_color, , );

RT_PROGRAM void diffuseEmitter()
{
    current_prd.radiance = current_prd.countEmitted ? emission_color : make_float3(0.f);
    current_prd.done = true;
	//printf("IN\n");
}


//-----------------------------------------------------------------------------
//
//  Lambertian surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,     diffuse_color, , );
rtDeclareVariable(float3,     geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3,     shading_normal,   attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray,              rtCurrentRay, );
rtDeclareVariable(float,      t_hit,            rtIntersectionDistance, );
rtDeclareVariable(float, tValue, attribute tValue, );

RT_PROGRAM void diffuse()
{
	//printf("sasd\n");
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	//printf("%f %f %f %f %f %f\n", world_shading_normal.x, world_shading_normal.y, world_shading_normal.z,
	//	shading_normal.x, shading_normal.y, shading_normal.z);
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    float3 hitpoint = ray.origin + t_hit * ray.direction;

    //
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //
    current_prd.origin = hitpoint;

    float z1=rnd(current_prd.seed);
    float z2=rnd(current_prd.seed);
    float3 p;
    cosine_sample_hemisphere(z1, z2, p);
    optix::Onb onb( ffnormal );
    onb.inverse_transform( p );
    current_prd.direction = p;

    // NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
    // with cosine density.
    current_prd.attenuation = current_prd.attenuation * diffuse_color;
    current_prd.countEmitted = false;

    //
    // Next event estimation (compute direct lighting).
    //
    unsigned int num_lights = lights.size();
    float3 result = make_float3(0.0f);
    for(int i = 0; i < num_lights; ++i)
    {
        // Choose random point on light
        ParallelogramLight light = lights[i];
        const float z1 = rnd(current_prd.seed);
        const float z2 = rnd(current_prd.seed);
        const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

        // Calculate properties of light sample (for area based pdf)
        const float  Ldist = length(light_pos - hitpoint);
        const float3 L     = normalize(light_pos - hitpoint);
        const float  nDl   = dot( ffnormal, L );
        const float  LnDl  = dot( light.normal, L );

        // cast shadow ray
        if ( nDl > 0.0f && LnDl > 0.0f )
        {
            PerRayData_pathtrace_shadow shadow_prd;
            shadow_prd.inShadow = false;
            // Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
            Ray shadow_ray = make_Ray( hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon );
            rtTrace(top_object, shadow_ray, shadow_prd);

			// do not calculate shadow in pre-pass, calculate in the second-pass


            if(!shadow_prd.inShadow)
            {
                const float A = length(cross(light.v1, light.v2));
                // convert area based pdf to solid angle
                const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
                result += light.emission * weight;
            }

        }
    }

    //current_prd.radiance = result;
	current_prd.radiance *= DECRESE_FACTOR;
	current_prd.tValue = tValue;
	
	//current_prd.radiance = make_float3(1.0f, 1.0f, 1.0f);
}

rtDeclareVariable(float3, world_normal, attribute world_normal, );
RT_PROGRAM void specular()
{
	float3 ffnormal = faceforward( world_normal, -ray.direction, world_normal );
	float3 hitpoint = ray.origin + t_hit * ray.direction;
	current_prd.origin = hitpoint;
	float3 R = reflect(ray.direction, ffnormal);
	current_prd.direction = R;
	current_prd.attenuation = current_prd.attenuation;
	current_prd.countEmitted = true;
	unsigned int num_lights = lights.size();
	float3 result = make_float3(0.0f);
	//current_prd.radiance = result;
	current_prd.radiance = current_prd.radiance;
	current_prd.tValue = tValue;
	current_prd.isSpecular = 1;
	//printf("SPECULAR\n");
 //   float3 result = make_float3(0.0f);
 //   current_prd.radiance = result;
	//current_prd.done = 1;
}

//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void shadow()
{
    current_prd_shadow.inShadow = true;
    rtTerminateRay();
}


//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_float4(bad_color, 1.0f);
}


//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, bg_color, , );

RT_PROGRAM void miss()
{
    current_prd.radiance = bg_color;
    current_prd.done = true;
}

RT_PROGRAM void glass_closest_hit_radiance()
{
	float n1 = 1.0f, n2 = 1.5f;
    float3 hitpoint = ray.origin + t_hit * ray.direction;
	
	float3 d = normalize(ray.direction);

	float cosTheta = dot(ray.direction, world_normal);
	float eta = n2 / n1;
	float3 realNormal;

	if (cosTheta > 0.0f)
	{
		realNormal = -world_normal;		
	}
	else
	{
		realNormal = world_normal;
		//eta = n1 / n2;
		cosTheta = -cosTheta;
		//current_prd.attenuation = make_float3(1.0f, 0.0f, 0.0f);
		//printf("OUTSIDE\n");
	}
	
	unsigned int seed = t_hit * frame_number;
	float u01 = rnd(seed);
	//thrust::uniform_real_distribution<float> u01(0, 1);
	//thrust::default_random_engine rng = makeSeededRandomEngine(frame_number, launch_index.x + launch_index.y, 0);

	if (u01 < (n2 - n1) / (n2 + n1) * (n2 - n1) / (n2 + n1) + (1 - (n2 - n1) / (n2 + n1) * (n2 - n1) / (n2 + n1)) * pow(1 - cosTheta, 5))
	{
		current_prd.direction = reflect(ray.direction, realNormal);
	}
	else
	{
		refract(current_prd.direction, ray.direction, world_normal, eta);
		//glm::vec3 a(d.x, d.y, d.z);
		//glm::vec3 b(realNormal.x, realNormal.y, realNormal.z);
		//glm::vec3 c = glm::refract(a, b, eta);
		//current_prd.direction = make_float3(c.x, c.y, c.z);
	}

	current_prd.origin = hitpoint;// + ray.direction * 0.01;
    current_prd.attenuation = current_prd.attenuation;
    current_prd.countEmitted = true;
	
    float3 result = make_float3(0.0f);
    //current_prd.radiance = result;
	current_prd.radiance *= DECRESE_FACTOR;
	current_prd.tValue = tValue;
	current_prd.isSpecular = 1;
}
