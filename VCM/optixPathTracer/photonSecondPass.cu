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
	float rayPdf;
};

struct PerRayData_pathtrace_shadow
{
    bool inShadow;
};

// 
rtBuffer<Photon> photonBuffer;
rtBuffer<int> gridStartIndexBuffer;
rtBuffer<int> gridEndIndexBuffer;
rtDeclareVariable(float,         gridLength, , );
rtDeclareVariable(float,         gridMin, , );
rtDeclareVariable(int,         gridSideCount, , );
rtDeclareVariable(int,         totalPhotons, , );

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

rtBuffer<float4, 2>              output_buffer;
rtBuffer<ParallelogramLight>     lights;


RT_PROGRAM void pathtrace_camera()
{
	
    size_t2 screen = output_buffer.size();

    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
    float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

    float2 jitter_scale = inv_screen / sqrt_num_samples;
    unsigned int samples_per_pixel = sqrt_num_samples*sqrt_num_samples;
    float3 result = make_float3(0.0f);

    unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame_number);
	
    {
		float2 d = pixel;
        float3 ray_origin = eye;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);

        // Initialze per-ray data
        PerRayData_pathtrace prd;
        prd.result = make_float3(0.f);
        prd.attenuation = make_float3(1.f);
        prd.countEmitted = true;
        prd.done = false;
        prd.seed = seed;
        prd.depth = 0;

        // Each iteration is a segment of the ray path.  The closest hit will
        // return new segments to be traced here.
        for(;;)
        {
			if (prd.depth > 8)
				break;
			
            Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
            rtTrace(top_object, ray, prd);
			
            if(prd.done)
            {
                // We have hit the background or a luminaire
                prd.result += prd.radiance * prd.attenuation;
                break;
            }

            // Russian roulette termination 
            if(prd.depth >= rr_begin_depth)
            {
                float pcont = fmaxf(prd.attenuation);
                if(rnd(prd.seed) >= pcont)
                    break;
                prd.attenuation /= pcont;
            }

            prd.depth++;
            prd.result += prd.radiance * prd.attenuation;

            // Update ray data for the next path segment
            ray_origin = prd.origin;
            ray_direction = prd.direction;
        }

        result += prd.result;
        seed = prd.seed;
    }

    //
    // Update the output buffer
    //
    float3 pixel_color = result;
	
	
    if (frame_number > 0)
    {
        float a = 1.0f / (float)frame_number;
        float3 old_color = make_float3(output_buffer[launch_index]);
        output_buffer[launch_index] = make_float4( lerp( old_color, pixel_color, a ), 1.0f );
    }
    else
    {
        output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
    }
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
}


//-----------------------------------------------------------------------------
//
//  Lambertian surface closest-hit
//
//-----------------------------------------------------------------------------

__device__ int gridIndex3Dto1D(int x, int y, int z)
{
	return x * gridSideCount * gridSideCount + y * gridSideCount + z;
}

rtDeclareVariable(float3,     diffuse_color, , );
rtDeclareVariable(float3,     geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3,     shading_normal,   attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray,              rtCurrentRay, );
rtDeclareVariable(float,      t_hit,            rtIntersectionDistance, );

RT_PROGRAM void diffuse()
{
	current_prd.done = true;

    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );

    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    float3 hitpoint = ray.origin + t_hit * ray.direction;

	float z1 = rnd(current_prd.seed);
	float z2 = rnd(current_prd.seed);
	float3 p;
	cosine_sample_hemisphere(z1, z2, p);
	optix::Onb onb(ffnormal);
	onb.inverse_transform(p);
	current_prd.direction = p;
	current_prd.rayPdf *= M_1_PIf;

	float radius = gridLength * 0.5f;
	float resultx = (hitpoint.x - gridMin) / gridLength;
	float resulty = (hitpoint.y - gridMin) / gridLength;
	float resultz = (hitpoint.z - gridMin) / gridLength;

	int flagx = int(resultx + 0.5f) - int(resultx) ? 0 : -1;
	int flagy = int(resulty + 0.5f) - int(resulty) ? 0 : -1;
	int flagz = int(resultz + 0.5f) - int(resultz) ? 0 : -1;

	int xx = resultx;
	int yy = resulty;
	int zz = resultz;

	float3 averageColor = make_float3(0.0f);
	int counter = 0;

	double totalWeight = 0;

	for (int i = xx + flagx; i <= xx + flagx + 1; i++)
		for (int j = yy + flagy; j <= yy + flagy + 1; j++)
			for (int k = zz + flagz; k <= zz + flagz + 1; k++)
			{
				int nGridIndex = gridIndex3Dto1D(i, j, k);
				int nStartIndex = gridStartIndexBuffer[nGridIndex];
				int nEndIndex = gridEndIndexBuffer[nGridIndex];

				if (nStartIndex == -1 && nEndIndex == -1)
					continue;

				for (int l = nStartIndex; l <= nEndIndex; l++)
				{
					if (length(photonBuffer[l].position - hitpoint) < radius)
					{
						float Wmis = current_prd.rayPdf / (current_prd.rayPdf + photonBuffer[l].rayPdf);
						averageColor += photonBuffer[l].color * photonBuffer[l].rayPdf;
						counter++;

						totalWeight += photonBuffer[l].rayPdf;
					}
				}
				
			}

	averageColor /= totalWeight;
	averageColor *= counter;

	float scale = 5.f;
	current_prd.attenuation = averageColor / (scale * radius * radius);
    current_prd.countEmitted = false;
    
	current_prd.radiance = make_float3(1.0f);
	
}

rtDeclareVariable(float3, world_normal, attribute world_normal, );
RT_PROGRAM void specular()
{
    float3 ffnormal = faceforward( world_normal, -ray.direction, world_normal );
	
    float3 hitpoint = ray.origin + t_hit * ray.direction;

    //
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //
    current_prd.origin = hitpoint;

    float3 R = reflect(ray.direction, ffnormal);
    current_prd.direction = R;

    // NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
    // with cosine density.
    current_prd.attenuation = current_prd.attenuation;
    current_prd.countEmitted = true;

    unsigned int num_lights = lights.size();
    float3 result = make_float3(0.0f);

    current_prd.radiance = result;
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
	}

	unsigned int seed = t_hit * frame_number;
	float u01 = rnd(seed);

	if (u01 < (n2 - n1) / (n2 + n1) * (n2 - n1) / (n2 + n1) + (1 - (n2 - n1) / (n2 + n1) * (n2 - n1) / (n2 + n1)) * pow(1 - cosTheta, 5))
	{
		current_prd.direction = reflect(ray.direction, realNormal);
	}
	else
	{
		refract(current_prd.direction, ray.direction, world_normal, eta);
	}

	current_prd.origin = hitpoint;
    current_prd.attenuation = current_prd.attenuation;
    current_prd.countEmitted = true;
	
    float3 result = make_float3(0.0f);
    current_prd.radiance = result;
}