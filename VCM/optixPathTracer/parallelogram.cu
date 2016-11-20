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

#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float4, plane, , );
rtDeclareVariable(float3, v1, , );
rtDeclareVariable(float3, v2, , );
rtDeclareVariable(float3, anchor, , );
//rtDeclareVariable(int, lgt_instance, , ) = {0};

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float, tValue, attribute tValue, );
//rtDeclareVariable(int, lgt_idx, attribute lgt_idx, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtBuffer<float3, 1>              Aabb_buffer;

RT_PROGRAM void intersect(int primIdx)
{
  float3 n = make_float3( plane );
  float dt = dot(ray.direction, n );
  float t = (plane.w - dot(n, ray.origin))/dt;
  if( t > ray.tmin && t < ray.tmax ) {
	  //printf("%f %f\n", ray.tmin, ray.tmax);
    float3 p = ray.origin + ray.direction * t;
    float3 vi = p - anchor;
    float a1 = dot(v1, vi);
    if(a1 >= 0 && a1 <= 1){
      float a2 = dot(v2, vi);
      if(a2 >= 0 && a2 <= 1){
        if( rtPotentialIntersection( t ) ) {
          shading_normal = geometric_normal = n;
          texcoord = make_float3(a1,a2,0);
		  tValue = t;
		  
          //lgt_idx = lgt_instance;
          rtReportIntersection( 0 );
        }
      }
    }
  }
  //printf("intersected:%d\n", primIdx);
}

RT_PROGRAM void bounds (int primIdx, float result[6])
{
  // v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
	const float3 tv1  = v1 / dot( v1, v1 );
	const float3 tv2  = v2 / dot( v2, v2 );
	const float3 p00  = anchor;
	const float3 p01  = anchor + tv1;
	const float3 p10  = anchor + tv2;
	const float3 p11  = anchor + tv1 + tv2;
	const float  area = length(cross(tv1, tv2));
  
	optix::Aabb* aabb = (optix::Aabb*)result;
  
	if(area > 0.0f && !isinf(area)) 
	{
		aabb->m_min = fminf( fminf( p00, p01 ), fminf( p10, p11 ) );
		aabb->m_max = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ) );
	} 
	else 
	{
		aabb->invalidate();
	}

  	Aabb_buffer[0] = aabb->m_min;
	Aabb_buffer[1] = aabb->m_max;
  //printf("bouding:%d\n", primIdx);
  
}

	//float r = sqrt(1.0f - u1u2.x * u1u2.x);
	//float phi = 2 * M_PI * u1u2.y;
	//float theta = M_PI * u1u2.x;
	//float3 dir = make_float3(cos(phi), cos(theta), sin(phi));
	//printf("%f %f %f\n", dir.x, dir.y, dir.z);