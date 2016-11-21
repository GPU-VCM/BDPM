#include <optix_world.h>
#include <math_constants.h>
#include "mesh.h"

using namespace optix;
rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3>   index_buffer;
rtDeclareVariable(int, lgt_instance, , ) = { 0 };

rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(int, lgt_idx, attribute lgt_idx, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void meshIntersect(int primIdx){
	const int3 v_idx = index_buffer[primIdx];

	const float3 p0 = vertex_buffer[v_idx.x];
	const float3 p1 = vertex_buffer[v_idx.y];
	const float3 p2 = vertex_buffer[v_idx.z];

	// Intersect ray with triangle
	float3 n;
	float  t, beta, gamma;
	if (intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma)) {
		
		if (rtPotentialIntersection(t)) {
			
			geometric_normal = normalize(n);
			if (normal_buffer.size() == 0) {
				shading_normal = geometric_normal;
			}
			else {
				float3 n0 = normal_buffer[v_idx.x];
				float3 n1 = normal_buffer[v_idx.y];
				float3 n2 = normal_buffer[v_idx.z];
				shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f - beta - gamma));
			}

			if (texcoord_buffer.size() == 0) {
				texcoord = make_float3(0.0f, 0.0f, 0.0f);
			}
			else {
				float2 t0 = texcoord_buffer[v_idx.x];
				float2 t1 = texcoord_buffer[v_idx.y];
				float2 t2 = texcoord_buffer[v_idx.z];
				texcoord = make_float3(t1*beta + t2*gamma + t0*(1.0f - beta - gamma));
			}
			lgt_idx = lgt_instance;
			rtPrintf("p0: %d\n", p0.x);
			rtReportIntersection(0);
		}
	}
}

RT_PROGRAM void boundingBoxMesh(int primIdx, float result[6]){
	//get indices
	int3 id = index_buffer[primIdx];
	//load vertices
	float3 v1 = vertex_buffer[id.x];
	float3 v2 = vertex_buffer[id.y];
	float3 v3 = vertex_buffer[id.z];

	/*float3 v1d=vertex_buffer[id.x]-normal_buffer[id.x]* 1.5f * BUMP_INTENSITY;
	float3 v2d=vertex_buffer[id.y]-normal_buffer[id.y]* 1.5f * BUMP_INTENSITY;
	float3 v3d=vertex_buffer[id.z]-normal_buffer[id.z]* 1.5f * BUMP_INTENSITY;*/
	const float area = optix::length(optix::cross(v2 - v1, v3 - v1));
	Aabb* aabb = (optix::Aabb*)result;
	if (area>0.0f)
	{
		/*aabb->m_min=fminf(fminf(fminf(v1, v1d),fminf(v2, v2d)), fminf(v3, v3d));
		aabb->m_max=fmaxf(fmaxf(fmaxf(v1, v1d),fmaxf(v2, v2d)), fmaxf(v3, v3d));*/

		aabb->m_min = fminf(fminf(v1, v2), v3);
		aabb->m_max = fmaxf(fmaxf(v1, v2), v3);
	}
	else
	{
		aabb->invalidate();
	}
}