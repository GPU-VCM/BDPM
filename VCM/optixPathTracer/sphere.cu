#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float3, center, , );
rtDeclareVariable(float, radius, , );
//rtDeclareVariable(int, lgt_instance, , ) = {0};

rtDeclareVariable(float3, world_normal, attribute world_normal, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
//rtDeclareVariable(int, lgt_idx, attribute lgt_idx, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tValue, attribute tValue, );
rtBuffer<float3, 1>              Aabb_buffer;

RT_PROGRAM void intersect(int primIdx)
{
	//printf("TESTING SPHERE\n");
	float3 v = ray.origin - center;
	float dv = dot(ray.direction, v); 
	float temp = dv * dv - (dot(v, v) - radius * radius);
	//bool flag = false;
	if (temp > 0)
	{
		
		float t = -dv - sqrt(temp);
		if (t <ray.tmin)
			t = -dv + sqrt(temp);
		//printf("%f\n", length(ray.direction));
		//float3 end = ray.origin + t * ray.direction;(length(end - center) - radius)<0.01f &&
		if(rtPotentialIntersection(t)) 
		{
			//if (flag)
			//	printf("IN\n");
			float3 p = ray.origin + t * ray.direction;
			//printf("Intersected:%f %f %f\n", p.x, p.y, p.z);
			geometric_normal = shading_normal = world_normal = normalize(ray.origin + t * ray.direction - center);
			tValue = t;
			rtReportIntersection( 0 );
        }
	}
}

RT_PROGRAM void bounds (int primIdx, float result[6])
{
	optix::Aabb* aabb = (optix::Aabb*)result;

	aabb->m_min = center - make_float3(radius);
	aabb->m_max = center + make_float3(radius);

	Aabb_buffer[0] = aabb->m_min;
	Aabb_buffer[1] = aabb->m_max;
	//printf("AABB\n");
}