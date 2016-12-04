CALLABLE float3 IlluminateAreaLight(
	const Light *const light,
	const float3       &iPos,
	const float2       &rnd2,
	float3             &DtL,
	float             &dist,
	float             &dirPdf,
	float             *EmPdf,
	float             *costheta)
{
	const float2 uv = SampleUniformTriangle(rnd2);
	const float3 ptOnLight = light->corner + light->e1 * uv.x + light->e2 * uv.y;

	DtL = ptOnLight - iPos;
	const float distSqr = dot(DtL, DtL);
	dist = sqrtf(distSqr);
	DtL = DtL / dist;

	const float cosNormalDir = dot(light->frame.Normal(), -DtL);
	if (cosNormalDir < EPS_COSINE)
	{
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	dirPdf = light->invArea * distSqr / cosNormalDir;
	*costheta = cosNormalDir;
	*EmPdf = light->invArea * cosNormalDir * INV_PI_F;

	return light->intensity;
}

CALLABLE float3 EmitAreaLight(
	const Light *const light,
	const float2      &rnd21,
	const float2      &rnd22,
	float3            &pos,
	float3            &dir,
	float             &EmPdf,
	float             *dirPdf,
	float             *costheta)
{
	const float2 uv = SampleUniformTriangle(rnd22);
	pos = light->corner + light->e1 * uv.x + light->e2 * uv.y;

	float3 localDirOut = SampleCosHemisphereW(rnd21, &EmPdf);

	EmPdf *= light->invArea;

	localDirOut.z = max(localDirOut.z, EPS_COSINE);
	dir = light->frame.ToWorld(localDirOut);

	*dirPdf = light->invArea;
	*costheta = localDirOut.z;

	return light->intensity * localDirOut.z;
}

CALLABLE float3 GetRadianceAreaLight(
	const Light *const light,
	const float3      &dir,
	const float3      &iPoint,
	float             *dirPdf,
	float             *EmPdf)
{
	const float costhetatoLight = max(0.0f, dot(light->frame.Normal(), -dir));
	if (costhetatoLight == 0)
		return make_float3(0.0f, 0.0f, 0.0f);

	*dirPdf = light->invArea;
	*EmPdf = CosHemispherePdfW(light->frame.Normal(), -dir) * light->invArea;

	return light->intensity;
}