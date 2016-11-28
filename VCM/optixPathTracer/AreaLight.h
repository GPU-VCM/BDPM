CALLABLE float3 IlluminateAreaLight(
	const Light *const light,
	const float3       &aReceivingPosition,
	const float2       &aRndTuple,
	float3             &oDirectionToLight,
	float             &oDistance,
	float             &oDirectPdfW,
	float             *oEmissionPdfW,
	float             *oCosAtLight)
{
	const float2 uv = SampleUniformTriangle(aRndTuple);
	const float3 lightPoint = light->p0 + light->e1 * uv.x + light->e2 * uv.y;

	oDirectionToLight = lightPoint - aReceivingPosition;
	const float distSqr = dot(oDirectionToLight, oDirectionToLight);
	oDistance = sqrtf(distSqr);
	oDirectionToLight = oDirectionToLight / oDistance;

	const float cosNormalDir = dot(light->mFrame.Normal(), -oDirectionToLight);

	// too close to, or under, tangent
	if (cosNormalDir < EPS_COSINE)
	{
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	oDirectPdfW = light->mInvArea * distSqr / cosNormalDir;
	*oCosAtLight = cosNormalDir;
	*oEmissionPdfW = light->mInvArea * cosNormalDir * INV_PI_F;

	return light->mIntensity;
}

CALLABLE float3 EmitAreaLight(
	const Light *const light,
	const float2      &aDirRndTuple,
	const float2      &aPosRndTuple,
	float3            &oPosition,
	float3            &oDirection,
	float             &oEmissionPdfW,
	float             *oDirectPdfA,
	float             *oCosThetaLight)
{
	const float2 uv = SampleUniformTriangle(aPosRndTuple);
	oPosition = light->p0 + light->e1 * uv.x + light->e2 * uv.y;

	float3 localDirOut = SampleCosHemisphereW(aDirRndTuple, &oEmissionPdfW);

	oEmissionPdfW *= light->mInvArea;

	// cannot really not emit the particle, so just bias it to the correct angle
	localDirOut.z = max(localDirOut.z, EPS_COSINE);
	oDirection = light->mFrame.ToWorld(localDirOut);

	*oDirectPdfA = light->mInvArea;
	*oCosThetaLight = localDirOut.z;

	return light->mIntensity * localDirOut.z;
}

CALLABLE float3 GetRadianceAreaLight(
	const Light *const light,
	const float3      &aRayDirection,
	const float3      &aHitPoint,
	float             *oDirectPdfA,
	float             *oEmissionPdfW)
{
	const float cosOutL = max(0.0f, dot(light->mFrame.Normal(), -aRayDirection));
	if (cosOutL == 0)
		return make_float3(0.0f, 0.0f, 0.0f);

	*oDirectPdfA = light->mInvArea;
	*oEmissionPdfW = CosHemispherePdfW(light->mFrame.Normal(), -aRayDirection) * light->mInvArea;

	return light->mIntensity;
}