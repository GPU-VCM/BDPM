CALLABLE float3 IlluminateDirectional(
		const Light *const light,
		const float		   mInvSceneRadiusSqr,
        float3             &oDirectionToLight,
        float              &oDistance,
        float              &oDirectPdfW,
        float              *oEmissionPdfW,
        float              *oCosAtLight)
{
	oDirectionToLight     = -light->mFrame.Normal();
    oDistance             = 1e36f;
    oDirectPdfW           = 1.f;

    *oCosAtLight = 1.f;
    *oEmissionPdfW = ConcentricDiscPdfA() * mInvSceneRadiusSqr;

	return light->mIntensity;
}

CALLABLE float3 EmitDirectional(
		const Light *const light,
        const float3      mSceneCenter,
		const float       mSceneRadius,
		const float		  mInvSceneRadiusSqr,
        const float2      &aPosRndTuple,
        float3            &oPosition,
        float3            &oDirection,
        float             &oEmissionPdfW,
        float             *oDirectPdfA,
        float             *oCosThetaLight)
{
	const float2 xy = SampleConcentricDisc(aPosRndTuple);

    oPosition = mSceneCenter +
				mSceneRadius * (
				-light->mFrame.Normal() + light->mFrame.Binormal() * xy.x + light->mFrame.Tangent() * xy.y);

    oDirection = light->mFrame.Normal();
    oEmissionPdfW = ConcentricDiscPdfA() * mInvSceneRadiusSqr;

	*oDirectPdfA = 1.f;
    *oCosThetaLight = 1.f;

    return light->mIntensity;
}

CALLABLE float3 GetRadianceDirectional()
{
	return make_float3(0, 0, 0);
}