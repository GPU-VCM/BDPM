CALLABLE float3 IlluminateBackground(
		const Light *const light,
		const float		   mInvSceneRadiusSqr,
        const float3       &aReceivingPosition,
        const float2       &aRndTuple,
        float3             &oDirectionToLight,
        float              &oDistance,
        float              &oDirectPdfW,
        float              *oEmissionPdfW,
        float              *oCosAtLight)
{
	// Replace these two lines with image sampling
	oDirectionToLight = SampleUniformSphereW(aRndTuple, &oDirectPdfW);

	// This stays even with image sampling
	oDistance = 1e36f;
	*oEmissionPdfW = oDirectPdfW * ConcentricDiscPdfA() * mInvSceneRadiusSqr;
	*oCosAtLight = 1.f;

	return light->mIntensity;
}

CALLABLE float3 EmitBackground(
		const Light *const light,
        const float3      mSceneCenter,
		const float       mSceneRadius,
		const float		  mInvSceneRadiusSqr,
        const float2      &aDirRndTuple,
        const float2      &aPosRndTuple,
        float3            &oPosition,
        float3            &oDirection,
        float             &oEmissionPdfW,
        float             *oDirectPdfA,
        float             *oCosThetaLight)
{
	float directPdf;

	// Replace these two lines with image sampling
	oDirection = SampleUniformSphereW(aDirRndTuple, &directPdf);
	//oDirection = -Vec3f(0.16123600f, -0.98195398f, 0.098840252f);

	// Stays even with image sampling
	const float2 xy = SampleConcentricDisc(aPosRndTuple);

	Frame frame;
	frame.SetFromZ(oDirection);
        
	oPosition = mSceneCenter + mSceneRadius * (
		-oDirection + frame.Binormal() * xy.x + frame.Tangent() * xy.y);

	//oPosition = Vec3f(-1.109054f, -2.15064538f, -1.087019148f);

	oEmissionPdfW = directPdf * ConcentricDiscPdfA() * mInvSceneRadiusSqr;

	// For background we lie about Pdf being in area measure
	*oDirectPdfA = directPdf;
        
	// Not used for infinite or delta lights
	*oCosThetaLight = 1.f;

	return light->mIntensity;
}

CALLABLE float3 GetRadianceBackground(
	const Light *const light,
	const float		   mInvSceneRadiusSqr,
    const float3       &aRayDirection,
    const float3       &aHitPoint,
    float             *oDirectPdfA,
    float             *oEmissionPdfW)
{
	// Replace this with image lookup (proper pdf and such)
	// use aRayDirection
	const float directPdf = UniformSpherePdfW();
	*oDirectPdfA = directPdf;
	*oEmissionPdfW = directPdf * ConcentricDiscPdfA() * mInvSceneRadiusSqr;
        
	return light->mIntensity;
}