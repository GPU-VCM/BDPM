#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <GLUTDisplay.h>
#include "commonStructs.h"
#include <ObjLoader.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include "Camera.h"

#if defined(__APPLE__)
#  include <OpenGL/gl.h>
#else
#  include <GL/glew.h>
#  include <GL/gl.h>
#endif

using namespace optix;

//-----------------------------------------------------------------------------
// 
// BidiPTracer Scene
//
//-----------------------------------------------------------------------------

class BidiPTracer : public SampleScene
{
public:
	// From SampleScene
	virtual void   initScene(InitialCameraData& camera_data);
	virtual void   trace(const RayGenCameraData& camera_data);
	virtual Buffer getOutputBuffer();
	virtual bool   keyPressed(unsigned char key, int x, int y);

	void createGeometry();
	void createLightBuffer(GLMmodel *model, std::vector<Light> &lights, const float3 &emissive);
	void doResize(unsigned int width, unsigned int height);

	static bool m_useGLBuffer;
	std::string prefix;

private:
	unsigned int num_devices;
	unsigned int max_num_devices;

	static unsigned int WIDTH;
	static unsigned int HEIGHT;

	int lightConditions, mMaxPathLength;
	int materials, object;
};

unsigned int BidiPTracer::WIDTH  = 512u;
unsigned int BidiPTracer::HEIGHT = 512u;

bool BidiPTracer::m_useGLBuffer = true;

bool BidiPTracer::keyPressed(unsigned char key, int x, int y)
{
	if ((key >= '0') && (key <= '9'))
	{
		mMaxPathLength = key - '0';
		if (mMaxPathLength == 0)
			mMaxPathLength = 10;
		m_context["mMaxPathLength"]->setInt(mMaxPathLength);
		m_camera_changed = true;
		return true;
	}
	switch (key)
	{
		case 'p':
		{
			num_devices = min(num_devices + 1, max_num_devices);
			std::vector<int> dev;
			for (unsigned int i = 0; i < num_devices; ++i)
				dev.push_back(i);
			m_context->setDevices(dev.begin(), dev.end());
			return true;
		}

		case 'P':
		{
			num_devices = max(num_devices - 1, 1);
			std::vector<int> dev;
			for(unsigned int i = 0; i < num_devices; ++i)
				dev.push_back(i);
			m_context->setDevices(dev.begin(), dev.end());
			return true;
		}

		case ' ':
		{
			++lightConditions;
			if (lightConditions > 2)
				lightConditions = 0;
			createGeometry();
			m_camera_changed = true;
			return true;
		}

		case 'n':
		{
			++materials;
			if (materials > 3)
				materials = 0;
			createGeometry();
			m_camera_changed = true;
			return true;
		}

		case 'o':
		{
			++object;
			if (object > 2)
				object = 0;
			createGeometry();
			m_camera_changed = true;
			return true;
		}
	}

	return false;
}

void BidiPTracer::initScene(InitialCameraData& camera_data)
{
	try
	{
		lightConditions = 2;
		materials = object = 0;
		mMaxPathLength = 5;

		// Setup state...

		m_context->setRayTypeCount(2);
		m_context->setEntryPointCount(3);
		m_context->setStackSize(3072);

		num_devices = Context::getDeviceCount();
		max_num_devices = num_devices;

		m_context["max_depth"]->setInt(1);
		m_context["radiance_ray_type"]->setUint(0u);
		m_context["shadow_ray_type"]->setUint(1u);
		m_context["scene_epsilon"]->setFloat(1.e-4f);
		const float vfov = 45.0f;
		m_context["vfov"]->setFloat(vfov);

		Variable output_buffer = m_context["output_buffer"];
		output_buffer->set(createOutputBuffer(RT_FORMAT_FLOAT4, WIDTH, HEIGHT));

		Variable temp_buffer = m_context["temp_buffer"];
		temp_buffer->set(createOutputBuffer(RT_FORMAT_FLOAT4, WIDTH, HEIGHT));

		// Set up camera chapel
		/*camera_data = InitialCameraData(make_float3(17.1677, 5.15077, 0.605223), // eye
										make_float3(3.69398, 7.25707, -0.428414), // lookat
										make_float3(-0.153785, -0.988065, -0.00879226), // up
										45.0f);                         // vfov*/

		// Set up camera cornell box
		camera_data = InitialCameraData(make_float3(2.75, 2.75, 7.5), // eye
										make_float3(2.75, 2.75, -2.5), // lookat
										make_float3(0.0f, 1.0f,  0.0f), // up
										vfov);                         // vfov

		// Set up camera
		/*camera_data = InitialCameraData(make_float3(0.0f, 9.7f, -20.22f), // eye
										make_float3(0.0f, 9.7f,  0.0f), // lookat
										make_float3(0.0f, 1.0f,  0.0f), // up
										vfov);                         // vfov*/

		// Declare camera variables. The values do not matter, they will be overwritten in trace...

		m_context["eye"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
		m_context["U"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
		m_context["V"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
		m_context["W"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));

		// Ray gen programs...
		std::string ptx_path = prefix + std::string("/pinhole_camera.cu.ptx");
		Program ray_gen_program = m_context->createProgramFromPTXFile(ptx_path, "pinhole_camera");
		m_context->setRayGenerationProgram(0, ray_gen_program);

		Program clear_temp_program = m_context->createProgramFromPTXFile(ptx_path, "clear_temp");
		m_context->setRayGenerationProgram(1, clear_temp_program);

		Program cumulate_samples_program = m_context->createProgramFromPTXFile(ptx_path, "cumulate_samples");
		m_context->setRayGenerationProgram(2, cumulate_samples_program);

		// Exception...

		Program exception_program = m_context->createProgramFromPTXFile(ptx_path, "exception");
		m_context->setExceptionProgram(0, exception_program);
		m_context["bad_color"]->setFloat(1.0f, 1.0f, 0.0f);

		// Miss programs...

		ptx_path = prefix + std::string("/constantbg.cu.ptx");
		m_context->setMissProgram(0, m_context->createProgramFromPTXFile(ptx_path, "miss"));
		m_context->setMissProgram(1, m_context->createProgramFromPTXFile(ptx_path, "miss_occlusion"));
		m_context["bg_color"]->setFloat(make_float3(108.0f/255.0f, 166.0f/255.0f, 205.0f/255.0f) * 0.5f);

		// Load and bind callable programs...
		
#ifdef USE_CALLABLE_PROGRAM
		ray_gen_program["IlluminateBackground"]->set(m_context->createProgramFromPTXFile(prefix + std::string("/BackgroundLight.cu.ptx"), "IlluminateBackground"));
		ray_gen_program["EmitBackground"]->set(m_context->createProgramFromPTXFile(prefix + std::string("/BackgroundLight.cu.ptx"), "EmitBackground"));
		ray_gen_program["GetRadianceBackground"]->set(m_context->createProgramFromPTXFile(prefix + std::string("/BackgroundLight.cu.ptx"), "GetRadianceBackground"));
		ray_gen_program["IlluminateAreaLight"]->set(m_context->createProgramFromPTXFile(prefix + std::string("/AreaLight.cu.ptx"), "IlluminateAreaLight"));
		ray_gen_program["EmitAreaLight"]->set(m_context->createProgramFromPTXFile(prefix + std::string("/AreaLight.cu.ptx"), "EmitAreaLight"));
		ray_gen_program["GetRadianceAreaLight"]->set(m_context->createProgramFromPTXFile(prefix + std::string("/AreaLight.cu.ptx"), "GetRadianceAreaLight"));
		ray_gen_program["IlluminateDirectional"]->set(m_context->createProgramFromPTXFile(prefix + std::string("/DirectionalLight.cu.ptx"), "IlluminateDirectional"));
		ray_gen_program["EmitDirectional"]->set(m_context->createProgramFromPTXFile(prefix + std::string("/DirectionalLight.cu.ptx"), "EmitDirectional"));
		ray_gen_program["GetRadianceDirectional"]->set(m_context->createProgramFromPTXFile(prefix + std::string("/DirectionalLight.cu.ptx"), "GetRadianceDirectional"));
#endif

		// Build the stratified grid...

		const int grid = 8;
		const int grid2 = grid * grid;
		Buffer strat_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
		strat_buffer->setFormat(RT_FORMAT_FLOAT2);
		strat_buffer->setSize(grid2);
		float *const map = (float *)strat_buffer->map();
		float *ptr = map;
		for (int y = 0; y < grid; ++y)
			for (int x = 0; x < grid; ++x, ptr += 2)
			{
				*ptr = ((float)x) / grid;
				*(ptr + 1) = ((float)y) / grid;
			}
		
		/*for (int i = 0; i < grid2; ++i)
		{
			const int i1 = (rand() % grid2) << 1;
			const int i2 = (rand() % grid2) << 1;
			const float t1 = map[i1];
			const float t2 = map[i1 + 1];
			map[i1] = map[i2];
			map[i1 + 1] = map[i2 + 1];
			map[i2] = t1;
			map[i2 + 1] = t2;
		}*/
		strat_buffer->unmap();
		m_context["strat_buffer"]->set(strat_buffer);

		m_context["aRadiusFactor"]->setFloat(0.003f);
		m_context["aRadiusAlpha"]->setFloat(0.75f);

		// Create scene geometry...

		createGeometry();

		// Finalize...

		m_context->validate();
		m_context->compile();
	}
	catch (Exception &e)
	{
		sutilReportError(e.getErrorString().c_str());
		exit(1);
	}
}

Buffer BidiPTracer::getOutputBuffer()
{
	return m_context["output_buffer"]->getBuffer();
}

void BidiPTracer::trace(const RayGenCameraData& camera_data)
{
	static unsigned int m_frame = 1;

	m_context["eye"]->setFloat(camera_data.eye);
	m_context["U"]->setFloat(camera_data.U);
	m_context["V"]->setFloat(camera_data.V);
	m_context["W"]->setFloat(camera_data.W);

	Buffer buffer = m_context["output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	buffer->getSize(buffer_width, buffer_height);

	if (m_camera_changed)
	{
		m_camera_changed = false;
		m_frame = 1;
	}

	m_context["frame_number"]->setUint(m_frame++);

	Camera cam;
	cam.Setup(camera_data.eye, normalize(camera_data.W), normalize(camera_data.V), make_float2((float)buffer_width, (float)buffer_height), 45.0f);
	m_context["camera"]->setUserData(sizeof(cam), &cam);

	m_context->launch(1, 
					  static_cast<unsigned int>(buffer_width),
					  static_cast<unsigned int>(buffer_height)
					  );
	m_context->launch(0,
					  static_cast<unsigned int>(buffer_width),
					  static_cast<unsigned int>(buffer_height)
					  );
	m_context->launch(2, 
					  static_cast<unsigned int>(buffer_width),
					  static_cast<unsigned int>(buffer_height)
					  );
}

/*// Chapel

void BidiPTracer::createGeometry()
{
	const int numMat = 2;
	optix::Material material[numMat];
	for (int i = 0; i < numMat; ++i)
		material[i] = m_context->createMaterial();

	// Load closest hit program...

	Program closest_hit = m_context->createProgramFromPTXFile(prefix + "/diffuse.cu.ptx", "closest_hit_diffuse");
	Program any_hit = m_context->createProgramFromPTXFile(prefix + "/diffuse.cu.ptx", "any_hit_occlusion");

	// Setup materials...

	// Diffuse material...

	material[0]->setClosestHitProgram(0, closest_hit);
	material[0]->setAnyHitProgram(1, any_hit);
	TriangleMaterial mat;
	mat.Reset();
	mat.mDiffuseReflectance = make_float3(0.76f, 0.75f, 0.5f);
	material[0]["mat"]->setUserData(sizeof(TriangleMaterial), &mat);

	// Light material...

	const float3 emissive = make_float3(0.803922f, 0.803922f, 0.803922f) * 10.0f;
	material[1]->setClosestHitProgram(0, closest_hit);
	material[1]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.mDiffuseReflectance = emissive;
	mat.isEmitter = true;
	material[1]["mat"]->setUserData(sizeof(TriangleMaterial), &mat);

	// Load OBJ files and set as geometry groups
	
	const int numGeomGroups = 2;
	GeometryGroup geomgroup[numGeomGroups];
	for (int i = 0; i < numGeomGroups; ++i)
		geomgroup[i] = m_context->createGeometryGroup();

	// Load intersect program...

	std::string ptx_path = prefix + std::string("/triangle_mesh_iterative.cu.ptx");
	Program mesh_intersect = m_context->createProgramFromPTXFile(ptx_path, "mesh_intersect");
	Program mesh_bounds = m_context->createProgramFromPTXFile(ptx_path, "mesh_bounds");

	// Load geometries...

	optix::Matrix4x4 m0 = m0.identity();
	ObjLoader objloader0("./Chapel.obj", m_context, geomgroup[0], material[0], false, "Sbvh", "Bvh");
	objloader0.setIntersectProgram(mesh_intersect);
	objloader0.setBboxProgram(mesh_bounds);
	//m0 = m0.scale(make_float3(15, 15, 15));
	objloader0.load(m0);

	ObjLoader objloader1("./ChapelLight.obj", m_context, geomgroup[1], material[1], false, "Sbvh", "Bvh");
	objloader1.setIntersectProgram(mesh_intersect);
	objloader1.setBboxProgram(mesh_bounds);
	objloader1.load(m0);

	// Build lights...

	std::vector<Light> lights;
	GLMmodel *model = glmReadOBJ("./ChapelLight.obj");
	if (lightConditions > 1)
		createLightBuffer(model, lights, emissive); 
	delete model;

	if (lightConditions == 1)
	{
		Light tl;
		tl.SetupBackgroundLight(make_float3(135/255.0f, 206/255.0f, 250/255.0f));
		lights.push_back(tl);
		m_context["backLightIx"]->setInt(lights.size() - 1);
	}
	else
		m_context["backLightIx"]->setInt(-1);

	if (lightConditions == 0)
	{
		Light tl2;
		tl2.SetupDirectionalLight(make_float3(-1.f, -1.5f, -4.f), make_float3(0.5f, 0.2f, 0.f) * 10.0f);
		lights.push_back(tl2);
	}

	// Create a buffer for the next-event estimation...

	Buffer m_light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
	m_light_buffer->setFormat(RT_FORMAT_USER);
	m_light_buffer->setElementSize(sizeof(Light));
	m_light_buffer->setSize(0);
	if (lights.size() != 0)
	{
	    m_light_buffer->setSize(lights.size());
		memcpy(m_light_buffer->map(), &lights[0], lights.size() * sizeof(Light));
		m_light_buffer->unmap();
	}
	m_context["lightBuffer"]->set(m_light_buffer);

	// Build scene...

	GeometryGroup maingroup = m_context->createGeometryGroup();
	int numChildren = 0;
	for (int i = 0; i < numGeomGroups; ++i)
		numChildren += geomgroup[i]->getChildCount();
	maingroup->setChildCount(numChildren);
	int ct = 0;
	for (int i = 0; i < numGeomGroups; ++i)
		for (unsigned int j = 0; j < geomgroup[i]->getChildCount(); ++j)
			maingroup->setChild(ct++, geomgroup[i]->getChild(j));
	maingroup->setAcceleration(m_context->createAcceleration("Bvh", "Bvh"));
	m_context["top_object"]->set(maingroup);

	// TODO: Compute bounding sphere...

	SceneSphere sp;
	sp.mSceneCenter = make_float3(5.5f/2.0f, 5.5f/2.0f, -5.5f/2.0f);
	sp.mSceneRadius = (9.5f/2.0f) * sqrtf(2.0f);
	sp.mInvSceneRadiusSqr = 1.0f / (sp.mSceneRadius * sp.mSceneRadius);
	m_context["mSceneSphere"]->setUserData(sizeof(sp), &sp);

	m_context["mMaxPathLength"]->setInt(mMaxPathLength);
}*/

void BidiPTracer::createGeometry()
{
	const int numMat = 7;
	optix::Material material[numMat];
	for (int i = 0; i < numMat; ++i)
		material[i] = m_context->createMaterial();

	// Load closest hit program...

	Program closest_hit = m_context->createProgramFromPTXFile(prefix + "/diffuse.cu.ptx", "closest_hit_diffuse");
	Program any_hit = m_context->createProgramFromPTXFile(prefix + "/diffuse.cu.ptx", "any_hit_occlusion");

	// Setup materials...

	// Diffuse material...

	material[0]->setClosestHitProgram(0, closest_hit);
	material[0]->setAnyHitProgram(1, any_hit);
	TriangleMaterial mat;
	mat.Reset();
	mat.mDiffuseReflectance = make_float3(0.76f, 0.75f, 0.5f);
	material[0]["mat"]->setUserData(sizeof(TriangleMaterial), &mat);

	material[1]->setClosestHitProgram(0, closest_hit);
	material[1]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.mDiffuseReflectance = make_float3(0.63f, 0.06f, 0.04f);
	material[1]["mat"]->setUserData(sizeof(TriangleMaterial), &mat);

	material[2]->setClosestHitProgram(0, closest_hit);
	material[2]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.mDiffuseReflectance = make_float3(0.15f, 0.48f, 0.09f);
	material[2]["mat"]->setUserData(sizeof(TriangleMaterial), &mat);

	// Light material...

	material[3]->setClosestHitProgram(0, closest_hit);
	material[3]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.mDiffuseReflectance = make_float3(0.803922f, 0.803922f, 0.803922f);
	mat.isEmitter = true;
	material[3]["mat"]->setUserData(sizeof(TriangleMaterial), &mat);

	material[4]->setClosestHitProgram(0, closest_hit);
	material[4]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.mMirrorReflectance = make_float3(1, 1, 1);
	mat.mIOR = 1.6f;
	material[4]["mat"]->setUserData(sizeof(TriangleMaterial), &mat);

	material[5]->setClosestHitProgram(0, closest_hit);
	material[5]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.mDiffuseReflectance = make_float3(0.1, 0.1, 0.1);
	mat.mPhongReflectance = make_float3(0.7f);
	mat.mPhongExponent = 90.0f;
	material[5]["mat"]->setUserData(sizeof(TriangleMaterial), &mat);

	material[6]->setClosestHitProgram(0, closest_hit);
	material[6]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.mMirrorReflectance = make_float3(1, 1, 1);
	material[6]["mat"]->setUserData(sizeof(TriangleMaterial), &mat);

	// Load OBJ files and set as geometry groups
	
	const int numGeomGroups = 5;
	GeometryGroup geomgroup[numGeomGroups];
	for (int i = 0; i < numGeomGroups; ++i)
		geomgroup[i] = m_context->createGeometryGroup();

	// Load intersect program...

	std::string ptx_path = prefix + std::string("/triangle_mesh_iterative.cu.ptx");
	Program mesh_intersect = m_context->createProgramFromPTXFile(ptx_path, "mesh_intersect");
	Program mesh_bounds = m_context->createProgramFromPTXFile(ptx_path, "mesh_bounds");

	// Load geometries...

	optix::Matrix4x4 m0 = m0.identity();
	ObjLoader objloader0("./CornellDiffuse2.obj", m_context, geomgroup[0], material[0], false, "TriangleKdTree", "KdTree");
	objloader0.setIntersectProgram(mesh_intersect);
	objloader0.setBboxProgram(mesh_bounds);
	//m0 = m0.scale(make_float3(15, 15, 15));
	objloader0.load(m0);

	ObjLoader objloader1("./CornellRedWall.obj", m_context, geomgroup[1], material[1]);
	objloader1.setIntersectProgram(mesh_intersect);
	objloader1.setBboxProgram(mesh_bounds);
	objloader1.load(m0);

	ObjLoader objloader2("./CornellBlueWall.obj", m_context, geomgroup[2], material[2]);
	objloader2.setIntersectProgram(mesh_intersect);
	objloader2.setBboxProgram(mesh_bounds);
	objloader2.load(m0);

	ObjLoader objloader3("./CornellLight.obj", m_context, geomgroup[3], material[(lightConditions < 2) ? 0 : 3]);
	objloader3.setIntersectProgram(mesh_intersect);
	objloader3.setBboxProgram(mesh_bounds);
	//m0 = m0.scale(make_float3(5, 5, 5));
	optix::Matrix4x4 m1 = m1.translate(make_float3(0, 0, 0));
	objloader3.load(m0 * m1);

	const char *objname[] = { "./CornellSphere.obj", "./CornellKnot.obj", "./cognacglass.obj" };
	const int mmap[4] = { 0, 4, 5, 6 };

	m0 = m0.identity();
	ObjLoader objloader4(objname[object], m_context, geomgroup[4], material[mmap[materials]], false, "TriangleKdTree", "KdTree");
	objloader4.setIntersectProgram(mesh_intersect);
	objloader4.setBboxProgram(mesh_bounds);
	if (object == 2)
	{
		m0 = m0.scale(make_float3(0.25, 0.25, 0.25));
		m1 = m1.translate(make_float3(15, 2, -12));
	}
	objloader4.load(m0 * m1);

	// Build lights...

	std::vector<Light> lights;
	GLMmodel *model = glmReadOBJ("./CornellLight.obj");
	if (lightConditions > 1)
		createLightBuffer(model, lights, make_float3(25.03329895614464f, 25.03329895614464f, 25.03329895614464f)); 
	delete model;

	if (lightConditions == 1)
	{
		Light tl;
		tl.SetupBackgroundLight(make_float3(135/255.0f, 206/255.0f, 250/255.0f));
		lights.push_back(tl);
		m_context["backLightIx"]->setInt(lights.size() - 1);
	}
	else
		m_context["backLightIx"]->setInt(-1);

	if (lightConditions == 0)
	{
		Light tl2;
		tl2.SetupDirectionalLight(make_float3(-1.f, -1.5f, -4.f), make_float3(0.5f, 0.2f, 0.f) * 10.0f);
		lights.push_back(tl2);
	}

	// Create a buffer for the next-event estimation...

	Buffer m_light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
	m_light_buffer->setFormat(RT_FORMAT_USER);
	m_light_buffer->setElementSize(sizeof(Light));
	m_light_buffer->setSize(0);
	if (lights.size() != 0)
	{
	    m_light_buffer->setSize(lights.size());
		memcpy(m_light_buffer->map(), &lights[0], lights.size() * sizeof(Light));
		m_light_buffer->unmap();
	}
	m_context["lightBuffer"]->set(m_light_buffer);

	// Build scene...

	GeometryGroup maingroup = m_context->createGeometryGroup();
	int numChildren = 0;
	for (int i = 0; i < numGeomGroups; ++i)
		numChildren += geomgroup[i]->getChildCount();
	maingroup->setChildCount(numChildren);
	int ct = 0;
	for (int i = 0; i < numGeomGroups; ++i)
		for (unsigned int j = 0; j < geomgroup[i]->getChildCount(); ++j)
			maingroup->setChild(ct++, geomgroup[i]->getChild(j));
	maingroup->setAcceleration(m_context->createAcceleration("Bvh", "Bvh"));
	m_context["top_object"]->set(maingroup);

	// TODO: Compute bounding sphere...

	SceneSphere sp;
	sp.mSceneCenter = make_float3(5.5f/2.0f, 5.5f/2.0f, -5.5f/2.0f);
	sp.mSceneRadius = (9.5f/2.0f) * sqrtf(2.0f);
	sp.mInvSceneRadiusSqr = 1.0f / (sp.mSceneRadius * sp.mSceneRadius);
	m_context["mSceneSphere"]->setUserData(sizeof(sp), &sp);

	m_context["mMaxPathLength"]->setInt(mMaxPathLength);
}

void BidiPTracer::doResize(unsigned int width, unsigned int height)
{
	try {
		Buffer buffer = m_context["temp_buffer"]->getBuffer();
		buffer->setSize( width, height );

		if(m_use_vbo_buffer)
		{
		  buffer->unregisterGLBuffer();
		  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer->getGLBOId());
		  glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer->getElementSize() * width * height, 0, GL_STREAM_DRAW);
		  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		  buffer->registerGLBuffer();
		}

	  } catch( Exception& e ){
		sutilReportError( e.getErrorString().c_str() );
		exit(2);
	  }
}

void BidiPTracer::createLightBuffer(GLMmodel *model, std::vector<Light> &lights, const float3 &emissive)
{
    for (GLMgroup* obj_group = model->groups; obj_group != 0; obj_group = obj_group->next) 
    {
		unsigned int num_triangles = obj_group->numtriangles;
		if (num_triangles == 0) continue; 
	
		// extract necessary data
		for (unsigned int i = 0; i < obj_group->numtriangles; ++i)
		{
			// indices for vertex data
			unsigned int tindex = obj_group->triangles[i];
			int3 vindices;
			vindices.x = model->triangles[tindex].vindices[0]; 
			vindices.y = model->triangles[tindex].vindices[1];
			vindices.z = model->triangles[tindex].vindices[2];

			Light light;
			light.SetupAreaLight(*((float3*)&model->vertices[vindices.x * 3]),
								 *((float3*)&model->vertices[vindices.y * 3]),
								 *((float3*)&model->vertices[vindices.z * 3]),
								 emissive);
			lights.push_back(light);
		}
	}
}

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
	std::cerr
	<< "Usage  : " << argv0 << " [options]\n"
	<< "App options:\n"
	<< "  -h  | --help                               Print this usage message\n"
	<< "  -P  | --pbo                                Use OpenGL PBO for output buffer\n"
	<< "  -n  | --nopbo                              Use OptiX internal output buffer (Default)\n"
	<< std::endl;
	GLUTDisplay::printUsage();

	std::cerr
	<< "App keystrokes:\n"
	<< "  p Decrease number of devices\n"
	<< "  P Increase number of devices\n"
	<< "  space change light conditions\n"
	<< "  o change object\n"
	<< "  n change object material\n"
	<< "  0-9 change number of bounces (0 = 10)\n"
	<< std::endl;

	if ( doExit ) exit(1);
}

int main( int argc, char** argv )
{
	GLUTDisplay::init( argc, argv );

	bool use_vbo_buffer = true;
	for(int i = 1; i < argc; ++i) {
	std::string arg = argv[i];
	if(arg == "-P" || arg == "--pbo") {
		use_vbo_buffer = true;
	} else if( arg == "-n" || arg == "--nopbo" ) {
		use_vbo_buffer = false;
	} else if( arg == "-h" || arg == "--help" ) {
		printUsageAndExit( argv[0] );
	} else {
		std::cerr << "Unknown option '" << arg << "'\n";
		printUsageAndExit( argv[0] );
	}
	}

	if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

	BidiPTracer scene;
#ifdef NDEBUG
	scene.prefix = std::string("./release");
#else
	scene.prefix = std::string("./debug");
#endif
	scene.setUseVBOBuffer(use_vbo_buffer);
	GLUTDisplay::setUseSRGB(true);
	GLUTDisplay::setProgressiveDrawingTimeout(0);
	GLUTDisplay::run("BidiPTracer", &scene, GLUTDisplay::CDProgressive);
	return 0;
}