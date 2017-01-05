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

//-----------------------------------------------------------------------------
//
// optixPathTracer: simple interactive path tracer
//
//-----------------------------------------------------------------------------

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

//#include "primeCommon.h"
//#include <optix_prime/optix_primepp.h>
#include "optixPathTracer.h"
#include <sutil.h>
#include <Arcball.h>
#include <OptiXMesh.h>
#include "commonStructs.h"
#include "Camera.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixPathTracer";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context        context = 0;
uint32_t       width = 1024;
uint32_t       height = 768;
bool           use_pbo = true;

int            frame_number = 1;
int            sqrt_num_samples = 2;
int            rr_begin_depth = 1;
Program        tri_intersection = 0;
Program        tri_bounding_box = 0;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
bool           camera_changed = true;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

int lightConditions, mMaxPathLength;
int materials, object;
float amount;


//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

std::string ptxPath(const std::string& cuda_file);
Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void loadGeometry();
void setupCamera();
void updateCamera();
void glutInitialize(int* argc, char** argv);
void glutRun();

void glutDisplay();
void glutKeyboardPress(unsigned char k, int x, int y);
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);
void glutResize(int w, int h);


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

std::string ptxPath(const std::string& cuda_file)
{
	return
		std::string(sutil::samplesPTXDir()) +
		"/" + std::string(SAMPLE_NAME) + "_generated_" +
		cuda_file +
		".ptx";
}


Buffer getOutputBuffer()
{
	return context["output_buffer"]->getBuffer();
}


void destroyContext()
{
	if (context)
	{
		context->destroy();
		context = 0;
	}
}


void registerExitHandler()
{
	// register shutdown handler
#ifdef _WIN32
	glutCloseFunc(destroyContext);  // this function is freeglut-only
#else
	atexit(destroyContext);
#endif
}


void setMaterial(
	GeometryInstance& gi,
	Material material,
	const std::string& color_name,
	const float3& color)
{
	gi->addMaterial(material);
	gi[color_name]->setFloat(color);
}

void createContext()
{
	context = Context::create();
	context->setRayTypeCount(3);
	context->setEntryPointCount(1);
	context->setStackSize(1800);

	lightConditions = 2;
	materials = object = 0;
	mMaxPathLength = 5;
	amount = 10.f;
	context["mMaxPathLength"]->setInt(10);
	context->setRayTypeCount(2);
	context->setEntryPointCount(3);
	context->setStackSize(3072);

	context["max_depth"]->setInt(1);
	context["radiance_ray_type"]->setUint(0u);
	context["shadow_ray_type"]->setUint(1u);
	context["scene_epsilon"]->setFloat(1.e-4f);

	const float vfov = 45.0f;
	context["vfov"]->setFloat(vfov);

	Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
	context["output_buffer"]->setBuffer(buffer);

	Buffer temp_buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, false);
	context["temp_buffer"]->setBuffer(temp_buffer);

	Buffer b = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	b->setFormat(RT_FORMAT_INT);
	int vals[] = { 0 };
	b->setSize(sizeof(vals));
	memcpy(b->map(), vals, sizeof(vals));
	b->unmap();
	int* buffers = static_cast<int*>(b->map());
	b->unmap();
	context["raysTracedThisFrame"]->setBuffer(b);

	//ray gen programs
	std::string ptx = ptxPath(std::string("pinhole_camera.cu"));
	context->setRayGenerationProgram(0, context->createProgramFromPTXFile(ptx, "pinhole_camera"));
	context->setRayGenerationProgram(1, context->createProgramFromPTXFile(ptx, "clear_temp"));
	context->setRayGenerationProgram(2, context->createProgramFromPTXFile(ptx, "cumulate_samples"));

	context->setExceptionProgram(0, context->createProgramFromPTXFile(ptx, "exception"));
	ptx = ptxPath(std::string("constantbg.cu"));
	context->setMissProgram(0, context->createProgramFromPTXFile(ptx, "miss"));
	context->setMissProgram(1, context->createProgramFromPTXFile(ptx, "miss_occlusion"));


	// Setup programs
	
	context["sqrt_num_samples"]->setUint(sqrt_num_samples);
	context["bad_color"]->setFloat(1000000.0f, 0.0f, 1000000.0f); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
	context["bg_color"]->setFloat(make_float3(108.0f / 255.0f, 166.0f / 255.0f, 205.0f / 255.0f) * 0.5f);

	// Build the stratified grid...

	const int grid = 16;
	const int grid2 = grid * grid;
	Buffer strat_buffer = context->createBuffer(RT_BUFFER_INPUT);
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

	strat_buffer->unmap();
	context["strat_buffer"]->setBuffer(strat_buffer);

	context["aRadiusFactor"]->setFloat(0.003f);
	context["aRadiusAlpha"]->setFloat(0.75f);
}

void createLightBuffer(Mesh model, std::vector<Light> &nlights, const float3 &emissive)
{
	float3* vert = reinterpret_cast<float3*>(model.positions);
	for (int32_t i = 0; i < model.num_triangles; ++i)
	{
		Light vlight;
		vlight.SetupAreaLight(
			//completely wrong: needs to be the three vertices of the triangle
			vert[model.tri_indices[i * 3 + 0]],
			vert[model.tri_indices[i * 3 + 1]],
			vert[model.tri_indices[i * 3 + 2]],
			emissive
			);
		nlights.push_back(vlight);
	}
}


void loadGeometry(const std::string mesh_file)
{
	

	context->setPrintEnabled(1);
	context->setPrintBufferSize(4096);

	std::vector<Light> nlights;
	Mesh model, model1, model2;
	std::string light_name = std::string(sutil::samplesDir()) + "/data/CornellLight.obj";
	Matrix4x4 matrix = Matrix4x4::scale(make_float3(10.f)) * Matrix4x4::translate(make_float3(1.8f, 4.f, 0.f)) * Matrix4x4::rotate(-0.785f, make_float3(0.f, 1.f, 0.f));
	const float* xform = matrix.getData();
	
	loadMesh(light_name, model, xform);
	
	matrix = Matrix4x4::scale(make_float3(10.f)) * Matrix4x4::translate(make_float3(-3.f, 4.f, 0.f)) * Matrix4x4::rotate(0.785f, make_float3(0.f, 1.f, 0.f));
	xform = matrix.getData();
	loadMesh(light_name, model1, xform);

	matrix = Matrix4x4::scale(make_float3(10.f)) * Matrix4x4::translate(make_float3(0.f, 5.f, 3.f)) * Matrix4x4::rotate(0.785f, make_float3(1.f, 0.f, 0.f));
	xform = matrix.getData();
	loadMesh(light_name, model2, xform);

	if (lightConditions > 1){
		createLightBuffer(model, nlights, make_float3(0.3f * 70.03329895614464f));
		createLightBuffer(model1, nlights, make_float3(0.3f * 80.f));
		createLightBuffer(model2, nlights, make_float3(0.3f * 90.f));
	}

	// Create a buffer for the next-event estimation...

	Buffer m_light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	m_light_buffer->setFormat(RT_FORMAT_USER);
	m_light_buffer->setElementSize(sizeof(Light));
	m_light_buffer->setSize(0);
	if (nlights.size() != 0)
	{
		m_light_buffer->setSize(nlights.size());
		memcpy(m_light_buffer->map(), &nlights[0], nlights.size() * sizeof(Light));
		m_light_buffer->unmap();
	}
	context["lightBuffer"]->setBuffer(m_light_buffer);


	const int numMat = 9;
	optix::Material material[numMat];
	for (int i = 0; i < numMat; ++i)
		material[i] = context->createMaterial();

	// Load closest hit program...

	Program closest_hit = context->createProgramFromPTXFile(ptxPath("diffuse.cu"), "closest_hit_diffuse");
	Program any_hit = context->createProgramFromPTXFile(ptxPath("diffuse.cu"), "any_hit_occlusion");

	material[0]->setClosestHitProgram(0, closest_hit);
	material[0]->setAnyHitProgram(1, any_hit);
	BaseMaterial mat;
	mat.Reset();
	mat.diffusePart = make_float3(0.5f);
	material[0]["mat"]->setUserData(sizeof(BaseMaterial), &mat);

	material[1]->setClosestHitProgram(0, closest_hit);
	material[1]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.diffusePart = make_float3(0.63f, 0.06f, 0.04f);
	material[1]["mat"]->setUserData(sizeof(BaseMaterial), &mat);

	material[2]->setClosestHitProgram(0, closest_hit);
	material[2]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.diffusePart = make_float3(0.15f, 0.48f, 0.09f);
	material[2]["mat"]->setUserData(sizeof(BaseMaterial), &mat);

	// Light material...

	material[3]->setClosestHitProgram(0, closest_hit);
	material[3]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.diffusePart = make_float3(0.803922f, 0.803922f, 0.803922f);
	mat.isEmitter = true;
	material[3]["mat"]->setUserData(sizeof(BaseMaterial), &mat);

	material[4]->setClosestHitProgram(0, closest_hit);
	material[4]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.mirror = make_float3(1, 1, 1);
	mat.ior = 1.6f;
	material[4]["mat"]->setUserData(sizeof(BaseMaterial), &mat);

	material[5]->setClosestHitProgram(0, closest_hit);
	material[5]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.diffusePart = make_float3(1.f);
	mat.phongPart = make_float3(0.7f);
	mat.exponent = 90.0f;
	material[5]["mat"]->setUserData(sizeof(BaseMaterial), &mat);

	material[6]->setClosestHitProgram(0, closest_hit);
	material[6]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.mirror = make_float3(1, 1, 1);
	material[6]["mat"]->setUserData(sizeof(BaseMaterial), &mat);

	material[7]->setClosestHitProgram(0, closest_hit);
	material[7]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.diffusePart = make_float3(0.f);
	mat.phongPart = make_float3(0.55f);
	mat.exponent = 70.0f;
	material[7]["mat"]->setUserData(sizeof(BaseMaterial), &mat);

	material[8]->setClosestHitProgram(0, closest_hit);
	material[8]->setAnyHitProgram(1, any_hit);
	mat.Reset();
	mat.diffusePart = make_float3(0.97f, 0.69f, 0.58f);
	material[8]["mat"]->setUserData(sizeof(BaseMaterial), &mat);

	const std::string ptx_path = ptxPath("triangle_mesh_iterative.cu");
	tri_bounding_box = context->createProgramFromPTXFile(ptx_path, "mesh_bounds");
	tri_intersection = context->createProgramFromPTXFile(ptx_path, "mesh_intersect");

	// create geometry instances
	std::vector<GeometryInstance> gis;

	const float3 white = make_float3(0.8f, 0.8f, 0.8f);
	const float3 green = make_float3(0.05f, 0.8f, 0.05f);
	const float3 red = make_float3(0.8f, 0.05f, 0.05f);
	const float3 light_em = make_float3(15.0f, 15.0f, 15.0f);

	//load mesh
	OptiXMesh mesh;
	mesh.context = context;
	mesh.intersection = tri_intersection;
	mesh.bounds = tri_bounding_box;
	mesh.material = material[1];

	mesh.material = material[0];
	loadMesh(std::string(sutil::samplesDir()) + "/data/infinity_plane.obj", mesh);
	gis.push_back(mesh.geom_instance);

	mesh.material = material[8];
	loadMesh(std::string(sutil::samplesDir()) + "/data/lucy.obj", mesh/*, Matrix4x4::translate(make_float3(0.f, 5.f, 0.f))*/);
	gis.push_back(mesh.geom_instance);
	//loadMesh(mesh_file, mesh, Matrix4x4::scale(make_float3(2500.f)) * Matrix4x4::translate(make_float3(0.12f, 0.f, -0.05f)));
	//gis.push_back(mesh.geom_instance);

	//mesh.material = material[4];
	//loadMesh(std::string(sutil::samplesDir()) + "/data/CornellKnot.obj", mesh, Matrix4x4::scale(make_float3(80.f)) * Matrix4x4::translate(make_float3(0.1f, 3.f, 0.f)));
	//gis.push_back(mesh.geom_instance);

	////loadMesh(std::string(sutil::samplesDir()) + "/data/dragon.obj", mesh, Matrix4x4::scale(make_float3(300.f)) * Matrix4x4::translate(make_float3(0.5f, 0.4f, -1.2f)) * Matrix4x4::rotate(-1.5709f, make_float3(0.f, 1.f, 0.f)));
	////gis.push_back(mesh.geom_instance);

	//mesh.material = material[5];
	//loadMesh(std::string(sutil::samplesDir()) + "/data/dragon.obj", mesh, Matrix4x4::scale(make_float3(300.f)) * Matrix4x4::translate(make_float3(1.f, 0.4f, -1.2f)) * Matrix4x4::rotate(-1.5709f, make_float3(0.f, 1.f, 0.f)));
	//gis.push_back(mesh.geom_instance);

	//mesh.material = material[7];
	//loadMesh(std::string(sutil::samplesDir()) + "/data/bunny.obj", mesh, Matrix4x4::scale(make_float3(900.f)) * Matrix4x4::translate(make_float3(0.1f, 0.2f, -0.1f)) * Matrix4x4::rotate(-1.5709f, make_float3(0.f, 1.f, 0.f)));
	//gis.push_back(mesh.geom_instance);

	//mesh.material = material[0];
	//loadMesh(std::string(sutil::samplesDir()) + "/data/CornellDiffuse2.obj", mesh, Matrix4x4::rotate(3.1412f, make_float3(0.f, 1.f, 0.f)) * Matrix4x4::scale(make_float3(100.f)) * Matrix4x4::translate(make_float3(-5.4f, 0.f, 5.f)));
	//gis.push_back(mesh.geom_instance);

	//mesh.material = material[0];
	///*loadMesh(std::string(sutil::samplesDir()) + "/data/CornellLight.obj", mesh, Matrix4x4::scale(make_float3(100.f)) * Matrix4x4::translate(make_float3(-0.1f, 5.f, -14.f)) * Matrix4x4::rotate(-M_PI_2f, make_float3(1.f, 0.f, 0.f)));
	//gis.push_back(mesh.geom_instance);*/

	//mesh.material = material[1];
	//loadMesh(std::string(sutil::samplesDir()) + "/data/CornellRedWall.obj", mesh, Matrix4x4::scale(make_float3(100.f)) * Matrix4x4::translate(make_float3(-0.1f, 0.f, 0.55f)));
	//gis.push_back(mesh.geom_instance);

	//mesh.material = material[2];
	//loadMesh(std::string(sutil::samplesDir()) + "/data/CornellBlueWall.obj", mesh, Matrix4x4::rotate(M_PIf, make_float3(0.f, 1.f, 0.f)) * Matrix4x4::scale(make_float3(100.f)) * Matrix4x4::translate(make_float3(-5.4f, 0.f, 5.f)));
	//gis.push_back(mesh.geom_instance);

	// Create shadow group (no light)
	GeometryGroup shadow_group = context->createGeometryGroup(gis.begin(), gis.end());
	shadow_group->setAcceleration(context->createAcceleration("Trbvh"));
	context["top_shadower"]->set(shadow_group);

	// Light
	OptiXMesh floor;
	floor.context = context;
	floor.intersection = tri_intersection;
	floor.bounds = tri_bounding_box;
	floor.material = material[3];
	loadMesh(std::string(sutil::samplesDir()) + "/data/CornellLight.obj", floor, Matrix4x4::scale(make_float3(10.f)) * Matrix4x4::translate(make_float3(1.8f, 4.f, 0.f)) * Matrix4x4::rotate(-0.785f, make_float3(0.f, 1.f, 0.f)));
	gis.push_back(floor.geom_instance);

	loadMesh(std::string(sutil::samplesDir()) + "/data/CornellLight.obj", floor, Matrix4x4::scale(make_float3(10.f)) * Matrix4x4::translate(make_float3(-3.f, 4.f, 0.f)) * Matrix4x4::rotate(0.785f, make_float3(0.f, 1.f, 0.f)));
	gis.push_back(floor.geom_instance);

	loadMesh(std::string(sutil::samplesDir()) + "/data/CornellLight.obj", floor, Matrix4x4::scale(make_float3(10.f)) * Matrix4x4::translate(make_float3(0.f, 5.f, 3.f)) *
		Matrix4x4::rotate(0.785f, make_float3(1.f, 0.f, 0.f)));
	gis.push_back(floor.geom_instance);
	// Create geometry group

	Acceleration acc = context->createAcceleration("Trbvh");

	GeometryGroup geometry_group = context->createGeometryGroup(gis.begin(), gis.end());
	geometry_group->setAcceleration(acc);
	acc->markDirty();

	context["top_object"]->set(geometry_group);
}


void setupCamera()
{
	camera_eye = make_float3(0.44f, 13.0f, 50.590f);
	camera_lookat = make_float3(0.44f, 13.0f, 3.0f);
	camera_up = make_float3(0.0f, 1.0f, 0.0f);

	camera_rotate = Matrix4x4::identity();
}


void updateCamera()
{
	const float fov = 35.0f;
	const float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);

	float3 camera_u, camera_v, camera_w;
	sutil::calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
		camera_u, camera_v, camera_w, /*fov_is_vertical*/ true);

	const Matrix4x4 frame = Matrix4x4::fromBasis(
		normalize(camera_u),
		normalize(camera_v),
		normalize(-camera_w),
		camera_lookat);
	const Matrix4x4 frame_inv = frame.inverse();
	// Apply camera rotation twice to match old SDK behavior
	const Matrix4x4 trans = frame*camera_rotate*camera_rotate*frame_inv;

	camera_eye = make_float3(trans*make_float4(camera_eye, 1.0f));
	camera_lookat = make_float3(trans*make_float4(camera_lookat, 1.0f));
	camera_up = make_float3(trans*make_float4(camera_up, 0.0f));

	sutil::calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
		camera_u, camera_v, camera_w, true);

	camera_rotate = Matrix4x4::identity();

	context["frame_number"]->setUint(frame_number++);
	context["eye"]->setFloat(camera_eye);
	context["U"]->setFloat(camera_u);
	context["V"]->setFloat(camera_v);
	context["W"]->setFloat(camera_w);

	if (camera_changed) // reset accumulation
		frame_number = 1;
	camera_changed = false;

	Buffer buffer = context["output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	buffer->getSize(buffer_width, buffer_height);

	Camera cam;
	cam.Setup(camera_eye, normalize(camera_w), normalize(camera_v), make_float2((float)buffer_width, (float)buffer_height), fov);
	context["camera"]->setUserData(sizeof(cam), &cam);
}


void glutInitialize(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(SAMPLE_NAME);
	glutHideWindow();
}


void glutRun()
{
	// Initialize GL state                                                            
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, width, height);

	glutShowWindow();
	glutReshapeWindow(width, height);

	// register glut callbacks
	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutDisplay);
	glutReshapeFunc(glutResize);
	glutKeyboardFunc(glutKeyboardPress);
	glutMouseFunc(glutMousePress);
	glutMotionFunc(glutMouseMotion);

	registerExitHandler();

	glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
	updateCamera();

	Buffer b = context->getBufferFromId(3);
	int vals[] = { 0 }; //re-initialize a zero value
	memcpy(b->map(), vals, sizeof(vals)); // copy the zero value to buffer
	b->unmap();

	context->launch(1, width, height);
	context->launch(0, width, height);
	context->launch(2, width, height);

	int* buffers = static_cast<int*>(b->map()); // grab the resulting buffer.
	//std::cout << "Rays This Frame: " << buffers[0] << " frame: " << frame_number << std::endl; // Output the rays in this frame
	b->unmap();

	sutil::displayBufferGL(getOutputBuffer());

	{
		static unsigned frame_count = 0;
		sutil::displayFps(frame_count++);
	}

	glutSwapBuffers();
}


void glutKeyboardPress(unsigned char k, int x, int y)
{

	switch (k)
	{
		case('m') :
		case(27) : // ESC
		{
			destroyContext();
			exit(0);
		}
		case('g') :
		{
			const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
			std::cerr << "Saving current frame to '" << outputImage << "'\n";
			sutil::displayBufferPPM(outputImage.c_str(), getOutputBuffer());
			break;
		}
		case('s') :
		{
			camera_eye = camera_eye - make_float3(0.f, 0.f, -amount);
			camera_changed = true;
			break;
		}
		case('d') :
		{
			camera_eye = camera_eye - make_float3(-amount, 0.f, 0.f);
			camera_changed = true;
			break;
		}
		case('a') :
		{
			camera_eye = camera_eye - make_float3(amount, 0.f, 0.f);
			camera_changed = true;
			break;
		}
		case('w') :
		{
			camera_eye = camera_eye - make_float3(0.f, 0.f, amount);
			camera_changed = true;
			break;
		}
		case('e') :
		{
			camera_eye = camera_eye - make_float3(0.f, amount, 0.f);
			camera_changed = true;
			break;
		}
		case('q') :
		{
			camera_eye = camera_eye - make_float3(0.f, -amount, 0.f);
			camera_changed = true;
			break;
		}
	}
}


void glutMousePress(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_button = button;
		mouse_prev_pos = make_int2(x, y);
	}
	else
	{
		// nothing
	}
}


void glutMouseMotion(int x, int y)
{
	if (mouse_button == GLUT_RIGHT_BUTTON)
	{
		const float dx = static_cast<float>(x - mouse_prev_pos.x) /
			static_cast<float>(width);
		const float dy = static_cast<float>(y - mouse_prev_pos.y) /
			static_cast<float>(height);
		const float dmax = fabsf(dx) > fabs(dy) ? dx : dy;
		const float scale = std::min<float>(dmax, 0.9f);
		camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
		camera_changed = true;
	}
	else if (mouse_button == GLUT_LEFT_BUTTON)
	{
		const float2 from = { static_cast<float>(mouse_prev_pos.x),
			static_cast<float>(mouse_prev_pos.y) };
		const float2 to = { static_cast<float>(x),
			static_cast<float>(y) };

		const float2 a = { from.x / width, from.y / height };
		const float2 b = { to.x / width, to.y / height };

		camera_rotate = arcball.rotate(b, a);
		camera_changed = true;
	}

	mouse_prev_pos = make_int2(x, y);
}


void glutResize(int w, int h)
{
	if (w == (int)width && h == (int)height) return;

	camera_changed = true;

	width = w;
	height = h;

	sutil::resizeBuffer(getOutputBuffer(), width, height);

	glViewport(0, 0, width, height);

	glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit(const std::string& argv0)
{
	std::cerr << "\nUsage: " << argv0 << " [options]\n";
	std::cerr <<
		"App Options:\n"
		"  -h | --help               Print this usage message and exit.\n"
		"  -f | --file               Save single frame to file and exit.\n"
		"  -n | --nopbo              Disable GL interop for display buffer.\n"
		"  -m | --mesh <mesh_file>   Specify path to mesh to be loaded.\n"
		"App Keystrokes:\n"
		"  q  Quit\n"
		"  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
		<< std::endl;

	exit(1);
}


int main(int argc, char** argv)
{
	std::string out_file;
	std::string mesh_file = std::string(sutil::samplesDir()) + "/data/cow.obj";
	for (int i = 1; i<argc; ++i)
	{
		const std::string arg(argv[i]);

		if (arg == "-h" || arg == "--help")
		{
			printUsageAndExit(argv[0]);
		}
		else if (arg == "-f" || arg == "--file")
		{
			if (i == argc - 1)
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit(argv[0]);
			}
			out_file = argv[++i];
		}
		else if (arg == "-n" || arg == "--nopbo")
		{
			use_pbo = false;
		}
		else if (arg == "-m" || arg == "--mesh")
		{
			if (i == argc - 1)
			{
				std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
				printUsageAndExit(argv[0]);
			}
			mesh_file = argv[++i];
		}
		else
		{
			std::cerr << "Unknown option '" << arg << "'\n";
			printUsageAndExit(argv[0]);
		}
	}

	try
	{
		glutInitialize(&argc, argv);

#ifndef __APPLE__
		glewInit();
#endif

		createContext();
		setupCamera();
		loadGeometry(mesh_file);

		context->validate();

		if (out_file.empty())
		{
			glutRun();
		}
		else
		{
			updateCamera();
			context->launch(0, width, height);
			sutil::displayBufferPPM(out_file.c_str(), getOutputBuffer());
			destroyContext();
		}

		return 0;
	}
	SUTIL_CATCH(context->get())
}
