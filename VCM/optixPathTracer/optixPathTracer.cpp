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

#include "optixPathTracer.h"
#include <sutil.h>
#include <Arcball.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <OptiXMesh.h>
#include <time.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixPathTracer";

#define MAX_PHOTON 20000000
//#define DRAWPHOTON
const std::string contextFileName = "photonSecondPass.cu";
//const std::string contextFileName = "optixPathTracer.cu";
//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context        context = 0;
uint32_t       width  = 600;
uint32_t       height = 600;
bool           use_pbo = true;

int            frame_number = 0;
int            sqrt_num_samples = 2;
int            rr_begin_depth = 1;
Program        pgram_intersection = 0;
Program        pgram_bounding_box = 0;
Program        sphere_intersection = 0;
Program		   sphere_bounding_box = 0;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
bool           camera_changed = true;
sutil::Arcball arcball;

// Pre-pass camera
float3	prepass_camera_up;
float3	prepass_camera_lookat;
float3	prepass_camera_eye;
Matrix4x4	prepass_camera_rotate;
bool	prepass_camera_changed = true;
int	prepass_frame_number = 0;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

std::vector<Buffer> vAabbBuffer;

Context	prepass_context = 0;
float3	lightPos; // used for pre-pass stage
Buffer	photonBuffer;
int photonSamples = 100; // number of samples in 360 degrees
const int nPrePassIteration = 1000;
const int maxDepth = 5;

std::vector<float3> photonPos;
std::vector<float3> photonColor;
int validPhoton;

// grid information used for second-pass of photon mapping
#define MAX_GRID 30000000
int gridStartIndex[MAX_GRID];
int gridEndIndex[MAX_GRID];
const float gridLength = 30.f;
const int gridSideCount = 20;
const float gridSideLength = gridLength * gridSideCount;
const float gridMin = -10.f;
std::vector<std::pair<Photon, int>> vPhotonGrid;
std::vector<Photon> secondPassPhoton;

//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

std::string ptxPath( const std::string& cuda_file );
Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void loadGeometry();
void setupCamera();
void updateCamera();
void glutInitialize( int* argc, char** argv );
void glutRun();

void glutDisplay();
void glutKeyboardPress( unsigned char k, int x, int y );
void glutMousePress( int button, int state, int x, int y );
void glutMouseMotion( int x, int y);
void glutResize( int w, int h );

void createPrePassContext();
void setupPrePassCamera();
void glutPrePassRun();
void glutPrePassDisplay();
void loadPrePassGeometry();
void glutPrePassMouseMotion(int x, int y);

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

std::string ptxPath( const std::string& cuda_file )
{
    return
        std::string(sutil::samplesPTXDir()) +
        "/" + std::string(SAMPLE_NAME) + "_generated_" +
        cuda_file +
        ".ptx";
}


Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}


void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}


void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc( destroyContext );  // this function is freeglut-only
#else
    atexit( destroyContext );
#endif
}


void setMaterial(
        GeometryInstance& gi,
        Material material,
        const std::string& color_name,
        const float3& color)
{
    gi->addMaterial(material);
	//printf("Material:%d\n", gi->getMaterialCount());
    gi[color_name]->setFloat(color);
}


GeometryInstance createParallelogram(
        const float3& anchor,
        const float3& offset1,
        const float3& offset2,
		Context& crtContext)
{
    Geometry parallelogram = crtContext->createGeometry();
    parallelogram->setPrimitiveCount( 1u );
    parallelogram->setIntersectionProgram( pgram_intersection );
    parallelogram->setBoundingBoxProgram( pgram_bounding_box );

    float3 normal = normalize( cross( offset1, offset2 ) );
    float d = dot( normal, anchor );
    float4 plane = make_float4( normal, d );

    float3 v1 = offset1 / dot( offset1, offset1 );
    float3 v2 = offset2 / dot( offset2, offset2 );

    parallelogram["plane"]->setFloat( plane );
    parallelogram["anchor"]->setFloat( anchor );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );

	Buffer AabbBuffer = crtContext->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, 2);
	parallelogram["Aabb_buffer"]->set(AabbBuffer);
	vAabbBuffer.push_back(AabbBuffer);

    GeometryInstance gi = crtContext->createGeometryInstance();
    gi->setGeometry(parallelogram);
    return gi;
}

GeometryInstance createSphere(
	const float3& center,
	const float& radius,
	Context& crtContext)
{
	Geometry sphere = crtContext->createGeometry();
	sphere->setPrimitiveCount( 1u );
	sphere->setIntersectionProgram( sphere_intersection );
	sphere->setBoundingBoxProgram( sphere_bounding_box );

	sphere["center"]->setFloat( center );
	sphere["radius"]->setFloat( radius );

	Buffer AabbBuffer = crtContext->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, 2);
	sphere["Aabb_buffer"]->set(AabbBuffer);
	vAabbBuffer.push_back(AabbBuffer);

	GeometryInstance gi = crtContext->createGeometryInstance();
	gi->setGeometry(sphere);
	return gi;
}

void createPrePassContext()
{
	prepass_context = Context::create();
	prepass_context->setRayTypeCount( 2 );
	prepass_context->setEntryPointCount( 1 );
	prepass_context->setStackSize( 1800 );

	prepass_context[ "scene_epsilon"                  ]->setFloat( 1.e-3f );
	prepass_context[ "pathtrace_ray_type"             ]->setUint( 0u );
	prepass_context[ "pathtrace_shadow_ray_type"      ]->setUint( 1u );
	prepass_context[ "rr_begin_depth"                 ]->setUint( rr_begin_depth );

	Buffer buffer = sutil::createOutputBuffer( prepass_context, RT_FORMAT_FLOAT4, photonSamples, photonSamples, use_pbo );
	prepass_context["output_buffer"]->set( buffer );
	Buffer photonBuffer = prepass_context->createBuffer( RT_BUFFER_OUTPUT);
	photonBuffer->setFormat( RT_FORMAT_USER );
	photonBuffer->setElementSize( sizeof( Photon ) );
	photonBuffer->setSize(maxDepth * photonSamples * photonSamples );
	prepass_context["photonBuffer"]->set(photonBuffer);
	Buffer isHitBuffer = prepass_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, maxDepth * photonSamples * photonSamples);
	prepass_context["isHitBuffer"]->set(isHitBuffer);

	// Setup programs
	const std::string cuda_file = "photonPrePass.cu";
	const std::string ptx_path = ptxPath( cuda_file );
	prepass_context->setRayGenerationProgram( 0, prepass_context->createProgramFromPTXFile( ptx_path, "pathtrace_camera" ) );
	prepass_context->setExceptionProgram( 0, prepass_context->createProgramFromPTXFile( ptx_path, "exception" ) );
	prepass_context->setMissProgram( 0, prepass_context->createProgramFromPTXFile( ptx_path, "miss" ) );

	prepass_context[ "sqrt_num_samples" ]->setUint( sqrt_num_samples );
	prepass_context[ "bad_color"        ]->setFloat( 1000000.0f, 0.0f, 1000000.0f ); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
	prepass_context[ "bg_color"         ]->setFloat( make_float3(0.0f) );
	prepass_context[ "row"]->setInt((int)(sqrt(nPrePassIteration + 1) + 0.5f));
	prepass_context[ "maxDepth"]->setInt(maxDepth);
}

void createContext(std::string filename)
{
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    context->setStackSize( 1800 );

    context[ "scene_epsilon"                  ]->setFloat( 1.e-3f );
    context[ "pathtrace_ray_type"             ]->setUint( 0u );
    context[ "pathtrace_shadow_ray_type"      ]->setUint( 1u );
    context[ "rr_begin_depth"                 ]->setUint( rr_begin_depth );

    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_FLOAT4, width, height, use_pbo );
    context["output_buffer"]->set( buffer );

    // Setup programs
    const std::string cuda_file = filename;
    const std::string ptx_path = ptxPath( cuda_file );
    context->setRayGenerationProgram( 0, context->createProgramFromPTXFile( ptx_path, "pathtrace_camera" ) );
    context->setExceptionProgram( 0, context->createProgramFromPTXFile( ptx_path, "exception" ) );
    context->setMissProgram( 0, context->createProgramFromPTXFile( ptx_path, "miss" ) );
    
    context[ "sqrt_num_samples" ]->setUint( sqrt_num_samples );
    context[ "bad_color"        ]->setFloat( 1000000.0f, 0.0f, 1000000.0f ); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
    context[ "bg_color"         ]->setFloat( make_float3(0.0f) );
}


void loadGeometry(Context& crtContext, std::string cudaFileName)
{
    // Light buffer
    ParallelogramLight light;
    light.corner   = make_float3( 343.0f, 548.6f, 227.0f);
    light.v1       = make_float3( -130.0f, 0.0f, 0.0f);
    light.v2       = make_float3( 0.0f, 0.0f, 105.0f);
    light.normal   = normalize( cross(light.v1, light.v2) );
    light.emission = make_float3( 15.0f, 15.0f, 5.0f );

    Buffer light_buffer = crtContext->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( ParallelogramLight ) );
    light_buffer->setSize( 1u );
    memcpy( light_buffer->map(), &light, sizeof( light ) );
    light_buffer->unmap();
    crtContext["lights"]->setBuffer( light_buffer );


    // Set up material
    const std::string cuda_file = cudaFileName;
    std::string ptx_path = ptxPath( cuda_file );
    Material diffuse = crtContext->createMaterial();
    Program diffuse_ch = crtContext->createProgramFromPTXFile( ptx_path, "diffuse" );
    Program diffuse_ah = crtContext->createProgramFromPTXFile( ptx_path, "shadow" );
    diffuse->setClosestHitProgram( 0, diffuse_ch );
    diffuse->setAnyHitProgram( 1, diffuse_ah );

	Material specular = crtContext->createMaterial();
	Program specular_ch = crtContext->createProgramFromPTXFile(ptx_path, "specular");
	Program specular_ah = crtContext->createProgramFromPTXFile(ptx_path, "shadow");
	specular->setClosestHitProgram(0, specular_ch);
	specular->setAnyHitProgram(1, specular_ah);

    Material diffuse_light = crtContext->createMaterial();
    Program diffuse_em = crtContext->createProgramFromPTXFile( ptx_path, "diffuseEmitter" );
    diffuse_light->setClosestHitProgram( 0, diffuse_em );
	// light doesn't have any-hit program so the shadow ray won't intersect with light

	Material glass = crtContext->createMaterial();
	Program glass_ch = crtContext->createProgramFromPTXFile(ptx_path, "glass_closest_hit_radiance");
	Program glass_ah = crtContext->createProgramFromPTXFile(ptx_path, "shadow");
	glass->setClosestHitProgram(0, glass_ch);
	glass->setAnyHitProgram(1, glass_ah);

    // Set up parallelogram programs
    ptx_path = ptxPath( "parallelogram.cu" );
    pgram_bounding_box = crtContext->createProgramFromPTXFile( ptx_path, "bounds" );
    pgram_intersection = crtContext->createProgramFromPTXFile( ptx_path, "intersect" );

	ptx_path = ptxPath("sphere.cu");
	sphere_bounding_box = crtContext->createProgramFromPTXFile(ptx_path, "bounds");
	sphere_intersection = crtContext->createProgramFromPTXFile(ptx_path, "intersect");

    // create geometry instances
    std::vector<GeometryInstance> gis;

    const float3 white = make_float3( 0.8f, 0.8f, 0.8f );
    const float3 green = make_float3( 0.05f, 0.8f, 0.05f );
    const float3 red   = make_float3( 0.8f, 0.05f, 0.05f );
    const float3 light_em = make_float3( 15.0f, 15.0f, 5.0f );
	const float3 gray = make_float3(0.5f, 0.5f, 0.5f);
	const float3 blue = make_float3(0.05f, 0.05f, 0.8f);
	const float3 yellow = make_float3(0.8f, 0.8f, 0.05f);

    // Floor
    gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 0.0f ),
                                        make_float3( 0.0f, 0.0f, 559.2f ),
                                        make_float3( 556.0f, 0.0f, 0.0f ),
										crtContext) );
    setMaterial(gis.back(), diffuse, "diffuse_color", gray);

    // Ceiling
    gis.push_back( createParallelogram( make_float3( 0.0f, 548.8f, 0.0f ),
                                        make_float3( 556.0f, 0.0f, 0.0f ),
										make_float3( 0.0f, 0.0f, 559.2f ),
										crtContext ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", gray);

    // Back wall
    gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 559.2f),
                                        make_float3( 0.0f, 548.8f, 0.0f),
										make_float3( 556.0f, 0.0f, 0.0f),
										crtContext ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", yellow);

    // Right wall
    gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 0.0f ),
                                        make_float3( 0.0f, 548.8f, 0.0f ),
										make_float3( 0.0f, 0.0f, 559.2f ),
										crtContext ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", green);

    // Left wall
    gis.push_back( createParallelogram( make_float3( 556.0f, 0.0f, 0.0f ),
                                        make_float3( 0.0f, 0.0f, 559.2f ),
										make_float3( 0.0f, 548.8f, 0.0f ),
										crtContext ) );
    setMaterial(gis.back(), diffuse, "diffuse_color", red);


#define COW

#ifdef SPHERE
	gis.push_back( createSphere( make_float3(150.0f, 350.0f, 350.0f), 100.0f,
		crtContext));
	setMaterial(gis.back(), specular, "diffuse_color", blue);

	//gis.push_back( createSphere( make_float3(220.0f, 250.0f, 220.0f), 100.0f,
	//	crtContext));
	//setMaterial(gis.back(), glass, "diffuse_color", blue);

	//gis.push_back( createSphere( make_float3(450.0f, 150.0f, 250.0f), 100.0f,
	//	crtContext));
	//setMaterial(gis.back(), glass, "diffuse_color", blue);
#endif

#ifdef COW
	OptiXMesh mesh;
	std::string filename = std::string(sutil::samplesDir()) + "/data/cow.obj";
	mesh.context = crtContext;
	mesh.material = glass;

	loadMesh(filename, mesh, Matrix4x4::translate(make_float3(250.f, 180.f, 250.f)) * Matrix4x4::scale(make_float3(50.f)));
	gis.push_back(mesh.geom_instance);
	gis.back()["diffuse_color"]->setFloat(red);
	//setMaterial(gis.back(), diffuse, "diffuse_color", red);
#endif

#ifdef DRAGON
	OptiXMesh mesh;
	std::string filename = std::string(sutil::samplesDir()) + "/data/dragon.obj";
	mesh.context = crtContext;
	mesh.material = specular;

	loadMesh(filename, mesh, Matrix4x4::translate(make_float3(270.f, 100.f, 360.f)) * Matrix4x4::scale(make_float3(35.f)));
	gis.push_back(mesh.geom_instance);
	gis.back()["diffuse_color"]->setFloat(red);
#endif

    GeometryGroup shadow_group = crtContext->createGeometryGroup(gis.begin(), gis.end());
    shadow_group->setAcceleration( crtContext->createAcceleration( "Trbvh" ) );
    crtContext["top_shadower"]->set( shadow_group );

    // Light
	gis.push_back( createParallelogram( make_float3( 343.0f, 548.6f, 227.0f),
		make_float3( -130.0f, 0.0f, 0.0f),
		make_float3( 0.0f, 0.0f, 105.0f),
		crtContext) );
	setMaterial(gis.back(), diffuse_light, "emission_color", light_em);

    // Create geometry group
    GeometryGroup geometry_group = crtContext->createGeometryGroup(gis.begin(), gis.end());
	//geometry_group->addChild(mesh.geom_instance);
    geometry_group->setAcceleration( crtContext->createAcceleration( "Trbvh" ) );
    crtContext["top_object"]->set( geometry_group );
}

void setupCamera()
{
    camera_eye    = make_float3( 278.0f, 273.0f, -900.0f );
    camera_lookat = make_float3( 278.0f, 273.0f,    0.0f );
    camera_up     = make_float3(   0.0f,   1.0f,    0.0f );

    camera_rotate  = Matrix4x4::identity();
}

void setupPrePassCamera()
{
	//prepass_camera_eye    = make_float3( 278.0f, 273.0f, -900.0f );
	prepass_camera_eye    = make_float3( 343.0f, 548.6f, 227.0f );
	//prepass_camera_lookat = make_float3( 278.0f, 273.0f,    0.0f );
	prepass_camera_lookat = make_float3( 278.0f, 0.0f,    250.0f );
	prepass_camera_up     = make_float3(   0.0f,   1.0f,    0.0f );

	//prepass_camera_eye    = lightPos - make_float3(0.0f, 50.0f, 0.0f); 
	//prepass_camera_lookat = make_float3( 278.0f, 273.0f,    0.0f );
	//prepass_camera_up     = make_float3(   0.0f,   1.0f,    0.0f );

	prepass_camera_rotate = Matrix4x4::identity();
}

void updateCamera()
{
    const float fov  = 35.0f;
    const float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    
    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
            camera_u, camera_v, camera_w, /*fov_is_vertical*/ true );

    const Matrix4x4 frame = Matrix4x4::fromBasis( 
            normalize( camera_u ),
            normalize( camera_v ),
            normalize( -camera_w ),
            camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans     = frame*camera_rotate*camera_rotate*frame_inv; 

    camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    camera_rotate = Matrix4x4::identity();

    context[ "frame_number" ]->setUint( frame_number++ );
    context[ "eye"]->setFloat( camera_eye );
    context[ "U"  ]->setFloat( camera_u );
    context[ "V"  ]->setFloat( camera_v );
    context[ "W"  ]->setFloat( camera_w );

	//float3 F = normalize(camera_lookat - camera_eye);
	//float3 R = normalize( cross( F, camera_up ) );
	//float3 U = normalize( cross( R, F ) );

	//glm::mat4x4 projectionMatrix = glm::perspective(fov, aspect_ratio, 0.01f, 1000.f);
	//glm::mat4x4 viewMatrix = glm::mat4x4(R.x, U.x, F.x, 0.0f,
	//									 R.y, U.y, F.y, 0.0f,
	//									 R.z, U.z, F.z, 0.0f,
	//									 -camera_eye.x, -camera_eye.y, -camera_eye.z, 1.0f);
	//mvp = projectionMatrix * viewMatrix;

    if( camera_changed ) // reset accumulation
        frame_number = 0;
    camera_changed = false;
}

void updatePrePassCamera()
{
	const float fov  = 150.0f;
	const float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);

	float3 camera_u, camera_v, camera_w;
	sutil::calculateCameraVariables(
		prepass_camera_eye, prepass_camera_lookat, prepass_camera_up, fov, aspect_ratio,
		camera_u, camera_v, camera_w, /*fov_is_vertical*/ true );

	const Matrix4x4 frame = Matrix4x4::fromBasis( 
		normalize( camera_u ),
		normalize( camera_v ),
		normalize( -camera_w ),
		prepass_camera_lookat);
	const Matrix4x4 frame_inv = frame.inverse();
	// Apply camera rotation twice to match old SDK behavior
	const Matrix4x4 trans     = frame*prepass_camera_rotate*prepass_camera_rotate*frame_inv; 

	prepass_camera_eye    = make_float3( trans*make_float4( prepass_camera_eye,    1.0f ) );
	prepass_camera_lookat = make_float3( trans*make_float4( prepass_camera_lookat, 1.0f ) );
	prepass_camera_up     = make_float3( trans*make_float4( prepass_camera_up,     0.0f ) );

	sutil::calculateCameraVariables(
		prepass_camera_eye, prepass_camera_lookat, prepass_camera_up, fov, aspect_ratio,
		camera_u, camera_v, camera_w, true );

	prepass_camera_rotate = Matrix4x4::identity();

	prepass_context[ "frame_number" ]->setUint( prepass_frame_number++ );
	prepass_context[ "eye"]->setFloat( prepass_camera_eye );
	prepass_context[ "U"  ]->setFloat( camera_u );
	prepass_context[ "V"  ]->setFloat( camera_v );
	prepass_context[ "W"  ]->setFloat( camera_w );

	if( prepass_camera_changed ) // reset accumulation
		prepass_frame_number = 0;
	prepass_camera_changed = false;

}

void glutInitialize( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );                                               
    glutCreateWindow( SAMPLE_NAME );
    glutHideWindow();                                                              
}


void glutRun()
{
    // Initialize GL state                                                            
    glMatrixMode(GL_PROJECTION);                                                   
    glLoadIdentity();                                                              
    glOrtho(0, 1, 0, 1, -1, 1 );                                                   

    glMatrixMode(GL_MODELVIEW);                                                    
    glLoadIdentity();                                                              

    glViewport(0, 0, width, height);                                 

    glutShowWindow();                                                              
    glutReshapeWindow( width, height);

    // register glut callbacks
    glutDisplayFunc( glutDisplay );
    glutIdleFunc( glutDisplay );
    glutReshapeFunc( glutResize );
    glutKeyboardFunc( glutKeyboardPress );
    glutMouseFunc( glutMousePress );
    glutMotionFunc( glutMouseMotion );

    registerExitHandler();

    glutMainLoop();
}

//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void drawBoundingBox()
{
	glMatrixMode (GL_PROJECTION);  
	glLoadIdentity ();  
	gluPerspective(35.0, (GLfloat) width/(GLfloat) height, 0.01, 20000.0);  
	glMatrixMode(GL_MODELVIEW);  
	glLoadIdentity();  
	gluLookAt(camera_eye.x, camera_eye.y, camera_eye.z,
		camera_lookat.x, camera_lookat.y, camera_lookat.z,
		camera_up.x, camera_up.y, camera_up.z);

	glColor3f(0.0f, 0.0f, 1.0f);

	glBegin(GL_LINES);

	int size = vAabbBuffer.size();
	for (int i = 0; i < size; i++)
	{
		GLvoid* data = 0;
		RT_CHECK_ERROR(rtBufferMap(vAabbBuffer[i]->get(), &data));
		float *f = (float*)data;
		glVertex3f(f[0], f[1], f[2]);
		glVertex3f(f[0], f[4], f[2]);
		glVertex3f(f[0], f[1], f[2]);
		glVertex3f(f[0], f[1], f[5]);
		glVertex3f(f[0], f[1], f[2]);
		glVertex3f(f[3], f[1], f[2]);

		glVertex3f(f[3], f[1], f[5]);
		glVertex3f(f[3], f[4], f[5]);
		glVertex3f(f[3], f[1], f[5]);
		glVertex3f(f[0], f[1], f[5]);
		glVertex3f(f[3], f[1], f[5]);
		glVertex3f(f[3], f[1], f[2]);

		glVertex3f(f[3], f[4], f[2]);
		glVertex3f(f[3], f[1], f[2]);
		glVertex3f(f[3], f[4], f[2]);
		glVertex3f(f[3], f[4], f[5]);
		glVertex3f(f[3], f[4], f[2]);
		glVertex3f(f[0], f[4], f[2]);

		glVertex3f(f[0], f[4], f[5]);
		glVertex3f(f[0], f[1], f[5]);
		glVertex3f(f[0], f[4], f[5]);
		glVertex3f(f[3], f[4], f[5]);
		glVertex3f(f[0], f[4], f[5]);
		glVertex3f(f[0], f[4], f[2]);
		//printf("%f %f %f %f %f %f\n", f[0], f[1], f[2], f[3], f[4], f[5]);
		RT_CHECK_ERROR(rtBufferUnmap(vAabbBuffer[i]->get()));
	}

	glEnd();
}
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

bool cmp(const std::pair<Photon, int> &a, const std::pair<Photon, int> &b)
{
	return a.second < b.second;
}

void getPrePassPhotonBuffer()
{
	GLvoid* data = 0;
	RT_CHECK_ERROR(rtBufferMap(prepass_context["isHitBuffer"]->getBuffer()->get(), &data));
	int *isHit = (int*)data;
	RT_CHECK_ERROR(rtBufferMap(prepass_context["photonBuffer"]->getBuffer()->get(), &data));
	Photon *photon = (Photon*)data;

	validPhoton = maxDepth * photonSamples * photonSamples;
	//printf("Before TotalPhoton:%d\n", nTotalPhoton);
	thrust::remove_if(photon, photon + validPhoton, isHit, thrust::logical_not<bool>());
	validPhoton = thrust::count_if(isHit, isHit + validPhoton, thrust::identity<bool>());
	//printf("After TotalPhoton:%d\n", nTotalPhoton);
	//printf("photon size:%d\n", sizeof(Photon));
	//FILE *fp1 = fopen("test1.txt", "w");
	for (int i = 0; i < validPhoton; i++)
	{
		photonPos.push_back(photon[i].position);
		photonColor.push_back(photon[i].color);
		
		//fprintf(fp1, "%f %f %f\n", photonColor[i].x, photonColor[i].y, photonColor[i].z);
		int gridIndexX = (photon[i].position.x - gridMin) / gridLength;
		int gridIndexY = (photon[i].position.y - gridMin) / gridLength;
		int gridIndexZ = (photon[i].position.z - gridMin) / gridLength;
		//printf("%d %d %d\n", gridIndexX, gridIndexY, gridIndexZ);
		std::pair<Photon, int> pair(photon[i], gridIndexX * gridSideCount * gridSideCount + gridIndexY * gridSideCount + gridIndexZ);
		vPhotonGrid.push_back(pair);
	}
	//fclose(fp1);
	RT_CHECK_ERROR(rtBufferUnmap(prepass_context["isHitBuffer"]->getBuffer()->get()));
	RT_CHECK_ERROR(rtBufferUnmap(prepass_context["photonBuffer"]->getBuffer()->get()));
}

void sortGridAndPassBuffer()
{
	std::sort(vPhotonGrid.begin(), vPhotonGrid.end(), cmp);

	validPhoton = vPhotonGrid.size();
	printf("total photons:%d\n", validPhoton);
	memset(gridStartIndex, -1, sizeof(gridStartIndex));
	memset(gridEndIndex, -1, sizeof(gridEndIndex));
	gridStartIndex[vPhotonGrid[0].second] = gridEndIndex[vPhotonGrid[0].second] = 0;
	for (int i = 0; i < validPhoton; i++)
	{
		secondPassPhoton.push_back(vPhotonGrid[i].first);

		if (i == validPhoton - 1)
		{
			gridEndIndex[vPhotonGrid[i].second] = i;
		}
		else if (vPhotonGrid[i].second != vPhotonGrid[i + 1].second)
		{
			gridStartIndex[vPhotonGrid[i + 1].second] = gridEndIndex[vPhotonGrid[i + 1].second] = i;
			gridEndIndex[vPhotonGrid[i].second] = i;
		}
	}

	Buffer gridPhotonBuffer = context->createBuffer( RT_BUFFER_INPUT );
	gridPhotonBuffer->setFormat( RT_FORMAT_USER );
	gridPhotonBuffer->setElementSize( sizeof( Photon ) );
	gridPhotonBuffer->setSize( validPhoton );
	memcpy(gridPhotonBuffer->map(), &secondPassPhoton[0], sizeof(Photon) * validPhoton);
	gridPhotonBuffer->unmap();
	context["photonBuffer"]->setBuffer(gridPhotonBuffer);

	Buffer gridStartIndexBuffer = context->createBuffer( RT_BUFFER_INPUT );
	gridStartIndexBuffer->setFormat( RT_FORMAT_USER );
	gridStartIndexBuffer->setElementSize( sizeof( int ) );
	gridStartIndexBuffer->setSize( MAX_GRID );
	memcpy(gridStartIndexBuffer->map(), gridStartIndex, sizeof(int) * MAX_GRID);
	gridStartIndexBuffer->unmap();
	context["gridStartIndexBuffer"]->setBuffer(gridStartIndexBuffer);

	Buffer gridEndIndexBuffer = context->createBuffer( RT_BUFFER_INPUT );
	gridEndIndexBuffer->setFormat( RT_FORMAT_USER );
	gridEndIndexBuffer->setElementSize( sizeof( int ) );
	gridEndIndexBuffer->setSize( MAX_GRID );
	memcpy(gridEndIndexBuffer->map(), gridEndIndex, sizeof(int) * MAX_GRID);
	gridEndIndexBuffer->unmap();
	context["gridEndIndexBuffer"]->setBuffer(gridEndIndexBuffer);

	context[ "gridLength" ]->setFloat(gridLength);
	context[ "gridMin" ]->setFloat(gridMin);
	context[ "gridSideCount" ]->setInt(gridSideCount);
	context[ "totalPhotons" ]->setInt(validPhoton);

	vPhotonGrid.clear();
	secondPassPhoton.clear();
}
void drawPhoton()
{
	glMatrixMode (GL_PROJECTION);  
	glLoadIdentity ();  
	gluPerspective(35.0, (GLfloat) width/(GLfloat) height, 0.01, 20000.0);  
	glMatrixMode(GL_MODELVIEW);  
	glLoadIdentity();  
	gluLookAt(camera_eye.x, camera_eye.y, camera_eye.z,
		camera_lookat.x, camera_lookat.y, camera_lookat.z,
		camera_up.x, camera_up.y, camera_up.z);

	glPointSize(0.001f);
	glEnable(GL_FRAMEBUFFER_SRGB_EXT);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_DEPTH_TEST);

	glVertexPointer(3, GL_FLOAT, 0, &photonPos[0]);
	glEnableClientState(GL_VERTEX_ARRAY);
	glColorPointer(3, GL_FLOAT, 0, &photonColor[0]);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, photonPos.size());

	glDisable(GL_DEPTH_TEST);
}

void glutDisplay()
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//testGetBuffer();
	printf("Iteration:%d\n", frame_number);
    updateCamera();
    context->launch( 0, width, height );

#ifdef DRAWPHOTON
    drawPhoton();
#else
	sutil::displayBufferGL( getOutputBuffer());
	
#endif	
	
	//drawBoundingBox();
	//glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    {
      static unsigned frame_count = 0;
      //sutil::displayFps( frame_count++ );
    }

    glutSwapBuffers();
	
}

void glutKeyboardPress( unsigned char k, int x, int y )
{

    switch( k )
    {
        case( 'q' ):
        case( 27 ): // ESC
        {
            destroyContext();
            exit(0);
        }
        case( 's' ):
        {
            const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer() );
            break;
        }
    }
}


void glutMousePress( int button, int state, int x, int y )
{
    if( state == GLUT_DOWN )
    {
        mouse_button = button;
        mouse_prev_pos = make_int2( x, y );
    }
    else
    {
        // nothing
    }
}


void glutMouseMotion( int x, int y)
{
    if( mouse_button == GLUT_RIGHT_BUTTON )
    {
        const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
                         static_cast<float>( width );
        const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
                         static_cast<float>( height );
        const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
        const float scale = std::min<float>( dmax, 0.9f );
        //camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
		camera_eye = camera_eye + normalize(camera_lookat - camera_eye) * 300.f *scale;
		camera_lookat = camera_lookat + normalize(camera_lookat - camera_eye) * 300.f *scale;
        camera_changed = true;
    }
    else if( mouse_button == GLUT_LEFT_BUTTON )
    {
        const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to   = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x   / width, to.y   / height };

        camera_rotate = arcball.rotate( b, a );
        camera_changed = true;
    }
	else if( mouse_button == GLUT_MIDDLE_BUTTON )
	{
		//printf("MIDDLE\n");
		const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
			static_cast<float>( width );
		const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
			static_cast<float>( height );

		float3 F = normalize(camera_lookat - camera_eye);
		float3 R = normalize( cross( F, camera_up ) );
		float3 U = normalize( cross( R, F ) );

		camera_lookat += (-U * dy + R * dx) * 1000.0f;
		camera_eye += (-U * dy + R * dx) * 1000.0f;
		camera_changed = true;
	}

    mouse_prev_pos = make_int2( x, y );
}

void glutResize( int w, int h )
{
    if ( w == (int)width && h == (int)height ) return;

    camera_changed = true;

    width  = w;
    height = h;
    
    sutil::resizeBuffer( getOutputBuffer(), width, height );

    glViewport(0, 0, width, height);                                               

    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
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


int main( int argc, char** argv )
 {
    std::string out_file;
    std::string mesh_file = std::string( sutil::samplesDir() ) + "/data/cow.obj";
    for( int i=1; i<argc; ++i )
    {
        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "-f" || arg == "--file"  )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            out_file = argv[++i];
        }
        else if( arg == "-n" || arg == "--nopbo"  )
        {
            use_pbo = false;
        }
        else if( arg == "-m" || arg == "--mesh" )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            mesh_file = argv[++i];
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif


		createPrePassContext();
		setupPrePassCamera();
		//loadPrePassGeometry();
		loadGeometry(prepass_context, "photonPrePass.cu");
		prepass_context->validate();
		//glutPrePassRun();
		updatePrePassCamera();

		
		int iteration = nPrePassIteration;
		while (iteration--)
		{
			clock_t start_time = clock();
			prepass_context[ "frame_number" ]->setUint( prepass_frame_number++ );
			prepass_context->launch(0, photonSamples, photonSamples);
			getPrePassPhotonBuffer(); 
			printf("Frame number:%d\n", prepass_frame_number);
			clock_t stop_time = clock();
			int time = (int)(stop_time - start_time);
			printf("%d\n", time);
		}

		createContext(contextFileName);
		setupCamera();
		loadGeometry(context, contextFileName);

		// photonSecondPass.cu should be created before this function call
		
		sortGridAndPassBuffer();

		prepass_context->destroy();


        context->validate();

        if ( out_file.empty() )
        {
            glutRun();
        }
        else
        {
            updateCamera();
            context->launch( 0, width, height );
            sutil::displayBufferPPM( out_file.c_str(), getOutputBuffer() );
            destroyContext();
        }
		
        return 0;
    }
    SUTIL_CATCH( context->get() )
}

