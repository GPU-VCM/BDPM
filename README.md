# Photon Mapping

#### What is Photon Mapping?

Photon mapping is a two-pass global illumination algorithm developed by Henrik Wann Jensen that approximately solves the rendering equation. Rays from the light source and rays from the camera are traced independently until some termination criterion is met, then they are connected in a second step to produce a radiance value.

Here's a [good introduction](https://web.cs.wpi.edu/~emmanuel/courses/cs563/write_ups/zackw/photon_mapping/PhotonMapping.html).

Screenshots
-----------

#### Results

<table class="image">
<tr>
	<td>Pre-pass</td>
	<td>Second-pass</td>
</tr>
<tr>
	<td><img src="VCM/img/p00.jpg"/></td>
	<td><img src="VCM/img/p01.jpg"/></td>
</tr>
<tr>
	<td><img src="VCM/img/p02.jpg"/></td>
	<td><img src="VCM/img/p03.jpg"/></td>
</tr>
<tr>
	<td><img src="VCM/img/p04.jpg"/></td>
	<td><img src="VCM/img/p05.jpg"/></td>
</tr>
</table>

#### Photon Visualization

The shadow looks not correct enough because we shoot photons only from one corner of our parallelogram light.

Photons are experiencing z-flipping because there're too many photons and if the camera isn't close enough, lots of them are overlapped.

<table class="image">
<tr>
	<td>Original image</td>
	<td>Photons</td>
</tr>
<tr>
	<td><img src="VCM/img/bbox0.jpg"/></td>
	<td><img src="VCM/img/2.gif"/></td>
</tr>
</table>

### Analysis

Time cost in the pre-pass:

![](VCM/img/chart0.png)

Time cost in the second-pass:(scene with only diffuse/specular material, which means)

![](VCM/img/chart1.png)

### Configure

* `cd ${PATH}/VCM`
* `mkdir build`
* `cd build`
* `cmake-gui ..`
* `configure` (with lots of warnings)
* `generate`
* Double click *VCM-PathTracer.sln*
* Set *optixPathTracer* as start-up project
* Run
