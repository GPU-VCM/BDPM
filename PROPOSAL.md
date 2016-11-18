# CIS565 Final Project Proposal: VCM (Vertex Connection and Merging)
#### Members : Akshay Shah, Kaixiang Miao
### Overview

Developing robust light transport simulation algorithms that are capable of dealing with arbitrary input scenes remains an elusive challenge. Although efficient global illumination algorithms exist, an acceptable approximation error in a reasonable amount of time is usually only achieved for specific types of input scenes.

For example, bidirectional path tracing has a very difficult time when rendering specular–diffuse–specular (SDS) effects (such as reflected and refracted caustics) because they can only be sampled with pure light tracing (t = 0) or pure eye tracing (s = 0), while Photon Mapping has difficulties in handling the illumination coming from a mirror.

To address this problem, Light Transport Simulation with Vertex Connection and Merging is proposed. This new technique combines photon mapping and bidirectional path tracing, which have so far been considered conceptually incompatible solutions to the light transport problem, into a more robust algorithm via multiple importance sampling. It efficiently handles a wide variety of lighting conditions, ranging from direct illumination, diffuse and glossy inter-reflections, to specular-diffuse-specular light transport.

What we want to do is to implement this new technique with NVidia’s Optix API (which uses CUDA). We plan to separate our work into two parts at first - the Photon Mapping and the Bidirectional Path Tracing, then merge these two parts together and finally implement the Vertex Merging algorithm.

##### Goals

* Photon Mapping
* Bidirectional Path Tracing
* Multiple Importance Sampling
* Vertex Merging
