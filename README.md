# Bidirectional Path Tracer with OptiX

Screenshots
-----------

![](VCM/img/bdpt_test_scene.PNG)

![](VCM/img/cornell_knot_bdpt.PNG)

![](VCM/img/stfd_drgn_bdpt.PNG)

![](VCM/img/bdpt_glass.PNG)

Debug Screenshots
-----------------

The more the pixel is white, the more time it took/spent in that space.

![](VCM/img/debug_heatmap.PNG)

With more iterations, it is clear that the refractive objects have more time spent with them. 

![](VCM/img/debug_heatmap2.PNG)

###### Comparison with Path tracing

| BDPT | MCPT |
| ---- | ---- |
| ![](VCM/img/bdpt_red.PNG) | ![](VCM/img/mcpt_red.PNG) |
