# Geometry and Transformations

## 1.3 Normals

A normal vector is a vector which is perpendicular to a surface at particular point. To calculate it, we take the cross product of any two nonparallel vectors which are tangent to the surface at a point.

For example, in the diagram below, C is a normal vector of the plane formed by A and B:

<p align="center">
    <img src="images/normal_vector.png" style="background-color: white" width="200">
</p>

Normals look a lot like vectors, but behave differently, particularly when it comes to transformations. My Rust implementation of pbrt will follow the original implementation in defining them as a separate type, `Normal3d`.