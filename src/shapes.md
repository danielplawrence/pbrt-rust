# Shapes

This section describes the implementation of geometric primitives such as spheres and triangles. All geometric primitives implement a common interface, such that the rest of the renderer can use this interface without needing any details about the underlying shape. This makes it possible to separate the geometric and shading subsystems.

`pbrt-rust` follows the original c++ implementation in using a two-level abstraction to represent shapes. The `Shape` trait provides access to the raw geometric properties of the primitive, such as its surface area and bounding box; it also provides a ray intersection routine. The `Primitive` trait encapsuraltes additional non-geometric information about the primitive, such as material properties. The rest of the render then deals only with the `Primitive` methods.

This chapter will focus on the geometry-only `Shape` interface, with the `Primitive` interface addressed in the following chapter.
