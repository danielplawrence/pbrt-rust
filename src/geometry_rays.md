# Geometry and Transformations

## 1.4 Rays

A ray is a semi-infinite line specified by a point \\(o\\) representing its origin and a vector \\(d\\) representing its direction.

<p align="center">
    <img src="images/ray_point_vector.svg" style="background-color: white" width="200">
</p>

The parametric form of a ray expresses it as a function of a scalar value \\(t\\), giving the set of points that the ray passes through: 

\\[
    r(t) = o + td  \quad   0 \le t < \infty
\\]

The `Ray` class implements an `at(t)` method, allowing a caller to retrieve a point along the ray:

```rust

pub struct Ray<T> {
    pub origin: Point3d<T>,
    pub direction: Vector3d<T>,
    pub t_max: Cell<T>,
    pub time: T
}
impl<T: Scalar> Ray<T> { 
    pub fn new(origin: Point3d<T>, direction: Vector3d<T>, tmax: T, time: T) -> Self {
        Ray {
            origin,
            direction,
            t_max: Cell::new(tmax),
            time
        }
    }
    pub fn default() -> Self {
        Ray {
            origin: Point3d::<T>::new(T::zero(), T::zero(), T::zero()),
            direction: Vector3d::<T>::new(T::zero(), T::zero(), T::zero()),
            t_max: Cell::new(T::inf()),
            time: T::zero()
        }
    }
    pub fn at(&self, t: T) -> Point3d<T> {
        self.origin + self.direction * t
    }
}
```

As with the c++ implementation, the `Ray` struct provides an all-argument constructor as well as a `default()` which sets the origin, direction and time values to 0, and t_max to infinity. The Ray also includes a member variable that limits the ray to a segment along its infinite extent. This field, tMax, allows us to restrict the ray to a segment of points. Following the c++ pbrt implementation, this fields needs to be mutable to allow a caller to modify it (this will be useful when recording the points where the ray intersects with an object). In order to make a single field mutable in rust, we need to use the `Cell<T>` struct. This will allow a caller to set and get the value using the `set` and `get` methods on `Cell<T>`.

# 1.4.2 Ray Differentials

The c++ pbrt implementation includes a subclass of `Ray` with additional information about two auxiliary rays. These extra rays represent camera rays offset by one sample in the \\(x\\) and \\(y\\) direction from the main ray on the film plane. By moving the `Ray` methods out to a trait and implementing them for both `Ray` and `RayDifferential`, we can enable geometry methods which act on both `Rays` and `RayDifferentials` without caring which underlying type they are dealing with. Here's what the `RayMethods` trait looks like:

```rust
trait RayMethods<T> {
    fn origin(&self) -> Point3d<T>;
    fn direction(&self) -> Vector3d<T>;
    fn at(&self, t: T) -> Point3d<T>;
    fn t_max(&self) -> &Cell<T>;
    fn time(&self) -> T;
}
```
We can then add impl blocks returning the appropriate fields for both `Ray` and `RayDifferential`. `RayDifferential` has a few extra fields and methods:

```rust
pub struct RayDifferential<T> {
    pub origin: Point3d<T>,
    pub direction: Vector3d<T>,
    pub t_max: Cell<T>,
    pub time: T,
    pub rx_origin: Point3d<T>,
    pub ry_origin: Point3d<T>,
    pub rx_direction: Vector3d<T>,
    pub ry_direction: Vector3d<T>,
    pub has_differentials: bool
}
impl<T: Scalar> RayDifferential<T> {
    pub fn new(origin: Point3d<T>, direction: Vector3d<T>, tmax: T, time: T) -> Self {
        RayDifferential {
            origin: origin,
            direction: direction,
            rx_origin: Point3d::<T>::new(T::zero(), T::zero(), T::zero()),
            rx_direction: Vector3d::<T>::new(T::zero(), T::zero(), T::zero()),
            ry_origin: Point3d::<T>::new(T::zero(), T::zero(), T::zero()),
            ry_direction: Vector3d::<T>::new(T::zero(), T::zero(), T::zero()),
            t_max: Cell::new(tmax),
            time: time,
            has_differentials: false
        }
    }
    pub fn default() -> Self {
        RayDifferential {
            origin: Point3d::<T>::new(T::zero(), T::zero(), T::zero()),
            direction: Vector3d::<T>::new(T::zero(), T::zero(), T::zero()),
            rx_origin: Point3d::<T>::new(T::zero(), T::zero(), T::zero()),
            rx_direction: Vector3d::<T>::new(T::zero(), T::zero(), T::zero()),
            ry_origin: Point3d::<T>::new(T::zero(), T::zero(), T::zero()),
            ry_direction: Vector3d::<T>::new(T::zero(), T::zero(), T::zero()),
            t_max: Cell::new(T::inf()),
            time: T::zero(),
            has_differentials: false
        }
    }
    pub fn from_ray(ray: &Ray<T>) -> Self {
        RayDifferential::new(ray.origin, ray.direction, ray.t_max.get(), ray.time)
    }
    pub fn scale_differentials(&mut self, s: T) {
        self.rx_origin = self.origin + (self.rx_origin - self.origin) * s;
        self.ry_origin = self.origin + (self.ry_origin - self.origin) * s;
        self.rx_direction = self.direction + (self.rx_direction - self.direction) * s;
        self.ry_direction = self.direction + (self.ry_direction - self.direction) * s;
    }
}
```
The `RayDifferential` the `rx_` fields describe origin and direction information about these rays, and the 'ScaleDifferentials' method updates the differential rays for an estimated sample spacing of s. Finally, a `from_ray_` constructor allows us to create a `RayDifferential` from a `Ray`.