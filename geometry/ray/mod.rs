use super::{point::Point3d, vector::Vector3d, Scalar};
use std::cell::Cell;

trait RayMethods<T> {
    fn origin(&self) -> Point3d<T>;
    fn direction(&self) -> Vector3d<T>;
    fn at(&self, t: T) -> Point3d<T>;
    fn t_max(&self) -> &Cell<T>;
    fn time(&self) -> T;
}
#[derive(Debug, Clone, PartialEq)]
pub struct Ray<T: Scalar> {
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
}
impl<T: Scalar> RayMethods<T> for Ray<T> {
    fn origin(&self) -> Point3d<T> {
        self.origin
    }
    fn direction(&self) -> Vector3d<T> {
        self.direction
    }
    fn at(&self, t: T) -> Point3d<T> {
        self.origin + self.direction * t
    }
    fn t_max(&self) -> &Cell<T> {
        &self.t_max
    }
    fn time(&self) -> T {
        self.time
    }
}
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
impl<T: Scalar> RayMethods<T> for RayDifferential<T> {
    fn origin(&self) -> Point3d<T> {
        self.origin
    }
    fn direction(&self) -> Vector3d<T> {
        self.direction
    }
    fn at(&self, t: T) -> Point3d<T> {
        self.origin + self.direction * t
    }
    fn t_max(&self) -> &Cell<T> {
        &self.t_max
    }
    fn time(&self) -> T {
        self.time
    }
}
#[test]
fn test_ray_new() {
    let ray = Ray::<f64>::new(Point3d::<f64>::new(1.0, 2.0, 3.0), 
    Vector3d::<f64>::new(4.0, 5.0, 6.0), 7.0, 8.0);
    assert_eq!(ray.origin, Point3d::<f64>::new(1.0, 2.0, 3.0));
    assert_eq!(ray.direction, Vector3d::<f64>::new(4.0, 5.0, 6.0));
    assert_eq!(ray.t_max.get(), 7.0);
    assert_eq!(ray.time, 8.0);
}
#[test]
fn test_ray_default() {
    let ray = Ray::<f64>::default();
    assert_eq!(ray.origin, Point3d::<f64>::new(0.0, 0.0, 0.0));
    assert_eq!(ray.direction, Vector3d::<f64>::new(0.0, 0.0, 0.0));
    assert_eq!(ray.t_max.get(), f64::inf());
    assert_eq!(ray.time, 0.0);
}
#[test]
fn test_ray_at() {
    let origin = Point3d::new(1.0, 2.0, 3.0);
    let direction = Vector3d::new(4.0, 5.0, 6.0);
    let tmax = 7.0;
    let time = 8.0;
    let ray = Ray::new(origin, direction, tmax, time);
    assert_eq!(ray.at(0.0), origin);
    assert_eq!(ray.at(1.0), Point3d::new(5.0, 7.0, 9.0));
    assert_eq!(ray.at(2.0), Point3d::new(9.0, 12.0, 15.0));
}
#[test]
fn test_ray_trait_methods_ray() {
    let origin = Point3d::new(1.0, 2.0, 3.0);
    let direction = Vector3d::new(4.0, 5.0, 6.0);
    let tmax = 7.0;
    let time = 8.0;
    let ray = Ray::new(origin, direction, tmax, time);
    assert_eq!(ray.origin(), origin);
    assert_eq!(ray.direction(), direction);
    assert_eq!(ray.t_max().get(), tmax);
    assert_eq!(ray.time(), time);
}
#[test]
fn test_ray_differential_new() {
    let ray = RayDifferential::<f64>::new(Point3d::<f64>::new(1.0, 2.0, 3.0), 
    Vector3d::<f64>::new(4.0, 5.0, 6.0), 7.0, 8.0);
    assert_eq!(ray.origin, Point3d::<f64>::new(1.0, 2.0, 3.0));
    assert_eq!(ray.direction, Vector3d::<f64>::new(4.0, 5.0, 6.0));
    assert_eq!(ray.t_max.get(), 7.0);
    assert_eq!(ray.time, 8.0);
}
#[test]
fn test_ray_differential_default() {
    let ray = RayDifferential::<f64>::default();
    assert_eq!(ray.origin, Point3d::<f64>::new(0.0, 0.0, 0.0));
    assert_eq!(ray.direction, Vector3d::<f64>::new(0.0, 0.0, 0.0));
    assert_eq!(ray.t_max.get(), f64::inf());
    assert_eq!(ray.time, 0.0);
}
#[test]
fn test_ray_differential_from_ray() {
    let ray = Ray::<f64>::new(Point3d::<f64>::new(1.0, 2.0, 3.0), 
    Vector3d::<f64>::new(4.0, 5.0, 6.0), 7.0, 8.0);
    let ray_differential = RayDifferential::from_ray(&ray);
    assert_eq!(ray_differential.origin, Point3d::<f64>::new(1.0, 2.0, 3.0));
    assert_eq!(ray_differential.direction, Vector3d::<f64>::new(4.0, 5.0, 6.0));
    assert_eq!(ray_differential.t_max.get(), 7.0);
    assert_eq!(ray_differential.time, 8.0);
    assert_eq!(ray_differential.rx_origin, Point3d::<f64>::new(0.0, 0.0, 0.0));
    assert_eq!(ray_differential.ry_origin, Point3d::<f64>::new(0.0, 0.0, 0.0));
    assert_eq!(ray_differential.rx_direction, Vector3d::<f64>::new(0.0, 0.0, 0.0));
    assert_eq!(ray_differential.ry_direction, Vector3d::<f64>::new(0.0, 0.0, 0.0));
    assert_eq!(ray_differential.has_differentials, false);
}
#[test]
fn test_ray_differential_at() {
    let origin = Point3d::new(1.0, 2.0, 3.0);
    let direction = Vector3d::new(4.0, 5.0, 6.0);
    let tmax = 7.0;
    let time = 8.0;
    let ray = RayDifferential::new(origin, direction, tmax, time);
    assert_eq!(ray.at(0.0), origin);
    assert_eq!(ray.at(1.0), Point3d::new(5.0, 7.0, 9.0));
    assert_eq!(ray.at(2.0), Point3d::new(9.0, 12.0, 15.0));
}
#[test]
fn test_ray_differential_trait_methods_ray() {
    let origin = Point3d::new(1.0, 2.0, 3.0);
    let direction = Vector3d::new(4.0, 5.0, 6.0);
    let tmax = 7.0;
    let time = 8.0;
    let ray = RayDifferential::new(origin, direction, tmax, time);
    assert_eq!(ray.origin(), origin);
    assert_eq!(ray.direction(), direction);
    assert_eq!(ray.t_max().get(), tmax);
    assert_eq!(ray.time(), time);
}
#[test]
fn test_ray_differential_scale_differentials() {
    let origin = Point3d::new(1.0, 2.0, 3.0);
    let direction = Vector3d::new(4.0, 5.0, 6.0);
    let mut ray = RayDifferential::default();
    ray.rx_origin = origin;
    ray.ry_origin = origin;
    ray.rx_direction = direction;
    ray.ry_direction = direction;
    ray.scale_differentials(2.0);
    assert_eq!(ray.rx_origin, Point3d::new(2.0, 4.0, 6.0));
    assert_eq!(ray.ry_origin, Point3d::new(2.0, 4.0, 6.0));
    assert_eq!(ray.rx_direction, Vector3d::new(8.0, 10.0, 12.0));
    assert_eq!(ray.ry_direction, Vector3d::new(8.0, 10.0, 12.0));
}