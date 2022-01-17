use std::ops::{Add, Sub, Mul, Index};
use num::Float;

use super::{Scalar, vector::{Vector2d, Vector3d}};

#[derive(Debug, Copy, Clone)]
pub struct Point2d<T> {
    pub x: T,
    pub y: T,
}
impl<T: Scalar> Point2d<T> {
    pub fn new(x: T, y: T) -> Self {
        Point2d{x, y}
    }
    pub fn from<U, V: From<U>>(p: Point2d<U>) -> Point2d<V> {
        Point2d{x: p.x.into(), y: p.y.into()}
    }
    pub fn from_3d(p: Point3d<T>) -> Self {
        Point2d{x: p.x, y: p.y}
    }
    pub fn to_vector<U: From<T> + Scalar>(&self) -> Vector2d<U> {
        Vector2d::new(self.x.into(), self.y.into())
    }
    pub fn distance(self, other: Self) -> T {
        return (self - other).length();
    }
    pub fn distance_squared(self, other: Self) -> T {
        return (self - other).squared_length();
    }
    pub fn lerp(self, other: Self, t: T) -> Self {
        return self + (other - self) * t;
    }
    pub fn max(self, other: Self) -> Self {
        let max_x = if self.x > other.x {self.x} else {other.x};
        let max_y = if self.y > other.y {self.y} else {other.y};
        return Point2d::new(max_x, max_y);
    }
    pub fn min(self, other: Self) -> Self {
        let min_x = if self.x < other.x {self.x} else {other.x};
        let min_y = if self.y < other.y {self.y} else {other.y};
        return Point2d::new(min_x, min_y);
    }
    pub fn abs(self) -> Self {
        Point2d::new(self.x.abs(), self.y.abs())
    }
    pub fn permute(self, x: usize, y: usize) -> Self {
        Point2d::new(self[x], self[y])
    }
}
impl<T: Scalar + Float> Point2d<T> {
    pub fn floor(self) -> Self {
        Point2d::new(self.x.floor(), self.y.floor())
    }
    pub fn ceil(self) -> Self {
        Point2d::new(self.x.ceil(), self.y.ceil())
    }
}
impl<T: Scalar> Add<Vector2d<T>> for Point2d<T> {
    type Output = Self;
    fn add(self, other: Vector2d<T>) -> Self {
        Point2d{x: self.x + other.x, y: self.y + other.y}
    }
}
impl<T: Scalar> Add for Point2d<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Point2d{x: self.x + other.x, y: self.y + other.y}
    }
}
impl<T: Scalar> Mul<T> for Point2d<T> {
    type Output = Self;
    fn mul(self, other: T) -> Self {
        Point2d{x:self.x * other, y:self.y * other}
    }
}
impl<T: Scalar> Sub<Vector2d<T>> for Point2d<T> {
    type Output = Self;
    fn sub(self, other: Vector2d<T>) -> Self {
        Point2d{x: self.x - other.x, y: self.y - other.y}
    }
}
impl<T: Scalar> Sub<Point2d<T>> for Point2d<T> {
    type Output = Vector2d<T>;
    fn sub(self, other: Self) -> Self::Output {
        Vector2d{x: self.x - other.x, y: self.y - other.y}
    }
}
impl<T: Scalar> Index<usize> for Point2d<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Index out of bounds")
        }
    }
}
#[derive(Debug, Copy, Clone)]
pub struct Point3d<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}
impl<T: Scalar> Point3d<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Point3d{x, y, z}
    }
    pub fn from<U, V: From<U>>(p: Point3d<U>) -> Point3d<V> {
        Point3d{x: p.x.into(), y: p.y.into(), z: p.z.into()}
    }
    pub fn to_vector<U: From<T> + Scalar>(&self) -> Vector3d<U> {
        Vector3d::new(self.x.into(), self.y.into(), self.z.into())
    }
    pub fn distance(self, other: Self) -> T {
        return (self - other).length();
    }
    pub fn distance_squared(self, other: Self) -> T {
        return (self - other).squared_length();
    }
    pub fn lerp(self, other: Self, t: T) -> Self {
        return self + (other - self) * t;
    }
    pub fn min(self, other: Self) -> Self {
        let min_x = if self.x < other.x {self.x} else {other.x};
        let min_y = if self.y < other.y {self.y} else {other.y};
        let min_z = if self.z < other.z {self.z} else {other.z};
        return Point3d::new(min_x, min_y, min_z);
    }
    pub fn max(self, other: Self) -> Self {
        let max_x = if self.x > other.x {self.x} else {other.x};
        let max_y = if self.y > other.y {self.y} else {other.y};
        let max_z = if self.z > other.z {self.z} else {other.z};
        return Point3d::new(max_x, max_y, max_z);
    }
    pub fn abs(self) -> Self {
        Point3d::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
    pub fn permute(self, x: usize, y: usize, z: usize) -> Self {
        Point3d::new(self[x], self[y], self[z])
    }
}
impl<T: Scalar + Float> Point3d<T> {
    pub fn floor(self) -> Self {
        Point3d::new(self.x.floor(), self.y.floor(), self.z.floor())
    }
    pub fn ceil(self) -> Self {
        Point3d::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }
}
impl<T: Scalar> Add<Vector3d<T>> for Point3d<T> {
    type Output = Self;
    fn add(self, other: Vector3d<T>) -> Self {
        Point3d{x: self.x + other.x, y: self.y + other.y, z: self.z + other.z}
    }
}
impl<T: Scalar> Mul<T> for Point3d<T> {
    type Output = Self;
    fn mul(self, other: T) -> Self {
        Point3d{x: self.x * other, y: self.y * other, z: self.z * other}
    }
}
impl<T: Scalar> Add for Point3d<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Point3d{x: self.x + other.x, y: self.y + other.y, z: self.z + other.z}
    }
}
impl<T: Scalar> Sub<Vector3d<T>> for Point3d<T> {
    type Output = Self;
    fn sub(self, other: Vector3d<T>) -> Self {
        Point3d{x: self.x - other.x, y: self.y - other.y, z: self.z - other.z}
    }
}
impl<T: Scalar> Sub<Point3d<T>> for Point3d<T> {
    type Output = Vector3d<T>;
    fn sub(self, other: Self) -> Self::Output {
        Vector3d{x: self.x - other.x, y: self.y - other.y, z: self.z - other.z}
    }
}
impl<T: Scalar> Index<usize> for Point3d<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds")
        }
    }
}

#[test]
fn test_point_2d_constructor() {
    let p = Point2d::new(0.0, 1.0);
    assert_eq!(p.x, 0.0);
    assert_eq!(p.y, 1.0);
}
#[test]
#[should_panic]
fn test_point_2d_index_panic() {
    let p = Point2d::new(0.0, 1.0);
    let _ = p[3];
}
#[test]
fn test_point_2d_index() {
    let p = Point2d::new(0.0, 1.0);
    assert_eq!(p[0], 0.0);
    assert_eq!(p[1], 1.0);
}
#[test]
fn test_point_2d_from_point_3d() {
    let p = Point3d::new(0.0, 1.0, 2.0);
    let q: Point2d<f64> = Point2d::from_3d(p);
    assert_eq!(q.x, 0.0);
    assert_eq!(q.y, 1.0);
}
#[test]
fn test_point_2d_from_point_2d() {
    let p = Point2d::new(0, 1);
    let q: Point2d<f64> = Point2d::<f64>::from(p);
    assert_eq!(q.x, 0.0);
    assert_eq!(q.y, 1.0);
}
#[test]
fn test_point_2d_to_vector() {
    let p = Point2d::new(0.0, 1.0);
    let v: Vector2d<f64> = p.to_vector();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 1.0);
}
#[test]
fn test_point_2d_add_vector() {
    let p = Point2d::new(0.0, 1.0);
    let q = Vector2d::new(1.0, 2.0);
    let r = p + q;
    assert_eq!(r.x, 1.0);
    assert_eq!(r.y, 3.0);
}
#[test]
fn test_point_2d_sub_vector() {
    let p = Point2d::new(1.0, 1.0);
    let q = Vector2d::new(1.0, 1.0);
    let r = p - q;
    assert_eq!(r.x, 0.0);
    assert_eq!(r.y, 0.0);
}
#[test]
fn test_point_2d_sub_point() {
    let p = Point2d::new(1.0, 1.0);
    let q = Point2d::new(0.0, 0.0);
    let r = p - q;
    assert_eq!(r.x, 1.0);
    assert_eq!(r.y, 1.0);
}
#[test]
fn test_point_2d_distance() {
    let p = Point2d::new(-1.0, 1.0);
    let q = Point2d::new(1.0, 1.0);
    let r = p.distance(q);
    assert_eq!(r, 2.0);
}
#[test]
fn test_point_2d_distance_squared() {
    let p = Point2d::new(-1.0, 1.0);
    let q = Point2d::new(1.0, 1.0);
    let r = p.distance_squared(q);
    assert_eq!(r, 4.0);
}
#[test]
fn test_point_2d_add_point() {
    let p = Point2d::new(1.0, 1.0);
    let q = Point2d::new(1.0, 1.0);
    let r = p + q;
    assert_eq!(r.x, 2.0);
    assert_eq!(r.y, 2.0);
}
#[test]
fn test_point_2d_mul_scalar() {
    let p = Point2d::new(1.0, 1.0);
    let q = p * 2.0;
    assert_eq!(q.x, 2.0);
    assert_eq!(q.y, 2.0);
}
#[test]
fn test_point_2d_lerp_p1() {
    let p = Point2d::new(0.0, 1.0);
    let q = Point2d::new(1.0, 2.0);
    let r = p.lerp(q, 0.0);
    assert_eq!(r.x, 0.0);
    assert_eq!(r.y, 1.0);
}
#[test]
fn test_point_2d_lerp_p2() {
    let p = Point2d::new(0.0, 1.0);
    let q = Point2d::new(1.0, 2.0);
    let r = p.lerp(q, 1.0);
    assert_eq!(r.x, 1.0);
    assert_eq!(r.y, 2.0);
}
#[test]
fn test_point_2d_lerp_mid() {
    let p = Point2d::new(1.0, 1.0);
    let q = Point2d::new(2.0, 2.0);
    let r = p.lerp(q, 0.5);
    assert_eq!(r.x, 1.5);
    assert_eq!(r.y, 1.5);
}
#[test]
fn test_point_2d_min() {
    let p = Point2d::new(1.0, 1.0);
    let q = Point2d::new(2.0, 2.0);
    let r = p.min(q);
    assert_eq!(r.x, 1.0);
    assert_eq!(r.y, 1.0);
}
#[test]
fn test_point_2d_max() {
    let p = Point2d::new(1.0, 1.0);
    let q = Point2d::new(2.0, 2.0);
    let r = p.max(q);
    assert_eq!(r.x, 2.0);
    assert_eq!(r.y, 2.0);
}
#[test]
fn test_point_2d_abs() {
    let p = Point2d::new(-1.0, -1.0);
    let q = p.abs();
    assert_eq!(q.x, 1.0);
    assert_eq!(q.y, 1.0);
}
#[test]
fn test_point_2d_floor() {
    let p = Point2d::new(1.5, 1.5);
    let q = p.floor();
    assert_eq!(q.x, 1.0);
    assert_eq!(q.y, 1.0);
}
#[test]
fn test_point_2d_ceil() {
    let p = Point2d::new(1.5, 1.5);
    let q = p.ceil();
    assert_eq!(q.x, 2.0);
    assert_eq!(q.y, 2.0);
}
#[test]
fn test_point_3d_constructor() {
    let p = Point3d::new(0.0, 1.0, 2.0);
    assert_eq!(p.x, 0.0);
    assert_eq!(p.y, 1.0);
    assert_eq!(p.z, 2.0);
}
#[test]
#[should_panic]
fn test_point_3d_index_panic() {
    let p = Point3d::new(0.0, 1.0, 2.0);
    let _ = p[3];
}
#[test]
fn test_point_3d_index() {
    let p = Point3d::new(0.0, 1.0, 2.0);
    assert_eq!(p[0], 0.0);
    assert_eq!(p[1], 1.0);
    assert_eq!(p[2], 2.0);
}
#[test]
fn test_point_3d_from_point_3d() {
    let p = Point3d::new(0, 1, 2);
    let q: Point3d<f64> = Point3d::<f64>::from(p);
    assert_eq!(q.x, 0.0);
    assert_eq!(q.y, 1.0);
    assert_eq!(q.z, 2.0);
}
#[test]
fn test_point_3d_to_vector() {
    let p = Point3d::new(0.0, 1.0, 2.0);
    let q = p.to_vector::<f64>();
    assert_eq!(q.x, 0.0);
    assert_eq!(q.y, 1.0);
    assert_eq!(q.z, 2.0);
}
#[test]
fn test_point_3d_add_vector() {
    let p = Point3d::new(0.0, 1.0, 2.0);
    let q = Vector3d::new(1.0, 2.0, 3.0);
    let r = p + q;
    assert_eq!(r.x, 1.0);
    assert_eq!(r.y, 3.0);
    assert_eq!(r.z, 5.0);
}
#[test]
fn test_point_3d_sub_vector() {
    let p = Point3d::new(0.0, 1.0, 2.0);
    let q = Vector3d::new(1.0, 2.0, 3.0);
    let r = p - q;
    assert_eq!(r.x, -1.0);
    assert_eq!(r.y, -1.0);
    assert_eq!(r.z, -1.0);
}
#[test]
fn test_point_3d_sub_point() {
    let p = Point3d::new(0.0, 1.0, 2.0);
    let q = Point3d::new(1.0, 2.0, 3.0);
    let r = p - q;
    assert_eq!(r.x, -1.0);
    assert_eq!(r.y, -1.0);
    assert_eq!(r.z, -1.0);
}
#[test]
fn test_point_3d_distance() {
    let p = Point3d::new(2.0, 0.0, 0.0);
    let q = Point3d::new(0.0, 0.0, 0.0);
    let r = p.distance(q);
    assert_eq!(r, 2.0);
}
#[test]
fn test_point_3d_distance_squared() {
    let p = Point3d::new(2.0, 0.0, 0.0);
    let q = Point3d::new(0.0, 0.0, 0.0);
    let r = p.distance_squared(q);
    assert_eq!(r, 4.0);
}
#[test]
fn test_point_3d_add_point() {
    let p = Point3d::new(1.0, 1.0, 1.0);
    let q = Point3d::new(1.0, 1.0, 1.0);
    let r = p + q;
    assert_eq!(r.x, 2.0);
    assert_eq!(r.y, 2.0);
    assert_eq!(r.z, 2.0);
}
#[test]
fn test_point_3d_mul_scalar() {
    let p = Point3d::new(1.0, 1.0, 1.0);
    let q = p * 2.0;
    assert_eq!(q.x, 2.0);
    assert_eq!(q.y, 2.0);
    assert_eq!(q.z, 2.0);
}
#[test]
fn test_point_3d_lerp_p1() {
    let p = Point3d::new(1.0, 1.0, 1.0);
    let q = Point3d::new(3.0, 3.0, 3.0);
    let r = p.lerp(q, 0.0);
    assert_eq!(r.x, 1.0);
    assert_eq!(r.y, 1.0);
    assert_eq!(r.z, 1.0);
}
#[test]
fn test_point_3d_lerp_mid() {
    let p = Point3d::new(1.0, 1.0, 1.0);
    let q = Point3d::new(3.0, 3.0, 3.0);
    let r = p.lerp(q, 0.5);
    assert_eq!(r.x, 2.0);
    assert_eq!(r.y, 2.0);
    assert_eq!(r.z, 2.0);
}
#[test]
fn test_point_3d_lerp_p2() {
    let p = Point3d::new(1.0, 1.0, 1.0);
    let q = Point3d::new(3.0, 3.0, 3.0);
    let r = p.lerp(q, 1.0);
    assert_eq!(r.x, 3.0);
    assert_eq!(r.y, 3.0);
    assert_eq!(r.z, 3.0);
}
#[test]
fn test_point_3d_max() {
    let p = Point3d::new(1.0, 2.0, 3.0);
    let q = Point3d::new(3.0, 2.0, 1.0);
    let r = p.max(q);
    assert_eq!(r.x, 3.0);
    assert_eq!(r.y, 2.0);
    assert_eq!(r.z, 3.0);
}
#[test]
fn test_point_3d_min() {
    let p = Point3d::new(1.0, 2.0, 3.0);
    let q = Point3d::new(3.0, 2.0, 1.0);
    let r = p.min(q);
    assert_eq!(r.x, 1.0);
    assert_eq!(r.y, 2.0);
    assert_eq!(r.z, 1.0);
}
#[test]
fn test_point_3d_abs() {
    let p = Point3d::new(-1.0, -2.0, -3.0);
    let q = p.abs();
    assert_eq!(q.x, 1.0);
    assert_eq!(q.y, 2.0);
    assert_eq!(q.z, 3.0);
}
#[test]
fn test_point_3d_floor() {
    let p = Point3d::new(1.2, 2.3, 3.4);
    let q = p.floor();
    assert_eq!(q.x, 1.0);
    assert_eq!(q.y, 2.0);
    assert_eq!(q.z, 3.0);
}
#[test]
fn test_point_3d_ceil() {
    let p = Point3d::new(1.2, 2.3, 3.4);
    let q = p.ceil();
    assert_eq!(q.x, 2.0);
    assert_eq!(q.y, 3.0);
    assert_eq!(q.z, 4.0);
}