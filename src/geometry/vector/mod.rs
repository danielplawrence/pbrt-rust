use std::ops::{Index, Add, Sub, Mul, Div, Neg};

use num::zero;

use super::Scalar;

pub trait Dot<T, U: Scalar> {
    fn dot(&self, other: &T) -> U;
    fn abs_dot(&self, other: &T) -> U {
        self.dot(other).abs()
    }
}

/// Flip a surface normal so that it lies in the same hemisphere as a given vector.
/// This is useful for computing the reflection direction.
/// This trait provides default behavior for any combination of vector types.
pub trait FaceForward<T: Dot<T, U> + Clone + Neg<Output = T>, U: Scalar>: Dot<T, U> {
    fn face_forward(&self, other: T) -> T {
        if self.dot(&other) < zero() {
            other.clone()
        } else {
            -other.clone()
        }
    }
}
/// Implements the FaceForward trait for any combination of vector types.
impl<U: Scalar, T: Dot<T, U> + Clone + Neg<Output = T>, V: Dot<T, U>> FaceForward<T, U> for V{}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vector2d<T> {
    pub x: T,
    pub y: T
}
impl<T: Scalar> Vector2d<T> {
    pub fn new(x: T, y: T) -> Self {
        Vector2d { x, y }
    }
    pub fn squared_length(&self) -> T {
        self.x * self.x + self.y * self.y
    }
    pub fn length(&self) -> T {
        self.squared_length().sqrt()
    }
    pub fn normalized(&self) -> Self {
        Vector2d {
            x: (self.x / self.length()),
            y: (self.y / self.length())
        }
    }
    pub fn min_component(&self) -> T {
        if self.x < self.y {
            self.x
        } else {
            self.y
        }
    }
    pub fn max_component(&self) -> T {
        if self.x > self.y {
            self.x
        } else {
            self.y
        }
    }
    pub fn min(&self, other: &Self) -> Self {
       let x = if self.x < other.x { self.x } else { other.x };
       let y = if self.y < other.y { self.y } else { other.y };
        Vector2d { x, y }
    }
    pub fn max(&self, other: &Self) -> Self {
        let x = if self.x > other.x { self.x } else { other.x };
        let y = if self.y > other.y { self.y } else { other.y };
         Vector2d { x, y }
    }
    pub fn permute(&self, x: usize, y: usize) -> Self {
        Vector2d {
            x: self[x],
            y: self[y]
        }
    }
    pub fn max_dimension(&self) -> usize {
        if self.x > self.y {
            0
        } else {
            1
        }
    }
}
impl<T: Scalar> Index<usize> for Vector2d<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Index out of bounds")
        }
    }
}
impl<T: Scalar> Add for Vector2d<T> {
    type Output = Vector2d<T>;
    fn add(self, other: Self) -> Self::Output {
        Vector2d::new(self.x + other.x, self.y + other.y)
    }
}
impl<T: Scalar> Sub for Vector2d<T> {
    type Output = Vector2d<T>;
    fn sub(self, other: Self) -> Self::Output {
        Vector2d::new(self.x - other.x, self.y - other.y)
    }
}
impl<T: Scalar> Mul<T> for Vector2d<T> {
    type Output = Vector2d<T>;
    fn mul(self, other: T) -> Self::Output {
        Vector2d::new(self.x * other, self.x * other)
    }
}
impl<T: Scalar> Div<T> for Vector2d<T> {
    type Output = Vector2d<T>;
    fn div(self, other: T) -> Self::Output {
        let recip = T::one() / other;
        Vector2d::new(self.x * recip, self.y * recip)
    }
}
impl<T: Scalar> Neg for Vector2d<T> {
    type Output = Vector2d<T>;
    fn neg(self) -> Self::Output {
        Vector2d::new(-self.x, -self.y)
    }
}
impl<U: Scalar> Dot<Vector2d<U>, U> for Vector2d<U> {
    fn dot(&self, other: &Vector2d<U>) -> U {
        self.x * other.x + self.y * other.y
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vector3d<T> {
   pub x: T,
   pub y: T,
   pub z: T
}
impl<T: Scalar> Vector3d<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Vector3d { x, y, z }
    }
    pub fn cross(&self, other: &Self) -> Self {
        Vector3d::new(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)
    }
    pub fn squared_length(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    pub fn length(&self) -> T {
        (self.squared_length()).sqrt()
    }
    pub fn normalized(&self) -> Self {
        Vector3d {
            x: (self.x / self.length()),
            y: (self.y / self.length()),
            z: (self.z / self.length())
        }
    }
    pub fn min_component(&self) -> T {
        if self.x < self.y {
            if self.x < self.z {
                self.x
            } else {
                self.z
            }
        } else {
            if self.y < self.z {
                self.y
            } else {
                self.z
            }
        }
    }
    pub fn max_component(&self) -> T {
        if self.x > self.y {
            if self.x > self.z {
                self.x
            } else {
                self.z
            }
        } else {
            if self.y > self.z {
                self.y
            } else {
                self.z
            }
        }
    }
    pub fn min(&self, other: &Self) -> Self {
         let x = if self.x < other.x { self.x } else { other.x };
         let y = if self.x < other.y { self.y } else { other.y };
         let z = if self.x < other.z { self.z } else { other.z };
          Vector3d { x, y, z }
    }
    pub fn max(&self, other: &Self) -> Self {
        let x = if self.x > other.x { self.x } else { other.x };
        let y = if self.x > other.y { self.y } else { other.y };
        let z = if self.x > other.z { self.z } else { other.z };
         Vector3d { x, y, z }
    }
    pub fn permute(&self, x: usize, y: usize, z: usize) -> Self {
        Vector3d {
            x: self[x],
            y: self[y],
            z: self[z]
        }
    }
    pub fn max_dimension(&self) -> usize {
        if self.x > self.y {
            if self.x > self.z {
                0
            } else {
                2
            }
        } else {
            if self.y > self.z {
                1
            } else {
                2
            }
        }
    }
    pub fn coordinate_system(&self) -> (Self, Self) {
        let y = if (self.x).abs() > (self.y).abs() {
            Vector3d::new(-self.z, zero(), self.x) / 
            (self.x * self.x + self.z * self.z).sqrt()
        } else {
            Vector3d::new(zero(), self.z, -self.y) / 
            (self.y * self.y + self.z * self.z).sqrt()
        };
        (self.cross(&y), y)
    }
}
impl<T: Scalar> Index<usize> for Vector3d<T> {
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
impl <T: Scalar> Add for Vector3d<T> {
    type Output = Vector3d<T>;
    fn add(self, other: Self) -> Self::Output {
        Vector3d::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}
impl <T: Scalar> Sub for Vector3d<T> {
    type Output = Vector3d<T>;
    fn sub(self, other: Self) -> Self::Output {
        Vector3d::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}
impl <T: Scalar> Mul<T> for Vector3d<T> {
    type Output = Vector3d<T>;
    fn mul(self, other: T) -> Self::Output {
        Vector3d::new(self.x * other, self.y * other, self.z * other)
    }
}
impl <T: Scalar> Div<T> for Vector3d<T> {
    type Output = Vector3d<T>;
    fn div(self, other: T) -> Self::Output {
        let recip = T::one() / other;
        Vector3d::new(self.x * recip, self.y * recip, self.z * recip)
    }
}
impl <T: Scalar> Neg for Vector3d<T> {
    type Output = Vector3d<T>;
    fn neg(self) -> Self::Output {
        Vector3d::new(-self.x, -self.y, -self.z)
    }
}
impl<U: Scalar> Dot<Vector3d<U>, U> for Vector3d<U> {
    fn dot(&self, other: &Vector3d<U>) -> U {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}
impl<U: Scalar> Dot<Normal3d<U>, U> for Vector3d<U> {
    fn dot(&self, other: &Normal3d<U>) -> U {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Normal3d<T> {
    pub x: T,
    pub y: T,
    pub z: T
}
impl<T: Scalar> Normal3d<T>{
    pub fn new(x: T, y: T, z: T) -> Self {
        Normal3d { x, y, z }
    }
    pub fn squared_length(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    pub fn length(&self) -> T {
        (self.squared_length()).sqrt()
    }
    pub fn normalized(&self) -> Self {
        Normal3d {
            x: (self.x / self.length()),
            y: (self.y / self.length()),
            z: (self.z / self.length())
        }
    }
}
impl<T: Scalar> Index<usize> for Normal3d<T> {
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
impl <T: Scalar> Add for Normal3d<T> {
    type Output = Normal3d<T>;
    fn add(self, other: Self) -> Self::Output {
        Normal3d::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}
impl <T: Scalar> Sub for Normal3d<T> {
    type Output = Normal3d<T>;
    fn sub(self, other: Self) -> Self::Output {
        Normal3d::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}
impl <T: Scalar> Mul<T> for Normal3d<T> {
    type Output = Normal3d<T>;
    fn mul(self, other: T) -> Self::Output {
        Normal3d::new(self.x * other, self.y * other, self.z * other)
    }
}
impl <T: Scalar> Div<T> for Normal3d<T> {
    type Output = Normal3d<T>;
    fn div(self, other: T) -> Self::Output {
        let recip = T::one() / other;
        Normal3d::new(self.x * recip, self.y * recip, self.z * recip)
    }
}
impl <T: Scalar> Neg for Normal3d<T> {
    type Output = Normal3d<T>;
    fn neg(self) -> Self::Output {
        Normal3d::new(-self.x, -self.y, -self.z)
    }
}
impl <T: Scalar> From<Vector3d<T>> for Normal3d<T> {
    fn from(v: Vector3d<T>) -> Self {
        Normal3d::new(v.x, v.y, v.z)
    }
}
impl<U: Scalar> Dot<Vector3d<U>, U> for Normal3d<U> {
    fn dot(&self, other: &Vector3d<U>) -> U {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}
impl<U: Scalar> Dot<Normal3d<U>, U> for Normal3d<U> {
    fn dot(&self, other: &Normal3d<U>) -> U {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

#[test]
fn test_vector_2d() {
    let v = Vector2d { x: 1.0, y: 2.0 };
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
}
#[test]
fn test_vector_2d_new() {
    let v = Vector2d::new(1.0, 2.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
}
#[test]
fn test_vector_2d_new_integer() {
    let v = Vector2d::new(1, 2);
    assert_eq!(v.x, 1);
    assert_eq!(v.y, 2);
}
#[test]
fn test_vector_2d_index() {
    let v = Vector2d { x: 1.0, y: 2.0 };
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
}
#[test]
#[should_panic]
fn test_vector_2d_index_panic() {
    let v = Vector2d { x: 1.0, y: 2.0 };
    assert_eq!(v[2], 2.0);
}
#[test]
fn test_vector_2d_add() {
    let v1 = Vector2d::new(1.0, 2.0);
    let v2 = Vector2d::new(4.0, 5.0);
    let v3 = v1 + v2;
    assert_eq!(v3.x, 5.0);
    assert_eq!(v3.y, 7.0);
}
#[test]
fn test_vector_2d_sub() {
    let v1 = Vector2d::new(1.0, 1.0);
    let v2 = Vector2d::new(1.0, 1.0);
    let v3 = v2 - v1;
    assert_eq!(v3.x, 0.0);
    assert_eq!(v3.y, 0.0);
}
#[test]
fn test_vector_2d_neg() {
    let v1 = Vector2d::new(1.0, 1.0);
    let v2 = -v1;
    assert_eq!(v2.x, -1.0);
    assert_eq!(v2.y, -1.0);
}
#[test]
fn test_vector_2d_mul() {
    let v1 = Vector2d::new(1.0, 1.0);
    let v2 = v1 * 2.0;
    assert_eq!(v2.x, 2.0);
    assert_eq!(v2.y, 2.0);
}
#[test]
fn test_vector_2d_div() {
    let v1 = Vector2d::new(1.0, 1.0);
    let v2 = v1 / 2.0;
    assert_eq!(v2.x, 0.5);
    assert_eq!(v2.y, 0.5);
}
#[test]
fn test_vector_2d_dot() {
    let v1 = Vector2d::new(1.0, 1.0);
    let v2 = Vector2d::new(1.0, 1.0);
    assert_eq!(v1.dot(&v2), 2.0);
}
#[test]
fn test_vector_2d_abs_dot() {
    let v1 = Vector2d::new(1.0, 1.0);
    let v2 = Vector2d::new(-1.0, -1.0);
    assert_eq!(v1.abs_dot(&v2), 2.0);
}
#[test]
fn test_vector_2d_squared_length() {
    let v = Vector2d::new(1.0, 1.0);
    assert_eq!(v.squared_length(), 2.0);
}
#[test]
fn test_vector_2d_length() {
    let v = Vector2d::new(1.0, 1.0);
    assert_eq!(v.length(), 2.0.sqrt());
}
#[test]
fn test_vector_2d_normalized() {
    let v = Vector2d::new(1.0, 1.0);
    let vn = v.normalized();
    assert_eq!(vn.x, 1.0 / 2.0.sqrt());
    assert_eq!(vn.y, 1.0 / 2.0.sqrt());
}
#[test]
fn test_vector_2d_min_component() {
    let v = Vector2d::new(1.0, 2.0);
    assert_eq!(v.min_component(), 1.0);
    let v2 = Vector2d::new(2.0, 1.0);
    assert_eq!(v2.min_component(), 1.0);
}
#[test]
fn test_vector_2d_max_component() {
    let v = Vector2d::new(1.0, 2.0);
    assert_eq!(v.max_component(), 2.0);
    let v2 = Vector2d::new(2.0, 1.0);
    assert_eq!(v2.max_component(), 2.0);
}
#[test]
fn test_vector_2d_min() {
    let v1 = Vector2d::new(5.0, 2.0);
    let v2 = Vector2d::new(3.0, 4.0);
    let v3 = Vector2d::new(3.0, 2.0);
    assert_eq!(v1.min(&v2), v3);
}
#[test]
fn test_vector_2d_max() {
    let v1 = Vector2d::new(5.0, 2.0);
    let v2 = Vector2d::new(3.0, 4.0);
    let v3 = Vector2d::new(5.0, 4.0);
    assert_eq!(v1.max(&v2), v3);
}
#[test]
fn test_vector_2d_permute() {
    let v = Vector2d::new(1.0, 2.0);
    let vp = v.permute(1, 0);
    assert_eq!(vp.x, 2.0);
    assert_eq!(vp.y, 1.0);
}
#[test]
fn test_vector_2d_max_dimension() {
    let v = Vector2d::new(1.0, 2.0);
    assert_eq!(v.max_dimension(), 1);
    let v2 = Vector2d::new(2.0, 1.0);
    assert_eq!(v2.max_dimension(), 0);
}
#[test]
fn test_vector_3d() {
    let v = Vector3d { x: 1.0, y: 2.0, z: 3.0 };
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
    assert_eq!(v.z, 3.0);
}
#[test]
fn test_vector_3d_new_integer() {
    let v = Vector3d::new(1, 2, 3);
    assert_eq!(v.x, 1);
    assert_eq!(v.y, 2);
    assert_eq!(v.z, 3);
}
#[test]
fn test_vector_3d_new() {
    let v = Vector3d::new(1.0, 2.0, 3.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
    assert_eq!(v.z, 3.0);
}
#[test]
fn test_vector_3d_index() {
    let v = Vector3d { x: 1.0, y: 2.0, z: 3.0 };
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);
}
#[test]
#[should_panic]
fn test_vector_3d_index_panic() {
    let v = Vector3d { x: 1.0, y: 2.0, z: 3.0 };
    assert_eq!(v[3], 3.0);
}
#[test]
fn test_vector_3d_add() {
    let v1 = Vector3d::new(1.0, 2.0, 3.0);
    let v2 = Vector3d::new(4.0, 5.0, 6.0);
    let v3 = v1 + v2;
    assert_eq!(v3.x, 5.0);
    assert_eq!(v3.y, 7.0);
    assert_eq!(v3.z, 9.0);
}
#[test]
fn test_vector_3d_sub() {
    let v1 = Vector3d::new(1.0, 1.0, 1.0);
    let v2 = Vector3d::new(1.0, 1.0, 1.0);
    let v3 = v2 - v1;
    assert_eq!(v3.x, 0.0);
    assert_eq!(v3.y, 0.0);
    assert_eq!(v3.z, 0.0);
}
#[test]
fn test_vector_3d_neg() {
    let v1 = Vector3d::new(1.0, 1.0, 1.0);
    let v2 = -v1;
    assert_eq!(v2.x, -1.0);
    assert_eq!(v2.y, -1.0);
    assert_eq!(v2.z, -1.0);
}
#[test]
fn test_vector_3d_mul() {
    let v1 = Vector3d::new(1.0, 1.0, 1.0);
    let v2 = v1 * 2.0;
    assert_eq!(v2.x, 2.0);
    assert_eq!(v2.y, 2.0);
    assert_eq!(v2.z, 2.0);
}
#[test]
fn test_vector_3d_div() {
    let v1 = Vector3d::new(1.0, 1.0, 1.0);
    let v2 = v1 / 2.0;
    assert_eq!(v2.x, 0.5);
    assert_eq!(v2.y, 0.5);
    assert_eq!(v2.z, 0.5);
}
#[test]
fn test_vector_3d_dot() {
    let v1 = Vector3d::new(1.0, 1.0, 1.0);
    let v2 = Vector3d::new(1.0, 1.0, 1.0);
    assert_eq!(v1.dot(&v2), 3.0);
}
#[test]
fn test_vector_3d_abs_dot() {
    let v1 = Vector3d::new(1.0, 1.0, 1.0);
    let v2 = Vector3d::new(-1.0, -1.0, -1.0);
    assert_eq!(v1.abs_dot(&v2), 3.0);
}
#[test]
fn test_vector_3d_cross() {
    let v1 = Vector3d::new(1.0, 0.0, 0.0);
    let v2 = Vector3d::new(0.0, 1.0, 0.0);
    let v3 = Vector3d::new(0.0, 0.0, 1.0);
    assert_eq!(v1.cross(&v2), v3);
    assert_eq!(v3.cross(&v1), v2);
    assert_eq!(v2.cross(&v3), v1);
}
#[test]
fn test_vector_3d_cross_negative() {
    let v1 = Vector3d::new(1.0, 0.0, 0.0);
    let v2 = Vector3d::new(0.0, 1.0, 0.0);
    let v3 = Vector3d::new(0.0, 0.0, 1.0);
    assert_eq!(v2.cross(&v1), -v3);
    assert_eq!(v1.cross(&v3), -v2);
    assert_eq!(v3.cross(&v2), -v1);
}
#[test]
fn test_vector_3d_squared_length() {
    let v = Vector3d::new(1.0, 1.0, 1.0);
    assert_eq!(v.squared_length(), 3.0);
}
#[test]
fn test_vector_3d_length() {
    let v = Vector3d::new(1.0, 1.0, 1.0);
    assert_eq!(v.length(), 3.0.sqrt());
}
#[test]
fn test_vector_3d_normalized() {
    let v = Vector3d::new(1.0, 1.0, 1.0);
    let vn = v.normalized();
    assert_eq!(vn.x, 1.0 / 3.0.sqrt());
    assert_eq!(vn.y, 1.0 / 3.0.sqrt());
    assert_eq!(vn.z, 1.0 / 3.0.sqrt());
}
#[test]
fn test_vector_3d_min_component() {
    let v = Vector3d::new(1.0, 2.0, 3.0);
    assert_eq!(v.min_component(), 1.0);
    let v2 = Vector3d::new(2.0, 1.0, 3.0);
    assert_eq!(v2.min_component(), 1.0);
    let v3 = Vector3d::new(2.0, 3.0, 1.0);
    assert_eq!(v3.min_component(), 1.0);
}
#[test]
fn test_vector_3d_max_component() {
    let v = Vector3d::new(1.0, 2.0, 3.0);
    assert_eq!(v.max_component(), 3.0);
    let v2 = Vector3d::new(2.0, 3.0, 1.0);
    assert_eq!(v2.max_component(), 3.0);
    let v3 = Vector3d::new(3.0, 2.0, 1.0);
    assert_eq!(v3.max_component(), 3.0);

}
#[test]
fn test_vector_3d_min() {
    let v1 = Vector3d::new(1.0, 2.0, 3.0);
    let v2 = Vector3d::new(4.0, 5.0, 6.0);
    let v3 = Vector3d::new(1.0, 2.0, 3.0);
    assert_eq!(v1.min(&v2), v3);
}
#[test]
fn test_vector_3d_max() {
    let v1 = Vector3d::new(1.0, 2.0, 3.0);
    let v2 = Vector3d::new(4.0, 5.0, 6.0);
    let v3 = Vector3d::new(4.0, 5.0, 6.0);
    assert_eq!(v1.max(&v2), v3);
}
#[test]
fn test_vector_3d_permute() {
    let v = Vector3d::new(1.0, 2.0, 3.0);
    let vp = v.permute(1, 2, 0);
    assert_eq!(vp.x, 2.0);
    assert_eq!(vp.y, 3.0);
    assert_eq!(vp.z, 1.0);
}
#[test]
fn test_vector_3d_max_dimension() {
    let v = Vector3d::new(1.0, 2.0, 3.0);
    assert_eq!(v.max_dimension(), 2);
    let v2 = Vector3d::new(1.0, 3.0, 1.0);
    assert_eq!(v2.max_dimension(), 1);
    let v3 = Vector3d::new(3.0, 1.0, 1.0);
    assert_eq!(v3.max_dimension(), 0);
}
#[test]
fn test_vector_3d_coordinate_system() {
    let v = Vector3d::new(1.0, 0.0, 0.0).normalized();
    let (v1, v2) = v.coordinate_system();
    assert_eq!(v1, Vector3d::new(0.0, -1.0, 0.0));
    assert_eq!(v2, Vector3d::new(0.0, 0.0, 1.0));
    assert_eq!(v1.dot(&v2), 0.0);
    assert_eq!(v2.dot(&v1), 0.0);
    assert_eq!(v.dot(&v1), 0.0);
    assert_eq!(v.dot(&v2), 0.0);
    let v2 = Vector3d::new(0.0, 1.0, 0.0).normalized();
    let (v1, v2) = v2.coordinate_system();
    assert_eq!(v1, Vector3d::new(-1.0, 0.0, 0.0));
    assert_eq!(v2, Vector3d::new(0.0, 0.0, -1.0));
}
#[test]
fn test_vector_3d_face_forward_vector() {
    let v = Vector3d::new(1.0, 1.0, 1.0);
    let v2 = Vector3d::new(1.0, 1.0, 1.0);
    let vf = v.face_forward(v2);
    assert_eq!(vf.x, -1.0);
    assert_eq!(vf.y, -1.0);
    assert_eq!(vf.z, -1.0);
}
#[test]
fn test_vector_3d_face_forward_normal() {
    let v = Vector3d::new(1.0, 1.0, 1.0);
    let v2 = Normal3d::new(1.0, 1.0, 1.0);
    let vf = v.face_forward(v2);
    assert_eq!(vf.x, -1.0);
    assert_eq!(vf.y, -1.0);
    assert_eq!(vf.z, -1.0);
}
#[test]
fn test_normal_3d_new() {
    let v = Normal3d::new(1.0, 2.0, 3.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
    assert_eq!(v.z, 3.0);
}
#[test]
fn test_normal_3d_index() {
    let v = Normal3d { x: 1.0, y: 2.0, z: 3.0 };
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);
}
#[test]
#[should_panic]
fn test_normal_3d_index_panic() {
    let v = Normal3d { x: 1.0, y: 2.0, z: 3.0 };
    assert_eq!(v[3], 3.0);
}
#[test]
fn test_normal_3d_add() {
    let v1 = Normal3d::new(1.0, 2.0, 3.0);
    let v2 = Normal3d::new(4.0, 5.0, 6.0);
    let v3 = v1 + v2;
    assert_eq!(v3.x, 5.0);
    assert_eq!(v3.y, 7.0);
    assert_eq!(v3.z, 9.0);
}
#[test]
fn test_normal_3d_sub() {
    let v1 = Normal3d::new(1.0, 1.0, 1.0);
    let v2 = Normal3d::new(1.0, 1.0, 1.0);
    let v3 = v2 - v1;
    assert_eq!(v3.x, 0.0);
    assert_eq!(v3.y, 0.0);
    assert_eq!(v3.z, 0.0);
}
#[test]
fn test_normal_3d_neg() {
    let v1 = Normal3d::new(1.0, 1.0, 1.0);
    let v2 = -v1;
    assert_eq!(v2.x, -1.0);
    assert_eq!(v2.y, -1.0);
    assert_eq!(v2.z, -1.0);
}
#[test]
fn test_normal_3d_mul() {
    let v1 = Normal3d::new(1.0, 1.0, 1.0);
    let v2 = v1 * 2.0;
    assert_eq!(v2.x, 2.0);
    assert_eq!(v2.y, 2.0);
    assert_eq!(v2.z, 2.0);
}
#[test]
fn test_normal_3d_div() {
    let v1 = Normal3d::new(1.0, 1.0, 1.0);
    let v2 = v1 / 2.0;
    assert_eq!(v2.x, 0.5);
    assert_eq!(v2.y, 0.5);
    assert_eq!(v2.z, 0.5);
}
#[test]
fn test_normal_3d_squared_length() {
    let v = Normal3d::new(1.0, 1.0, 1.0);
    assert_eq!(v.squared_length(), 3.0);
}
#[test]
fn test_normal_3d_length() {
    let v = Normal3d::new(1.0, 1.0, 1.0);
    assert_eq!(v.length(), 3.0.sqrt());
}
#[test]
fn test_normal_3d_normalized() {
    let v = Normal3d::new(1.0, 1.0, 1.0);
    let vn = v.normalized();
    assert_eq!(vn.x, 1.0 / 3.0.sqrt());
    assert_eq!(vn.y, 1.0 / 3.0.sqrt());
    assert_eq!(vn.z, 1.0 / 3.0.sqrt());
}
#[test]
fn test_normal_3d_dot() {
    let v1 = Normal3d::new(1.0, 1.0, 1.0);
    let v2 = Normal3d::new(1.0, 1.0, 1.0);
    assert_eq!(v1.dot(&v2), 3.0);
}
#[test]
fn test_normal_3d_abs_dot() {
    let v1 = Normal3d::new(1.0, 1.0, 1.0);
    let v2 = Normal3d::new(-1.0, -1.0, -1.0);
    assert_eq!(v1.abs_dot(&v2), 3.0);
}
#[test]
fn test_normal_3d_dot_vector_3d() {
    let v1 = Normal3d::new(1.0, 1.0, 1.0);
    let v2 = Vector3d::new(1.0, 1.0, 1.0);
    assert_eq!(v1.dot(&v2), 3.0);
}
#[test]
fn test_vector_3d_dot_normal_3d() {
    let v1 = Vector3d::new(1.0, 1.0, 1.0);
    let v2 = Normal3d::new(1.0, 1.0, 1.0);
    assert_eq!(v1.dot(&v2), 3.0);
}
#[test]
fn test_normal_3d_faceforward_vector() {
    let n = Normal3d::new(1.0, 1.0, 1.0);
    let v = Vector3d::new(1.0, 1.0, 1.0);
    let nf = n.face_forward(v);
    assert_eq!(nf.x, -1.0);
    assert_eq!(nf.y, -1.0);
    assert_eq!(nf.z, -1.0);
}
#[test]
fn test_normal_3d_faceforward_normal() {
    let n = Normal3d::new(1.0, 1.0, 1.0);
    let v = Normal3d::new(1.0, 1.0, 1.0);
    let nf = n.face_forward(v);
    assert_eq!(nf.x, -1.0);
    assert_eq!(nf.y, -1.0);
    assert_eq!(nf.z, -1.0);
}