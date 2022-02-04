use num::{Num, Signed};
use std::ops::*;
use std::fmt::Debug;
pub mod vector;
pub mod point;
pub mod ray;
pub mod bounds;
pub mod transform;
pub mod quaternion;

/// Describes the shared behavior of scalar types in the geometry module.
/// This allows us to define generic Vector and Point types which can support
/// both integer and floating point types.
pub trait Scalar:
Debug +
Copy +
Num +
Copy + 
PartialOrd + 
Add<Output=Self> + 
Sub<Output=Self> + 
Mul<Output=Self> + 
Div<Output=Self> + 
Signed +
Neg<Output=Self>{
    fn sqrt(self) -> Self;
    fn inf() -> Self;
    fn min(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }
    fn max(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }
    fn two() -> Self {
        Self::one() + Self::one()
    }
    fn approximately_equal(self, other: Self) -> bool {
        (self - other).abs() < Self::epsilon()
    }
    fn epsilon() -> Self;
}
impl Scalar for f64{
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn inf() -> Self {
        f64::INFINITY
    }
    fn epsilon() -> Self {
        0.00001
    }
}
impl Scalar for i64{
    fn sqrt(self) -> Self {
        (self as f64).sqrt() as i64
    }
    fn inf() -> Self {
        i64::MAX
    }
    fn epsilon() -> Self {
        1
    }
}
impl Scalar for f32{
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn inf() -> Self {
        f32::INFINITY
    }
    fn epsilon() -> Self {
        0.00001
    }
}
impl Scalar for i32{
    fn sqrt(self) -> Self {
        ((self as f64).sqrt() as i64).try_into().unwrap()
    }
    fn inf() -> Self {
        i32::MAX
    }
    fn epsilon() -> Self {
        1
    }
}
#[test]
fn test_f64_sqrt(){
    assert_eq!(f64::sqrt(4.0), 2.0);
}
#[test]
fn test_i64_sqrt(){
    assert_eq!(i64::sqrt(4), 2);
}
#[test]
fn test_f32_sqrt(){
    assert_eq!(f32::sqrt(4.0), 2.0);
}
#[test]
fn test_i32_sqrt(){
    assert_eq!(i32::sqrt(4), 2);
}
#[test]
fn test_f64_inf(){
    assert_eq!(f64::inf(), f64::INFINITY);
}
#[test]
fn test_i64_inf(){
    assert_eq!(i64::inf(), i64::MAX);
}
#[test]
fn test_f32_inf(){
    assert_eq!(f32::inf(), f32::INFINITY);
}
#[test]
fn test_i32_inf(){
    assert_eq!(i32::inf(), i32::MAX);
}
#[test]
fn test_f64_min(){
    assert_eq!(f64::min(1.0, 2.0), 1.0);
}
#[test]
fn test_i64_min(){
    assert_eq!(Scalar::min(1, 2), 1);
}
#[test]
fn test_f32_min(){
    assert_eq!(f32::min(1.0, 2.0), 1.0);
}
#[test]
fn test_i32_min(){
    assert_eq!(Scalar::min(1, 2), 1);
}
#[test]
fn test_f64_max(){
    assert_eq!(f64::max(1.0, 2.0), 2.0);
}
#[test]
fn test_i64_max(){
    assert_eq!(Scalar::max(1, 2), 2);
}
#[test]
fn test_f32_max(){
    assert_eq!(f32::max(1.0, 2.0), 2.0);
}
#[test]
fn test_i32_max(){
    assert_eq!(Scalar::max(1, 2), 2);
}
#[test]
fn test_f64_approximately_equal(){
    assert!(f64::approximately_equal(1.0, 1.0));
    assert!(!f64::approximately_equal(1.0, 2.0));
}
#[test]
fn test_i64_approximately_equal(){
    assert!(Scalar::approximately_equal(1, 1));
    assert!(!Scalar::approximately_equal(1, 2));
}
#[test]
fn test_f32_approximately_equal(){
    assert!(f32::approximately_equal(1.0, 1.0));
    assert!(!f32::approximately_equal(1.0, 2.0));
}
#[test]
fn test_i32_approximately_equal(){
    assert!(Scalar::approximately_equal(1, 1));
    assert!(!Scalar::approximately_equal(1, 2));
}