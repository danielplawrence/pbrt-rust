use num::{Num, Signed};
use std::ops::*;
use std::fmt::Debug;
pub mod vector;
pub mod point;

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
}
impl Scalar for f64{
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}
impl Scalar for i64{
    fn sqrt(self) -> Self {
        (self as f64).sqrt() as i64
    }
}
impl Scalar for f32{
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}
impl Scalar for i32{
    fn sqrt(self) -> Self {
        ((self as f64).sqrt() as i64).try_into().unwrap()
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