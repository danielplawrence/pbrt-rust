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