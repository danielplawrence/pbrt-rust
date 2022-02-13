
use std::{ops::{Index, Mul, Add}, f64::consts::PI, cell::Cell};

use num::Float;

use crate::geometry::bounds::Bounds3d;

use super::{Scalar, vector::{Vector3d, Normal3d}, point::Point3d, ray::{Ray, RayDifferential}, quaternion::Quaternion};

#[derive(Debug, Clone, Copy)]
pub struct Matrix4x4<T> {
    data: [[T;4]; 4]
}
impl<T: Scalar> Matrix4x4<T> {
    pub fn new(data: [[T;4]; 4]) -> Self {
        return Matrix4x4 {
            data
        }
    }
    pub fn default() -> Self {
        let data = [
            [T::one(), T::zero(), T::zero(), T::zero()],
            [T::zero(), T::one(), T::zero(), T::zero()],
            [T::zero(), T::zero(), T::one(), T::zero()],
            [T::zero(), T::zero(), T::zero(), T::one()]
        ];
        return Matrix4x4{
            data
        }
    }
    pub fn from_values(t00: T, t01: T, t02: T, t03: T,
        t10: T, t11: T, t12: T, t13: T,
        t20: T, t21: T, t22: T, t23: T,
        t30: T, t31: T, t32: T, t33: T) -> Self {
            Matrix4x4{
                data: [
                    [t00, t01, t02, t03],
                    [t10, t11, t12, t13],
                    [t20, t21, t22, t23],
                    [t30, t31, t32, t33]
                ]
            }
        }
    pub fn transpose(&self) -> Self {
        return Matrix4x4::from_values(
            self.data[0][0], self.data[1][0], self.data[2][0], self.data[3][0],
            self.data[0][1], self.data[1][1], self.data[2][1], self.data[3][1],
            self.data[0][2], self.data[1][2], self.data[2][2], self.data[3][2],
            self.data[0][3], self.data[1][3], self.data[2][3], self.data[3][3]);
    }
    pub fn determinant(&self) -> T {
        self[0][0] * self[1][1] * self[2][2] * self[3][3]
        + self[0][0] * self[1][2] * self[2][3] * self[3][1]
        + self[0][0] * self[1][3] * self[2][1] * self[3][2]
    
        + self[0][1] * self[1][0] * self[2][3] * self[3][2]
        + self[0][1] * self[1][2] * self[2][0] * self[3][3]
        + self[0][1] * self[1][3] * self[2][2] * self[3][0]
    
        + self[0][2] * self[1][0] * self[2][1] * self[3][3]
        + self[0][2] * self[1][1] * self[2][3] * self[3][0]
        + self[0][2] * self[1][3] * self[2][0] * self[3][1]
    
        + self[0][3] * self[1][0] * self[2][2] * self[3][1]
        + self[0][3] * self[1][1] * self[2][0] * self[3][2]
        + self[0][3] * self[1][2] * self[2][1] * self[3][0]
    
        - self[0][0] * self[1][1] * self[2][3] * self[3][2]
        - self[0][0] * self[1][2] * self[2][1] * self[3][3]
        - self[0][0] * self[1][3] * self[2][2] * self[3][1]
    
        - self[0][1] * self[1][0] * self[2][2] * self[3][3]
        - self[0][1] * self[1][2] * self[2][3] * self[3][0]
        - self[0][1] * self[1][3] * self[2][0] * self[3][2]
    
        - self[0][2] * self[1][0] * self[2][3] * self[3][1]
        - self[0][2] * self[1][1] * self[2][0] * self[3][3]
        - self[0][2] * self[1][3] * self[2][1] * self[3][0]
    
        - self[0][3] * self[1][0] * self[2][1] * self[3][2]
        - self[0][3] * self[1][1] * self[2][2] * self[3][0]
        - self[0][3] * self[1][2] * self[2][0] * self[3][1]
    }
    pub fn inverse(&self) -> Self {
        let inv_det = T::one() / self.determinant();
        let data = [
            [   (
                    self[1][1] * self[2][2] * self[3][3]
                    + self[1][2] * self[2][3] * self[3][1]
                    + self[1][3] * self[2][1] * self[3][2]
                    - self[1][1] * self[2][3] * self[3][2]
                    - self[1][2] * self[2][1] * self[3][3]
                    - self[1][3] * self[2][2] * self[3][1]
                ) * inv_det,
                (
                    self[0][1] * self[2][3] * self[3][2]
                    + self[0][2] * self[2][1] * self[3][3]
                    + self[0][3] * self[2][2] * self[3][1]
                    - self[0][1] * self[2][2] * self[3][3]
                    - self[0][2] * self[2][3] * self[3][1]
                    - self[0][3] * self[2][1] * self[3][2]
                ) * inv_det,
                (
                    self[0][1] * self[1][2] * self[3][3]
                    + self[0][2] * self[1][3] * self[3][1]
                    + self[0][3] * self[1][1] * self[3][2]
                    - self[0][1] * self[1][3] * self[3][2]
                    - self[0][2] * self[1][1] * self[3][3]
                    - self[0][3] * self[1][2] * self[3][1]
                ) * inv_det,
                (
                    self[0][1] * self[1][3] * self[2][2]
                    + self[0][2] * self[1][1] * self[2][3]
                    + self[0][3] * self[1][2] * self[2][1]
                    - self[0][1] * self[1][2] * self[2][3]
                    - self[0][2] * self[1][3] * self[2][1]
                    - self[0][3] * self[1][1] * self[2][2]
                ) * inv_det
            ],
            [
                (
                    self[1][0] * self[2][3] * self[3][2]
                    + self[1][2] * self[2][0] * self[3][3]
                    + self[1][3] * self[2][2] * self[3][0]
                    - self[1][0] * self[2][2] * self[3][3]
                    - self[1][2] * self[2][3] * self[3][0]
                    - self[1][3] * self[2][0] * self[3][2]
                ) * inv_det,
                (
                    self[0][0] * self[2][2] * self[3][3]
                    + self[0][2] * self[2][3] * self[3][0]
                    + self[0][3] * self[2][0] * self[3][2]
                    - self[0][0] * self[2][3] * self[3][2]
                    - self[0][2] * self[2][0] * self[3][3]
                    - self[0][3] * self[2][2] * self[3][0]
                ) * inv_det,
                (
                    self[0][0] * self[1][3] * self[3][2]
                    + self[0][2] * self[1][0] * self[3][3]
                    + self[0][3] * self[1][2] * self[3][0]
                    - self[0][0] * self[1][2] * self[3][3]
                    - self[0][2] * self[1][3] * self[3][0]
                    - self[0][3] * self[1][0] * self[3][2]
                ) * inv_det,
                (
                    self[0][0] * self[1][2] * self[2][3]
                    + self[0][2] * self[1][3] * self[2][0]
                    + self[0][3] * self[1][0] * self[2][2]
                    - self[0][0] * self[1][3] * self[2][2]
                    - self[0][2] * self[1][0] * self[2][3]
                    - self[0][3] * self[1][2] * self[2][0]
                ) * inv_det
            ],
            [
                (
                    self[1][0] * self[2][1] * self[3][3]
                    + self[1][1] * self[2][3] * self[3][0]
                    + self[1][3] * self[2][0] * self[3][1]
                    - self[1][0] * self[2][3] * self[3][1]
                    - self[1][1] * self[2][0] * self[3][3]
                    - self[1][3] * self[2][1] * self[3][0]
                ) * inv_det,
                (
                    self[0][0] * self[2][3] * self[3][1]
                    + self[0][1] * self[2][0] * self[3][3]
                    + self[0][3] * self[2][1] * self[3][0]
                    - self[0][0] * self[2][1] * self[3][3]
                    - self[0][1] * self[2][3] * self[3][0]
                    - self[0][3] * self[2][0] * self[3][1]
                ) * inv_det,
                (
                    self[0][0] * self[1][1] * self[3][3]
                    + self[0][1] * self[1][3] * self[3][0]
                    + self[0][3] * self[1][0] * self[3][1]
                    - self[0][0] * self[1][3] * self[3][1]
                    - self[0][1] * self[1][0] * self[3][3]
                    - self[0][3] * self[1][1] * self[3][0]
                ) * inv_det,
                (
                    self[0][0] * self[1][3] * self[2][1]
                    + self[0][1] * self[1][0] * self[2][3]
                    + self[0][3] * self[1][1] * self[2][0]
                    - self[0][0] * self[1][1] * self[2][3]
                    - self[0][1] * self[1][3] * self[2][0]
                    - self[0][3] * self[1][0] * self[2][1]
                ) * inv_det
            ],
            [
                (
                    self[1][0] * self[2][2] * self[3][1]
                    + self[1][1] * self[2][0] * self[3][2]
                    + self[1][2] * self[2][1] * self[3][0]
                    - self[1][0] * self[2][1] * self[3][2]
                    - self[1][1] * self[2][2] * self[3][0]
                    - self[1][2] * self[2][0] * self[3][1]
                ) * inv_det,
                (
                    self[0][0] * self[2][1] * self[3][2]
                    + self[0][1] * self[2][2] * self[3][0]
                    + self[0][2] * self[2][0] * self[3][1]
                    - self[0][0] * self[2][2] * self[3][1]
                    - self[0][1] * self[2][0] * self[3][2]
                    - self[0][2] * self[2][1] * self[3][0]
                ) * inv_det,
                (
                    self[0][0] * self[1][2] * self[3][1]
                    + self[0][1] * self[1][0] * self[3][2]
                    + self[0][2] * self[1][1] * self[3][0]
                    - self[0][0] * self[1][1] * self[3][2]
                    - self[0][1] * self[1][2] * self[3][0]
                    - self[0][2] * self[1][0] * self[3][1]
                ) * inv_det,
                (
                    self[0][0] * self[1][1] * self[2][2]
                    + self[0][1] * self[1][2] * self[2][0]
                    + self[0][2] * self[1][0] * self[2][1]
                    - self[0][0] * self[1][2] * self[2][1]
                    - self[0][1] * self[1][0] * self[2][2]
                    - self[0][2] * self[1][1] * self[2][0]
                ) * inv_det
            ]
        ];
        return Matrix4x4::new(data)
    }
}
impl<T: Scalar> Index<usize> for Matrix4x4<T> {
    type Output = [T;4];
    fn index(&self, x: usize) -> &[T; 4] {
        return &self.data[x];
    }
}
impl<T: Scalar> Mul for Matrix4x4<T> {
    type Output = Matrix4x4<T>;
    fn mul(self, other: Matrix4x4<T>) -> Matrix4x4<T> {
        let mut data: [[T;4];4] = [[T::zero();4];4];
        for i in 0..4{
            for j in 0..4{
                data[i][j] = self[i][0] * other[0][j] + 
                        self[i][1] * other[1][j] + 
                        self[i][2] * other[2][j] + 
                        self[i][3] * other[3][j];
            }
        }
        return Matrix4x4{
            data
        }
    }
}
impl<T: Scalar> Mul<T> for Matrix4x4<T> {
    type Output = Matrix4x4<T>;
    fn mul(self, scalar: T) -> Matrix4x4<T> {
        let mut data: [[T;4];4] = [[T::zero();4];4];
        for i in 0..4{
            for j in 0..4{
                data[i][j] = self.data[i][j] * scalar;
            }
        }
        return Matrix4x4{
            data
        }
    }
}
impl<T: Scalar> Mul<Matrix4x4<T>> for Vector3d<T> {
    type Output = Vector3d<T>;
    fn mul(self, other: Matrix4x4<T>) -> Vector3d<T> {
        return other * self;
    }
}
impl<T: Scalar> Mul<Matrix4x4<T>> for Point3d<T> {
    type Output = Point3d<T>;
    fn mul(self, other: Matrix4x4<T>) -> Point3d<T> {
        return other * self;
    }
}
impl<T: Scalar> Mul<Matrix4x4<T>> for Normal3d<T> {
    type Output = Normal3d<T>;
    fn mul(self, other: Matrix4x4<T>) -> Normal3d<T> {
        return other * self;
    }
}
impl<T: Scalar> Mul<Vector3d<T>> for Matrix4x4<T> {
    type Output = Vector3d<T>;
    fn mul(self, other: Vector3d<T>) -> Vector3d<T> {
        let x = other.x;
        let y = other.y;
        let z = other.z;
        return Vector3d{
            x: x * self[0][0] + y * self[1][0] + z * self[2][0],
            y: x * self[0][1] + y * self[1][1] + z * self[2][1],
            z: x * self[0][2] + y * self[1][2] + z * self[2][2],
        }
    }
}
impl<T: Scalar> Mul<Point3d<T>> for Matrix4x4<T> {
    type Output = Point3d<T>;
    fn mul(self, other: Point3d<T>) -> Point3d<T> {
        let x = other.x;
        let y = other.y;
        let z = other.z;
        let xp = self[0][0] * x + self[0][1] * y + self[0][2] * z + self[0][3];
        let yp = self[1][0] * x + self[1][1] * y + self[1][2] * z + self[1][3];
        let zp = self[2][0] * x + self[2][1] * y + self[2][2] * z + self[2][3];
        let wp = self[3][0] * x + self[3][1] * y + self[3][2] * z + self[3][3];
        if wp == T::one(){
            return Point3d{
                x: xp,
                y: yp,
                z: zp
            }
        } else{
            return Point3d{
                x: xp / wp,
                y: yp / wp,
                z: zp / wp
            }
        }
    }
}
impl<T: Scalar> Mul<Normal3d<T>> for Matrix4x4<T> {
    type Output = Normal3d<T>;
    fn mul(self, other: Normal3d<T>) -> Normal3d<T> {
        let x = other.x;
        let y = other.y;
        let z = other.z;
        return Normal3d{
            x: x * self[0][0] + y * self[1][0] + z * self[2][0],
            y: x * self[0][1] + y * self[1][1] + z * self[2][1],
            z: x * self[0][2] + y * self[1][2] + z * self[2][2],
        }
    }
}
impl<T: Scalar> Add<Matrix4x4<T>> for Matrix4x4<T> {
    type Output = Matrix4x4<T>;
    fn add(self, other: Matrix4x4<T>) -> Matrix4x4<T> {
        let mut data: [[T;4];4] = [[T::zero();4];4];
        for i in 0..4{
            for j in 0..4{
                data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        return Matrix4x4{
            data
        }
    }
}
impl<T: Scalar> PartialEq for Matrix4x4<T> {
    fn eq(&self, other: &Matrix4x4<T>) -> bool {
        for i in 0..4{
            for j in 0..4{
                if self.data[i][j].approximately_equal(other.data[i][j]) == false {
                    return false;
                }
            }
        }
        return true;
    }
}
#[derive(Debug, PartialEq, Clone)]
pub struct Transform<T: Scalar> {
    m: Matrix4x4<T>,
    m_inv: Matrix4x4<T>
}
impl<T: Scalar + Float> Transform<T> {
    pub fn default() -> Self {
        return Transform { 
            m: Matrix4x4::from_values(
                T::one(), T::zero(), T::zero(), T::zero(),
                T::zero(), T::one(), T::zero(), T::zero(),
                T::zero(), T::zero(), T::one(), T::zero(),
                T::zero(), T::zero(), T::zero(), T::one()
            ), 
            m_inv: Matrix4x4::from_values(
                T::one(), T::zero(), T::zero(), T::zero(),
                T::zero(), T::one(), T::zero(), T::zero(),
                T::zero(), T::zero(), T::one(), T::zero(),
                T::zero(), T::zero(), T::zero(), T::one()
            )}
    }
    pub fn new(m: Matrix4x4<T>, m_inv: Matrix4x4<T>) -> Self {
        return Transform { m, m_inv }
    }
    pub fn matrix(&self) -> &Matrix4x4<T> {
        return &self.m;
    }
    pub fn inverse(&self) -> Self {
        return Transform { m: self.m_inv, m_inv: self.m };
    }
    pub fn transpose(&self) -> Self {
        return Transform { m:self.m.transpose(), m_inv: self.m_inv.transpose() }
    }
    pub fn is_identity(&self) -> bool {
        let identity = Matrix4x4::from_values(
            T::one(), T::zero(), T::zero(), T::zero(),
            T::zero(), T::one(), T::zero(), T::zero(),
            T::zero(), T::zero(), T::one(), T::zero(),
            T::zero(), T::zero(), T::zero(), T::one()
        );
        return self.m.eq(&identity);
    }
    pub fn swaps_handedness(&self) -> bool {
        let det = 
        self.m[0][0] * (self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1]) -
        self.m[0][1] * (self.m[1][0] * self.m[2][2] - self.m[1][2] * self.m[2][0]) +
        self.m[0][2] * (self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0]);
        return det < T::zero();
    }
    pub fn translate(delta: Vector3d<T>) -> Transform<T> {
        let m = Matrix4x4::from_values(
            T::one(), T::zero(), T::zero(), delta.x,
             T::zero(), T::one(), T::zero(), delta.y,
              T::zero(), T::zero(), T::one(), delta.z,
               T::zero(), T::zero(), T::zero(),T::one());
        let m_inv = Matrix4x4::from_values(
            T::one(), T::zero(), T::zero(), -delta.x,
            T::zero(), T::one(), T::zero(), -delta.y,
             T::zero(), T::zero(), T::one(), -delta.z,
              T::zero(),T::zero(), T::zero(), T::one());
        return Transform{m, m_inv};
    }
    pub fn scale(x:T, y:T, z:T) -> Transform<T> {
        let m = Matrix4x4::from_values(            
        x, T::zero(), T::zero(), T::zero(),
        T::zero(), y, T::zero(), T::zero(),
        T::zero(), T::zero(), z, T::zero(),
        T::zero(), T::zero(), T::zero(), T::one());
        let m_inv = Matrix4x4::from_values(            
            T::one()/x, T::zero(), T::zero(), T::zero(),
            T::zero(), T::one()/y, T::zero(), T::zero(),
            T::zero(), T::zero(), T::one()/z, T::zero(),
            T::zero(), T::zero(), T::zero(), T::one());
        return Transform{m, m_inv};
    }
    pub fn has_scale(&self) -> bool {
        let la2 = (self * Vector3d::new(T::one(), T::zero(), T::zero())).squared_length();
        let lb2 = (self * Vector3d::new(T::zero(), T::one(), T::zero())).squared_length();
        let lc2 = (self * Vector3d::new(T::zero(), T::zero(), T::one())).squared_length();
        return la2 != T::one() || lb2 != T::one() || lc2 != T::one();
    }
    pub fn rotate_z(angle: T) -> Self {
        let (sin, cos) = angle.to_radians().sin_cos();
        let m = Matrix4x4::from_values(
            cos, -sin, T::zero(), T::zero(),
            sin, cos, T::zero(), T::zero(),
            T::zero(), T::zero(), T::one(), T::zero(),
            T::zero(), T::zero(), T::zero(), T::one()
        );
        return Transform { 
            m, 
            m_inv: m.transpose()}
    }
    pub fn rotate_y(angle: T) -> Self {
        let (sin, cos) = angle.to_radians().sin_cos();
        let m = Matrix4x4::from_values(
            cos, T::zero(), sin, T::zero(),
            T::zero(), T::one(), T::zero(), T::zero(),
            -sin, T::zero(), cos, T::zero(),
            T::zero(), T::zero(), T::zero(), T::one()
        );
        return Transform { 
            m, 
            m_inv: m.transpose()}
    }
    pub fn rotate_x(angle: T) -> Self {
        let (sin, cos) = angle.to_radians().sin_cos();
        let m = Matrix4x4::from_values(
            T::one(), T::zero(), T::zero(), T::zero(),
            T::zero(), cos, -sin, T::zero(),
            T::zero(), sin, cos, T::zero(),
            T::zero(), T::zero(), T::zero(), T::one()
        );
        return Transform { 
            m, 
            m_inv: m.transpose()}
    }
    pub fn rotate(angle: T, axis: Vector3d<T>) -> Transform<T> {
        let a = axis.normalized();
        let (sin, cos) = angle.to_radians().sin_cos();
        let mut m = Matrix4x4::default();
        // Compute rotation of first basis vector
        m.data[0][0] = a.x * a.x + (T::one() - a.x * a.x) * cos;
        m.data[0][1] = a.x * a.y * (T::one() - cos) - a.z * sin;
        m.data[0][2] = a.x * a.z * (T::one() - cos) + a.y * sin;
        m.data[0][3] = T::zero();
        // Compute rotations of second and third basis vectors
        m.data[1][0] = a.x * a.y * (T::one() - cos) + a.z * sin;
        m.data[1][1] = a.y * a.y + (T::one() - a.y * a.y) * cos;
        m.data[1][2] = a.y * a.z * (T::one() - cos) - a.x * sin;
        m.data[1][3] = T::zero();
        
        m.data[2][0] = a.x * a.z * (T::one() - cos) - a.y * sin;
        m.data[2][1] = a.y * a.z * (T::one() - cos) + a.x * sin;
        m.data[2][2] = a.z * a.z + (T::one() - a.z * a.z) * cos;
        m.data[2][3] = T::zero();
        return Transform{m, m_inv: m.transpose()};
    }
    pub fn look_at(pos: Point3d<T>, target: Point3d<T>, up: Vector3d<T>) -> Self {
        let dir = (target - pos).normalized();
        let right = up.cross(&dir).normalized();
        let new_up = dir.cross(&right);
        let m = Matrix4x4::from_values(
            right.x, new_up.x, dir.x, pos.x,
            right.y, new_up.y, dir.y, pos.y,
            right.z, new_up.z, dir.z, pos.z,
            T::zero(), T::zero(), T::zero(), T::one()
        );
        return Transform{m, m_inv: m.inverse()};
    }
}
impl<T: Scalar> Mul<Transform<T>> for &Transform<T> {
    type Output = Transform<T>;
    fn mul(self, rhs: Transform<T>) -> Transform<T> {
        return Transform{m: self.m * rhs.m, m_inv: rhs.m_inv * self.m_inv};
    }
}
impl<T: Scalar> Mul<Point3d<T>> for &Transform<T> {
    type Output = Point3d<T>;
    fn mul(self, rhs: Point3d<T>) -> Point3d<T> {
        return self.m * rhs;
    }
}
impl<T: Scalar> Mul<Vector3d<T>> for &Transform<T> {
    type Output = Vector3d<T>;
    fn mul(self, rhs: Vector3d<T>) -> Vector3d<T> {
        return self.m * rhs;
    }
}
impl<T: Scalar> Mul<Normal3d<T>> for &Transform<T> {
    type Output = Normal3d<T>;
    fn mul(self, rhs: Normal3d<T>) -> Normal3d<T> {
        let x = rhs.x;
        let y = rhs.y;
        let z = rhs.z;
        return Normal3d::new(
            self.m_inv.data[0][0] * x + self.m_inv.data[1][0] * y + self.m_inv.data[2][0] * z,
            self.m_inv.data[0][1] * x + self.m_inv.data[1][1] * y + self.m_inv.data[2][1] * z,
            self.m_inv.data[0][2] * x + self.m_inv.data[1][2] * y + self.m_inv.data[2][2] * z,
        );
    }
}
impl<T: Scalar> Mul<Ray<T>> for &Transform<T> {
    type Output = Ray<T>;
    fn mul(self, rhs: Ray<T>) -> Ray<T> {
        return Ray{
            origin: self * rhs.origin,
            direction: self * rhs.direction,
            t_max: rhs.t_max,
            time: rhs.time
        };
    }
}
impl<T: Scalar> Mul<RayDifferential<T>> for &Transform<T> {
    type Output = RayDifferential<T>;
    fn mul(self, rhs: RayDifferential<T>) -> RayDifferential<T> {
        return RayDifferential{
            origin: self * rhs.origin,
            direction: self * rhs.direction,
            rx_origin: self * rhs.rx_origin,
            ry_origin: self * rhs.ry_origin,
            rx_direction: self * rhs.rx_direction,
            ry_direction: self * rhs.ry_direction,
            has_differentials: rhs.has_differentials,
            time: rhs.time,
            t_max: rhs.t_max
        };
    }
}
impl<T: Scalar> Mul<Bounds3d<T>> for &Transform<T> {
    type Output = Bounds3d<T>;
    fn mul(self, rhs: Bounds3d<T>) -> Bounds3d<T> {
        let p_min = self * rhs.min;
        let p_max = self * rhs.max;
        return Bounds3d::new(p_min, p_max);
    }
}

#[derive(Debug, PartialEq)]
pub struct AnimatedTransform<T: Scalar> {
    start_transform: Transform<T>,
    end_transform: Transform<T>,
    start_time: T,
    end_time: T,
    translations: [Vector3d<T>; 2],
    scales: [Matrix4x4<T>; 2],
    rotations: [Quaternion<T>; 2],
    actually_animated: bool,
}
impl<T: Scalar + Float> AnimatedTransform<T> {
    pub fn new(start_transform: Transform<T>, end_transform: Transform<T>, start_time: T, end_time: T) -> Self {
        let (start_translation, start_rotation, start_scale) = 
        AnimatedTransform::decompose(start_transform.m);
        let (end_translation, end_rotation, end_scale) = 
        AnimatedTransform::decompose(end_transform.m);
        let translations = [start_translation, end_translation];
        let scales = [start_scale, end_scale];
        let mut rotations = [start_rotation, end_rotation];
        // Flip the second rotation if needed to select shortest path
        if start_rotation.dot(&end_rotation) < T::zero() {
            rotations[1] = -rotations[1];
        }
        return AnimatedTransform{
            start_transform: start_transform.clone(),
            end_transform: end_transform.clone(),
            start_time,
            end_time,
            translations,
            scales,
            rotations,
            actually_animated: start_transform != end_transform,
        };
    }
    pub fn decompose(matrix: Matrix4x4<T>) -> (Vector3d<T>, Quaternion<T>, Matrix4x4<T>) {
        // Extract translation
        let translation = Vector3d::new(
            matrix.data[0][3],
            matrix.data[1][3],
            matrix.data[2][3]
        );
        // Compute new transformation matrix with translation removed
        let mut m = matrix;
        m.data[0][3] = T::zero();
        m.data[1][3] = T::zero();
        m.data[2][3] = T::zero();
        m.data[3][3] = T::one();
        // Compute rotation
        let mut norm;
        let mut count = 0;
        let mut rotation = m;
        loop {
            let mut next = Matrix4x4::default();
            // Compute average of matrix and its inverse transpose 
            let r_inverse_transpose = rotation.inverse().transpose();
            for i in 0..4 {
                for j in 0..4 {
                    next.data[i][j] = T::half() * (rotation.data[i][j] + r_inverse_transpose.data[i][j])
                }
            }
            // Compute norm of difference -- when this is small enough, we're done
            norm = T::zero();
            for i in 0..3 {
                let n  = (rotation.data[i][0] - next.data[i][0]).abs() + 
                (rotation.data[i][1] - next.data[i][1]).abs() + 
                (rotation.data[i][2] - next.data[i][2]).abs();
                norm = Scalar::max(norm, n);
            }
            rotation = next;
            count+=1;
            if norm < Scalar::epsilon() || count > 100 {
                break;
            }
        }
        // At this point, next contains the scale and rotation part of the matrix
        // To compute the scale, we need to find the matrix that satisfies 
        // the equation M = RS. To get this, we just multiply M with the 
        // inverse of R
        let scale = rotation.inverse() * m;
        let rotation_qat = Quaternion::from_transform(&Transform::new(rotation, rotation.transpose()));
        return (translation, rotation_qat, scale);
    }
    pub fn interpolate(&self, time: T) -> Transform<T> {
        if !self.actually_animated {
            return self.start_transform.clone();
        }
        let mut t = (time - self.start_time) / (self.end_time - self.start_time);
        if t < T::zero() {
            t = T::zero();
        }
        if t > T::one() {
            t = T::one();
        }
        let translation = Transform::translate(self.translations[0] * (T::one() - t) + self.translations[1] * t);
        let scale = self.scales[0] * (T::one() - t) + self.scales[1] * t;
        let rotation = self.rotations[0].slerp(&self.rotations[1], t);
        let interpolated = &(&translation * rotation.to_transform()) * Transform::new(scale, scale.inverse());
        return interpolated;
    }
    pub fn apply_to_ray(&self, ray: Ray<T>) -> Ray<T> {
        let ray_time = ray.time;
        let mut interpolated_ray = &self.interpolate(ray_time) * ray;
        interpolated_ray.time = ray_time;
        return interpolated_ray;
    }
    pub fn apply_to_ray_differential(&self, ray: RayDifferential<T>) -> RayDifferential<T> {
        let ray_time = ray.time;
        let mut interpolated_ray = &self.interpolate(ray_time) * ray;
        interpolated_ray.time = ray_time;
        return interpolated_ray;
    }
    pub fn apply_to_point(&self, point: Point3d<T>, time: T) -> Point3d<T> {
        return &self.interpolate(time) * point;
    }
    pub fn apply_to_vector(&self, vector: Vector3d<T>, time: T) -> Vector3d<T> {
        return &self.interpolate(time) * vector;
    }
}
#[test]
fn test_matrix_new() {
    let mat = Matrix4x4::new([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ]);
    assert_eq!(mat[0][0], 0.0);
}
#[test]
fn test_matrix_default() {
    let mat = Matrix4x4::<f64>::default();
    assert_eq!(mat.data, [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]);
}
#[test]
fn test_matrix_new_from_values() {
    let mat = Matrix4x4::from_values(
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    );
    assert_eq!(mat[0][0], 0.0);
}
#[test]
fn test_matrix_transpose() {
    let mat1 = Matrix4x4::from_values(
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0
    );
    let mat2 = Matrix4x4::from_values(
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    );
    assert_eq!(mat1.transpose(), mat2);
}
#[test]
fn test_matrix_transpose_identity() {
    let mat1 = Matrix4x4::from_values(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    assert_eq!(mat1.transpose(), mat1);
}
#[test]
fn test_matrix_add() {
    let mat1 = Matrix4x4::from_values(
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0
    );
    let mat2 = Matrix4x4::from_values(
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0
    );
    let expected = Matrix4x4::from_values(
        4.0, 4.0, 4.0, 4.0,
        4.0, 4.0, 4.0, 4.0,
        4.0, 4.0, 4.0, 4.0,
        4.0, 4.0, 4.0, 4.0
    );
    assert_eq!(mat1 + mat2, expected);
}
#[test]
fn test_matrix_mul() {
    let mat1 = Matrix4x4::from_values(
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0
    );
    let mat2 = Matrix4x4::from_values(
        16.0, 16.0, 16.0, 16.0,
        16.0, 16.0, 16.0, 16.0,
        16.0, 16.0, 16.0, 16.0,
        16.0,16.0,16.0,16.0
    );
    assert_eq!(mat1.clone() * mat1, mat2);
}
#[test]
fn test_matrix_inverse_identity() {
    let mat1 = Matrix4x4::<f64>::default();
    let mat2 = Matrix4x4::<f64>::default();
    assert_eq!(mat1.inverse(), mat2);
}
#[test]
fn test_matrix_inverse() {
        let mat1 = Matrix4x4::from_values(
        5.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0,
        0.0, 1.0, 3.0, 0.0,
        1.0, 0.0, 0.0, 1.0
    );
    let mat1_inverse = Matrix4x4::from_values(
        0.2, -0.0, -0.0, -0.0,
        -0.0, -1.0, 1.0, -0.0,
        -0.0, 0.3333333333333333, -0.0, -0.0,
        -0.2, -0.0, -0.0, 1.0
    );
    assert_eq!(mat1.inverse(), mat1_inverse);
    let mat2 = mat1 * mat1.inverse();
    assert_eq!(mat2, Matrix4x4::<f64>::default());
}
#[test]
fn test_transform_default() {
    let identity = Matrix4x4::from_values(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    let t = Transform::<f64>::default();
    assert_eq!(t.m, identity);
    assert_eq!(t.m_inv, identity);
}
#[test]
fn test_transform_new() {
    let identity = Matrix4x4::from_values(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    let t = Transform::new(identity, identity);
    assert_eq!(t.m, identity);
    assert_eq!(t.m_inv, identity);
}
#[test]
fn test_transform_inverse() {
    let mat1 = Matrix4x4::from_values(
        5.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0,
        0.0, 1.0, 3.0, 0.0,
        1.0, 0.0, 0.0, 1.0
    );
    let mat1_inverse = Matrix4x4::from_values(
        0.2, 0.0, 0.0, 0.0,
        0.0, -1.0, 3.0, 0.0,
        0.0, 0.33, 0.0, 0.0,
        -0.2, 0.0, 0.0, 1.0
    );
    let t = Transform::new(mat1, mat1_inverse);
    assert_eq!(t.inverse(), Transform::new(mat1_inverse, mat1));
}
#[test]
fn test_transform_transpose() {
    let mat1 = Matrix4x4::from_values(
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0
    );
    let mat2 = Matrix4x4::from_values(
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    );
    let t = Transform::new(mat1.clone(), mat1);
    let t_t = t.transpose();
    assert_eq!(t_t.m, mat2);
    assert_eq!(t_t.m_inv, mat2);
}
#[test]
fn test_transform_is_identity() {
    let identity = Matrix4x4::from_values(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    let t = Transform::new(identity.clone(),identity);
    assert!(t.is_identity());
}
#[test]
fn test_transform_translate() {
    let t = Transform::translate(Vector3d::new(1.0, 1.0, 1.0));
    let expected_m = Matrix4x4::from_values(
        1.0, 0.0, 0.0, 1.0, 
        0.0, 1.0, 0.0, 1.0, 
        0.0, 0.0, 1.0, 1.0, 
        0.0, 0.0, 0.0, 1.0);
    let expected_inv = Matrix4x4::from_values(
        1.0, 0.0, 0.0, -1.0, 
        0.0, 1.0, 0.0, -1.0, 
        0.0, 0.0, 1.0, -1.0, 
        0.0, 0.0, 0.0, 1.0);
    assert_eq!(t.m, expected_m);
    assert_eq!(t.m_inv, expected_inv);
}
#[test]
fn test_transform_scale() {
    let t = Transform::scale(2.0, 2.0, 2.0);
    let expected_m = Matrix4x4::from_values(
        2.0, 0.0, 0.0, 0.0, 
        0.0, 2.0, 0.0, 0.0, 
        0.0, 0.0, 2.0, 0.0, 
        0.0, 0.0, 0.0, 1.0);
    let expected_inv = Matrix4x4::from_values(
        0.5, 0.0, 0.0, 0.0, 
        0.0, 0.5, 0.0, 0.0, 
        0.0, 0.0, 0.5, 0.0, 
        0.0, 0.0, 0.0, 1.0);
    assert_eq!(t.m, expected_m);
    assert_eq!(t.m_inv, expected_inv);
}
#[test]
fn test_transform_rotate_z() {
    let t = Transform::<f64>::rotate_z(90.0);
    let expected_m = Matrix4x4::from_values(
        0.0, -1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    let expected_inv = Matrix4x4::from_values(
        0.0, 1.0, 0.0, 0.0,
        -1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    assert_eq!(t.m, expected_m);
    assert_eq!(t.m_inv, expected_inv);
}
#[test]
fn test_transform_rotate_x() {
    let t = Transform::<f64>::rotate_x(90.0);
    let expected_m = Matrix4x4::from_values(
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    let expected_inv = Matrix4x4::from_values(
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    assert_eq!(t.m, expected_m);
    assert_eq!(t.m_inv, expected_inv);
}
#[test]
fn test_transform_rotate_y() {
    let t = Transform::<f64>::rotate_y(90.0);
    let expected_m = Matrix4x4::from_values(
        0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    let expected_inv = Matrix4x4::from_values(
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    assert_eq!(t.m, expected_m);
    assert_eq!(t.m_inv, expected_inv);
}
#[test]
fn test_transform_rotate_vec_y() {
    let t = Transform::<f64>::rotate(90.0, Vector3d::new(0.0, 1.0, 0.0));
    let expected_m = Matrix4x4::from_values(
        0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    let expected_inv = Matrix4x4::from_values(
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    assert_eq!(t.m, expected_m);
    assert_eq!(t.m_inv, expected_inv);
}
#[test]
fn test_transform_rotate_vec_z() {
    let t = Transform::<f64>::rotate(90.0, Vector3d::new(0.0, 0.0, 1.0));
    let expected_m = Matrix4x4::from_values(
        0.0, -1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    let expected_inv = Matrix4x4::from_values(
        0.0, 1.0, 0.0, 0.0,
        -1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    assert_eq!(t.m, expected_m);
    assert_eq!(t.m_inv, expected_inv);
}
#[test]
fn test_transform_rotate_vec_x() {
    let t = Transform::<f64>::rotate(90.0, Vector3d::new(1.0, 0.0, 0.0));
    let expected_m = Matrix4x4::from_values(
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    let expected_inv = Matrix4x4::from_values(
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    assert_eq!(t.m, expected_m);
    assert_eq!(t.m_inv, expected_inv);
}
#[test]
fn test_look_at() {
    let eye = Point3d::new(0.0, 0.0, 0.0);
    let center = Point3d::new(1.0, 1.0, 1.0);
    let up = Vector3d::new(0.0, 1.0, 0.0);
    let t = Transform::look_at(eye, center, up);
    let expected_m = Matrix4x4::new(
        [[0.7071067811865476, -0.4082482904638631, 0.5773502691896258, 0.0], 
        [0.0, 0.8164965809277261, 0.5773502691896258, 0.0], 
        [-0.7071067811865476, -0.4082482904638631, 0.5773502691896258, 0.0], 
        [0.0, 0.0, 0.0, 1.0]]);
    assert_eq!(t.m, expected_m);
}
#[test]
fn test_has_scale_true() {
    let t = Transform::scale(2.0, 3.0, 4.0);
    assert!(t.has_scale());
}
#[test]
fn test_has_scale_false() {
    let t = Transform::<f64>::default();
    assert!(!t.has_scale());
}
#[test]
fn test_transform_mul_vector() {
    let t = Transform::scale(2.0, 3.0, 4.0);
    let v = Vector3d::new(1.0, 2.0, 3.0);
    let expected = Vector3d::new(2.0, 6.0, 12.0);
    assert_eq!(&t * v, expected);
}
#[test]
fn test_transform_mul_point() {
    let t = Transform::scale(2.0, 3.0, 4.0);
    let p = Point3d::new(1.0, 2.0, 3.0);
    let expected = Point3d::new(2.0, 6.0, 12.0);
    assert_eq!(&t * p, expected);
}
#[test]
fn test_transform_mul_transform() {
    let t1 = Transform::scale(2.0, 3.0, 4.0);
    let t2 = Transform::scale(3.0, 2.0, 1.0);
    let expected = Transform::scale(6.0, 6.0, 4.0);
    assert_eq!(&t1 * t2, expected);
}
#[test]
fn test_transform_mul_normal() {
    let t = Transform::scale(2.0, 3.0, 4.0);
    let n = Normal3d::new(1.0, 2.0, 3.0);
    let expected = Normal3d::new(0.5, 0.6666666666666666, 0.75);
    assert_eq!(&t * n, expected);
}
#[test]
fn test_transform_mul_ray() {
    let t = Transform::scale(2.0, 3.0, 4.0);
    let r = Ray::new(Point3d::new(1.0, 2.0, 3.0), Vector3d::new(1.0, 0.0, 0.0), 0.0, 0.0);
    let expected = Ray::new(Point3d::new(2.0, 6.0, 12.0), Vector3d::new(2.0, 0.0, 0.0), 0.0, 0.0);
    assert_eq!(&t * r, expected);
}
#[test]
fn test_transform_bounds() {
    let t = Transform::scale(2.0, 3.0, 4.0);
    let b = Bounds3d::new(Point3d::new(1.0, 2.0, 3.0), Point3d::new(4.0, 6.0, 8.0));
    let expected = Bounds3d::new(Point3d::new(2.0, 6.0, 12.0), Point3d::new(8.0, 18.0, 32.0));
    assert_eq!(&t * b, expected);
}
#[test]
fn test_transform_swaps_handedness_true() {
    let t = Transform::scale(1.0, -1.0, 1.0);
    assert!(t.swaps_handedness());
}
#[test]
fn test_transform_swaps_handedness_false() {
    let t = Transform::scale(1.0, 1.0, 1.0);
    assert!(!t.swaps_handedness());
}
#[test]
fn test_animated_transform_new() {
    let t = AnimatedTransform::new(
        Transform::scale(2.0, 3.0, 4.0), 
        Transform::scale(3.0, 2.0, 1.0),
        0.0,
        1.0
    );

    assert_eq!(t.start_time, 0.0);
    assert_eq!(t.end_time, 1.0);
    assert_eq!(t.start_transform, Transform::scale(2.0, 3.0, 4.0));
    assert_eq!(t.end_transform, Transform::scale(3.0, 2.0, 1.0));
    assert_eq!(t.actually_animated, true);
}
#[test]
fn test_animated_transform_not_actually_animated() {
    let t = AnimatedTransform::new(
        Transform::scale(2.0, 3.0, 4.0), 
        Transform::scale(2.0, 3.0, 4.0),
        0.0,
        0.0
    );
    assert_eq!(t.actually_animated, false);
}
#[test]
fn test_animated_transform_decompose_default() {
    let transform = Transform::new(Matrix4x4::<f64>::default(), Matrix4x4::<f64>::default());
    let expected = (
        Vector3d::new(0.0, 0.0, 0.0),
        Quaternion { v: Vector3d { x: 0.0, y: 0.0, z: 0.0 }, w: 1.0 },
        *Transform::scale(1.0, 1.0, 1.0).matrix()
    );
    assert_eq!(AnimatedTransform::decompose(*transform.matrix()), expected);
}
#[test]
fn test_animated_transform_decompose_translation() {
    let transform = Transform::translate(Vector3d::new(1.0, 1.0, 1.0));
    let expected = (
        Vector3d::new(1.0, 1.0, 1.0),
        Quaternion { v: Vector3d { x: 0.0, y: 0.0, z: 0.0 }, w: 1.0 },
        *Transform::scale(1.0, 1.0, 1.0).matrix()
    );
    assert_eq!(AnimatedTransform::decompose(*transform.matrix()), expected);
}
#[test]
fn test_animated_transform_decompose_rotation() {
    let transform = Transform::rotate_x(PI / 2.0);
    let expected = (
        Vector3d::new(0.0, 0.0, 0.0),
        Quaternion::from_transform(&Transform::rotate_x(PI / 2.0)),
        *Transform::scale(1.0, 1.0, 1.0).matrix()
    );
    assert_eq!(AnimatedTransform::decompose(*transform.matrix()), expected);
}
#[test]
fn test_animated_transform_decompose_scaling() {
    let transform = Transform::scale(2.0, 2.0, 2.0);
    let expected = (
        Vector3d::new(0.0, 0.0, 0.0),
        Quaternion { v: Vector3d { x: 0.0, y: 0.0, z: 0.0 }, w: 1.0 },
        *Transform::scale(2.0, 2.0, 2.0).matrix()
    );
    assert_eq!(AnimatedTransform::decompose(*transform.matrix()), expected);
}
#[test]
fn test_animated_transform_interpolate_translation() {
    let t1 = Transform::translate(Vector3d::new(1.0, 1.0, 1.0));
    let t2 = Transform::translate(Vector3d::new(3.0, 3.0, 3.0));
    let t = AnimatedTransform::new(t1, t2, 0.0, 1.0);
    assert_eq!(t.interpolate(1.0), Transform::translate(Vector3d::new(3.0, 3.0, 3.0)));
    assert_eq!(t.interpolate(0.5), Transform::translate(Vector3d::new(2.0, 2.0, 2.0)));
    assert_eq!(t.interpolate(0.0), Transform::translate(Vector3d::new(1.0, 1.0, 1.0)));
}
#[test]
fn test_animated_transform_interpolate_scale() {
    let t1 = Transform::scale(1.0, 1.0, 1.0);
    let t2 = Transform::scale(3.0, 3.0, 3.0);
    let t = AnimatedTransform::new(t1, t2, 0.0, 1.0);
    assert_eq!(t.interpolate(1.0), Transform::scale(3.0, 3.0, 3.0));
    assert_eq!(t.interpolate(0.5), Transform::scale(2.0, 2.0, 2.0));
    assert_eq!(t.interpolate(0.0), Transform::scale(1.0, 1.0, 1.0));
}
#[test]
fn test_animated_transform_interpolate_rotation() {
    let t1 = Transform::rotate_x(0.0);
    let t2 = Transform::rotate_x(90.0);
    let t = AnimatedTransform::new(t1, t2, 0.0, 1.0);
    assert_eq!(t.interpolate(1.0), Transform::rotate_x(90.0));
    assert_eq!(t.interpolate(0.0), Transform::rotate_x(0.0));
    assert_eq!(t.interpolate(0.5), Transform::rotate_x(45.0));
}
#[test]
fn test_animated_transform_apply_to_ray() {
    let t1 = Transform::translate(Vector3d::new(0.0, 0.0, 0.0));
    let t2 = Transform::translate(Vector3d::new(3.0, 3.0, 3.0));
    let t = AnimatedTransform::new(t1, t2, 0.0, 1.0);
    let ray = Ray::new(Point3d::new(0.0, 0.0, 0.0), Vector3d::new(0.0, 0.0, 1.0), 1.0, 0.0);
    assert_eq!(t.apply_to_ray(ray), Ray::new(Point3d::new(0.0, 0.0, 0.0), Vector3d::new(0.0, 0.0, 1.0), 1.0, 0.0));
    let ray = Ray::new(Point3d::new(0.0, 0.0, 0.0), Vector3d::new(0.0, 0.0, 1.0), 1.0, 0.5);
    assert_eq!(t.apply_to_ray(ray), Ray::new(Point3d::new(1.5, 1.5, 1.5), Vector3d::new(0.0, 0.0, 1.0), 1.0, 0.5));
    let ray = Ray::new(Point3d::new(0.0, 0.0, 0.0), Vector3d::new(0.0, 0.0, 1.0), 1.0, 1.0);
    assert_eq!(t.apply_to_ray(ray), Ray::new(Point3d::new(3.0, 3.0, 3.0), Vector3d::new(0.0, 0.0, 1.0), 1.0, 1.0));
}
#[test]
fn test_animated_transform_apply_to_point() {
    let t1 = Transform::translate(Vector3d::new(0.0, 0.0, 0.0));
    let t2 = Transform::translate(Vector3d::new(3.0, 3.0, 3.0));
    let t = AnimatedTransform::new(t1, t2, 0.0, 1.0);
    assert_eq!(t.apply_to_point(Point3d::new(0.0, 0.0, 0.0), 0.0), Point3d::new(0.0, 0.0, 0.0));
    assert_eq!(t.apply_to_point(Point3d::new(0.0, 0.0, 0.0), 1.0), Point3d::new(3.0, 3.0, 3.0));
    assert_eq!(t.apply_to_point(Point3d::new(0.0, 0.0, 0.0), 0.5), Point3d::new(1.5, 1.5, 1.5));
}
#[test]
fn test_animated_transform_apply_to_vector() {
    let t1 = Transform::rotate_x(0.0);
    let t2 = Transform::rotate_x(180.0);
    let t = AnimatedTransform::new(t1, t2, 0.0, 1.0);
    assert!(t.apply_to_vector(Vector3d::new(1.0, 1.0, 1.0), 0.0).approximately_equal(&Vector3d::new(1.0, 1.0, 1.0)));
    assert!(t.apply_to_vector(Vector3d::new(1.0, 1.0, 1.0), 1.0).approximately_equal(&Vector3d::new(1.0, -1.0, -1.0)));
    assert!(t.apply_to_vector(Vector3d::new(1.0, 1.0, 1.0), 0.5).approximately_equal(&Vector3d::new(1.0, 1.0, -1.0)));
}
#[test]
fn test_animated_transform_apply_to_ray_differential() {
    let t1 = Transform::translate(Vector3d::new(0.0, 0.0, 0.0));
    let t2 = Transform::translate(Vector3d::new(3.0, 3.0, 3.0));
    let t = AnimatedTransform::new(t1, t2, 0.0, 1.0);
    let ray = Ray::new(Point3d::new(0.0, 0.0, 0.0), Vector3d::new(0.0, 0.0, 1.0), 1.0, 0.0);
    let ray_differential = RayDifferential::from_ray(&ray);
    assert_eq!(t.apply_to_ray_differential(ray_differential), RayDifferential::from_ray(&Ray::new(Point3d::new(0.0, 0.0, 0.0), Vector3d::new(0.0, 0.0, 1.0), 1.0, 0.0)));
    let ray = Ray::new(Point3d::new(0.0, 0.0, 0.0), Vector3d::new(0.0, 0.0, 1.0), 1.0, 0.5);
    let ray_differential = RayDifferential::from_ray(&ray);
    assert_eq!(
        t.apply_to_ray_differential(ray_differential),
            RayDifferential{
                origin: Point3d::new(1.5, 1.5, 1.5), 
                direction: Vector3d::new(0.0, 0.0, 1.0),
                rx_origin: Point3d::new(1.5, 1.5, 1.5),
                rx_direction: Vector3d::new(0.0, 0.0, 0.0),
                ry_origin: Point3d::new(1.5, 1.5, 1.5),
                ry_direction: Vector3d::new(0.0, 0.0, 0.0),
                time: 0.5,
                t_max: Cell::new(1.0),
                has_differentials: false,
            }
        );
    let ray = Ray::new(Point3d::new(0.0, 0.0, 0.0), Vector3d::new(0.0, 0.0, 1.0), 1.0, 1.0);
    let ray_differential = RayDifferential::from_ray(&ray);
    assert_eq!(
        t.apply_to_ray_differential(ray_differential),
            RayDifferential{
                origin: Point3d::new(3.0, 3.0, 3.0), 
                direction: Vector3d::new(0.0, 0.0, 1.0),
                rx_origin: Point3d::new(3.0, 3.0, 3.0),
                rx_direction: Vector3d::new(0.0, 0.0, 0.0),
                ry_origin: Point3d::new(3.0, 3.0, 3.0),
                ry_direction: Vector3d::new(0.0, 0.0, 0.0),
                time: 1.0,
                t_max: Cell::new(1.0),
                has_differentials: false,
            }
        );
}