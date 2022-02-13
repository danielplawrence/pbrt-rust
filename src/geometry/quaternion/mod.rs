use std::ops::{Add, Sub, Div, Mul, Neg};
use num::{clamp, Float};

use crate::geometry::vector::Vector3d;

use super::{Scalar, vector::Dot, transform::{Matrix4x4, Transform}};

#[derive(Debug, Copy, Clone)]
pub struct Quaternion<T>{
    pub(crate) v: Vector3d<T>,
    pub(crate) w: T,
}
impl<T: Scalar + Float > Quaternion<T> {
    pub fn new(v: Vector3d<T>, w: T) -> Self {
        Quaternion{v, w}
    }
    pub fn dot(&self, other: &Self) -> T {
        self.v.dot(&other.v) + self.w * other.w
    }
    pub fn normalize(&self) -> Self {
        let norm = Scalar::sqrt(self.dot(&self));
        Quaternion{v: self.v / norm, w: self.w / norm}
    }
    pub fn slerp(&self, other: &Self, t: T) -> Self {
        let cos_theta = self.dot(other);
        if cos_theta > (T::one() - Scalar::epsilon()) {
            return (*self * (T::one() - t) + *other * t).normalize();
        } else {
            let theta = clamp(cos_theta, -T::one(), T::one()).acos();
            let theta_p = theta * t;
            let qperp = (*other - *self * cos_theta).normalize();
            return *self * theta_p.cos() + qperp * theta_p.sin();
        }
    }
    pub fn from_transform(transform: &Transform<T>) -> Self {
        let m = transform.matrix();
        let mut v = Vector3d::new(T::zero(), T::zero(), T::zero());
        let trace = m[0][0] + m[1][1] + m[2][2];
        if trace > T::zero() {
            // Compute w from matrix trace, then xyz
            // 4w^2 = m[0][0] + m[1][1] + m[2][2] + m[3][3] (but m[3][3] == 1)
            let mut s = Scalar::sqrt(trace + T::one());
            let w = s / T::two();
            s = T::half() / s;
            v.x = (m[2][1] - m[1][2]) * s;
            v.y = (m[0][2] - m[2][0]) * s;
            v.z = (m[1][0] - m[0][1]) * s;
            return Self{v, w};
        } else {
            // Compute largest of $x$, $y$, or $z$, then remaining components
            let nxt = [1, 2, 0];
            let mut q = [T::zero(); 3];
            let mut i = 0;
            if m[1][1] > m[0][0] {i = 1};
            if m[2][2] > m[i][i] {i = 2};
            let j = nxt[i];
            let k = nxt[j];
            let mut s = Scalar::sqrt((m[i][i] - (m[j][j] + m[k][k])) + T::one());
            q[i] = s * T::half();
            if s != T::zero(){
                s = T::half() / s;
            }
            let w = (m[k][j] - m[j][k]) * s;
            q[j] = (m[j][i] + m[i][j]) * s;
            q[k] = (m[k][i] + m[i][k]) * s;
            v.x = q[0];
            v.y = q[1];
            v.z = q[2];
            return Self{v, w};
        }
    }
    pub fn to_transform(&self) -> Transform<T> {
        let xx = self.v.x * self.v.x;
        let yy = self.v.y * self.v.y;
        let zz = self.v.z * self.v.z;
        let xy = self.v.x * self.v.y;
        let xz = self.v.x * self.v.z;
        let yz = self.v.y * self.v.z;
        let wx = self.v.x * self.w;
        let wy = self.v.y * self.w;
        let wz = self.v.z * self.w;

        let mut m_data = [[T::zero(); 4]; 4];
        m_data[0][0] = T::one() - T::two() * (yy + zz);
        m_data[0][1] = T::two() * (xy + wz);
        m_data[0][2] = T::two() * (xz - wy);
        m_data[1][0] = T::two() * (xy - wz);
        m_data[1][1] = T::one() - T::two() * (xx + zz);
        m_data[1][2] = T::two() * (yz + wx);
        m_data[2][0] = T::two() * (xz + wy);
        m_data[2][1] = T::two() * (yz - wx);
        m_data[2][2] = T::one() - T::two() * (xx + yy);
        m_data[3][3] = T::one();

        let mat = Matrix4x4::new(m_data);
        return Transform::new(mat.transpose(), mat);
    }
}
impl<T:Scalar> Add for Quaternion<T> {
    type Output = Quaternion<T>;
    fn add(self, other: Quaternion<T>) -> Quaternion<T> {
        Quaternion {
            v: self.v + other.v,
            w: self.w + other.w,
        }
    }
}
impl<T: Scalar> Sub for Quaternion<T> {
    type Output = Quaternion<T>;
    fn sub(self, other: Quaternion<T>) -> Quaternion<T> {
        Quaternion {
            v: self.v - other.v,
            w: self.w - other.w,
        }
    }
}
impl<T: Scalar> Mul<T> for Quaternion<T> {
    type Output = Quaternion<T>;
    fn mul(self, other: T) -> Quaternion<T> {
        Quaternion {
            v: self.v * other,
            w: self.w * other,
        }
    }
}
impl<T: Scalar> Div<T> for Quaternion<T> {
    type Output = Quaternion<T>;
    fn div(self, other: T) -> Quaternion<T> {
        Quaternion {
            v: self.v / other,
            w: self.w / other,
        }
    }
}
impl<T: Scalar> Neg for Quaternion<T> {
    type Output = Quaternion<T>;
    fn neg(self) -> Quaternion<T> {
        Quaternion {
            v: -self.v,
            w: -self.w,
        }
    }
}
impl<T: Scalar> PartialEq for Quaternion<T> {
    fn eq(&self, other: &Quaternion<T>) -> bool {
        self.v.approximately_equal(&other.v) && self.w.approximately_equal(other.w)
    }
}
#[test]
fn test_quaternion_new() {
    let q = Quaternion::new(Vector3d::new(1.0, 2.0, 3.0), 4.0);
    assert_eq!(q.v, Vector3d::new(1.0, 2.0, 3.0));
    assert_eq!(q.w, 4.0);
}
#[test]
fn test_quaternion_normalize() {
    let q = Quaternion::new(Vector3d::new(1.0, 2.0, 3.0), 4.0);
    let q_norm = q.normalize();
    assert_eq!(q_norm.dot(&q_norm), 0.9999999999999998);
}
#[test]
fn test_quaternion_add() {
    let a = Quaternion {
        v: Vector3d::new(1.0, 2.0, 3.0),
        w: 4.0,
    };
    let b = Quaternion {
        v: Vector3d::new(5.0, 6.0, 7.0),
        w: 8.0,
    };
    let c = a + b;
    assert_eq!(c.v, Vector3d::new(6.0, 8.0, 10.0));
    assert_eq!(c.w, 12.0);
}
#[test]
fn test_quaternion_sub() {
    let a = Quaternion {
        v: Vector3d::new(1.0, 2.0, 3.0),
        w: 4.0,
    };
    let b = Quaternion {
        v: Vector3d::new(5.0, 6.0, 7.0),
        w: 8.0,
    };
    let c = a - b;
    assert_eq!(c.v, Vector3d::new(-4.0, -4.0, -4.0));
    assert_eq!(c.w, -4.0);
}
#[test]
fn test_quaternion_mul_t() {
    let a = Quaternion {
        v: Vector3d::new(1.0, 2.0, 3.0),
        w: 4.0,
    };
    let b = a * 5.0;
    assert_eq!(b.v, Vector3d::new(5.0, 10.0, 15.0));
    assert_eq!(b.w, 20.0);
}
#[test]
fn test_quaternion_div_t() {
    let a = Quaternion {
        v: Vector3d::new(1.0, 2.0, 3.0),
        w: 4.0,
    };
    let b = a / 2.0;
    assert_eq!(b.v, Vector3d::new(0.5, 1.0, 1.5));
    assert_eq!(b.w, 2.0);
}
#[test]
fn test_quaternion_slerp() {
    let a = Quaternion::new(Vector3d::new(0.0, 0.0, 0.0), 1.0); // Reference position
    let b = Quaternion::new(Vector3d::new(0.0, 1.0, 0.0), 0.0); // -180 degree rotation around y-axis
    let mut c = a.slerp(&b, 0.0);
    assert_eq!(c.v, Vector3d::new(0.0, 0.0, 0.0));
    assert!(c.w.approximately_equal(1.0));
    c = a.slerp(&b, 1.0);
    assert!(c.v.approximately_equal(&Vector3d::new(0.0, 1.0, 0.0)));
    assert!(c.w.approximately_equal(0.0));
    c = a.slerp(&b, 0.5);
    assert!(c.v.approximately_equal(&Vector3d::new(0.0, 0.7071, 0.0))); // -90 degree rotation around y-axis
    assert!(c.w.approximately_equal(0.7071));
}
#[test]
fn test_quaternion_slerp_rotations() {
    let t1 = Transform::rotate_x(0.0);
    let q1 = Quaternion::from_transform(&t1);
    let t2 = Transform::rotate_x(180.0);
    let q2 = Quaternion::from_transform(&t2);
    let t3 = Transform::rotate_x(90.0);
    let q3 = Quaternion::from_transform(&t3);
    let q4 = q1.slerp(&q2, 0.5);
    assert_eq!(q3, q4);
}
#[test]
fn test_quaternion_to_transform() {
    let q = Quaternion::new(Vector3d::new(1.0, 1.0, 1.0), 1.0);
    let t = q.to_transform();
    assert_eq!(t.matrix(), &Matrix4x4::new([
        [-3.0, 0.0, 4.0, 0.0], 
        [4.0, -3.0, 0.0, 0.0], 
        [0.0, 4.0, -3.0, 0.0], 
        [0.0, 0.0, 0.0, 1.0]
        ]));
}
#[test]
fn test_quaternion_from_transform() {
    let t = Transform::new(Matrix4x4::new([
            [1.0, 0.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0, 0.0], 
            [0.0, 0.0, 0.0, 1.0]
        ]), Matrix4x4::new([
            [1.0, 0.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0, 0.0], 
            [0.0, 0.0, 0.0, 1.0]
        ]));
    let q = Quaternion::from_transform(&t);
    assert_eq!(q.v, Vector3d::new(0.0, 0.0, 0.0));
    assert_eq!(q.w, 1.0);
    let p = Quaternion::new(Vector3d{ x: 0.0, y: 0.0, z: 0.0}, 1.0);
    let t = p.to_transform();
    let r = Quaternion::from_transform(&t);
    assert_eq!(p, r);
}
#[test]
fn test_quaternion_from_transform_x_rotation() {
    let t = Transform::rotate_x(0.0);
    let q = Quaternion::from_transform(&t);
    assert_eq!(q.v, Vector3d::new(0.0, 0.0, 0.0));
    assert_eq!(q.w, 1.0);
    let t = Transform::rotate_x(90.0);
    let q = Quaternion::from_transform(&t);
    assert!(q.v.approximately_equal(&Vector3d::new(0.707106, 0.0, 0.0)));
    assert!(q.w.approximately_equal(0.707106));
    let t = Transform::rotate_x(180.0);
    let q = Quaternion::from_transform(&t);
    assert!(q.v.approximately_equal(&Vector3d::new(1.0, 0.0, 0.0)));
    assert!(q.w.approximately_equal(0.0));
}
#[test]
fn test_quaternion_from_transform_y_rotation() {
    let t = Transform::rotate_y(0.0);
    let q = Quaternion::from_transform(&t);
    assert_eq!(q.v, Vector3d::new(0.0, 0.0, 0.0));
    assert_eq!(q.w, 1.0);
    let t = Transform::rotate_y(90.0);
    let q = Quaternion::from_transform(&t);
    assert!(q.v.approximately_equal(&Vector3d::new(0.0, 0.707106, 0.0)));
    assert!(q.w.approximately_equal(0.707106));
    let t = Transform::rotate_y(180.0);
    let q = Quaternion::from_transform(&t);
    assert!(q.v.approximately_equal(&Vector3d::new(0.0, 1.0, 0.0)));
    assert!(q.w.approximately_equal(0.0));
}
#[test]
fn test_quaternion_from_transform_z_rotation() {
    let t = Transform::rotate_z(0.0);
    let q = Quaternion::from_transform(&t);
    assert_eq!(q.v, Vector3d::new(0.0, 0.0, 0.0));
    assert_eq!(q.w, 1.0);
    let t = Transform::rotate_z(90.0);
    let q = Quaternion::from_transform(&t);
    assert!(q.v.approximately_equal(&Vector3d::new(0.0, 0.0, 0.707106)));
    assert!(q.w.approximately_equal(0.707106));
    let t = Transform::rotate_z(180.0);
    let q = Quaternion::from_transform(&t);
    assert!(q.v.approximately_equal(&Vector3d::new(0.0, 0.0, 1.0)));
    assert!(q.w.approximately_equal(0.0));
}
#[test]
fn test_quaternion_neg() {
    let q = Quaternion::new(Vector3d::new(1.0, 2.0, 3.0), 4.0);
    let q_neg = -q;
    assert_eq!(q_neg.v, Vector3d::new(-1.0, -2.0, -3.0));
    assert_eq!(q_neg.w, -4.0);
}