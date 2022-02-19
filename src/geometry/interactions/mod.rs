use std::ops::Mul;

use super::{vector::{Vector3d, Vector2d, Normal3d, FaceForward}, point::Point3d, ray::Ray, Scalar, transform::Transform};

pub trait MediumInterface{}
pub trait InteractionData<T: Scalar> {
    fn point(&self) -> Point3d<T>;
    fn time(&self) -> T;
    fn p_error(&self) -> Vector3d<T>;
    fn wo(&self) -> Vector3d<T>;
    fn n(&self) -> Option<Normal3d<T>> {
        None
    }
    fn is_surface_interaction(&self) -> bool {
        return self.n().is_some();
    }
    fn medium_interface(&self) -> Option<&dyn MediumInterface>;
}
pub trait InteractionMethods<T: Scalar>: InteractionData<T> {
    fn spawn_ray(&self, direction: Vector3d<T>) -> Ray<T> {
        Ray::new(self.point(), direction, T::inf(), self.time())
    }
    fn spawn_ray_to_point(&self, point: Point3d<T>) -> Ray<T> {
        Ray::new(self.point(), point - self.point(), T::inf(), self.time())
    }
    fn spawn_ray_to_interaction(&self, other: &dyn InteractionMethods<T>) -> Ray<T> {
        Ray::new(self.point(), other.point() - self.point(), T::inf(), self.time())
    }
}
struct Shading<T> {
    pub n: Normal3d<T>,
    pub dp_du: Vector3d<T>,
    pub dp_dv: Vector3d<T>,
    pub dn_du: Normal3d<T>,
    pub dn_dv: Normal3d<T>,
}
trait Shape<T> {
    fn transform_swaps_handedness(&self) -> bool;
    fn reverse_orientation(&self) -> bool;
}
struct SurfaceInteraction<T: Scalar> {
    pub point: Point3d<T>,
    pub time: T,
    pub p_error: Vector3d<T>,
    pub wo: Vector3d<T>,
    pub n: Normal3d<T>,
    pub uv: Vector2d<T>,
    pub dp_du: Vector3d<T>,
    pub dp_dv: Vector3d<T>,
    pub dn_du: Normal3d<T>,
    pub dn_dv: Normal3d<T>,
    pub shading: Shading<T>,
    pub shape: Option<Box<dyn Shape<T>>>,
}
impl<T: Scalar> InteractionData<T> for SurfaceInteraction<T>{
    fn point(&self) -> Point3d<T> {
        self.point
    }
    fn time(&self) -> T {
        self.time
    }
    fn p_error(&self) -> Vector3d<T> {
        self.p_error
    }
    fn wo(&self) -> Vector3d<T> {
        self.wo
    }
    fn n(&self) -> Option<Normal3d<T>> {
        Some(self.n)
    }
    fn is_surface_interaction(&self) -> bool {
        true
    }
    fn medium_interface(&self) -> Option<&dyn MediumInterface> {
        None
    }
}
impl<T: Scalar> SurfaceInteraction<T> {
    pub fn new(
        point: Point3d<T>, 
        time: T, 
        p_error: Vector3d<T>, 
        wo: Vector3d<T>, 
        uv: Vector2d<T>, 
        dp_du: Vector3d<T>, 
        dp_dv: Vector3d<T>, 
        dn_du: Normal3d<T>,
        dn_dv: Normal3d<T>,
        shape: Option<Box<dyn Shape<T>>>,
    ) -> Self {
            let n = Normal3d::from_vec(dp_du.cross(&dp_dv).normalized());
            let adjusted_normal = if shape.is_some() {
                if (shape.as_ref().unwrap().transform_swaps_handedness()) 
                ^ (shape.as_ref().unwrap().reverse_orientation()) {
                    -n
                } else {
                    n
                }
            } else {
                n
            };
            SurfaceInteraction {
            point,
            time,
            p_error,
            wo,
            n: adjusted_normal,
            uv,
            dp_du,
            dp_dv,
            dn_du,
            dn_dv,
            shading: Shading {
                n,
                dp_du,
                dp_dv,
                dn_du,
                dn_dv
            },
            shape
        }
    }
    pub fn set_shading_geometry(
        &mut self, 
        dp_du: Vector3d<T>, 
        dp_dv: Vector3d<T>, 
        dn_du: Normal3d<T>, 
        dn_dv: Normal3d<T>,
        orientation_is_authoritative: bool,
    ) {
        self.shading.n = Normal3d::from_vec(dp_du.cross(&dp_dv).normalized());
        if self.shape.is_some() {
            if (self.shape.as_ref().unwrap().transform_swaps_handedness()) ^ 
            (self.shape.as_ref().unwrap().reverse_orientation()) {
                self.shading.n = -self.shading.n;
            }
        }
        if orientation_is_authoritative {
            self.n = self.n.face_forward(self.shading.n);
        } else{
           self.shading.n = self.shading.n.face_forward(self.n);
        }
        self.shading.dp_du = dp_du;
        self.shading.dp_dv = dp_dv;
        self.shading.dn_du = dn_du;
        self.shading.dn_dv = dn_dv;
    }
}
impl<T: Scalar> Mul<SurfaceInteraction<T>> for Transform<T> {
    type Output = SurfaceInteraction<T>;
    fn mul(self, other: SurfaceInteraction<T>) -> Self::Output {
        SurfaceInteraction {
            point: &self * other.point,
            time: other.time,
            p_error: &self * other.p_error,
            wo: &self * other.wo,
            n: &self * other.n,
            uv: other.uv,
            dp_du: &self * other.dp_du,
            dp_dv: &self * other.dp_dv,
            dn_du: &self * other.dn_du,
            dn_dv: &self * other.dn_dv,
            shading: Shading {
                n: &self * other.shading.n,
                dp_du: &self * other.shading.dp_du,
                dp_dv: &self * other.shading.dp_dv,
                dn_du: &self * other.shading.dn_du,
                dn_dv: &self * other.shading.dn_dv
            },
            shape: other.shape
        }
    }
}
#[test]
fn test_surface_interaction_new_no_shape() {
    let point = Point3d::new(1.0, 2.0, 3.0);
    let time = 1.0;
    let p_error = Vector3d::new(1.0, 2.0, 3.0);
    let wo = Vector3d::new(1.0, 2.0, 3.0);
    let uv = Vector2d::new(1.0, 2.0);
    let dp_du = Vector3d::new(1.0, 0.0, 0.0);
    let dp_dv = Vector3d::new(0.0, 1.0, 0.0);
    let dn_du = Normal3d::new(0.0, 0.0, 1.0);
    let dn_dv = Normal3d::new(0.0, 1.0, 0.0);
    let shape = None;
    let si = SurfaceInteraction::new(point, time, p_error, wo, uv, 
        dp_du, dp_dv, dn_du, dn_dv, shape);
    assert_eq!(si.point, point);
    assert_eq!(si.time, time);
    assert_eq!(si.p_error, p_error);
    assert_eq!(si.wo, wo);
    assert_eq!(si.uv, uv);
    assert_eq!(si.dp_du, dp_du);
    assert_eq!(si.dp_dv, dp_dv);
    assert_eq!(si.dn_du, dn_du);
    assert_eq!(si.dn_dv, dn_dv);
    assert_eq!(si.n, Normal3d::from_vec(dp_du.cross(&dp_dv).normalized()));
    assert_eq!(si.shading.n, Normal3d::from_vec(dp_du.cross(&dp_dv).normalized()));
    assert_eq!(si.shading.dp_du, dp_du);
}
#[test]
fn test_surface_interaction_new_with_shape() {
    struct TestShape {
        transform_swaps_handedness: bool,
        reverse_orientation: bool,
    }
    impl Shape<f64> for TestShape {
        fn transform_swaps_handedness(&self) -> bool {
            self.transform_swaps_handedness
        }
        fn reverse_orientation(&self) -> bool {
            self.reverse_orientation
        }
    }
    let point = Point3d::new(1.0, 2.0, 3.0);
    let time = 1.0;
    let p_error = Vector3d::new(1.0, 2.0, 3.0);
    let wo = Vector3d::new(1.0, 2.0, 3.0);
    let uv = Vector2d::new(1.0, 2.0);
    let dp_du = Vector3d::new(1.0, 0.0, 0.0);
    let dp_dv = Vector3d::new(0.0, 1.0, 0.0);
    let dn_du = Normal3d::new(0.0, 0.0, 1.0);
    let dn_dv = Normal3d::new(0.0, 1.0, 0.0);
    let shape = TestShape {
        transform_swaps_handedness: true,
        reverse_orientation: true,
    };
    let si = SurfaceInteraction::new(point, time, p_error, wo, uv, 
        dp_du, dp_dv, dn_du, dn_dv, Some(Box::new(shape)));
    let expected_n = Normal3d::from_vec(dp_du.cross(&dp_dv).normalized());
    assert_eq!(si.n, expected_n);
    assert_eq!(si.shading.n, expected_n);
    let shape = TestShape {
        transform_swaps_handedness: true,
        reverse_orientation: false,
    };
    let si = SurfaceInteraction::new(point, time, p_error, wo, uv, 
        dp_du, dp_dv, dn_du, dn_dv, Some(Box::new(shape)));
    assert_eq!(si.n, -expected_n);
    assert_eq!(si.shading.n, expected_n);
    let shape = TestShape {
        transform_swaps_handedness: false,
        reverse_orientation: true,
    };
    let si = SurfaceInteraction::new(point, time, p_error, wo, uv, 
        dp_du, dp_dv, dn_du, dn_dv, Some(Box::new(shape)));
    assert_eq!(si.n, -expected_n);
    assert_eq!(si.shading.n, expected_n);
}
#[test]
fn test_set_shading_geometry() {
    let point = Point3d::new(1.0, 2.0, 3.0);
    let time = 1.0;
    let p_error = Vector3d::new(1.0, 2.0, 3.0);
    let wo = Vector3d::new(1.0, 2.0, 3.0);
    let uv = Vector2d::new(1.0, 2.0);
    let dp_du = Vector3d::new(1.0, 0.0, 0.0);
    let dp_dv = Vector3d::new(0.0, 1.0, 0.0);
    let dn_du = Normal3d::new(0.0, 0.0, 1.0);
    let dn_dv = Normal3d::new(0.0, 1.0, 0.0);
    let shape = None;
    let mut si = SurfaceInteraction::new(point, time, p_error, wo, uv, 
        dp_du, dp_dv, dn_du, dn_dv, shape);
    si.set_shading_geometry(dp_du, dp_dv, dn_du, dn_dv, true);
    assert_eq!(si.shading.n, Normal3d::from_vec(dp_du.cross(&dp_dv).normalized()));
    assert_eq!(si.dp_du, dp_du);
    assert_eq!(si.dp_dv, dp_dv);
}
#[test]
fn test_set_shading_geometry_with_shape() {
    struct TestShape {
        transform_swaps_handedness: bool,
        reverse_orientation: bool,
    }
    impl Shape<f64> for TestShape {
        fn transform_swaps_handedness(&self) -> bool {
            self.transform_swaps_handedness
        }
        fn reverse_orientation(&self) -> bool {
            self.reverse_orientation
        }
    }
    let point = Point3d::new(1.0, 2.0, 3.0);
    let time = 1.0;
    let p_error = Vector3d::new(1.0, 2.0, 3.0);
    let wo = Vector3d::new(1.0, 2.0, 3.0);
    let uv = Vector2d::new(1.0, 2.0);
    let dp_du = Vector3d::new(1.0, 0.0, 0.0);
    let dp_dv = Vector3d::new(0.0, 1.0, 0.0);
    let dn_du = Normal3d::new(0.0, 0.0, 1.0);
    let dn_dv = Normal3d::new(0.0, 1.0, 0.0);
    let shape = TestShape {
        transform_swaps_handedness: true,
        reverse_orientation: true,
    };
    let mut si = SurfaceInteraction::new(point, time, p_error, wo, uv, 
        dp_du, dp_dv, dn_du, dn_dv, Some(Box::new(shape)));
    si.set_shading_geometry(dp_du, dp_dv, dn_du, dn_dv, true);
    assert_eq!(si.shading.n, Normal3d::from_vec(dp_du.cross(&dp_dv).normalized()));
    assert_eq!(si.dp_du, dp_du);
    assert_eq!(si.dp_dv, dp_dv);
    let shape = TestShape {
        transform_swaps_handedness: false,
        reverse_orientation: true,
    };
    let mut si = SurfaceInteraction::new(point, time, p_error, wo, uv, 
        dp_du, dp_dv, dn_du, dn_dv, Some(Box::new(shape)));
    si.set_shading_geometry(dp_du, dp_dv, dn_du, dn_dv, true);
    assert_eq!(si.shading.n, -Normal3d::from_vec(dp_du.cross(&dp_dv).normalized()));
    assert_eq!(si.dp_du, dp_du);
    assert_eq!(si.dp_dv, dp_dv);
    let shape = TestShape {
        transform_swaps_handedness: true,
        reverse_orientation: false,
    };
    let mut si = SurfaceInteraction::new(point, time, p_error, wo, uv, 
        dp_du, dp_dv, dn_du, dn_dv, Some(Box::new(shape)));
    si.set_shading_geometry(dp_du, dp_dv, dn_du, dn_dv, true);
    assert_eq!(si.shading.n, -Normal3d::from_vec(dp_du.cross(&dp_dv).normalized()));
    assert_eq!(si.dp_du, dp_du);
    assert_eq!(si.dp_dv, dp_dv);
}
#[test]
fn test_set_shading_geometry_orientation_is_authoritative() {
    let point = Point3d::new(1.0, 2.0, 3.0);
    let time = 1.0;
    let p_error = Vector3d::new(1.0, 2.0, 3.0);
    let wo = Vector3d::new(1.0, 2.0, 3.0);
    let uv = Vector2d::new(1.0, 2.0);
    let dp_du = Vector3d::new(1.0, 0.0, 0.0);
    let dp_dv = Vector3d::new(0.0, 1.0, 0.0);
    let dn_du = Normal3d::new(0.0, 0.0, 1.0);
    let dn_dv = Normal3d::new(0.0, 1.0, 0.0);
    let shape = None;
    let mut si = SurfaceInteraction::new(point, time, p_error, wo, uv, 
        dp_du, dp_dv, dn_du, dn_dv, shape);
    si.set_shading_geometry(dp_du, dp_dv, dn_du, dn_dv, false);
    assert_eq!(si.shading.n, -Normal3d::from_vec(dp_du.cross(&dp_dv).normalized()));
    assert_eq!(si.n, Normal3d::from_vec(dp_du.cross(&dp_dv).normalized()));
    assert_eq!(si.dp_du, dp_du);
    assert_eq!(si.dp_dv, dp_dv);
    let shape = None;
    let mut si = SurfaceInteraction::new(point, time, p_error, wo, uv, 
        dp_du, dp_dv, dn_du, dn_dv, shape);
    si.set_shading_geometry(dp_du, dp_dv, dn_du, dn_dv, true);
    assert_eq!(si.shading.n, Normal3d::from_vec(dp_du.cross(&dp_dv).normalized()));
    assert_eq!(si.n, -Normal3d::from_vec(dp_du.cross(&dp_dv).normalized()));
}
#[test]
fn test_transform_surface_interaction() {
    let point = Point3d::new(1.0, 2.0, 3.0);
    let time = 1.0;
    let p_error = Vector3d::new(1.0, 2.0, 3.0);
    let wo = Vector3d::new(1.0, 2.0, 3.0);
    let uv = Vector2d::new(1.0, 2.0);
    let dp_du = Vector3d::new(1.0, 0.0, 0.0);
    let dp_dv = Vector3d::new(0.0, 1.0, 0.0);
    let dn_du = Normal3d::new(0.0, 0.0, 1.0);
    let dn_dv = Normal3d::new(0.0, 1.0, 0.0);
    let shape = None;
    let si = SurfaceInteraction::new(point, time, p_error, wo, uv, 
        dp_du, dp_dv, dn_du, dn_dv, shape);
    let transform = Transform::translate(Vector3d::new(1.0, 2.0, 3.0));
    let res = transform * si;
    assert_eq!(res.point(), Point3d::new(2.0, 4.0, 6.0));
    assert_eq!(res.time, 1.0);
    assert_eq!(res.p_error, Vector3d::new(1.0, 2.0, 3.0));
    assert_eq!(res.wo, Vector3d::new(1.0, 2.0, 3.0));
    assert_eq!(res.uv, Vector2d::new(1.0, 2.0));
    assert_eq!(res.dp_du, Vector3d::new(1.0, 0.0, 0.0));
    assert_eq!(res.dp_dv, Vector3d::new(0.0, 1.0, 0.0));
    assert_eq!(res.dn_du, Normal3d::new(0.0, 0.0, 1.0));
    assert_eq!(res.dn_dv, Normal3d::new(0.0, 1.0, 0.0));
    assert_eq!(res.shading.n, Normal3d::from_vec(dp_du.cross(&dp_dv).normalized()));
    assert_eq!(res.n, Normal3d::from_vec(dp_du.cross(&dp_dv).normalized()));
    assert_eq!(res.dp_du, dp_du);
}