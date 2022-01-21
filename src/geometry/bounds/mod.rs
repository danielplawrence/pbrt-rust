use std::ops::{Index};

use super::{point::{Point2d, Point3d}, Scalar, vector::{Vector2d, Vector3d}};

#[derive(PartialEq, Debug, Clone)]
pub struct Bounds2d<T> {
    pub min: Point2d<T>,
    pub max: Point2d<T>,
}
impl <T: Scalar> Bounds2d<T> {
    pub fn new(min: Point2d<T>, max: Point2d<T>) -> Self {
        Bounds2d {
            min,
            max
        }
    }
    pub fn new_from_point(p: Point2d<T>) -> Self {
        Bounds2d {
            min: p,
            max: p
        }
    }
    pub fn default() -> Self {
        Bounds2d {
            min: Point2d{x: T::inf(), y: T::inf()},
            max: Point2d{x: -T::inf(), y: -T::inf()}
        }
    }
    pub fn intersection(&self, other: &Self) -> Self {
        Bounds2d {
            min: self.min.max(other.min),
            max: self.max.min(other.max)
        }
    }
    pub fn overlaps(&self, other: &Self) -> bool {
        let x = (self.max.x >= other.min.x) && (self.min.x <= other.max.x);
        let y = (self.max.y >= other.min.y) && (self.min.y <= other.max.y);
        return x && y;
    }
    pub fn contains(&self, p: &Point2d<T>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }
    pub fn contains_exclusive(&self, p: &Point2d<T>) -> bool {
        p.x >= self.min.x && p.x < self.max.x && p.y >= self.min.y && p.y < self.max.y
    }
    pub fn expand(&self, p: T) -> Self {
        Bounds2d {
            min: self.min - Vector2d::new(p, p),
            max: self.max + Vector2d::new(p, p)
        }
    }
    pub fn diagonal(&self) -> Vector2d<T> {
        self.max - self.min
    }
    pub fn area(&self) -> T {
        let d = self.diagonal();
        d.x * d.y
    }
    pub fn maximum_extent(&self) -> usize {
        let d = self.diagonal();
        if d.x > d.y {
            0
        } else {
            1
        }
    }
    pub fn lerp(&self, p: Point2d<T>) -> Point2d<T> {
        Point2d {
            x: self.min.x + (self.diagonal().x * p.x),
            y: self.min.y + (self.diagonal().y * p.y)
        }
    }
    pub fn offset(&self, p: &Point2d<T>) -> Vector2d<T> {
        let mut o = *p - self.min;
        if self.max.x > self.min.x {
            o.x = o.x / self.max.x - self.min.x;
        }
        if self.max.y > self.min.y {
            o.y = o.y / self.max.y - self.min.y
        };
        return o;
    }
    pub fn bounding_sphere(&self) -> (Point2d<T>, T) {
        let center = (self.min + self.max) / T::two();
        let radius = if self.contains(&center) {
            center.distance(self.max)
         } else {T::zero()};
        return (center, radius);
    }
    pub fn corner(&self, i: usize) -> Point2d<T> {
        match i {
            0 => self.min,
            1 => Point2d{x: self.max.x, y: self.min.y},
            2 => self.max,
            3 => Point2d{x: self.min.x, y: self.max.y},
            _ => panic!("Invalid corner index")
        }
    }

}
trait Union<T> {
    fn union(&self, other: &T) -> Self;
}
impl<T: Scalar> Union<Point2d<T>> for Bounds2d<T> {
    fn union(&self, other: &Point2d<T>) -> Self {
        Bounds2d {
            min: Point2d {
                x: self.min.x.min(other.x),
                y: self.min.y.min(other.y)
            },
            max: Point2d {
                x: self.max.x.max(other.x),
                y: self.max.y.max(other.y)
            }
        }
    }
}
impl<T: Scalar> Union<Bounds2d<T>> for Bounds2d<T> {
    fn union(&self, other: &Bounds2d<T>) -> Self {
        Bounds2d {
            min: Point2d {
                x: self.min.x.min(other.min.x),
                y: self.min.y.min(other.min.y)
            },
            max: Point2d {
                x: self.max.x.max(other.max.x),
                y: self.max.y.max(other.max.y)
            }
        }
    }
}
impl<T: Scalar> Index<usize> for Bounds2d<T> {
    type Output = Point2d<T>;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.min,
            1 => &self.max,
            _ => panic!("Index out of bounds")
        }
    }
} 
#[derive(PartialEq, Debug, Clone)]
pub struct Bounds3d<T> {
    pub min: Point3d<T>,
    pub max: Point3d<T>,
}
impl<T: Scalar> Bounds3d<T> {
    pub fn new(p1: Point3d<T>, p2: Point3d<T>) -> Self {
        let min = Point3d::<T>::new(
            p1.x.min(p2.x),
            p1.y.min(p2.y),
            p1.z.min(p2.z)
        );
        let max = Point3d::<T>::new(
            p1.x.max(p2.x),
            p1.y.max(p2.y),
            p1.z.max(p2.z)
        );
        Bounds3d {
            min,
            max
        }
    }
    pub fn new_from_point(p: Point3d<T>) -> Self {
        Bounds3d {
            min: p,
            max: p
        }
    }
    pub fn default() -> Self {
        Bounds3d {
            min: Point3d{x: T::inf(), y: T::inf(), z: T::inf()},
            max: Point3d{x: -T::inf(), y: -T::inf(), z: -T::inf()}
        }
    }
    pub fn intersection(&self, other: &Self) -> Self {
        Bounds3d {
            min: self.min.max(other.min),
            max: self.max.min(other.max)
        }
    }
    pub fn overlaps(&self, other: &Self) -> bool {
        let x = (self.max.x >= other.min.x) && (self.min.x <= other.max.x);
        let y = (self.max.y >= other.min.y) && (self.min.y <= other.max.y);
        let z = (self.max.z >= other.min.z) && (self.min.z <= other.max.z);
        return x && y && z;
    }
    pub fn contains(&self, p: &Point3d<T>) -> bool {
        return p.x >= self.min.x && p.x <= self.max.x &&
            p.y >= self.min.y && p.y <= self.max.y &&
            p.z >= self.min.z && p.z <= self.max.z;
    }
    pub fn contains_exclusive(&self, p: &Point3d<T>) -> bool {
        return p.x >= self.min.x && p.x < self.max.x &&
        p.y >= self.min.y && p.y < self.max.y &&
        p.z >= self.min.z && p.z < self.max.z;
    }
    pub fn expand(&self, s: T) -> Self {
        Bounds3d {
            min: self.min - Vector3d::new(s, s, s),
            max: self.max + Vector3d::new(s, s, s)
        }
    }
    pub fn diagonal(&self) -> Vector3d<T> {
        self.max - self.min
    }
    pub fn surface_area(&self) -> T {
        let d = self.diagonal();
        (d.x * d.y + d.x * d.z + d.y * d.z) + (d.x * d.y + d.x * d.z + d.y * d.z)
    }
    pub fn volume(&self) -> T {
        let d = self.diagonal();
        d.x * d.y * d.z
    }
    pub fn maximum_extent(&self) -> usize {
        let xlen = self.max.x - self.min.x;
        let ylen = self.max.y - self.min.y;
        let zlen = self.max.z - self.min.z;
        if xlen > ylen && xlen > zlen {
            0
        } else if ylen > zlen {
            1
        } else {
            2
        }
    }
    pub fn lerp(&self, p: Point3d<T>) -> Point3d<T> {
        Point3d::new(
            self.min.x + p.x * (self.max.x - self.min.x),
            self.min.y + p.y * (self.max.y - self.min.y),
            self.min.z + p.z * (self.max.z - self.min.z)
        )
    }
    pub fn offset(&self, p: &Point3d<T>) -> Vector3d<T> {
        let mut o = *p - self.min;
        if self.max.x > self.min.x {
            o.x = o.x / self.max.x - self.min.x;
        }
        if self.max.y > self.min.y {
            o.y = o.y / self.max.y - self.min.y
        };
        if self.max.z > self.min.z {
            o.z = o.z / self.max.z - self.min.z
        };
        return o;
    }
    pub fn bounding_sphere(&self) -> (Point3d<T>, T) {
        let center = (self.min + self.max) / T::two();
        let radius = if self.contains(&center) {
            center.distance(self.max)
         } else {T::zero()};
        return (center, radius);
    }
    pub fn corner(&self, index: usize) -> Point3d<T> {
        match index {
            0 => self.min,
            1 => Point3d::new(self.max.x, self.min.y, self.min.z),
            2 => Point3d::new(self.min.x, self.max.y, self.min.z),
            3 => Point3d::new(self.min.x, self.min.y, self.max.z),
            4 => Point3d::new(self.max.x, self.max.y, self.min.z),
            5 => Point3d::new(self.max.x, self.min.y, self.max.z),
            6 => Point3d::new(self.min.x, self.max.y, self.max.z),
            7 => self.max,
            _ => panic!("Invalid corner index")
        }
    }
}

impl<T: Scalar> Union<Point3d<T>> for Bounds3d<T> {
    fn union(&self, other: &Point3d<T>) -> Self {
        Bounds3d {
            min: Point3d {
                x: self.min.x.min(other.x),
                y: self.min.y.min(other.y),
                z: self.min.z.min(other.z)
            },
            max: Point3d {
                x: self.max.x.max(other.x),
                y: self.max.y.max(other.y),
                z: self.max.z.max(other.z)
            }
        }
    }
}
impl<T: Scalar> Union<Bounds3d<T>> for Bounds3d<T> {
    fn union(&self, other: &Bounds3d<T>) -> Self {
        Bounds3d {
            min: Point3d {
                x: self.min.x.min(other.min.x),
                y: self.min.y.min(other.min.y),
                z: self.min.z.min(other.min.z)
            },
            max: Point3d {
                x: self.max.x.max(other.max.x),
                y: self.max.y.max(other.max.y),
                z: self.max.z.max(other.max.z)
            }
        }
    }
}
impl<T: Scalar> Index<usize> for Bounds3d<T> {
    type Output = Point3d<T>;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.min,
            1 => &self.max,
            _ => panic!("Index out of bounds")
        }
    }
} 
#[test]
fn test_bounds_2d() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    assert_eq!(b.min.x, 1.0);
    assert_eq!(b.min.y, 2.0);
    assert_eq!(b.max.x, 3.0);
    assert_eq!(b.max.y, 4.0);
}
#[test]
fn test_bounds_2d_default() {
    let b = Bounds2d::<f64>::default();
    assert_eq!(b.min.x, f64::inf());
    assert_eq!(b.min.y, f64::inf());
    assert_eq!(b.max.x, -f64::inf());
    assert_eq!(b.max.y, -f64::inf());
}
#[test]
fn test_bounds_2d_new_from_point() {
    let b = Bounds2d::new_from_point(Point2d{x: 1.0, y: 2.0});
    assert_eq!(b.min.x, 1.0);
    assert_eq!(b.min.y, 2.0);
    assert_eq!(b.max.x, 1.0);
    assert_eq!(b.max.y, 2.0);
}
#[test]
fn test_bounds_2d_index() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    assert_eq!(b[0].x, 1.0);
    assert_eq!(b[0].y, 2.0);
    assert_eq!(b[1].x, 3.0);
    assert_eq!(b[1].y, 4.0);
}
#[test]
#[should_panic]
fn test_bounds_2d_index_panic() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    b[2];
}
#[test]
fn test_bounds_2d_union_point() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    let b2 = b.union(&Point2d{x: 5.0, y: 6.0});
    assert_eq!(b2.min.x, 1.0);
    assert_eq!(b2.min.y, 2.0);
    assert_eq!(b2.max.x, 5.0);
    assert_eq!(b2.max.y, 6.0);
}
#[test]
fn test_bounds_2d_union_bounds() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    let b2 = b.union(&Bounds2d::new(Point2d{x: 5.0, y: 6.0}, Point2d{x: 7.0, y: 8.0}));
    assert_eq!(b2.min.x, 1.0);
    assert_eq!(b2.min.y, 2.0);
    assert_eq!(b2.max.x, 7.0);
    assert_eq!(b2.max.y, 8.0);
}
#[test]
fn test_bounds_2d_overlaps() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    let b2 = Bounds2d::new(Point2d{x: 5.0, y: 6.0}, Point2d{x: 7.0, y: 8.0});
    assert!(!b.overlaps(&b2));
    let b3 = Bounds2d::new(Point2d{x: 5.0, y: 6.0}, Point2d{x: 7.0, y: 8.0});
    assert!(b2.overlaps(&b3));
}
#[test]
fn test_bounds_2d_intersection() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    let b2 = b.intersection(&Bounds2d::new(Point2d{x: 5.0, y: 6.0}, Point2d{x: 7.0, y: 8.0}));
    assert_eq!(b2.min.x, 5.0);
    assert_eq!(b2.min.y, 6.0);
    assert_eq!(b2.max.x, 3.0);
    assert_eq!(b2.max.y, 4.0);
}
#[test]
fn test_bounds_2d_contains() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    assert!(b.contains(&Point2d{x: 2.0, y: 3.0}));
    assert!(!b.contains(&Point2d{x: 4.0, y: 3.0}));
}
#[test]
fn test_bounds_2d_contains_exclusive() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    assert!(b.contains_exclusive(&Point2d{x: 2.0, y: 3.0}));
    assert!(!b.contains_exclusive(&Point2d{x: 4.0, y: 3.0}));
}
#[test]
fn test_bounds_2d_expand() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    let b2 = b.expand(1.0);
    assert_eq!(b2.min.x, 0.0);
    assert_eq!(b2.min.y, 1.0);
    assert_eq!(b2.max.x, 4.0);
    assert_eq!(b2.max.y, 5.0);
}
#[test]
fn test_bounds_2d_diagonal() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    assert_eq!(b.diagonal(), Vector2d{x: 2.0, y: 2.0});
}
#[test]
fn test_bounds_2d_area() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    assert_eq!(b.area(), 4.0);
}
#[test]
fn test_bounds_2d_maximum_extent() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    assert_eq!(b.maximum_extent(), 1);
}
#[test]
fn test_bounds_2d_lerp() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 2.0}, Point2d{x: 3.0, y: 4.0});
    assert_eq!(b.lerp(Point2d{x: 0.5, y: 0.5}), Point2d{x: 2.0, y: 3.0});
}
#[test]
fn test_bounds_2d_offset() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 1.0}, Point2d{x: 5.0, y: 5.0});
    let b2 = b.offset(&Point2d{x: 1.0, y: 1.0});
    assert_eq!(b2, Vector2d{x: -1.0, y: -1.0});
}
#[test]
fn test_bounds_2d_bounding_sphere() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 1.0}, Point2d{x: 5.0, y: 5.0});
    let (b, r) = b.bounding_sphere();
    assert_eq!(b, Point2d{x: 3.0, y: 3.0});
    assert_eq!(r, 2.8284271247461903);
}
#[test]
fn test_bounds_2d_corner() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 1.0}, Point2d{x: 5.0, y: 5.0});
    assert_eq!(b.corner(0), Point2d{x: 1.0, y: 1.0});
    assert_eq!(b.corner(1), Point2d{x: 5.0, y: 1.0});
    assert_eq!(b.corner(2), Point2d{x: 5.0, y: 5.0});
    assert_eq!(b.corner(3), Point2d{x: 1.0, y: 5.0});
}
#[test]
#[should_panic]
fn test_bounds_2d_corner_panic() {
    let b = Bounds2d::new(Point2d{x: 1.0, y: 1.0}, Point2d{x: 5.0, y: 5.0});
    b.corner(4);
}
#[test]
fn test_bounds3d() {
    let b = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 4.0, y: 5.0, z: 6.0});
    assert_eq!(b.min.x, 1.0);
    assert_eq!(b.min.y, 2.0);
    assert_eq!(b.min.z, 3.0);
    assert_eq!(b.max.x, 4.0);
    assert_eq!(b.max.y, 5.0);
    assert_eq!(b.max.z, 6.0);
}
#[test]
fn test_bounds3d_default() {
    let b = Bounds3d::<f64>::default();
    assert_eq!(b.min.x, f64::inf());
    assert_eq!(b.min.y, f64::inf());
    assert_eq!(b.min.z, f64::inf());
    assert_eq!(b.max.x, -f64::inf());
    assert_eq!(b.max.y, -f64::inf());
    assert_eq!(b.max.z, -f64::inf());
}
#[test]
fn test_bounds3d_new_from_point() {
    let b = Bounds3d::new_from_point(Point3d{x: 1.0, y: 2.0, z: 3.0});
    assert_eq!(b.min.x, 1.0);
    assert_eq!(b.min.y, 2.0);
    assert_eq!(b.min.z, 3.0);
    assert_eq!(b.max.x, 1.0);
    assert_eq!(b.max.y, 2.0);
    assert_eq!(b.max.z, 3.0);
}
#[test]
fn test_bounds3d_index() {
    let b = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 4.0, y: 5.0, z: 6.0});
    assert_eq!(b[0].x, 1.0);
    assert_eq!(b[0].y, 2.0);
    assert_eq!(b[0].z, 3.0);
    assert_eq!(b[1].x, 4.0);
    assert_eq!(b[1].y, 5.0);
    assert_eq!(b[1].z, 6.0);
}
#[test]
#[should_panic]
fn test_bounds3d_index_panic() {
    let b = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 4.0, y: 5.0, z: 6.0});
    b[2];
}
#[test]
fn test_bounds3d_union_point() {
    let b = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 4.0, y: 5.0, z: 6.0});
    let b2 = b.union(&Point3d{x: 7.0, y: 8.0, z: 9.0});
    assert_eq!(b2.min.x, 1.0);
    assert_eq!(b2.min.y, 2.0);
    assert_eq!(b2.min.z, 3.0);
    assert_eq!(b2.max.x, 7.0);
    assert_eq!(b2.max.y, 8.0);
    assert_eq!(b2.max.z, 9.0);
}
#[test]
fn test_bounds3d_union_bounds() {
    let b = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 4.0, y: 5.0, z: 6.0});
    let b2 = b.union(&Bounds3d::new(Point3d{x: 7.0, y: 8.0, z: 9.0}, Point3d{x: 10.0, y: 11.0, z: 12.0}));
    assert_eq!(b2.min.x, 1.0);
    assert_eq!(b2.min.y, 2.0);
    assert_eq!(b2.min.z, 3.0);
    assert_eq!(b2.max.x, 10.0);
    assert_eq!(b2.max.y, 11.0);
    assert_eq!(b2.max.z, 12.0);
}
#[test]
fn test_bounds_3d_intersection() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 4.0, y: 5.0, z: 6.0});
    let b = Bounds3d::new(Point3d{x: 7.0, y: 8.0, z: 9.0}, Point3d{x: 10.0, y: 11.0, z: 12.0});
    let c = a.intersection(&b);
    assert_eq!(c.min.x, 7.0);
    assert_eq!(c.min.y, 8.0);
    assert_eq!(c.min.z, 9.0);
    assert_eq!(c.max.x, 4.0);
    assert_eq!(c.max.y, 5.0);
    assert_eq!(c.max.z, 6.0);
}
#[test]
fn test_bounds_3d_overlaps() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 4.0, y: 5.0, z: 6.0});
    let b = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 10.0, y: 11.0, z: 12.0});
    assert!(a.overlaps(&b));
    assert!(b.overlaps(&a));
    let c = Bounds3d::new(Point3d{x: -1.0, y: -2.0, z: -3.0}, Point3d{x: -4.0, y: -5.0, z: -6.0});
    assert!(!a.overlaps(&c));
    assert!(!c.overlaps(&a));
}
#[test]
fn test_bounds_3d_contains() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 4.0, y: 5.0, z: 6.0});
    assert!(a.contains(&Point3d{x: 2.0, y: 3.0, z: 4.0}));
    assert!(!a.contains(&Point3d{x: 5.0, y: 6.0, z: 7.0}));
}
#[test]
fn test_bounds_3d_contains_exclusive() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 4.0, y: 5.0, z: 6.0});
    assert!(a.contains_exclusive(&Point3d{x: 2.0, y: 3.0, z: 4.0}));
    assert!(!a.contains_exclusive(&Point3d{x: 5.0, y: 6.0, z: 7.0}));
}
#[test]
fn test_bounds_3d_diagonal() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 4.0, y: 5.0, z: 6.0});
    assert_eq!(a.diagonal().x, 3.0);
    assert_eq!(a.diagonal().y, 3.0);
    assert_eq!(a.diagonal().z, 3.0);
}
#[test]
fn test_bounds_3d_surface_area() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 4.0, y: 5.0, z: 6.0});
    assert_eq!(a.surface_area(), 54.0);
}
#[test]
fn test_bounds_3d_volume() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 4.0, y: 5.0, z: 6.0});
    assert_eq!(a.volume(), 27.0);
}
#[test]
fn test_bounds_3d_maximum_extent() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 2.0, z: 3.0}, Point3d{x: 4.0, y: 5.0, z: 6.0});
    assert_eq!(a.maximum_extent(), 2);
}
#[test]
fn test_bounds_3d_lerp() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 1.0, z: 1.0}, Point3d{x: 2.0, y: 2.0, z: 2.0});
    let b = a.lerp(Point3d::new(0.5, 0.5, 0.5));
    assert_eq!(b.x, 1.5);
    assert_eq!(b.y, 1.5);
    assert_eq!(b.z, 1.5);
}
#[test]
fn test_bounds_3d_offset() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 1.0, z: 1.0}, Point3d{x: 5.0, y: 5.0, z: 5.0});
    let b = a.offset(&Point3d{x: 1.0, y: 1.0, z: 1.0});
    assert_eq!(b, Vector3d{x: -1.0, y: -1.0, z: -1.0});
}
#[test]
fn test_bounds_3d_expand() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 1.0, z: 1.0}, Point3d{x: 5.0, y: 5.0, z: 5.0});
    let b = a.expand(1.0);
    assert_eq!(b, Bounds3d::new(Point3d{x: 0.0, y: 0.0, z: 0.0}, Point3d{x: 6.0, y: 6.0, z: 6.0}));
}
#[test]
fn test_bounds_3d_bounding_sphere() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 1.0, z: 1.0}, Point3d{x: 2.0, y: 2.0, z: 2.0});
    let (b, r) = a.bounding_sphere();
    assert_eq!(b.x, 1.5);
    assert_eq!(b.y, 1.5);
    assert_eq!(b.z, 1.5);
    assert_eq!(r, 0.8660254037844386);
}
#[test]
fn test_bounds_3d_corner() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 1.0, z: 1.0}, Point3d{x: 5.0, y: 5.0, z: 5.0});
    assert_eq!(a.corner(0), Point3d{x: 1.0, y: 1.0, z: 1.0});
    assert_eq!(a.corner(1), Point3d{x: 5.0, y: 1.0, z: 1.0});
    assert_eq!(a.corner(2), Point3d{x: 1.0, y: 5.0, z: 1.0});
    assert_eq!(a.corner(3), Point3d{x: 1.0, y: 1.0, z: 5.0});
    assert_eq!(a.corner(4), Point3d{x: 5.0, y: 5.0, z: 1.0});
    assert_eq!(a.corner(5), Point3d{x: 5.0, y: 1.0, z: 5.0});
    assert_eq!(a.corner(6), Point3d{x: 1.0, y: 5.0, z: 5.0});
    assert_eq!(a.corner(7), Point3d{x: 5.0, y: 5.0, z: 5.0});
}
#[test]
#[should_panic]
fn test_bounds_3d_corner_panic() {
    let a = Bounds3d::new(Point3d{x: 1.0, y: 1.0, z: 1.0}, Point3d{x: 5.0, y: 5.0, z: 5.0});
    a.corner(8);
}