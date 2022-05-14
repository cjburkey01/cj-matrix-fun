#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod matrix;

pub use matrix::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_array_sanity() {
        let matrix = Matrix2::empty();
        assert_eq!([0.0; 4], matrix.elems);
    }

    #[test]
    fn matrix_array_index() {
        // Matrix of the form:
        // 0.5  10.0
        // 0.0   2.0
        let matrix = Matrix2::from_array([0.5, 10.0, 0.0, 2.0]);

        // Check that indexing works
        assert_eq!(0.5, matrix[(0, 0)]);
        assert_eq!(10.0, matrix[(0, 1)]);
        assert_eq!(0.0, matrix[(1, 0)]);
        assert_eq!(2.0, matrix[(1, 1)]);
    }

    #[test]
    fn matrix_mul() {
        // Matrix of the form:
        // 0.1    92.3
        // 653.0   2.0
        let a = Matrix2::from_array([0.1, 92.3, 653.0, 2.0]);
        // Matrix of the form:
        // 29.0  0.2
        //  9.2  1.2
        let b = Matrix2::from_array([29.0, 0.2, 9.2, 1.2]);

        // Multiply and assert
        let c = a * b;
        assert_eq_array_epsilon(&[852.06, 110.78, 18955.4, 133.0], &c.elems, 0.0001);
    }

    #[test]
    fn matrix_size_mul() {
        // Matrix of the form:
        // 19.3  193.0  12.0
        // 10.2  19.3    1.0
        // 0.2   10.0    8.1
        let a = Matrix3::from_array([19.3, 193.0, 12.0, 10.2, 19.3, 1.0, 0.2, 10.0, 8.1]);
        // Vector of the form:
        // 1.9
        // 1.1
        // 0.5
        let b = Vector3::from_array([1.9, 1.1, 0.5]);

        // Multiply and assert
        let c = a * b;
        assert_eq_array_epsilon(&[254.97, 41.11, 15.43], &c.elems, 0.0001);
    }

    #[test]
    fn matrix_scalar_mul() {
        // Matrix of the form:
        // 1.0  2.0
        // 3.0  4.0
        let a = Matrix2::from_array([1.0, 2.0, 3.0, 4.0]);
        // Scalar
        let b = 2.0;

        // Multiply and assert
        let c = a * b;
        assert_eq_array_epsilon(&[2.0, 4.0, 6.0, 8.0], &c.elems, 0.0001);
    }

    // Utility function to check if the given arrays' values are within `epsilon` of each other and
    // panic if they aren't
    fn assert_eq_array_epsilon<const LEN: usize>(a: &[f64; LEN], b: &[f64; LEN], epsilon: f64) {
        for i in 0..LEN {
            let aa = a[i];
            let bb = b[i];
            if (aa - bb).abs() > epsilon {
                panic!(
                    "Expected {aa} but got {bb} on index {i} which exceeded epsilon of {epsilon}"
                );
            }
        }
    }
}
