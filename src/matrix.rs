use std::ops::{Add, Index, IndexMut, Mul, Sub};

/// Type alias for the array of matrix elements.
pub type MatrixArray<const ROWS: usize, const COLS: usize> = [f64; ROWS * COLS];

/// Type alias for a 2x2 matrix.
pub type Matrix2 = Matrix<2, 2>;
/// Type alias for a 3x3 matrix.
pub type Matrix3 = Matrix<3, 3>;
/// Type alias for a 4x4 matrix.
pub type Matrix4 = Matrix<4, 4>;

/// Type alias for a 1-column matrix (row vector).
pub type Vector<const ROWS: usize> = Matrix<ROWS, 1>;

/// Type alias for a 2-row vector.
pub type Vector2 = Vector<2>;
/// Type alias for a 3-row vector.
pub type Vector3 = Vector<3>;
/// Type alias for a 4-row vector.
pub type Vector4 = Vector<4>;

/// A generic constant-size matrix.
pub struct Matrix<const ROWS: usize, const COLS: usize>
where
    [f64; ROWS * COLS]: Sized,
{
    /// The elements of this matrix, stored in row-major order.
    pub elems: MatrixArray<ROWS, COLS>,
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS>
where
    [f64; ROWS * COLS]: Sized,
{
    /// Create and return a matrix with the provided value as every element.
    pub fn from_array(elems: [f64; ROWS * COLS]) -> Self {
        Self { elems }
    }

    /// Create and return a matrix with the provided value as every element.
    pub fn filled(elem: f64) -> Self {
        Self {
            elems: [elem; ROWS * COLS],
        }
    }

    /// Create and return a matrix with every element as `0.0f64`.
    pub fn empty() -> Self {
        Self::filled(0.0)
    }

    /// Retrieve `Some` reference to the element at the provided at the row and column location or
    /// `None` if the provided element is out of this matrix's bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<&f64> {
        Self::index(row, col).map(|index| &self.elems[index])
    }

    /// Retrieve a reference to the element at the provided row and column. This will panic if the
    /// row or column is out of this matrix's bounds.
    pub fn get_unsafe(&self, row: usize, col: usize) -> &f64 {
        &self.elems[Self::index(row, col).unwrap_or_else(|| Self::bounds_panic(row, col))]
    }

    /// Retrieve `Some` mutable reference to the element at the provided at the row and column
    /// location or `None` if the provided element is out of this matrix's bounds.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f64> {
        Self::index(row, col).map(|index| &mut self.elems[index])
    }

    /// Retrieve a mutable reference to the element at the provided row and column. This will panic
    /// if the row or column is out of this matrix's bounds.
    pub fn get_mut_unsafe(&mut self, row: usize, col: usize) -> &mut f64 {
        &mut self.elems[Self::index(row, col).unwrap_or_else(|| Self::bounds_panic(row, col))]
    }

    // Helper function to get the array index of the element at the provided row and column. Returns
    // `None` if the row or column is out of bounds
    fn index(row: usize, col: usize) -> Option<usize> {
        if row < ROWS && col < COLS {
            Some(row * COLS + col)
        } else {
            None
        }
    }

    // Helper function to return a string for out-of-bounds matrix index calls.
    fn bounds_panic(_row: usize, _col: usize) -> ! {
        panic!("matrix index {_row},{_col} out of bounds for matrix of size {ROWS}x{COLS}")
    }
}

// Implement default as a matrix with elements of `0.0f64`
impl<const ROWS: usize, const COLS: usize> Default for Matrix<ROWS, COLS>
where
    [f64; ROWS * COLS]: Sized,
{
    fn default() -> Self {
        Self::empty()
    }
}

// Allow indexing the matrix using a `(row, col)` tuple.
impl<const ROWS: usize, const COLS: usize> Index<(usize, usize)> for Matrix<ROWS, COLS>
where
    [f64; ROWS * COLS]: Sized,
{
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get_unsafe(index.0, index.1)
    }
}

// Allow mutable indexing with a `(row, col)` tuple.
impl<const ROWS: usize, const COLS: usize> IndexMut<(usize, usize)> for Matrix<ROWS, COLS>
where
    [f64; ROWS * COLS]: Sized,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut_unsafe(index.0, index.1)
    }
}

// Implement partial equality
impl<const ROWS: usize, const COLS: usize> PartialEq for Matrix<ROWS, COLS>
where
    [f64; ROWS * COLS]: Sized,
{
    fn eq(&self, other: &Self) -> bool {
        self.elems.eq(&other.elems)
    }
}

// Implement matrix addition
impl<const ROWS: usize, const COLS: usize> Add<Matrix<ROWS, COLS>> for Matrix<ROWS, COLS>
where
    [f64; ROWS * COLS]: Sized,
{
    type Output = Self;

    fn add(self, rhs: Matrix<ROWS, COLS>) -> Self::Output {
        let mut elems = self.elems;
        for i in 0..self.elems.len() {
            elems[i] += rhs.elems[i];
        }
        Self::from_array(elems)
    }
}

// Implement matrix subtraction
impl<const ROWS: usize, const COLS: usize> Sub<Matrix<ROWS, COLS>> for Matrix<ROWS, COLS>
where
    [f64; ROWS * COLS]: Sized,
{
    type Output = Self;

    fn sub(self, rhs: Matrix<ROWS, COLS>) -> Self::Output {
        let mut elems = self.elems;
        for i in 0..self.elems.len() {
            elems[i] -= rhs.elems[i];
        }
        Self::from_array(elems)
    }
}

// Implement matrix scalar multiplication
impl<const ROWS: usize, const COLS: usize> Mul<f64> for Matrix<ROWS, COLS>
where
    [f64; ROWS * COLS]: Sized,
{
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            elems: self.elems.map(|elem| elem * rhs),
        }
    }
}

// Multiplication is implemented for matrices when the left matrix has the same number of columns as
// the right side has rows. Now we have const generics to make this all compile-time!
impl<const A_ROW: usize, const A_COL_B_ROW: usize, const B_COL: usize>
    Mul<Matrix<A_COL_B_ROW, B_COL>> for Matrix<A_ROW, A_COL_B_ROW>
where
    [f64; A_ROW * A_COL_B_ROW]: Sized,
    [f64; A_COL_B_ROW * B_COL]: Sized,
    [f64; A_ROW * B_COL]: Sized,
{
    // The output matrix will have the same number of rows as the left matrix and the same number of
    // columns as the right
    type Output = Matrix<A_ROW, B_COL>;

    fn mul(self, rhs: Matrix<A_COL_B_ROW, B_COL>) -> Self::Output {
        let mut output_matrix = Matrix::empty();

        // Loop through each column for each row in the output matrix
        for row in 0..A_ROW {
            for col in 0..B_COL {
                // Sum up the products of the matrix values
                let mut sum = 0.0;
                for cell in 0..A_COL_B_ROW {
                    sum += self[(row, cell)] * rhs[(cell, col)];
                }
                // And set the output to this
                output_matrix[(row, col)] = sum;
            }
        }

        output_matrix
    }
}
