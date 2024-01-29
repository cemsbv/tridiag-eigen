//! Matrix extensions.

use nalgebra::{Complex, ComplexField, DMatrix, DVector, RealField};
use num_traits::{One, Zero};
use smallvec::SmallVec;

/// How many layers are allocated in the smallvec structures.
const ALLOCATED_LAYERS: usize = 8;

/// Extend the nalgebra DMatrix with some helper functions.
pub trait DMatrixComplexExt {
    /// Create a square diagonal $n$-by-$n$.
    ///
    /// $n$ is the size of the diagonal arary.
    fn from_complex_diagonal_vec(diagonal: Vec<Complex<f64>>) -> DMatrix<Complex<f64>>;

    /// Apply the function on all diagonal elements.
    fn apply_complex_diagonal<F>(&mut self, func: F)
    where
        F: Fn(&Complex<f64>) -> Complex<f64>;

    /// Apply the function on all diagonal elements.
    fn apply_complex_diagonal_enumerated<F>(&mut self, func: F)
    where
        F: Fn(usize, &Complex<f64>) -> Complex<f64>;

    /// Create a square $n$-by-$n$ matrix $\bm{M}^{-1/2}$.
    ///
    /// $n$ is the size of the diagonal arary.
    fn from_complex_diagonal_power_negative_half(
        diagonal: &[Complex<f64>],
    ) -> DMatrix<Complex<f64>> {
        let mut matrix = DMatrix::from_diagonal(&DVector::from_row_slice(diagonal));
        matrix.apply_complex_diagonal(|e| e.sqrt().recip());

        matrix
    }

    /// Create a square $n$-by-$n$ matrix $\bm{M}^{1/2}$.
    ///
    /// $n$ is the size of the diagonal arary.
    /// This is the same as $\sqrt{\bm{M}}$.
    fn from_complex_diagonal_power_half(diagonal: &[Complex<f64>]) -> DMatrix<Complex<f64>> {
        let mut matrix = DMatrix::from_diagonal(&DVector::from_row_slice(diagonal));
        matrix.apply_complex_diagonal(|e| e.sqrt());

        matrix
    }

    /// Create a $n$-by-$m$ matrix with a $1$ for each $m$ position in $n$.
    ///
    /// $m$ is the size of `layers_with_identity`.
    ///
    /// Each element of `layers_with_identity` must be unique and smaller than
    /// $n$.
    fn from_complex_layer_identity(
        n: usize,
        layers_with_identity: SmallVec<[usize; ALLOCATED_LAYERS]>,
    ) -> DMatrix<Complex<f64>>;
}

/// Extend the nalgebra DMatrix with some helper functions.
pub trait DMatrixExt<T: RealField + One + Zero + Copy> {
    /// Create a square diagonal $n$-by-$n$.
    ///
    /// $n$ is the size of the diagonal arary.
    fn from_diagonal_vec(diagonal: Vec<T>) -> DMatrix<T>;

    /// Apply the function on all diagonal elements.
    fn apply_diagonal<F>(&mut self, func: F)
    where
        F: Fn(&T) -> T;

    /// Apply the function on all diagonal elements.
    fn apply_diagonal_enumerated<F>(&mut self, func: F)
    where
        F: Fn(usize, &T) -> T;

    /// Create a square $n$-by-$n$ matrix $\bm{M}^{-1/2}$.
    ///
    /// $n$ is the size of the diagonal arary.
    fn from_diagonal_power_negative_half(diagonal: &[T]) -> DMatrix<T> {
        let mut matrix = DMatrix::from_diagonal(&DVector::from_row_slice(diagonal));
        matrix.apply_diagonal(|e| e.sqrt().recip());

        matrix
    }

    /// Create a square $n$-by-$n$ matrix $\bm{M}^{1/2}$.
    ///
    /// $n$ is the size of the diagonal arary.
    /// This is the same as $\sqrt{\bm{M}}$.
    fn from_diagonal_power_half(diagonal: &[T]) -> DMatrix<T> {
        let mut matrix = DMatrix::from_diagonal(&DVector::from_row_slice(diagonal));
        matrix.apply_diagonal(|e| e.sqrt());

        matrix
    }

    /// Create a $n$-by-$m$ matrix with a $1$ for each $m$ position in $n$.
    ///
    /// $m$ is the size of `layers_with_identity`.
    ///
    /// Each element of `layers_with_identity` must be unique and smaller than
    /// $n$.
    fn from_layer_identity(
        n: usize,
        layers_with_identity: SmallVec<[usize; ALLOCATED_LAYERS]>,
    ) -> DMatrix<T>;
}

impl<T: RealField + One + Zero + Copy> DMatrixExt<T> for DMatrix<T> {
    #[inline(always)]
    fn from_diagonal_vec(diagonal: Vec<T>) -> DMatrix<T> {
        DMatrix::from_diagonal(&DVector::from(diagonal))
    }

    #[inline(always)]
    fn apply_diagonal<F>(&mut self, func: F)
    where
        F: Fn(&T) -> T,
    {
        for i in 0..self.nrows() {
            self[(i, i)] = func(&self[(i, i)]);
        }
    }

    #[inline(always)]
    fn apply_diagonal_enumerated<F>(&mut self, func: F)
    where
        F: Fn(usize, &T) -> T,
    {
        for i in 0..self.nrows() {
            self[(i, i)] = func(i, &self[(i, i)]);
        }
    }

    #[inline(always)]
    fn from_layer_identity(
        n: usize,
        mut layers_with_identity: SmallVec<[usize; ALLOCATED_LAYERS]>,
    ) -> DMatrix<T> {
        // Define the layout of the matrix
        let mut matrix = DMatrix::from_element(n, layers_with_identity.len(), nalgebra::zero());

        // Sort the layers so the matrix will look like an identity matrix
        layers_with_identity.sort_unstable();

        // Fill the layers with a 1 for each identity
        for (index, layer) in layers_with_identity.iter().enumerate() {
            // TODO: handle out of bounds layer numbers
            matrix[(*layer, index)] = nalgebra::one();
        }

        matrix
    }
}

impl DMatrixComplexExt for DMatrix<Complex<f64>> {
    #[inline(always)]
    fn from_complex_diagonal_vec(diagonal: Vec<Complex<f64>>) -> DMatrix<Complex<f64>> {
        DMatrix::from_diagonal(&DVector::from(diagonal))
    }

    #[inline(always)]
    fn apply_complex_diagonal<F>(&mut self, func: F)
    where
        F: Fn(&Complex<f64>) -> Complex<f64>,
    {
        for i in 0..self.nrows() {
            self[(i, i)] = func(&self[(i, i)]);
        }
    }

    #[inline(always)]
    fn apply_complex_diagonal_enumerated<F>(&mut self, func: F)
    where
        F: Fn(usize, &Complex<f64>) -> Complex<f64>,
    {
        for i in 0..self.nrows() {
            self[(i, i)] = func(i, &self[(i, i)]);
        }
    }

    #[inline(always)]
    fn from_complex_layer_identity(
        n: usize,
        mut layers_with_identity: SmallVec<[usize; ALLOCATED_LAYERS]>,
    ) -> DMatrix<Complex<f64>> {
        // Define the layout of the matrix
        let mut matrix = DMatrix::from_element(n, layers_with_identity.len(), nalgebra::zero());

        // Sort the layers so the matrix will look like an identity matrix
        layers_with_identity.sort_unstable();

        // Fill the layers with a 1 for each identity
        for (index, layer) in layers_with_identity.iter().enumerate() {
            // Complex<f64>ODO: handle out of bounds layer numbers
            matrix[(*layer, index)] = nalgebra::one();
        }

        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, Matrix4, Matrix6x3, RowVector3, RowVector4};

    #[test]
    fn test_diagonal_power_half() {
        assert_eq!(
            DMatrix::from_diagonal_power_half(&[1.0, 4.0, 9.0, 16.0]),
            Matrix4::from_rows(&[
                RowVector4::new(1.0, 0.0, 0.0, 0.0),
                RowVector4::new(0.0, 2.0, 0.0, 0.0),
                RowVector4::new(0.0, 0.0, 3.0, 0.0),
                RowVector4::new(0.0, 0.0, 0.0, 4.0),
            ])
        );
    }

    #[test]
    fn test_diagonal_power_negative_half() {
        assert_eq!(
            DMatrix::from_diagonal_power_negative_half(&[1.0, 4.0, 9.0, 16.0]),
            Matrix4::from_rows(&[
                RowVector4::new(1.0, 0.0, 0.0, 0.0),
                RowVector4::new(0.0, 1.0 / 2.0, 0.0, 0.0),
                RowVector4::new(0.0, 0.0, 1.0 / 3.0, 0.0),
                RowVector4::new(0.0, 0.0, 0.0, 1.0 / 4.0),
            ])
        );
    }

    #[test]
    fn test_layer_identity() {
        assert_eq!(
            DMatrix::<f64>::from_layer_identity(6, smallvec::smallvec![1, 2, 3]),
            Matrix6x3::from_rows(&[
                RowVector3::new(0.0, 0.0, 0.0),
                RowVector3::new(1.0, 0.0, 0.0),
                RowVector3::new(0.0, 1.0, 0.0),
                RowVector3::new(0.0, 0.0, 1.0),
                RowVector3::new(0.0, 0.0, 0.0),
                RowVector3::new(0.0, 0.0, 0.0),
            ])
        );
    }
}
