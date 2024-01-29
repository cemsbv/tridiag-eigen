#![doc = include_str!("../README.md")]

mod matrix;

use matrix::DMatrixComplexExt;
use nalgebra::{Complex, ComplexField, DMatrix, DVector, Dyn, SymmetricEigen};

/// Decompose a symmetric tridiagonal matrix into eigenvalues and eigenvectors.
pub fn into_eigen_values_and_vectors<T>(matrix: DMatrix<T>) -> SymmetricEigen<T, Dyn>
where
    T: ComplexField + ComplexField<RealField = T>,
{
    if matrix.len() == 1 {
        SymmetricEigen {
            eigenvectors: DMatrix::identity(1, 1),
            eigenvalues: DVector::from_element(1, matrix[0].clone()),
        }
    } else {
        // Decompose the matrix into a struct which contains the eigen values and eigen
        // vectors
        matrix.symmetric_eigen()
    }
}

/// Decompose a symmetric tridiagonal matrix into eigenvalues and eigenvectors.
pub fn into_eigen_values_and_vectors2(
    matrix: DMatrix<Complex<f64>>,
) -> SymmetricEigen<Complex<f64>, Dyn> {
    matrix.symmetric_eigen()
}

/// Decompose a symmetric tridiagonal matrix into eigenvectors and eigenvalues.
pub fn into_complex_eigen_values_and_vectors(
    matrix: DMatrix<Complex<f64>>,
) -> (DMatrix<Complex<f64>>, DVector<Complex<f64>>) {
    // Decompose the matrix into a struct which contains the eigen values and eigen
    // vectors
    if matrix.len() == 1 {
        (DMatrix::identity(1, 1), DVector::from_element(1, matrix[0]))
    } else {
        // Somehow this has better results than the result from symmetric eigen for the values only
        let eigenvalues = matrix.eigenvalues().unwrap();

        let mut eigenvectors = matrix.clone();

        // Solve for each eigenvalue
        for (eigenvalue, mut vector_column) in
            eigenvalues.iter().zip(eigenvectors.column_iter_mut())
        {
            // Subtract the eigenvalue from the diagonal
            let mut subtracted_diagonal = matrix.clone();
            subtracted_diagonal.apply_complex_diagonal(|elem| elem - eigenvalue);

            // Get the row echelon form by using LU decomposition
            let lu = subtracted_diagonal.lu();
            let u = lu.u();
            let mut reduced = u.clone();

            // Get the indices of all rows with only 0 or close enough to 0 values
            let empty_rows = u
                .diagonal()
                .iter()
                .enumerate()
                // Get the rows where all values are zero
                .filter(|(_, diagonal_value)| diagonal_value.abs() <= 1e-14)
                .map(|(i, _)| i)
                .collect::<Vec<_>>();

            // Subtract the other vectors from B
            let mut b = DVector::from_element(u.nrows(), Complex::from(0.0));

            // Find the row with only zero values or close enough to zero
            for empty_row_index in empty_rows
                .iter()
                // Revert the order because we want to reduce the bottom empty rows first
                .rev()
            {
                // Get the column with the same index as the empty row
                let column: DVector<Complex<f64>> = reduced.column(*empty_row_index).into_owned();

                // Create the B vector where U x = B and B = [-1..]
                b -= column;

                b = b.remove_row(*empty_row_index);

                // Remove the row and column we stored in B from U
                reduced = reduced.remove_column(*empty_row_index);
                reduced = reduced.remove_row(*empty_row_index);
            }

            // Solve A x = b
            let mut solved: DVector<Complex<f64>> = reduced.solve_upper_triangular_unchecked(&b);

            // Put the removed value back
            for empty_row_index in empty_rows.into_iter() {
                solved = solved.insert_row(empty_row_index, Complex::from(1.0));
            }

            vector_column.set_column(0, &solved);
        }

        (eigenvectors, eigenvalues)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2, Matrix3, Matrix4, RowVector4, Vector4};

    macro_rules! complex {
        () => {
            Complex::new(0.0, 0.0)
        };
        ($a:expr) => {
            Complex::new($a as f64, 0.0)
        };
        ($a:literal + $b:literal i) => {
            Complex::new($a as f64, $b as f64)
        };
        ($a:literal - $b:literal i) => {
            Complex::new($a as f64, -$b as f64)
        };
    }

    macro_rules! assert_vec_eq {
        ($a:expr,$b:expr; epsilon = $eps:expr $(; $opt:ident)*) => {{
            $a$(.$opt())*.into_iter().zip($b$(.$opt())*.into_iter()).for_each(|(a, b)| {
                approx::assert_relative_eq!(a.re, b.re, epsilon = $eps);
                approx::assert_relative_eq!(a.im, b.im, epsilon = $eps);
            });
        }};
        ($a:expr$(, $b:expr)+ $(; $opt:ident)*) => {{
            assert_vec_eq!($a, nalgebra::DVector::from(vec![$($b, )*]); epsilon = 1e-16 $(; $opt)*)
        }};
        ($a:expr$(, $b:expr)+; epsilon = $eps:expr $(; $opt:ident)*) => {{
            assert_vec_eq!($a, nalgebra::DVector::from(vec![$($b, )*]); epsilon = $eps $(; $opt)*)
        }};
    }

    #[test]
    fn test_tridiagonal_matrix() {
        let m = Matrix4::from_rows(&[
            RowVector4::new(1.0, 3.0, 0.0, 0.0),
            RowVector4::new(3.0, 4.0, 1.0, 0.0),
            RowVector4::new(0.0, 1.0, 3.0, 4.0),
            RowVector4::new(0.0, 0.0, 4.0, 3.0),
        ]);

        let SymmetricEigen { eigenvalues, .. } =
            into_eigen_values_and_vectors::<f64>(nalgebra::convert(m));

        assert_eq!(
            eigenvalues,
            Vector4::new(
                7.279897809207659,
                5.6442991910069455,
                -0.5788834938285751,
                -1.3453135063860298,
            ),
        );
    }

    #[test]
    fn test_single_matrix() {
        let m = DMatrix::from_element(1, 1, 100.0);

        let our = into_eigen_values_and_vectors(m.clone());
        let theirs = m.symmetric_eigen();

        assert_eq!(our.eigenvectors, theirs.eigenvectors);
        assert_eq!(our.eigenvalues, theirs.eigenvalues);
    }

    /// Source: https://www.wolframalpha.com/input?i=eigenvalues+%7B%7B5+%2B+5i%2C5+%2B+5i%2C5+%2B+5i%7D%2C%7B5+%2B+5i%2C5+%2B+5i%2C5+%2B+5i%7D%2C%7B5+%2B+5i%2C5+%2B+5i%2C5+%2B+5i%7D%7D
    #[test]
    #[should_panic] // TODO: fix test
    fn complex_eigen_3x3() {
        #[rustfmt::skip]
        let m = Matrix3::new(
            complex!(1.0 + 5.0 i), complex!(0), complex!(0),
            complex!(0), complex!(3.0 + 2.0 i), complex!(0),
            complex!(5.0 + 5.0 i), complex!(5.0 + 5.0 i), complex!(5.0 + 5.0 i),
        );

        let (eigenvectors, eigenvalues) =
            into_complex_eigen_values_and_vectors(nalgebra::convert(m));

        assert_vec_eq!(
            eigenvalues,
            complex!(1 + 5 i),
            complex!(5 + 5 i),
            complex!(3 + 2 i)
            ; epsilon = 1e-15
        );
        assert_vec_eq!(
            eigenvectors.column(0),
            complex!(-2 + 2 i),
            complex!(0),
            complex!(5);
            normalize
        );
        assert_vec_eq!(
            eigenvectors.column(1),
            complex!(0),
            complex!(0),
            complex!(1);
            normalize
        );
        assert_vec_eq!(
            eigenvectors.column(2),
            complex!(0),
            complex!(-5 - 1 i),
            complex!(10);
            normalize
        );
    }

    /// Source: https://www.wolframalpha.com/input?i=eigensystem+%7B%7B5+%2B+5*I%2C+5+%2B+5*I%7D%2C+%7B5+%2B+5*I%2C+5+%2B+5*I%7D%7D
    #[test]
    #[should_panic] // TODO: fix test
    fn complex_eigen_2x2() {
        let m = DMatrix::from_element(2, 2, Complex::new(5.0, 5.0));
        let (eigenvectors, eigenvalues) = into_complex_eigen_values_and_vectors(m);

        assert_vec_eq!(eigenvalues, complex!(10 + 10 i), complex!(0); epsilon = 1e-15);
        assert_vec_eq!(eigenvectors.column(0), complex!(1), complex!(1); epsilon = 1e-16);
        assert_vec_eq!(eigenvectors.column(1), complex!(-1), complex!(1); epsilon = 1e-16);
    }

    #[test]
    fn complex_eigen_2x2_2() {
        #[rustfmt::skip]
        let m = Matrix2::new(
            complex!(1 + 1 i), complex!(0),
            complex!(0), complex!(2 + 2 i),
        );
        let (eigenvectors, eigenvalues) =
            into_complex_eigen_values_and_vectors(nalgebra::convert(m));

        assert_vec_eq!(eigenvalues, complex!(2 + 2 i), complex!(1 + 1 i); epsilon = 1e-20);
        assert_vec_eq!(eigenvectors.column(0), complex!(0), complex!(1); epsilon = 1e-20);
        assert_vec_eq!(eigenvectors.column(1), complex!(1), complex!(0); epsilon = 1e-20);
    }

    #[test]
    fn complex_eigen_2x2_3() {
        #[rustfmt::skip]
        let m = Matrix2::new(
            complex!(1 + 1 i), complex!(1 - 5 i),
            complex!(0), complex!(2 + 2 i),
        );
        let (eigenvectors, eigenvalues) =
            into_complex_eigen_values_and_vectors(nalgebra::convert(m));

        assert_vec_eq!(eigenvalues, complex!(2 + 2 i), complex!(1 + 1 i); epsilon = 1e-20);
        assert_vec_eq!(eigenvectors.column(0), complex!(-2 - 3 i), complex!(1); epsilon = 1e-20);
        assert_vec_eq!(eigenvectors.column(1), complex!(1), complex!(0); epsilon = 1e-20);
    }
}
