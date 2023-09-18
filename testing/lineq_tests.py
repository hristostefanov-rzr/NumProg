import unittest
import numpy as np

from lineq.gauss import gauss_elimination, forward_substitution, backward_substitution
from lineq.lu_decomposition import lu_solve, lu_decomposition, l_backward_substitution
from lineq.qr_decomposition import qr_solve, qr_decomposition, qr_linear_least_squares


class TestGaussElimination(unittest.TestCase):
    def test_forward_substitution(self):
        A = np.array([[2, 1, -1], [0, 3, 2], [0, 0, 1]])
        b = np.array([2, 3, 1])
        A_result, b_result = forward_substitution(A, b)
        self.assertTrue(np.allclose(A_result, np.triu(A_result)))
        self.assertTrue(np.allclose(b, b_result))

    def test_backward_substitution(self):
        A = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]])
        b = np.array([2, 3, 1])
        x = backward_substitution(A, b)
        expected_x = np.array([1.0, 1.0, 1.0])
        self.assertTrue(np.allclose(x, expected_x))

    def test_gauss_elimination(self):
        A = np.array([[2, 1, -1], [0, 3, 2], [0, 0, 1]])
        b = np.array([2, 5, 1])
        x = gauss_elimination(A, b)
        expected_x = np.array([1, 1, 1])
        self.assertTrue(np.allclose(x, expected_x))

    def test_backward_substitution_singular_matrix(self):
        A = np.array([[0, 1], [0, 0]])
        b = np.array([2, 0])
        with self.assertRaises(Exception):
            backward_substitution(A, b)

    def test_gauss_elimination_singular_matrix(self):
        A = np.array([[0, 1], [0, 0]])
        b = np.array([2, 0])
        with self.assertRaises(Exception):
            gauss_elimination(A, b)

    def test_forward_substitution_zero_diagonal(self):
        A = np.array([[0, 1], [1, 0]])
        b = np.array([2, 3])
        A_result, b_result = forward_substitution(A, b)
        self.assertTrue(np.allclose(A_result, np.triu(A_result)))
        self.assertTrue(np.allclose(np.array([3, 2]), b_result))

    def test_backward_substitution_large_values(self):
        A = np.array([[1000, 0], [0, 1000000]])
        b = np.array([2000, 2000000])
        x = backward_substitution(A, b)
        expected_x = np.array([2, 2])
        self.assertTrue(np.allclose(x, expected_x))

    def test_gauss_elimination_small_values(self):
        A = np.array([[0.001, 0.002], [0.003, 0.004]])
        b = np.array([0.005, 0.011])
        x = gauss_elimination(A, b)
        expected_x = np.array([1.0, 2.0])
        self.assertTrue(np.allclose(x, expected_x))

    def test_forward_substitution_large_matrix(self):
        A = np.eye(100)
        b = np.ones(100)
        A_result, b_result = forward_substitution(A, b)
        self.assertTrue(np.allclose(A_result, np.triu(A_result)))
        self.assertTrue(np.allclose(b, b_result))

    def test_gauss_elimination_random_cases(self):
        for _ in range(5):
            n = np.random.randint(500, 1000)
            A = np.random.rand(n, n)
            b = np.random.rand(n)
            x = gauss_elimination(A, b)

            self.assertTrue(np.allclose(np.dot(A, x), b, rtol=1e-6, atol=1e-6))

    def test_gauss_elimination_zero_diagonal(self):
        A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        b = np.array([2, 3, 0])
        with self.assertRaises(Exception):
            gauss_elimination(A, b)


class TestLUDecomposition(unittest.TestCase):
    def test_lu_decomposition_identity_matrix(self):
        # Test lu_decomposition with an identity matrix
        A = np.eye(4)
        L, U = lu_decomposition(A)
        self.assertTrue(np.allclose(L, np.eye(4)))  # L should be an identity matrix
        self.assertTrue(np.allclose(U, A))  # U should be equal to the input matrix A

    def test_lu_decomposition_singular_matrix(self):
        # Test lu_decomposition with a singular matrix
        A = np.array([[1, 2, 3], [1, 2, 3], [7, 8, 9]])
        L, U = lu_decomposition(A)
        print(L, U)
        print(np.dot(L, U))
        with self.assertRaises(Exception):
            lu_decomposition(A)  # Should raise an exception for singular matrix

    def test_l_backward_substitution(self):
        # Test l_backward_substitution with a lower triangular matrix
        L = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
        b = np.array([1, 2, 4])
        x = l_backward_substitution(L, b)
        expected_x = np.array([1, 0, 0])
        self.assertTrue(
            np.allclose(x, expected_x)
        )  # Check if x is the expected solution

    def test_lu_solve_identity_matrix(self):
        # Test lu_solve with an identity matrix
        A = np.eye(4)
        b = np.array([1, 2, 3, 4])
        x = lu_solve(A, b)
        expected_x = b
        self.assertTrue(
            np.allclose(x, expected_x)
        )  # Check if x is equal to the input b

    def test_lu_solve_random_cases(self):
        # Test lu_solve with random matrices and vectors
        for _ in range(5):
            n = np.random.randint(3, 10)  # Random matrix size between 3 and 10
            A = np.random.rand(n, n)  # Random matrix with values between 0 and 1
            b = np.random.rand(n)  # Random vector with values between 0 and 1
            x = lu_solve(A, b)

            # Check if the solution satisfies Ax = b within a tolerance
            self.assertTrue(np.allclose(np.dot(A, x), b, rtol=1e-6, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
