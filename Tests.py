# Tests
import numpy as np

from Warping import logSe3Mapping, expSe3Mapping, poseConcatenation

from external import SE3_Exp

def testLieExpWorksAsExpected():
	xi = np.array([
		[1],
		[1],
		[1],
		[1],
		[1],
		[1]
	])

	matrix_1 = expSe3Mapping(xi)
	matrix_2 = SE3_Exp(xi)
	print("testLieExpWorksAsExpected - Works as expected: ", np.array_equal(matrix_1, matrix_2))


def testLie():
	xi = np.array([
		[0.5],
		[0.5],
		[1.],
		[1.],
		[0.5],
		[0.5]
	])

	matrix = expSe3Mapping(xi)
	xi_2 = logSe3Mapping(matrix)

	print("testLie - Works as expected: ", np.allclose(xi, xi_2))


def testPoseConcatenation():
	matrix_1 = np.array([
		[0, 0, 1, 1],
		[0, 1, 0, 0],
		[-1, 0, 0, 0],
		[0, 0, 0, 1]
	])

	matrix_2 =  np.array([
		[0, 1, 0, 0],
		[-1, 0, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	])

	matrix_3 = np.matmul(matrix_1, matrix_2)

	xi_1 = logSe3Mapping(matrix_1)
	xi_2 = logSe3Mapping(matrix_2)

	xi_result = poseConcatenation(xi_1, xi_2)

	xi_3 = logSe3Mapping(matrix_3)

	matrix_result = expSe3Mapping(xi_result)

	print("testPoseConcatenation - Works as expected", (np.allclose(xi_3, xi_result) and np.allclose(matrix_result, matrix_3)))


def testsWarping():
	testLieExpWorksAsExpected()
	testLie()
	testPoseConcatenation()


if __name__ == "__main__":
	testsWarping()