package main

import (
	"testing"
)

func TestAt(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	matrix := InitMatrixWithData[float64](2, 3, data)

	if matrix.At(1, 2) != 6.0 {
		t.Errorf("At(1, 2) = %f; want 6.0", matrix.At(1, 2))
	}
}

func TestEquals(t *testing.T) {
	matrixA := InitMatrixWithData[float64](2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	matrixB := InitMatrixWithData[float64](2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})

	if !matrixA.Equals(matrixB) {
		t.Errorf("Expected matrixA to be equal to matrixB")
	}
}

func TestTranspose(t *testing.T) {
	matrixA := InitMatrixWithData[float64](2, 3, []float64{
			1, 2, 3,
			4, 5, 6,
	})

	expected := InitMatrixWithData[float64](3, 2, []float64{
			1, 4,
			2, 5,
			3, 6,
	})

	// Perform transpose
	transposed := matrixA.Transpose()

	// Check if the transposed matrix matches the expected matrix
	if !transposed.Equals(expected) {
		t.Errorf("Transpose was incorrect, got: %v, want: %v.", transposed, expected)
	}
}

func TestNaiveMult(t *testing.T) {
	matrixA := InitMatrixWithData[float64](2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	matrixB := InitMatrixWithData[float64](3, 2, []float64{7.0, 8.0, 9.0, 10.0, 11.0, 12.0})
	expected := InitMatrixWithData[float64](2, 2, []float64{58.0, 64.0, 139.0, 154.0})

	result := matrixA.NaiveMult(matrixB)

	if !result.Equals(expected) {
		t.Errorf("NaiveMult result was incorrect, got: %v, want: %v.", result, expected)
	}
}

func TestTransposeMult(t *testing.T) {
	matrixA := InitMatrixWithData[float64](2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	matrixB := InitMatrixWithData[float64](3, 2, []float64{7.0, 8.0, 9.0, 10.0, 11.0, 12.0})
	expected := InitMatrixWithData[float64](2, 2, []float64{58.0, 64.0, 139.0, 154.0})

	result := matrixA.TransposeMult(matrixB)

	if !result.Equals(expected) {
		t.Errorf("NaiveMult result was incorrect, got: %v, want: %v.", result, expected)
	}
}

func TestTransposeParallelMult(t *testing.T) {
	matrixA := InitMatrixWithData[float64](2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	matrixB := InitMatrixWithData[float64](3, 2, []float64{7.0, 8.0, 9.0, 10.0, 11.0, 12.0})
	expected := InitMatrixWithData[float64](2, 2, []float64{58.0, 64.0, 139.0, 154.0})

	result := matrixA.TransposeMultParallel(matrixB)

	if !result.Equals(expected) {
		t.Errorf("NaiveMult result was incorrect, got: %v, want: %v.", result, expected)
	}
}

func TestNaiveMultWithZeroMatrix(t *testing.T) {
	matrixA := InitMatrixWithData[float64](2, 3, []float64{1, 2, 3, 4, 5, 6})
	zeroMatrix := InitMatrixWithData[float64](3, 2, []float64{0, 0, 0, 0, 0, 0})
	expected := InitMatrixWithData[float64](2, 2, []float64{0, 0, 0, 0})

	result := matrixA.NaiveMult(zeroMatrix)
	if !result.Equals(expected) {
		t.Errorf("Expected zero matrix, got: %v", result)
	}
}

func TestNaiveMultWithIdentityMatrix(t *testing.T) {
	matrixA := InitMatrixWithData[float64](2, 2, []float64{1, 2, 3, 4})
	identityMatrix := InitMatrixWithData[float64](2, 2, []float64{1, 0, 0, 1})

	result := matrixA.NaiveMult(identityMatrix)
	if !result.Equals(matrixA) {
		t.Errorf("Multiplying with identity matrix should return the original matrix, got: %v", result)
	}
}

func TestNaiveMultNonSquareMatrices(t *testing.T) {
	matrixA := InitMatrixWithData[float64](2, 3, []float64{1, 2, 3, 4, 5, 6})
	matrixB := InitMatrixWithData[float64](3, 2, []float64{7, 8, 9, 10, 11, 12})
	expected := InitMatrixWithData[float64](2, 2, []float64{58, 64, 139, 154})

	result := matrixA.NaiveMult(matrixB)
	if !result.Equals(expected) {
		t.Errorf("NaiveMult result was incorrect, got: %v, want: %v.", result, expected)
	}
}
