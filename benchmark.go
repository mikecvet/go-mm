package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/blas/gonum"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/netlib/blas/netlib"
)

const DEFAULT_TRIALS = 3

/**
 * Run a series of timed trials of various matrix multiplication methods; extract average 
 * completion times and print them out. 
 */
func TimeMultiplication(fast, gpuOnly bool, increment int) {
	n := 400
	bdata := &BenchmarkData{}

	// loading and warmup
	trial(128, 128, 128, 128, bdata, false, false)

	if !fast && !gpuOnly {
		fmt.Println("elements naive transpose transpose_parallel metal_naive metal_transpose mps gonum openblas")
	} else if gpuOnly {
		fmt.Println("elements metal_naive metal_transpose mps")
	} else {
		// Fast
		fmt.Println("elements metal_naive metal_transpose mps gonum openblas")
	}

	bdata = &BenchmarkData{}

	for ;; {
		for i := 0; i < DEFAULT_TRIALS; i++ {
			trial(n, n, n, n, bdata, fast, gpuOnly)
		}

		printTimes(n * n, bdata, fast, gpuOnly)

		// reset benchmark data for test of next matrix size
		bdata = &BenchmarkData{}
		n += increment
	}
}


/**
 * Generates vectors of random floating-point values, constructs matrix structs, and
 * executes matrix multiplication according to input arguments. Each multiplication
 * operation is timed.
 */
func trial (aRows, aCols, bRows, bCols int, bdata *BenchmarkData, fast, gpuOnly bool) {
	bdata.Iterations += 1

	a32Data := make([]float32, aRows * aCols)
	b32Data := make([]float32, bRows * bCols)
	a64Data := make([]float64, aRows * aCols)
	b64Data := make([]float64, bRows * bCols)

	for i := 0; i < aRows * aCols; i++ {
		a32Data[i] = rand.Float32();
	}

	for i := 0; i < bRows * bCols; i++ {
		b32Data[i] = rand.Float32();
	}

	// Create double-precision copies of the 32-bit input data; Gonum
	// works only with float64s
	for i := 0; i < aRows * aCols; i++ {
		a64Data[i] = float64(a32Data[i]);
	}

	for i := 0; i < bRows * bCols; i++ {
		b64Data[i] = float64(b32Data[i]);
	}

	start := time.Now()

	a32 := InitMatrixWithData(aRows, aRows, a32Data)
	b32 := InitMatrixWithData(bRows, bCols, b32Data)
	c32 := NewMatrix[float32](aRows, bCols)
	d32 := NewMatrix[float32](aRows, bCols)
	e32 := NewMatrix[float32](aRows, bCols)

	var naiveResult *Matrix[float32];
	var transposeResult *Matrix[float32];
	var transposeParallelResult *Matrix[float32];

	var gonumA *mat.Dense
	var gonumB *mat.Dense
	var gonumC *mat.Dense
	var gonumD *mat.Dense

	var gonumC32 *Matrix[float32]
	var gonumD32 *Matrix[float32]

	// Run hand-coded Go implementations
	if (!fast && !gpuOnly) {
		start = time.Now()
		naiveResult = a32.NaiveMult(b32)
		bdata.TimeNaive(start)

		start = time.Now()
		transposeResult = a32.TransposeMult(b32)
		bdata.TimeTranspose(start)

		start = time.Now()
		transposeParallelResult = a32.TransposeMultParallel(b32)
		bdata.TimeTransposeParallel(start)

		assertEqualsOrLog[float32]("naive", "transpose", naiveResult, transposeResult)
		assertEqualsOrLog[float32]("transpose", "transpose_parallel", transposeResult, transposeParallelResult)
	}

	// If not gpu-only, run Gonum and OpenBLAS operations
	if !gpuOnly {
		gonumA = mat.NewDense(aRows, aCols, a64Data)
		gonumB = mat.NewDense(bRows, bCols, b64Data)
		gonumC = mat.NewDense(aRows, bCols, nil)
		gonumD = mat.NewDense(aRows, bCols, nil)

		blas64.Use(gonum.Implementation{})
		start = time.Now()
		gonumC.Mul(gonumA, gonumB)
		bdata.TimeGonumNative(start)

		blas64.Use(netlib.Implementation{})
		start = time.Now()
		gonumD.Mul(gonumA, gonumB)
		bdata.TimeGonumOpenBLAS(start)

		gonumC32 = InitMatrixWithData(aRows, aRows, convertDoubleToFloat(gonumC.RawMatrix().Data))
		gonumD32 = InitMatrixWithData(aRows, aRows, convertDoubleToFloat(gonumD.RawMatrix().Data))

		assertEqualsOrLog[float32]("gonum_native", "gonum_openblas", gonumC32, gonumD32)

		if !fast {
			assertEqualsOrLog[float32]("naive", "gonum_native", naiveResult, gonumC32)
		}
	}

	// Run GPU-based multiplication
	start = time.Now()
	MetalNaive(a32, b32, c32)
	bdata.TimeMetalNaive(start)

	start = time.Now()
	MetalTranspose(a32, b32, d32)
	bdata.TimeMetalTranspose(start)

	start = time.Now()
	MPS(a32, b32, e32)
	bdata.TimeMPS(start)

	assertEqualsOrLog[float32]("metal_naive", "metal_transpose", c32, d32)
	assertEqualsOrLog[float32]("metal_transpose", "mps", d32, e32)
	assertEqualsOrLog[float32]("metal_naive", "mps", c32, e32)

	if (!gpuOnly) {
		assertEqualsOrLog[float32]("gonum_native", "mps", gonumC32, e32)
		assertEqualsOrLog[float32]("metal_naive", "gonum_openblas", c32, gonumD32)
	}
}

/**
 * Convert an array of []float64 to []float32 of the same values
 */
func convertDoubleToFloat(data []float64) []float32 {
	x := make([]float32, len(data))
	for i := 0; i < len(data); i++ {
		x[i] = float32(data[i])
	}

	return x
}

/**
 * Asserts the two Matrices are equal; if not, prints out the indices which do not match
 */
func assertEqualsOrLog[T FloatingPoint](s, t string, a, b *Matrix[T]) {
	if !a.Equals(b) {
		fmt.Printf("%s != %s\n", s, t)
		printMismatches[T](a, b)
	}
}

/**
 * Prints index and value of divergent entries in the two matrices
 */
func printMismatches[T FloatingPoint](c, d *Matrix[T]) {
	for i := range c.Data {
		if math.Abs(float64(c.Data[i] - d.Data[i])) > 0.01 {
			fmt.Printf("Difference at index %d: lhs = %f, rhs = %f\n", i, c.Data[i], d.Data[i])
		}
	}
}

/**
 * Format printing of benchmark times. Yes, its hacky
 */
func printTimes(elements int, bdata *BenchmarkData, fast, gpuOnly bool) {
	if !fast && !gpuOnly {
		fmt.Printf("%v %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", 
			elements, 
			bdata.NaiveAverage(), 
			bdata.TransposeAverage(),
			bdata.TransposeParallelAverage(), 
			bdata.MetalNaiveAverage(),
			bdata.MetalTransposeAverage(), 
			bdata.MPSAverage(),
			bdata.GonumNativeAverage(),
			bdata.GonumOpenBLASAverage())
	} else if gpuOnly {
		fmt.Printf("%v %.2f %.2f %.2f\n", 
			elements, 
			bdata.MetalNaiveAverage(),
			bdata.MetalTransposeAverage(), 
			bdata.MPSAverage())
	} else {
		// Fast
		fmt.Printf("%v %.2f %.2f %.2f %.2f %.2f\n", 
			elements, 
			bdata.MetalNaiveAverage(),
			bdata.MetalTransposeAverage(), 
			bdata.MPSAverage(),
			bdata.GonumNativeAverage(),
			bdata.GonumOpenBLASAverage())
	}
}