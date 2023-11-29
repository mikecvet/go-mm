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

type BenchmarkData struct {
	NaiveCumulative int64
	TransposeCumulative int64
	TransposeParallelCumulative int64
	MetalNaiveCumulative int64
	MetalTransposeCumulative int64
	MPSCumulative int64
	GonumNativeCumulative int64
	GonumOpenBLASCumulative int64
	Iterations int
}

func (b *BenchmarkData) TimeNaive(start time.Time) {
	b.NaiveCumulative += time.Since(start).Milliseconds()
}

func (b *BenchmarkData) TimeTranspose(start time.Time) {
	b.TransposeCumulative += time.Since(start).Milliseconds()
}

func (b *BenchmarkData) TimeTransposeParallel(start time.Time) {
	b.TransposeParallelCumulative += time.Since(start).Milliseconds()
}

func (b *BenchmarkData) TimeMetalNaive(start time.Time) {
	b.MetalNaiveCumulative += time.Since(start).Milliseconds()
}

func (b *BenchmarkData) TimeMetalTranspose(start time.Time) {
	b.MetalTransposeCumulative += time.Since(start).Milliseconds()
}

func (b *BenchmarkData) TimeMPS(start time.Time) {
	b.MPSCumulative += time.Since(start).Milliseconds()
}

func (b *BenchmarkData) TimeGonumNative(start time.Time) {
	b.GonumNativeCumulative += time.Since(start).Milliseconds()
}

func (b *BenchmarkData) TimeGonumOpenBLAS(start time.Time) {
	b.GonumOpenBLASCumulative += time.Since(start).Milliseconds()
}

func (b *BenchmarkData) NaiveAverage() float64 {
	return float64(b.NaiveCumulative) / float64(b.Iterations)
}

func (b *BenchmarkData) TransposeAverage() float64 {
	return float64(b.TransposeCumulative) / float64(b.Iterations)
}

func (b *BenchmarkData) TransposeParallelAverage() float64 {
	return float64(b.TransposeParallelCumulative) / float64(b.Iterations)
}

func (b *BenchmarkData) MetalNaiveAverage() float64 {
	return float64(b.MetalNaiveCumulative) / float64(b.Iterations)
}

func (b *BenchmarkData) MetalTransposeAverage() float64 {
	return float64(b.MetalTransposeCumulative) / float64(b.Iterations)
}

func (b *BenchmarkData) MPSAverage() float64 {
	return float64(b.MPSCumulative) / float64(b.Iterations)
}

func (b *BenchmarkData) GonumNativeAverage() float64 {
	return float64(b.GonumNativeCumulative) / float64(b.Iterations)
}

func (b *BenchmarkData) GonumOpenBLASCumulativeAverage() float64 {
	return float64(b.GonumOpenBLASCumulative) / float64(b.Iterations)
}

func trial (a_rows, a_cols, b_rows, b_cols int, bdata *BenchmarkData) {
	bdata.Iterations += 1

	a32_data := make([]float32, a_rows * a_cols)
	b32_data := make([]float32, b_rows * b_cols)
	a64_data := make([]float64, a_rows * a_cols)
	b64_data := make([]float64, b_rows * b_cols)

	for i := 0; i < a_rows * a_cols; i++ {
		a32_data[i] = rand.Float32();
	}

	for i := 0; i < b_rows * b_cols; i++ {
		b32_data[i] = rand.Float32();
	}

	// Create double-precision copies of the 32-bit input data
	for i := 0; i < a_rows * a_cols; i++ {
		a64_data[i] = float64(a32_data[i]);
	}

	for i := 0; i < b_rows * b_cols; i++ {
		b64_data[i] = float64(b32_data[i]);
	}

	a32 := InitMatrixWithData(a_rows, a_rows, a32_data)
	b32 := InitMatrixWithData(b_rows, b_cols, b32_data)
	c32 := NewMatrix[float32](a_rows, b_cols)
	d32 := NewMatrix[float32](a_rows, b_cols)
	e32 := NewMatrix[float32](a_rows, b_cols)

	start := time.Now()
	naiveResult := a32.NaiveMult(b32)
	bdata.TimeNaive(start)

	start = time.Now()
	transposeResult := a32.TransposeMult(b32)
	bdata.TimeTranspose(start)

	start = time.Now()
	transposeParallelResult := a32.TransposeMultParallel(b32)
	bdata.TimeTransposeParallel(start)

	start = time.Now()
	MetalNaive(a32, b32, c32)
	bdata.TimeMetalNaive(start)

	start = time.Now()
	MetalNaive(a32, b32, d32)
	bdata.TimeMetalTranspose(start)

	start = time.Now()
	MPS(a32, b32, e32)
	bdata.TimeMPS(start)

	gonum_a := mat.NewDense(a_rows, a_cols, a64_data)
	gonum_b := mat.NewDense(b_rows, b_cols, b64_data)
	gonum_c := mat.NewDense(a_rows, b_cols, nil)
	gonum_d := mat.NewDense(a_rows, b_cols, nil)

	blas64.Use(gonum.Implementation{})
	start = time.Now()
	gonum_c.Mul(gonum_a, gonum_b)
	bdata.TimeGonumNative(start)

	blas64.Use(netlib.Implementation{})
	start = time.Now()
	gonum_d.Mul(gonum_a, gonum_b)
	bdata.TimeGonumOpenBLAS(start)

	gonum_c32 := InitMatrixWithData(a_rows, a_rows, convertDoubleToFloat(gonum_c.RawMatrix().Data))
	gonum_d32 := InitMatrixWithData(a_rows, a_rows, convertDoubleToFloat(gonum_d.RawMatrix().Data))

	assertEqualsOrLog[float32]("naive", "transpose", naiveResult, transposeResult)
	assertEqualsOrLog[float32]("transpose", "transpose_parallel", transposeResult, transposeParallelResult)
	assertEqualsOrLog[float32]("metal_naive", "metal_transpose", c32, e32)
	assertEqualsOrLog[float32]("metal_naive", "mps", c32, e32)
	assertEqualsOrLog[float32]("naive", "mps", naiveResult, e32)
	assertEqualsOrLog[float32]("gonum_native", "gonum_openblas", gonum_c32, gonum_d32)
	assertEqualsOrLog[float32]("naive", "gonum_native", naiveResult, gonum_c32)
}

func convertDoubleToFloat(data []float64) []float32 {
	x := make([]float32, len(data))
	for i := 0; i < len(data); i++ {
		x[i] = float32(data[i])
	}

	return x
}

func assertEqualsOrLog[T FloatingPoint](s, t string, a, b *Matrix[T]) {
	if !a.Equals(b) {
		fmt.Printf("%s != %s\n", s, t)
		printMismatches[T](a, b)
	}
}

func printMismatches[T FloatingPoint](c, d *Matrix[T]) {
	for i := range c.Data {
		if math.Abs(float64(c.Data[i] - d.Data[i])) > 0.01 {
				fmt.Printf("Difference at index %d: lhs = %f, rhs = %f\n", i, c.Data[i], d.Data[i])
		}
	}

	//fmt.Printf("first: %v\n", c)
	//fmt.Printf("second: %v\n", d)
}

func TimeMultiplication() {
	n := 400
  //n := 33
	// n := 2
	bdata := &BenchmarkData{}

	// loading and warmup
	trial(128, 128, 128, 128, bdata)

	fmt.Println("[elements naive transpose transpose_parallel metal_naive metal_transpose mps gonum openblas]")

	bdata = &BenchmarkData{}

	for ;; {
		for i := 0; i < 3; i++ {
			trial(n, n, n, n, bdata)
		}

		fmt.Printf("%v %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", 
			n * n, 
		  bdata.NaiveAverage(), 
		  bdata.TransposeAverage(),
			bdata.TransposeParallelAverage(), 
			bdata.MetalNaiveAverage(),
			bdata.MetalTransposeAverage(), 
			bdata.MPSAverage(),
			bdata.GonumNativeAverage(),
			bdata.GonumOpenBLASCumulativeAverage())

			//  break

		n += 100
		bdata = &BenchmarkData{}
	}
}