package main

import "time"

/**
 * Accumulates timing information for a series of benchmark runs
 * of matrix multiplication methods.
 */
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