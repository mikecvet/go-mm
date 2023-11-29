//go:build darwin
// +build darwin

package main

/*
#cgo LDFLAGS: -framework Metal -framework CoreGraphics -framework Foundation
#include <stdlib.h>
#include <stdbool.h>
#include "metal.h"
*/
import "C"
import (
	"unsafe"
)

// Params matches the definitions in metal.h
type MatrixParams struct {
	a_rows, a_cols int32
	b_rows, b_cols int32
}

// Compile the shader source code and initialize pipelines.
func Compile (metalSourceFile string) {
	src := C.CString(metalSourceFile)
	defer C.free(unsafe.Pointer(src))
	C.initializePipelineAndCommandQueue(src)
}

func prepareUnsafe (a *Matrix[float32], b *Matrix[float32], c *Matrix[float32]) *(C.MatrixParams) {
	var a_data unsafe.Pointer
	var b_data unsafe.Pointer

	if len(a.Data) > 0 {
		a_data = unsafe.Pointer(&a.Data[0])
	}

	if len(b.Data) > 0 {
		b_data = unsafe.Pointer(&b.Data[0])
	}

	C.initializeMTLBuffers(
		a_data,                  // Input opaque pointer for A
		b_data,                  // Input opaque pointer for B
		C.int(4),                // sizeof(float32)
		C.int(a.Size()),         // A.Size(), number of elements 
		C.int(b.Size()),         // B.Size(), number of elements
		C.int(a.Rows * b.Cols))  // Result matrix size

	params := MatrixParams{
		a_rows: int32(a.Rows),
		a_cols: int32(a.Cols),
		b_rows: int32(b.Rows),
		b_cols: int32(b.Cols),
	}

	return (*C.MatrixParams)(unsafe.Pointer(&params));
}

func MetalNaive (a *Matrix[float32], b *Matrix[float32], c *Matrix[float32]) {
	unsafeParams := prepareUnsafe(a, b, c);
	metalResults := C.metal_mult_naive(unsafeParams)
	c.Data = unsafe.Slice((*float32)(metalResults), a.Rows * b.Cols)

	return
}

func MetalTranspose (a *Matrix[float32], b *Matrix[float32], c *Matrix[float32]) {
	unsafeParams := prepareUnsafe(a, b, c);
	metalResults := C.metal_mult_transpose(unsafeParams)
	c.Data = unsafe.Slice((*float32)(metalResults), a.Rows * b.Cols)

	return
}

func MPS (a *Matrix[float32], b *Matrix[float32], c *Matrix[float32]) {
	unsafeParams := prepareUnsafe(a, b, c);
	metalResults := C.mps_mult(unsafeParams)
	c.Data = unsafe.Slice((*float32)(metalResults), a.Rows * b.Cols)

	return
}
