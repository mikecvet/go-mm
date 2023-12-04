package main

/*
#include <stdlib.h>
#include "metal.h"
*/
import "C"
import (
  "unsafe"
)

// MatrixParams matches the definitions in metal.h
type MatrixParams struct {
  a_rows, a_cols int32
  b_rows, b_cols int32
}

// Compile the shader source code and initialize pipelines. The metalSource
// param contains the contents of an embedded Metal Shading Language file.
func Compile (metalSource string) {
  // Wrap string in a C string
  src := C.CString(metalSource)

  // Free the above string after command queue is initialized
  defer C.free(unsafe.Pointer(src))

  // Compile the source, initialize pipelines and command queue
  C.initializePipelineAndCommandQueue(src)
}

/**
 * Initializes Metal buffers ahead of matrix operaetions. Creates a C-native MatrixParams struct,
 * returns an unsafe pointer to be passed into GPU space.
 */
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

/**
 * Calls the naive matrix multiplication algorithm written in Metal Shading Language, runs
 * on the GPU.
 */
func MetalNaive (a *Matrix[float32], b *Matrix[float32], c *Matrix[float32]) {
  unsafeParams := prepareUnsafe(a, b, c);
  metalResults := C.metal_mult_naive(unsafeParams)
  c.Data = unsafe.Slice((*float32)(metalResults), a.Rows * b.Cols)

  return
}

/**
 * Calls the transposed naive matrix multiplication algorithm written in Metal Shading Language, 
 * runs on the GPU.
 */
func MetalTranspose (a *Matrix[float32], b *Matrix[float32], c *Matrix[float32]) {
  t := b.Transpose()
  unsafeParams := prepareUnsafe(a, t, c);
  metalResults := C.metal_mult_transpose(unsafeParams)
  c.Data = unsafe.Slice((*float32)(metalResults), a.Rows * b.Cols)

  return
}

/**
 * Calls the matrix multiplication functionality in the Metal Performance Shaders framework,
 * runs on the GPU.
 */
func MPS (a *Matrix[float32], b *Matrix[float32], c *Matrix[float32]) {
  unsafeParams := prepareUnsafe(a, b, c);
  metalResults := C.mps_mult(unsafeParams)
  c.Data = unsafe.Slice((*float32)(metalResults), a.Rows * b.Cols)

  return
}
