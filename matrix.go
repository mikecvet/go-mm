package main

import (
  "math"
  "sync"
)

type FloatingPoint interface {
  ~float32 | ~float64
}

type Matrix[T FloatingPoint] struct {
  Rows, Cols int
  Data       []T
}

func NewMatrix[T FloatingPoint](rows, cols int) *Matrix[T] {
  m := &Matrix[T]{
    Rows: rows,
    Cols: cols,
    Data: make([]T, rows*cols),
  }

  return m
}

func InitMatrixWithData[T FloatingPoint](rows, cols int, data []T) *Matrix[T] {
  m := NewMatrix[T](rows, cols)

  for i := 0; i < rows; i++ {
    for j := 0; j < cols; j++ {
      m.Set(i, j, data[m.Index(i, j)])
    }
  }

  return m
}

func (m Matrix[T]) Transpose() *Matrix[T] {
  c_data := make([]T, m.Size())
  ptr := 0

  for i := 0; i < m.Cols; i++ {
    for j := 0; j < m.Rows; j++ {
      c_data[ptr] = m.At(j, i)
      ptr++
    }
  }

  return InitMatrixWithData[T](m.Cols, m.Rows, c_data)
}

func (m Matrix[T]) Index(i, j int) int {
  return i*m.Cols + j;
}

func (m *Matrix[T]) Set(i, j int, v T) {
  m.Data[m.Index(i, j)] = v
}

func (m Matrix[T]) At(i, j int) T {
  return m.Data[m.Index(i, j)]
}

func (m Matrix[T]) Size() int {
  return m.Rows * m.Cols
}

func (a Matrix[T]) Equals(b *Matrix[T]) bool {
  if a.Rows != b.Rows || a.Cols != b.Cols {
    return false
  }

  for i := 0; i < a.Size(); i++ {
    if math.Abs(float64(a.Data[i] - b.Data[i])) > 0.01 {
      return false
    }
  }

  return true
}

func (a Matrix[T]) NaiveMult(b *Matrix[T]) *Matrix[T] {
  if a.Cols == b.Rows {
    c_data := make([]T, a.Cols * b.Rows)
    ptr := 0

    for i := 0; i < a.Rows; i++ {
      for j := 0; j < b.Cols; j++ {
        var sum T = 0.0
        for k := 0; k < a.Cols; k++ {
          sum += a.At(i, k) * b.At(k, j)
        }

        c_data[ptr] = sum
        ptr++
      }
    }

    return InitMatrixWithData(a.Rows, b.Cols, c_data)
  } else {
    panic("matrices are the wrong size for multiplication")
  }
}

func (a Matrix[T]) TransposeMult(b *Matrix[T]) *Matrix[T] {
  if a.Cols == b.Rows {
    c_data := make([]T, a.Cols * b.Rows)
    ptr := 0

    t := b.Transpose()

    for i := 0; i < a.Rows; i++ {
      for j := 0; j < b.Cols; j++ {
        var sum T = 0.0
        for k := 0; k < a.Cols; k++ {
          sum += a.At(i, k) * t.At(j, k)
        }

        c_data[ptr] = sum
        ptr++
      }
    }

    return InitMatrixWithData(a.Rows, b.Cols, c_data)
  } else {
    panic("matrices are the wrong size for multiplication")
  }
}

func (a Matrix[T]) TransposeMultParallel(b *Matrix[T]) *Matrix[T] {
  if a.Cols != b.Rows {
    panic("matrices are the wrong size for multiplication")
  }

  c_data := make([]T, a.Rows*b.Cols)
  t := b.Transpose()
  var wg sync.WaitGroup

  for i := 0; i < a.Rows; i++ {
    wg.Add(1) // Add a count to the WaitGroup for the new goroutine
    go func(i int) {
      defer wg.Done() // Decrease the count when the goroutine completes
      ptr := i * b.Cols
      for j := 0; j < b.Cols; j++ {
        var sum T = 0.0
        for k := 0; k < a.Cols; k++ {
          sum += a.At(i, k) * t.At(j, k)
        }
        c_data[ptr+j] = sum
      }
    }(i)
  }

  wg.Wait() // Wait for all goroutines to complete
  return InitMatrixWithData(a.Rows, b.Cols, c_data)
}
