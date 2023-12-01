#include <metal_stdlib>
using namespace metal;

typedef struct MatrixParams {
  int a_rows, a_cols;
  int b_rows, b_cols;
} MatrixParams;

kernel void matrix_multiply_naive(
  device const MatrixParams *params,
  constant float *A,
  constant float *B,
  device float *C,
  // Indicates the thread's unique position within the entire grid of threads being executed
  // The uint2 type is a 2D coordinate, with values x and y representing its indices on each axis
  // This specific value is not provided from the calling code, but provided by the Metal framework
  uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= params->a_rows || gid.y >= params->b_cols) {
    return; // This thread is out of matrix dimensionality range, do nothing
  }

  float sum = 0.0;
  int k;

  // Loop unrolling; improves performance by a notable margin
  for (k = 0; k <= params->a_cols - 4; k += 4) {
    sum += A[gid.x * params->a_cols + k]     * B[k * params->b_cols + gid.y];
    sum += A[gid.x * params->a_cols + k + 1] * B[(k + 1) * params->b_cols + gid.y];
    sum += A[gid.x * params->a_cols + k + 2] * B[(k + 2) * params->b_cols + gid.y];
    sum += A[gid.x * params->a_cols + k + 3] * B[(k + 3) * params->b_cols + gid.y];
  }

  // Handle any remaining elements
  for (; k < params->a_cols; ++k) {
    sum += A[gid.x * params->a_cols + k] * B[k * params->b_cols + gid.y];
  }

  C[gid.x * params->b_cols + gid.y] = sum;
}

kernel void matrix_multiply_transpose(
  device const MatrixParams *params,
  constant float *A,
  constant float *B,
  device float *C,
  // Indicates the thread's unique position within the entire grid of threads being executed
  // The uint2 type is a 2D coordinate, with values x and y representing its indices on each axis
  // This specific value is not provided from the calling code, but provided by the Metal framework
  uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params->a_rows || gid.y >= params->b_cols) {
      return; // This thread is out of matrix dimensionality range, do nothing
    }

    float sum = 0.0;
    int k;

    // Loop unrolling; improves performance by a notable margin
    for (k = 0; k <= params->a_cols - 4; k += 4) {
      sum += A[gid.x * params->a_cols + k]     * B[gid.x * params->b_cols + k];
      sum += A[gid.x * params->a_cols + k + 1] * B[gid.x * params->b_cols + k + 1];
      sum += A[gid.x * params->a_cols + k + 2] * B[gid.x * params->b_cols + k + 2];
      sum += A[gid.x * params->a_cols + k + 3] * B[gid.x * params->b_cols + k + 3];
    }

    // Handle any remaining elements
    for (; k < params->a_cols; ++k) {
      sum += A[gid.x * params->a_cols + k] * B[gid.y * params->b_cols + k];
    }

    C[gid.x * params->b_cols + gid.y] = sum;
}
