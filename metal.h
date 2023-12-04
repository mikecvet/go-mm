// Matches with MatrixParams type in metal.go
typedef struct MatrixParams {
  int a_rows, a_cols;
  int b_rows, b_cols;
} MatrixParams;

void initializePipelineAndCommandQueue(char* source_path);
void initializeMTLBuffers(
  void* a, 
  void* b, 
  int in_data_size_bytes, 
  int in_array_size,
  int out_data_size_bytes, 
  int out_array_size
);

void* metal_mult_naive(MatrixParams *params);
void* metal_mult_transpose(MatrixParams *params);
void* mps_mult(MatrixParams *params);
