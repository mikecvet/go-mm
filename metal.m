// +build darwin

#include "metal.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// Interface to the local GPU device
id<MTLDevice> device;

// Representations of compiled compute programs to execute on the GPU
id<MTLComputePipelineState> pipelineStateNaive;
id<MTLComputePipelineState> pipelineStateTranspose;

// Used to create and submit command buffers to the GPU device
id<MTLCommandQueue> commandQueue;

// Buffers of input and output data being passed to and from the GPU
id<MTLBuffer> bufferA;
id<MTLBuffer> bufferB;
id<MTLBuffer> bufferC;

/**
 * Compiles and creates the Metal shader library used later on to execute commands on the GPU.
 * Initializes the pipeline state objects for the relevant public functions defined in the 
 * Metal shader code.
 */
void
initializePipelineAndCommandQueue (char *source_path) 
{
  device = MTLCreateSystemDefaultDevice();
  NSLog(@"Using default device %s", [device.name UTF8String]);

  NSError *error = nil;

  // Compile and initialize a new library located at the provided source path.
  MTLCompileOptions *compileOptions = [MTLCompileOptions new];
  compileOptions.languageVersion = MTLLanguageVersion3_0;
  NSString *ss = [NSString stringWithUTF8String:source_path];

  id<MTLLibrary> lib = [device newLibraryWithSource:ss
    options:compileOptions
    error:&error];

  if (lib == nil) {
    NSLog(@"Failed to create library, error %@.", error);
    return;
  }

  // Create a representation of the naive multiplication public shader function in 
  // the Metal library created above
  id<MTLFunction> naiveFunction =
      [lib newFunctionWithName:@"matrix_multiply_naive"];
  if (naiveFunction == nil) {
    NSLog(@"Failed to find the matrix_multiply_naive function.");
    return;
  }

  pipelineStateNaive = [device newComputePipelineStateWithFunction:naiveFunction
    error:&error];
  if (pipelineStateNaive == nil) {
    NSLog(@"Failed to create naive pipeline state object, error %@.", error);
    return;
  }

  // Create a representation of the transpose multiplication public shader function in 
  // the Metal library created above
  id<MTLFunction> transposeFunction =
    [lib newFunctionWithName:@"matrix_multiply_transpose"];
  if (transposeFunction == nil) {
    NSLog(@"Failed to find the matrix_multiply_transpose function.");
    return;
  }

  pipelineStateTranspose = [device newComputePipelineStateWithFunction:transposeFunction
    error:&error];
  if (pipelineStateTranspose == nil) {
    NSLog(@"Failed to create transpose pipeline state object, error %@.", error);
    return;
  }

  commandQueue = [device newCommandQueue];
  if (commandQueue == nil) {
    NSLog(@"Failed to find the command queue.");
    return;
  }
}

/**
 * Initialize the two input buffers containing matrix data, and also prepare the output buffer
 * for the resulting matrix multiplication result.
 */
void 
initializeMTLBuffers (
  void* a, 
  void* b, 
  int data_size_bytes, 
  int a_array_size,
  int b_array_size,
  int out_array_size
  ) {
  bufferA = [device newBufferWithBytes:a 
    length:a_array_size*data_size_bytes 
    options:MTLResourceStorageModeShared];

  bufferB = [device newBufferWithBytes:b 
    length:b_array_size*data_size_bytes 
    options:MTLResourceStorageModeShared];

  bufferC = [device newBufferWithLength:out_array_size*data_size_bytes 
    options:MTLResourceStorageModeShared];
}

/**
 * Execute a matrix multiplication of the previously-initialized matrix buffers using Apple's MPS
 * (Metal Performance Shaders) library. This doesn't execute any of the hand-written multiplication
 * code in the accompanying Metal shader library.
 */
void*
mps_mult (MatrixParams *params) 
{
  int a_rows = params->a_rows;
  int a_cols = params->a_cols;
  int b_rows = params->b_rows;
  int b_cols = params->b_cols;

  // Define Matrix "descriptions", accounting for matrix dimensionality and byte size
  MPSMatrixDescriptor *descriptorA = [MPSMatrixDescriptor matrixDescriptorWithDimensions:a_rows
    columns:a_cols
    rowBytes:a_cols * sizeof(float)
    dataType:MPSDataTypeFloat32];

  MPSMatrixDescriptor *descriptorB = [MPSMatrixDescriptor matrixDescriptorWithDimensions:b_rows
    columns:b_cols
    rowBytes:b_cols * sizeof(float)
    dataType:MPSDataTypeFloat32];

  MPSMatrixDescriptor *descriptorC = [MPSMatrixDescriptor matrixDescriptorWithDimensions:a_rows
    columns:b_cols
    rowBytes:b_cols * sizeof(float)
    dataType:MPSDataTypeFloat32];

  // Initialize matrix representations using above descriptions and matrix buffers
  MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descriptorA];
  MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descriptorB];
  MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descriptorC];

  // Creates the multiplication instance
  MPSMatrixMultiplication *matrixMultiplication = [[MPSMatrixMultiplication alloc] initWithDevice:device
  resultRows:a_rows
  resultColumns:b_cols
  interiorColumns:a_cols];

  // Encodes the multiplication command into the command buffer for the GPU
  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  [matrixMultiplication encodeToCommandBuffer:commandBuffer
    leftMatrix:matrixA
    rightMatrix:matrixB
    resultMatrix:matrixC];
  
  // Executes and passes the command buffer to the GPU device
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  // Extract raw result data from response buffer
  float *resultData = (float *)[bufferC contents];

  [matrixA release];
  [matrixB release];
  [matrixC release];
  [matrixMultiplication release];
  [commandBuffer release];

  return resultData;
}

/**
 * Configures GPU grids, serializes input parameters and buffers into the compute encoder, and executes the commands.
 * This assumes that the correct pipeline state has already been set to either the naive or transpose
 * metal matrix multiplication kernel functions.
 */
void*
metal_mult (MatrixParams *params, id<MTLComputePipelineState> pipelineState)
{
    @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) {
      NSLog(@"Failed to get the command buffer.");
      return nil;
    }
    // Get the compute encoder.
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) {
      NSLog(@"Failed to get the compute encoder.");
      return nil;
    }

    [computeEncoder setComputePipelineState:pipelineState];

    MTLSize threadsPerGrid = MTLSizeMake(params->a_cols, params->a_rows, 1);

    // Calculate a threadgroup size.
    // https://developer.apple.com/documentation/metal/calculating_threadgroup_and_grid_sizes?language=objc
    NSUInteger w = pipelineStateNaive.threadExecutionWidth;
    NSUInteger h = pipelineStateNaive.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);

    [computeEncoder setBytes:params length:16 atIndex:0];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:2];
    [computeEncoder setBuffer:bufferC offset:0 atIndex:3];

    // Encode the compute command.
    [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

    // End the compute pass.
    [computeEncoder endEncoding];

    // Execute the command.
    [commandBuffer commit];

    // We could add a completion event handler here instead and do other work, but since 
    // the matrix result is needed right away, we'll just block.
    // https://developer.apple.com/documentation/metal/mtlcommandbuffer/1442997-addcompletedhandler
    [commandBuffer waitUntilCompleted];

    return bufferC.contents;
  }
}

/**
 * Configures the GPU command encoder to call the naive matrix multiplication Metal kernel function.
 */
void*
metal_mult_naive (MatrixParams *params) 
{
  return metal_mult(params, pipelineStateNaive);
}

/**
 * Configures the GPU command encoder to call the transpose matrix multiplication Metal kernel function.
 */
void*
metal_mult_transpose (MatrixParams *params) 
{
  return metal_mult(params, pipelineStateTranspose);
}