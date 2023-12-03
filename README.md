# go-mm
This project explores the usage of Apple's Metal-compatible GPUs from Go code, using Objective-C bindings, Metal Shading Library source files, and the Metal Performance Shaders library. The performance of matrix multiplication tasks is compared across a couple different hand-written MSL implementations, the MPS library, Gonum and OpenBLAS, and hand-written Go implementations.

![High Level Overview](https://github.com/mikecvet/go-mm/blob/main/static/high_level.jpg)

This project is discussed in detail in this blog post, which also explains more about how cgo is used and how the Metal APIs work.

This program is fairly simple:

```
  ~/code/go-mm ~>> ./go-mm --help
  Usage of ./go-mm:
  -fast
    	Enable fast mode
  -gpu-only
    	Use GPU only
  -incr string
    	Incremental integer value

  ~/code/go-mm ~>> go build -o go-mm
  ~/code/go-mm ~>> ./go-mm 
  2023-12-01 18:29:27.097 go-mm[82758:22587913] Using default device Apple M2
  elements naive transpose transpose_parallel metal_naive metal_transpose mps gonum openblas
  160000 206.33 199.33 42.67 8.33 5.67 0.33 5.00 1.33
  250000 382.33 401.00 89.33 11.33 7.33 0.00 9.33 3.67
  360000 663.00 683.00 146.33 18.33 11.00 0.00 16.67 10.33 
  ^C
  ~/code/go-mm ~>> ./go-mm --gpu-only
  2023-12-01 18:33:07.687 go-mm[82789:22590787] Using default device Apple M2
  elements metal_naive metal_transpose mps
  160000 7.33 6.67 0.67
  250000 7.33 9.00 0.00
  360000 9.00 10.33 0.00
  490000 12.33 14.33 1.00
  640000 18.00 20.67 1.00
  810000 24.67 28.33 1.33
  1000000 35.00 38.67 2.00
```

This data can be used to plot a graph of multiplication time in ms, via the provided Python script

![Performance Graph](https://github.com/mikecvet/go-mm/blob/main/static/graph.png)

Plotting results removing the slowest three methods:

![Performance Graph - Fast Implementations](https://github.com/mikecvet/go-mm/blob/main/static/fast.png)

Plotting results is trivial through the provided Python script

`~/code/go-mm ~>> python3 plot.py ./z --gpuonly`
