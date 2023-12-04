package main

/*
#cgo LDFLAGS: -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -L/opt/homebrew/opt/openblas/lib -lopenblas
*/
import "C"

import (
  _ "embed"
  "flag"
  "fmt"
  "os"
  "strconv"
)

const DEFAULT_INCREMENT = 100

//go:embed mm.metal
var source string

func main() {
  incrValue := DEFAULT_INCREMENT
  fastFlag := flag.Bool("fast", false, "Enable fast mode")
  gpuOnlyFlag := flag.Bool("gpu-only", false, "Use GPU only")
  incrFlag := flag.String("incr", "", "Incremental integer value")

  flag.Parse()

  if *incrFlag != "" {
    v, err := strconv.Atoi(*incrFlag)
    if err != nil {
      fmt.Fprintf(os.Stderr, "Error: Invalid value for --incr: %s\n", *incrFlag)
      os.Exit(1)
    }
    incrValue = v
  }

  Compile(source)

  // Run benchmarking
  TimeMultiplication(*fastFlag, *gpuOnlyFlag, incrValue)
}
