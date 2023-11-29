//go:build darwin
// +build darwin

package main

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation -L/opt/homebrew/opt/openblas/lib -lopenblas
#include <stdlib.h>
#include <stdbool.h>
*/
import "C"

import (
	_ "embed"
)

//go:embed mm.metal
var source string

func main() {
	Compile(source)
	TimeMultiplication()
}
