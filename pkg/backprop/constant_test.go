package backprop

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestConstMat(t *testing.T) {
	one := NewConstantMat(2, 5, 1.0)
	fmt.Println("All ones, 2x5")
	fmt.Println(mat.Formatted(one, mat.Squeeze()))
	one.T()
	fmt.Println("All ones, 5x2")
	fmt.Println(mat.Formatted(one, mat.Squeeze()))
}
