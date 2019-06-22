package backprop

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLayer1(t *testing.T) {
	fmt.Println("Testing layer")
	lay := NewLayer(5, 3, Sigmoid)
	lay.Dump()
	x := mat.NewDense(10, 5, nil)
	z, a := lay.Forward(x)
	fmt.Println("z result", mat.Formatted(z))
	fmt.Println("a result", mat.Formatted(a))
}
