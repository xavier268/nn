package backprop

import (
	"fmt"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLayer1(t *testing.T) {

	fmt.Println("Testing layer")
	lay := NewLayer(5, 3, Sigmoid)
	lay.RandomizeWeight()
	lay.Dump()
	x := mat.NewDense(10, 5, nil)
	x.Apply(func(i, j int, v float64) float64 { return 2*rand.Float64() - 1 }, x)
	a := lay.Forward(x)
	fmt.Println("a result", mat.Formatted(a))
}
