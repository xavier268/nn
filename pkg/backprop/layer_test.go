package backprop

import (
	"fmt"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLayer1(t *testing.T) {

	fmt.Println("Testing layer")
	lay := NewFCLayer(5, 3).SetActivation(ActivationSigmoid)
	lay.InitWB(InitializationRandom)

	x := mat.NewDense(10, 5, nil)
	x.Apply(func(i, j int, v float64) float64 { return 2*rand.Float64() - 1 }, x)
	a := lay.Forward(x)

	t.SkipNow()

	lay.Dump()
	fmt.Println("a result", mat.Formatted(a))
}
