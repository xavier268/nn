package mynn

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func getRdm(r, c int) *mat.Dense {
	x := mat.NewDense(r, c, nil)
	x.Apply(func(i, j int, v float64) float64 { return math.Floor(10. * rand.Float64()) }, x)
	return x
}

func TestNewMlp(t *testing.T) {
	// Create network
	m := NewMLP(3, 11, 5)
	m.dump()
	// random input
	x := mat.NewDense(7, 3, nil)
	x.Apply(func(i, j int, v float64) float64 { return 0.77 }, x)
	// Process input
	fmt.Println("Input\n", mat.Formatted(x))
	y := m.Forward(x)
	fmt.Println("Result\n", mat.Formatted(y))
}
