// Demo the mat package
package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func main() {
	rand.Seed(42)
	x := randomMat(3, 5)
	x.Apply(func(i, j int, v float64) float64 { return math.Floor(v * 100.) }, x)
	dump(x)
}

func dump(m mat.Matrix) {
	r, c := m.Dims()
	fmt.Println("Dimensions (rows, cols) : ", r, c)
	fmt.Println("Dump : ", m)
	fmt.Println("Formatted : ")
	fmt.Println(mat.Formatted(m, mat.Squeeze()))
}

// Generate a random matrix
func randomMat(r, c int) *mat.Dense {
	m := mat.NewDense(r, c, nil)
	m.Apply(func(i, j int, v float64) float64 { return rand.Float64() }, m)
	return m
}
