// Package cost contains costing functions.
package cost

import (
	"github.com/xavier268/nn/pkg/ntwk"
	"gonum.org/v1/gonum/mat"
)

// MSE : Mean Square Costing function (1/2n x sum of squares)
// used as parameter for training
var MSE = ntwk.Coster{
	F:    mse,
	G:    grad,
	Name: "MSE Costing",
}

func mse(yest, ytrue *mat.Dense) float64 {
	delta := new(mat.Dense)
	delta.Sub(yest, ytrue)
	r, _ := delta.Dims()
	delta.MulElem(delta, delta)
	return mat.Sum(delta) / (2 * float64(r))
}

func grad(yest, ytrue *mat.Dense) *mat.Dense {
	delta := new(mat.Dense)
	delta.Sub(yest, ytrue)
	r, _ := delta.Dims()
	delta.Scale(1/float64(r), delta)
	return delta
}
