package backprop

import "gonum.org/v1/gonum/mat"

// Cost is the interface for a cost function,
// providing both direct result and gradient
// It takes as input  estimated results (yest) and ground truth (ytruth).
// The gradient has the exact same dimension as the yest matrix.
type Cost interface {
	cost(yest, ytrue *mat.Dense) float64
	grad(yest, ytrue *mat.Dense) *mat.Dense
	name() string
}

// CostMSE defines a cost function using Mean Squered Error
var CostMSE = new(mseImpl)

type mseImpl struct{}

func (*mseImpl) cost(yest, ytrue *mat.Dense) float64 {
	delta := new(mat.Dense)
	delta.Sub(yest, ytrue)
	r, c := delta.Dims()
	res := 0.
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			res = res + delta.At(i, j)*delta.At(i, j)
		}
	}
	return res / (2 * float64(r))
}
func (*mseImpl) grad(yest, ytrue *mat.Dense) *mat.Dense {
	delta := new(mat.Dense)
	delta.Sub(yest, ytrue)
	r, _ := delta.Dims()
	delta.Scale(1/float64(r), delta)
	return delta
}
func (*mseImpl) name() string { return "MSE (Mean Squared Error) cost function" }
