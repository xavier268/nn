package cost

import (
	"github.com/xavier268/nn/pkg/graph"
	"gonum.org/v1/gonum/mat"
)

// MSE : Mean Square Costing function (1/2n x sum of squares)
// used as parameter for training
type MSE struct {
	*graph.Node
}

// Compiler checks
var _ Coster = NewMSE()

//NewMSE constructor
func NewMSE() *MSE {
	return new(MSE)
}

// Name generic human friendly name
func (m *MSE) Name() string {
	return "MSE Activation"
}

// Cost compute cost
func (m *MSE) Cost(yest, ytrue *mat.Dense) float64 {
	delta := new(mat.Dense)
	delta.Sub(yest, ytrue)
	r, _ := delta.Dims()
	delta.MulElem(delta, delta)
	return mat.Sum(delta) / (2 * float64(r))
}

// Grad computes gradient
func (m *MSE) Grad(yest, ytrue *mat.Dense) *mat.Dense {
	delta := new(mat.Dense)
	delta.Sub(yest, ytrue)
	r, _ := delta.Dims()
	delta.Scale(1/float64(r), delta)
	return delta
}
