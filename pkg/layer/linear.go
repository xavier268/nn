package layer

import (
	"math/rand"

	"github.com/xavier268/nn/pkg/graph"
	"github.com/xavier268/nn/pkg/util"
	"gonum.org/v1/gonum/mat"
)

// Linear is a fully connected layer without activation
type Linear struct {
	*graph.Node
	w *mat.Dense
}

// Compiler check
var _ Processor = new(Linear)
var _ Trainable = new(Linear)

// NewLinear constructor. Weights are not initialized.
func NewLinear(name string, nin, nout int) *Linear {
	return &Linear{graph.NewNode(name), mat.NewDense(nin+1, nout, nil)}
}

// Forward computes and store forward values
// No recursion, assumes input is already available.
func (l *Linear) Forward() {
	x := util.AppendOnes(l.GetVin())
	x.Mul(x, l.w)
	l.SetVout(x)
}

// Init weights betwween -1 and +1
// Ignore params
func (l *Linear) Init(_ InitParam) {
	l.w.Apply(func(_, _ int, _ float64) float64 {
		return rand.Float64()*2. - 1.
	}, l.w)

}

//Backward : assume all inputs (X and DOut) are available,
// no recursion performed
// No need for previous forward (included)
func (l *Linear) Backward(doGrad bool) (grad *mat.Dense) {
	xx := util.AppendOnes(l.GetVin())
	dout := l.GetDeltaOut()
	din := new(mat.Dense)
	din.Mul(dout, l.w.T())
	l.SetDeltaIn(din)

	if !doGrad {
		return nil
	}
	grad = new(mat.Dense)
	grad.Mul(xx.T(), dout)
	return grad
}

// Train does backward, then performs a single training step on this block
func (l *Linear) Train(param TrainParam) {

	grad := l.Backward(true)
	// Adjust weigts with learning rate and regul
	if param.L2 > 0 { // Ignore regularization if l2 <= 0
		reg := new(mat.Dense)
		reg.Scale(param.L2, l.w)
		grad.Add(grad, reg)
	}
	grad.Scale(param.Learning, grad)
	l.w.Sub(l.w, grad)
}
