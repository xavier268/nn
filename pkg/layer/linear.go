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

// Forward computes and stores forward values
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

// Backward : compute and stores deltas.
// Assumes values are available (both V and Delta)
func (l *Linear) Backward(doGrad bool) (grad *mat.Dense) {

	dout := l.GetDeltaOut() // n x out
	din := new(mat.Dense)
	din.Mul(dout, l.w.T()) // (n x out )x(out x in+1) => (n x in+1)
	l.SetDeltaIn(din)      // Removing extra column not necessary as it will not be passed to the Edges ?
	if !doGrad {
		// No grad requested
		return nil
	}
	x := util.AppendOnes(l.GetVin())
	grad = new(mat.Dense)
	grad.Mul(x.T(), dout)
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