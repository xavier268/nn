package layer

import (
	"github.com/xavier268/nn/pkg/graph"
	"gonum.org/v1/gonum/mat"
)

// Relu activation node
type Relu struct {
	*graph.Node
}

// Compiler checks
var _ Processor = new(Relu)
var _ Trainable = new(Relu)

// NewRelu constructor
func NewRelu() *Relu {
	return new(Relu)
}

// Forward updates edges forward
func (r *Relu) Forward() {
	vin := r.GetVin()
	vout := new(mat.Dense)
	vout.Apply(func(_, _ int, v float64) float64 {
		if v < 0 {
			return 0.
		}
		return v
	}, vin)
	r.SetVout(vout)
}

// Backward - no gradient
func (r *Relu) Backward(_ bool) *mat.Dense {
	vin := r.GetVin()
	dout := r.GetDeltaOut()
	din := new(mat.Dense)
	din.Apply(func(i, j int, v float64) float64 {
		if v < 0 {
			return 0.
		}
		return dout.At(i, j)
	}, vin)
	r.SetDeltaIn(din)
	return nil
}

// Init does nothing
func (r *Relu) Init(_ InitParam) {}

// Train just updates backwards, no training
func (r *Relu) Train(_ TrainParam) { r.Backward(true) }
