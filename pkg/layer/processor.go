package layer

import (
	"github.com/xavier268/nn/pkg/cost"
	"gonum.org/v1/gonum/mat"
)

// Processor are Node that can compute for/backward and update Edges
type Processor interface {
	Forward()
	Backward(computeGrad bool) *mat.Dense
}

// Trainable is a Node that can both Process and be Trained
type Trainable interface {
	Processor
	Init(param InitParam)
	Train(param TrainParam)
}

// TrainParam parameters
type TrainParam struct {
	Learning, L2 float64
	Cost         cost.Coster
	Rounds       int
}

// InitParam parameters
type InitParam struct {
	// Reserved for future use
}
