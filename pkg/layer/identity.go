package layer

import (
	"github.com/xavier268/nn/pkg/graph"
	"gonum.org/v1/gonum/mat"
)

// Identity is a basic Node that does nothing.
type Identity struct {
	*graph.Node
	size int
}

// Compiler check
var _ Processor = new(Identity)
var _ Trainable = new(Identity)

// NewIdentity constructor
func NewIdentity(name string, size int) *Identity {
	return &Identity{graph.NewNode(name), size}
}

// Forward computation
func (i *Identity) Forward() {
	i.SetVout(i.GetVin())
}

// Backward ignore gradient calculation
func (i *Identity) Backward(_ bool) *mat.Dense {
	i.SetDeltaIn(i.GetDeltaOut())
	return nil
}

// Init does nothing
func (i *Identity) Init(_ InitParam) {}

// Train does nothing
func (i *Identity) Train(_ TrainParam) {}
