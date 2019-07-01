package ntwk

import (
	"gonum.org/v1/gonum/mat"
)

// Processor is the core processing node in the network.
// Nodes are used to process back/forward, input data, get output,
// log, evaluate, or or store weights.
// Activations are special cases of nodes
// There are connected from/to 0 or more processors
type Processor interface {
	ConnectTo(p Processor) error
	ConnectFrom(p Processor) error

	GetName() string
	GetID() int64

	// Forward computation
	Forward() *mat.Dense
	// Backward() *mat.Dense

	// Size in and out
	GetNin() int
	GetNout() int

	// Verify if input/output size match
	Verify() error
}
