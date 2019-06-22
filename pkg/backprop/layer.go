package backprop

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Layer represent a single layer of neurons.
type Layer struct {
	nbin, nbout int
	w           *mat.Dense
	act         Activation
}

// NewLayer creates a new Layer with provided input, output entries.
func NewLayer(in, out int, activation Activation) *Layer {
	lay := new(Layer)
	lay.nbin, lay.nbout = in, out
	// The w matrix contains both weights and biais(its last LINE).
	lay.w = mat.NewDense(in+1, out, nil)
	lay.act = activation
	return lay
}

// Weights returns a view (slice) fo the weights
func (lay *Layer) Weights() mat.Matrix {
	return lay.w.Slice(0, lay.nbin, 0, lay.nbout)
}

// Biais returns a view (slice) of the Biais
func (lay *Layer) Biais() mat.Matrix {
	return lay.w.Slice(lay.nbin, lay.nbin+1, 0, lay.nbout)
}

// Dump will printout the layer in a readable format.
func (lay *Layer) Dump() {
	fmt.Printf("Layer dump : from %d nodes => %d nodes\n", lay.nbin, lay.nbout)
	fmt.Printf("Weight :\n%v\n", mat.Formatted(lay.Weights()))
	fmt.Printf("Bias :\n%v\n", mat.Formatted(lay.Biais()))
}

// Forward pass on mini batch x
// x has a line per record (n), and as many columns as entry nodes (in)
// returns both the linear combination (z) and (if requested from the flag)
// the activated value (a)
func (lay *Layer) Forward(x *mat.Dense, activate bool) (z, a *mat.Dense) {
	// x is (n x in )
	xx := new(mat.Dense)
	row, _ := x.Dims()
	xx.Augment(x, NewConstantMat(row, 1, 1.0)) //  ( n x in+1 )
	z = new(mat.Dense)
	z.Mul(xx, lay.w) // (n x in + 1 ).(in +1  x out) = (n x out)
	if !activate {
		return z, nil // activation was not computed
	}
	// Compute activation only if requested.
	a = new(mat.Dense)
	a.Apply(func(i, j int, v float64) float64 { return lay.act(v, true) }, z)
	return z, a
}

// Backprop applies the backpropagation algorith to calculate the errors vectors
// deltaOut is n x out matrix, where n is the number of records,
// and out is the number of output of the layer
// deltaIn is an n x in matrix, where in is the number of inputs from the layer
func (lay *Layer) Backprop(x *mat.Dense, deltaOut *mat.Dense) (deltaIn *mat.Dense) {

	// TODO
	return nil
}
