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

// NewFCLayer creates a new Fully Connected Layer with provided input, output entries.
// Default activation is identity
func NewFCLayer(in, out int) *Layer {
	lay := new(Layer)
	lay.nbin, lay.nbout = in, out
	// The w matrix contains both weights and biais(its last LINE), (in + 1) x (out)
	lay.w = mat.NewDense(in+1, out, nil)
	lay.act = ActivationIdentity
	return lay
}

// SetActivation to the requested activation mode
func (lay *Layer) SetActivation(activation Activation) *Layer {
	if activation != nil {
		lay.act = activation
	}
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
	fmt.Printf("Layer dump : %d nodes => %d nodes\n", lay.nbin, lay.nbout)
	fmt.Printf("Activation : %s\n", lay.act.name())
	fmt.Printf("Weight :\n%v\n", mat.Formatted(lay.Weights()))
	fmt.Printf("Bias :\n%v\n", mat.Formatted(lay.Biais()))
}

// Forward pass on mini batch x
// x has a line per record (n), and as many columns as entry nodes (in)
// returns the activated value (a)
// x is NOT modified
func (lay *Layer) Forward(x *mat.Dense) (a *mat.Dense) {
	// x is (n x in )
	row, _ := x.Dims()

	xx := new(mat.Dense)
	xx.Augment(x, NewConstantMat(row, 1, 1.0)) //  ( n x in+1 )
	z := new(mat.Dense)
	z.Mul(xx, lay.w) // (n x in + 1 ).(in +1  x out) = (n x out)

	z.Apply(func(i, j int, v float64) float64 { return lay.act.f(v) }, z)
	return z
}

// Backprop applies the backpropagation algorith to calculate the errors vectors
// deltaOut is n x out or n x out+1 matrix, where n is the number of records,
// and out is the number of output of the layer
// deltaOut is truncated (sliced) : only the n x out part is taken into account.
// deltaIn is an n x in + 1  matrix, where in is the number of inputs from the layer
// If deltaOut was the grad of C w.r. to the activations,
// then deltaIn is the grad of C w.r. to the activations one layer down.
// wgrad is the gradient of C w.r. to the weight and biais
//
//             TO DO : NEEDS REVISION & TEST !!
//
func (lay *Layer) Backprop(x *mat.Dense, deltaOut *mat.Dense) (deltaIn *mat.Dense, wgrad *mat.Dense) {
	// First, we do a partail forward calculation
	xx := new(mat.Dense)
	row, _ := x.Dims()
	xx.Augment(x, NewConstantMat(row, 1, 1.0)) //  ( n x in+1 )
	z := new(mat.Dense)
	z.Mul(xx, lay.w) // (n x in + 1 ).(in +1  x out) = (n x out)

	// res = deltaOut (n x out) *elementMultiply* sigmaPrime(z) (n x out)
	res := mat.NewDense(row, lay.nbout, nil)
	res.Apply(
		func(i, j int, v float64) float64 {
			return v * lay.act.df(z.At(i, j))
		}, deltaOut.Slice(0, row, 0, lay.nbout))

	deltaIn = mat.NewDense(row, lay.nbin+1, nil) // row x in+1
	deltaIn.Mul(res, lay.w.T())                  // (row x out) . (out x in + 1)) = row x in+1

	// Compute the gradient of the weight+biais matrix
	wgrad = mat.NewDense(lay.nbin+1, lay.nbout, nil) // in+1 x out
	wgrad.Mul(xx.T(), res)                           //   (in+1 x n)  x  (n x out) = (in+1 x out)

	return deltaIn, wgrad // (nxin+1), (in+1 x out)
}

// InitWB generates random weights and biaises
func (lay *Layer) InitWB(initialization Initialization) {
	initialization(lay)
}
