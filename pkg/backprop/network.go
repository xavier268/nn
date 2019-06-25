package backprop

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Network is a multilayer network
type Network struct {
	layers []*Layer
	cost   Cost
}

// NewMLNetwork creates a multilayer network
// default cost is MSE
func NewMLNetwork(layers ...*Layer) *Network {
	// Check consistency ...

	for i := 0; i < len(layers)-1; i++ {
		if layers[i].nbout != layers[i+1].nbin {
			panic("Mismatch layers dimensions ?!")
		}
	}
	net := new(Network)
	net.layers = layers
	net.cost = CostMSE
	return net
}

// SetCost defines a new cost function for network
func (net *Network) SetCost(cost Cost) *Network {
	net.cost = cost
	if net.cost == nil {
		net.cost = CostMSE
	}
	return net
}

// Dims provides the input/output dimensions of the network
func (net *Network) Dims() (in, out int) {
	in = net.layers[0].nbin
	out = net.layers[len(net.layers)-1].nbout
	return
}

// Dump the network
func (net *Network) Dump() {
	in, out := net.Dims()
	fmt.Printf("Dumping Network. \nSize : %d x %d \nCost function: %s\nNb of layers : %d\n",
		in, out, net.cost.name(), len(net.layers))
	for i, l := range net.layers {
		fmt.Printf("Layer N° %d/%d\n", i, len(net.layers))
		l.Dump()
	}
}

// Predict computes an estimate, using forward through the layers
func (net *Network) Predict(x *mat.Dense) *mat.Dense {
	y := new(mat.Dense)
	xx := x
	for _, l := range net.layers {
		y = l.Forward(xx)
		xx = y
	}
	return y
}

// RandomizeWeight generates random weights
func (net *Network) RandomizeWeight() *Network {
	for _, l := range net.layers {
		l.InitWB(InitializationRandom)
	}
	return net
}

// Cost estimate yest vs ground truth ytrue
func (net *Network) Cost(yest, ytrue *mat.Dense) float64 {
	return net.cost.cost(yest, ytrue)
}

// bruteForcePartialDerivative gets the brute force derivative
// for cost of input x and ground truth ytrue
// w.r.t. the weight+biais matrix of the l-th layer
// No check on l  - will panic if out of range
// Use for TESTING only, very slow
// This is NOT THREAD SAFE as it MODIFIES NETWORK, then put it back on exit
func (net *Network) bruteForcePartialDerivative(x, ytrue *mat.Dense, epsilon float64, l int) *mat.Dense {

	r, c := net.layers[l].w.Dims()
	g := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			// Initial cost
			c1 := net.Cost(net.Predict(x), ytrue)
			// Now, we slightly change the weight
			v := net.layers[l].w.At(i, j)
			net.layers[l].w.Set(i, j, v+epsilon)
			// Compute modified cost
			c2 := net.Cost(net.Predict(x), ytrue)
			// Restore weight back to former value
			net.layers[l].w.Set(i, j, v)
			// Store partial derivative
			g.Set(i, j, (c2-c1)/epsilon)
		}
	}
	return g
}

// gradDump displays the gradients for all layers using back prop
// Used  for testing/debugging ...
func (net *Network) gradDump(x, ytrue *mat.Dense) {

	var a []*mat.Dense
	y := x
	a = append(a, y)
	// We store successif activation vectors in a, starting with input
	for _, ll := range net.layers {
		y = ll.Forward(y)
		a = append(a, y)
	}
	yest := y
	// Compute initial delta
	delta := net.cost.grad(yest, ytrue)
	// Apply backprop backwards
	for i := len(net.layers) - 1; i >= 0; i-- {
		// Compute backprop gradient
		deltaIn, grad := net.layers[i].Backprop(a[i], delta)
		delta = deltaIn
		// compute brute force gradient, with epsilon = 1e-6
		gradbf := net.bruteForcePartialDerivative(x, ytrue, 1e-6, i)
		// display Mean Squared Error between both gradients.
		fmt.Printf("Comparing backprop gradient with brute force gradient for layer %d, MSE = %e \n", i, CostMSE.cost(grad, gradbf))

	}

}
