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

// InitWB generates random weights according to provided initialization method
func (net *Network) InitWB(initialization Initialization) *Network {
	for _, l := range net.layers {
		l.InitWB(initialization)
	}
	return net
}

// Cost estimate yest vs ground truth ytrue
func (net *Network) Cost(yest, ytrue *mat.Dense) float64 {
	return net.cost.cost(yest, ytrue)
}

// Evaluate the prediction on a validation batch
func (net *Network) Evaluate(x, ytrue *mat.Dense) float64 {
	return net.Cost(net.Predict(x), ytrue)
}

// Train learns from a minibatch,
// applying learning rate for 'steps'  steps
// Return the resulting achieved cost
func (net *Network) Train(x, ytrue *mat.Dense, learning float64, steps int) float64 {
	f := 0.
	for i := 0; i < steps; i++ {
		f = net.train1(x, ytrue, learning)
		if i%1000 == 0 {
			fmt.Printf("%d\tcost : %f\n", i, f)
		}
	}
	return f
}

// train1 learns from a minibatch,
// applying learning rate for ONE steps
// Return the resulting achieved cost
func (net *Network) train1(x, ytrue *mat.Dense, learning float64) float64 {

	// We store successif activation vectors in a, starting with input
	var a []*mat.Dense // activation(s) for each layers
	y := x
	a = append(a, y)
	for _, ll := range net.layers {
		y = ll.Forward(y)
		a = append(a, y)
	}
	yest := y // Last activation obtained

	// Compute initial delta
	delta := net.cost.grad(yest, ytrue)
	// Apply backprop backwards
	for i := len(net.layers) - 1; i >= 0; i-- {
		// Compute backprop gradient
		deltaIn, grad := net.layers[i].Backprop(a[i], delta)
		delta = deltaIn

		// Adjust weigts with learning rate
		grad.Scale(learning, grad)
		net.layers[i].w.Sub(net.layers[i].w, grad)
	}
	return net.Cost(yest, ytrue)
}
