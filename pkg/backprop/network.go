package backprop

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Network is a multilayer network
type Network struct {
	layers []*Layer
	cost   Cost
	lr     float64 // Learning rate >0
	l2     float64 // l2 regularization ratio 0 : none, otherwise >0
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
	net.lr = 1e-10
	net.l2 = 0
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
