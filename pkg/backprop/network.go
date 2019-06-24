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
	net.cost = MSE
	return net
}

// SetCost defines a new cost function for network
func (net *Network) SetCost(cost Cost) *Network {
	net.cost = cost
	if net.cost == nil {
		net.cost = MSE
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
	fmt.Printf("Dumping Network. \nSize : %d x %d \nNb of layers : %d\n",
		in, out, len(net.layers))
	for i, l := range net.layers {
		fmt.Printf("Layer N° %d/%d - ", i, len(net.layers))
		l.Dump()
	}
}

// Forward computes an estimate
func (net *Network) Forward(x *mat.Dense) *mat.Dense {
	y := new(mat.Dense)
	xx := x
	for _, l := range net.layers {
		y = l.Forward(xx)
	}
	return y
}

// RandomizeWeight generates random weights
func (net *Network) RandomizeWeight() *Network {
	for _, l := range net.layers {
		l.RandomizeWeight()
	}
	return net
}
