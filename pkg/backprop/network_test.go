package backprop

import "testing"

func TestNewNetwork1(t *testing.T) {

	l1 := NewLayer(3, 4, Sigmoid)
	l2 := NewLayer(4, 5, Relu)
	l3 := NewLayer(5, 2, Identity)
	net := NewMLNetwork(l1, l2, l3)
	net.SetCost(MSE)
	net.RandomizeWeight().Dump()
}
