package backprop

import (
	"fmt"
	"testing"
)

func TestNewNetwork1(t *testing.T) {
	fmt.Println("Testing Network1")
	l1 := NewLayer(3, 4, ActivationSigmoid)
	l2 := NewLayer(4, 5, ActivationRelu)
	l3 := NewLayer(5, 2, ActivationIdentity)
	net := NewMLNetwork(l1, l2, l3)
	net.SetCost(MSE)
	net.RandomizeWeight()

	t.SkipNow()
	net.Dump()
}
