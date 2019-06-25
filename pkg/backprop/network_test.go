package backprop

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestNewNetwork1(t *testing.T) {
	fmt.Println("Testing Network1")
	l0 := NewFCLayer(3, 4).SetActivation(ActivationRelu)
	l1 := NewFCLayer(4, 5).SetActivation(ActivationIdentity)
	l2 := NewFCLayer(5, 2).SetActivation(ActivationSigmoid)
	net := NewMLNetwork(l0, l1, l2)
	net.SetCost(CostMSE)
	net.RandomizeWeight()

	x := mat.NewDense(2, 3, []float64{
		0, 1, 2,
		3, 4, 5,
	})

	y := net.Predict(x)
	e := net.Cost(y, y)
	check(t, e)

	t.SkipNow()
	net.Dump()
	fmt.Println("x", mat.Formatted(x))
	fmt.Println("y", mat.Formatted(y))

}

func TestNetworkNetwork2(t *testing.T) {

	fmt.Println("Testing Network2 - grad on 1 line")
	l0 := NewFCLayer(3, 4).SetActivation(ActivationSigmoid)
	l1 := NewFCLayer(4, 5).SetActivation(ActivationSigmoid)
	l2 := NewFCLayer(5, 2).SetActivation(ActivationSigmoid)
	net := NewMLNetwork(l0, l1, l2)
	net.SetCost(CostMSE)
	net.RandomizeWeight()

	x := mat.NewDense(1, 3, []float64{
		0, 1, 2,
	})
	// Assume ground truth is zero
	ytrue := mat.NewDense(1, 2, []float64{
		2, 4,
	})

	net.bruteForcePartialDerivative(x, ytrue, 1e-5, 0) // Weight
	net.bruteForcePartialDerivative(x, ytrue, 1e-5, 1) // Weight
	net.bruteForcePartialDerivative(x, ytrue, 1e-5, 2) // Weight

	//t.SkipNow()

	net.gradDump(t, x, ytrue)
}

func TestNetworkNetwork3(t *testing.T) {

	fmt.Println("Testing Network3 - grad on multiple(3)  lines")
	l0 := NewFCLayer(3, 4).SetActivation(ActivationSigmoid)
	l1 := NewFCLayer(4, 5).SetActivation(ActivationSigmoid)
	l2 := NewFCLayer(5, 2).SetActivation(ActivationSigmoid)
	net := NewMLNetwork(l0, l1, l2)
	net.SetCost(CostMSE)
	net.RandomizeWeight()

	x := mat.NewDense(3, 3, []float64{
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
	})
	// Assume ground truth is zero
	ytrue := mat.NewDense(3, 2, []float64{
		2, 4,
		7, 8,
		5, 5,
	})

	net.gradDump(t, x, ytrue)
}

// Utilities

// gradDump displays the gradients for all layers using back prop
// Used  for testing/debugging ...
func (net *Network) gradDump(t *testing.T, x, ytrue *mat.Dense) {

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
		mse := CostMSE.cost(grad, gradbf)
		fmt.Printf("Comparing backprop gradient with brute force gradient for layer %d, MSE = %e \n", i, mse)
		check(t, mse)
	}
}
