package backprop

import (
	"fmt"
	"testing"

	"github.com/xavier268/gonum-demo/pkg/iris"
	"gonum.org/v1/gonum/mat"
)

func TestNewNetwork1(t *testing.T) {
	fmt.Println("Testing Network1")
	l0 := NewFCLayer(3, 4).SetActivation(ActivationRelu)
	l1 := NewFCLayer(4, 5).SetActivation(ActivationIdentity)
	l2 := NewFCLayer(5, 2).SetActivation(ActivationSigmoid)
	net := NewMLNetwork(l0, l1, l2)
	net.SetCost(CostMSE)
	net.InitWB(InitializationRandom)

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

	fmt.Println("Testing Network2 - grad on single line")
	l0 := NewFCLayer(3, 4).SetActivation(ActivationSigmoid)
	l1 := NewFCLayer(4, 5).SetActivation(ActivationIdentity)
	l2 := NewFCLayer(5, 2).SetActivation(ActivationRelu)
	net := NewMLNetwork(l0, l1, l2)
	net.SetCost(CostMSE)
	net.InitWB(InitializationRandom)

	x := mat.NewDense(1, 3, []float64{
		0, 1, 2,
	})
	// Assume ground truth is zero
	ytrue := mat.NewDense(1, 2, []float64{
		2, 4,
	})

	net.gradDump(t, x, ytrue)
}

func TestNetworkNetwork3Sigmoid(t *testing.T) {

	fmt.Println("Testing Network Sigmoid - grad on multiple(3)  lines")
	l0 := NewFCLayer(3, 4).SetActivation(ActivationSigmoid)
	l1 := NewFCLayer(4, 5).SetActivation(ActivationSigmoid)
	l2 := NewFCLayer(5, 2).SetActivation(ActivationSigmoid)
	net := NewMLNetwork(l0, l1, l2)
	net.SetCost(CostMSE)
	net.InitWB(InitializationRandom)

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
func TestNetworkNetwork3Relu(t *testing.T) {

	fmt.Println("Testing Network Relu - grad on multiple(3)  lines")
	l0 := NewFCLayer(3, 4).SetActivation(ActivationRelu)
	l1 := NewFCLayer(4, 5).SetActivation(ActivationRelu)
	l2 := NewFCLayer(5, 2).SetActivation(ActivationRelu)
	net := NewMLNetwork(l0, l1, l2)
	net.SetCost(CostMSE)
	net.InitWB(InitializationRandom)

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

func TestTrainIris(t *testing.T) {
	fmt.Println("Testing trainig on iris dataset")
	x, y := iris.GetIrisXY()

	NewMLNetwork(
		NewFCLayer(4, 5),
		NewFCLayer(5, 3)).
		SetCost(CostMSE).
		InitWB(InitializationRandom).
		Train(x, y, 1e-2, 1000)
}

// ******************************************************
//         Network specific testing utilities
// ******************************************************

// gradDump checks the gradients for all layers using back prop
// comparing with brute force gradient
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
