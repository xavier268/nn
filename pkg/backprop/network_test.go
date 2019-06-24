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

	net.gradDump(x, ytrue)
}

func TestNetworkNetwork3(t *testing.T) {

	fmt.Println("Testing Network3 - grad on multiple(3)  lines")
	l0 := NewFCLayer(3, 4).SetActivation(ActivationRelu)
	l1 := NewFCLayer(4, 5).SetActivation(ActivationRelu)
	l2 := NewFCLayer(5, 2).SetActivation(ActivationRelu)
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

	net.bruteForcePartialDerivative(x, ytrue, 1e-5, 0) // Weight
	net.bruteForcePartialDerivative(x, ytrue, 1e-5, 1) // Weight
	net.bruteForcePartialDerivative(x, ytrue, 1e-5, 2) // Weight

	//t.SkipNow()

	net.gradDump(x, ytrue)
}
