package backprop

import (
	"fmt"
	"math"
	"testing"
)

func TestRelu(test *testing.T) {
	fmt.Println("Testing Relu")
	// Format is : input, direct value, derivative value
	table := [][]float64{
		{1, 1, 1},
		{2, 2, 1},
		{-1, 0, 0},
		{0, 0, 0},
	}

	for _, t := range table {
		check(test, ActivationRelu.f(t[0])-t[1])
		check(test, ActivationRelu.df(t[0])-t[2])
	}
}

func TestSigmoid(t *testing.T) {
	fmt.Println("Testing Sigmoid")
	check(t, ActivationSigmoid.f(0)-(1/(1+math.Exp(0))))
	check(t, ActivationSigmoid.df(0)-0.25)
	check(t, ActivationSigmoid.f(3)-(1/(1+math.Exp(-3))))
	check(t, ActivationSigmoid.df(12.56)-ActivationSigmoid.df(-12.56))
	check(t, ActivationSigmoid.f(200)-1)
	check(t, ActivationSigmoid.f(-200))

}
