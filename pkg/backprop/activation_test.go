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
		check(test, Relu(t[0], true)-t[1])
		check(test, Relu(t[0], false)-t[2])
	}
}

func TestSigmoid(t *testing.T) {
	fmt.Println("Testing Sigmoid")
	check(t, Sigmoid(0, true)-(1/(1+math.Exp(0))))
	check(t, Sigmoid(0, false)-0.25)
	check(t, Sigmoid(3, true)-(1/(1+math.Exp(-3))))
	check(t, Sigmoid(12.56, false)-Sigmoid(-12.56, false))
	check(t, Sigmoid(200, true)-1)
	check(t, Sigmoid(-200, true))
}

func check(t *testing.T, v float64) {
	if math.Abs(v) >= 1e-20 {
		fmt.Printf("The value %f should have been nul or very small ?", v)
		t.Fail()
	}
}
