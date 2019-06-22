package backprop

import "math"

// Activation defines an activation function and its derivative
// direct = true for the function, false for the derivative
type Activation func(v float64, direct bool) float64

// Sigmoid activation
func Sigmoid(v float64, direct bool) float64 {
	if direct {
		return 1 / (1 + math.Exp(-v))
	}
	return 1 / ((1 + math.Exp(-v)) * (1 + math.Exp(v)))

}

// Relu is max(0,v)
func Relu(v float64, direct bool) float64 {
	if direct {
		if v > 0 {
			return v
		}
	}
	if v > 0 {
		return 1.
	}
	return 0.
}

// Identity activation
func Identity(v float64, direct bool) float64 {
	if direct {
		return v
	}
	return 1.
}
