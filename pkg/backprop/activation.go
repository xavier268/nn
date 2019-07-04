package backprop

import "math"

// Activation defines an activation function , its derivative and its name
type Activation interface {
	f(float64) float64
	df(float64) float64
	name() string
}

// Providing gloal activation variables

// ActivationIdentity activation
var ActivationIdentity = new(identity)

// ActivationSigmoid provides an Activation structure for Sigmoid
var ActivationSigmoid = new(sigmoid)

// ActivationRelu is max(0,v)
var ActivationRelu = new(relu)

type sigmoid struct{}

func (*sigmoid) f(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}
func (*sigmoid) df(v float64) float64 {
	return 1 / ((1 + math.Exp(-v)) * (1 + math.Exp(v)))
}
func (*sigmoid) name() string { return "Sigmoid activation" }

type relu struct{}

func (*relu) f(v float64) float64 {
	if v > 0 {
		return v
	}
	return 0.
}
func (*relu) df(v float64) float64 {
	if v > 0 {
		return 1.
	}
	return 0.
}
func (*relu) name() string { return "Relu activation" }

type identity struct{}

func (*identity) f(v float64) float64 { return v }
func (*identity) df(float64) float64  { return 1. }
func (*identity) name() string        { return "Identity activation" }
