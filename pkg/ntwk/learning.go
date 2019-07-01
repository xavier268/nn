package ntwk

import "gonum.org/v1/gonum/mat"

// TrainParam parameter setting for training
type TrainParam struct {
	Learning, L2 float64
	Iteration    int
}

// InitParam param setting for init
type InitParam struct {
	F func(w *mat.Dense)
}

// Coster is the interface for all costing functions
type Coster struct {
	// Function
	F func(yest, ytrue *mat.Dense) float64
	// Gradient
	G func(yest, ytrue *mat.Dense) *mat.Dense
	// Name
	Name string
}
