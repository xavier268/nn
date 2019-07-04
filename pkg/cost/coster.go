package cost

import "gonum.org/v1/gonum/mat"

// Coster is iùmplemented for all cost functions
type Coster interface {
	Cost(yest, ytrue *mat.Dense) float64    // cost
	Grad(yest, ytrue *mat.Dense) *mat.Dense // gradient
	Name() string                           // human friendly name
}
