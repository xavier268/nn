package backprop

import "gonum.org/v1/gonum/mat"

// Cost is a cost function, providing either direct result or gradient
// It is applied on the delta between estimated results and ground truth.
// The gradient has the exact same dimension as the input delta matrix.
type Cost func(delta *mat.Dense, direct bool) (cost float64, grad *mat.Dense)

// MSE Mean Square Error ( 1/2 * sum of squares errors )
func MSE(delta *mat.Dense, direct bool) (cost float64, grad *mat.Dense) {

	if direct {
		// Direct evaluation
		res := 0.
		r, c := delta.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				res = res + delta.At(i, j)*delta.At(i, j)
			}
		}
		return res / 2, nil
	}
	// Return gradient
	return -1, delta
}
