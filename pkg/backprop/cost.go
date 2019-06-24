package backprop

import "gonum.org/v1/gonum/mat"

// Cost is a cost function, providing either direct result or gradient
// It takes as input  estimated results (yest) and ground truth (ytruth).
// The gradient has the exact same dimension as the yest matrix.
type Cost func(yest, ytrue *mat.Dense, direct bool) (cost float64, grad *mat.Dense)

// MSE Mean Square Error ( 1/2n * sum of squares errors )
func MSE(yest, ytrue *mat.Dense, direct bool) (cost float64, grad *mat.Dense) {

	delta := new(mat.Dense)
	delta.Sub(ytrue, yest)
	r, c := delta.Dims()
	delta.Scale(1/float64(r), delta)

	if direct {
		// Direct evaluation
		res := 0.
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
