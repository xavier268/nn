package util

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Shuffle input matrix rows IN PLACE
// All matrix should have same number of rows
// Shufle happens based on the number of rows of the first matrix.
func Shuffle(xx ...*mat.Dense) {

	r, _ := xx[0].Dims()

	rand.Shuffle(r, func(i, j int) {
		for _, x := range xx {
			rr, c := x.Dims()
			if rr != r {
				panic("All input matrices should have same number of rows")
			}
			for cc := 0; cc < c; cc++ {
				a, b := x.At(i, cc), x.At(j, cc)
				x.Set(i, cc, b)
				x.Set(j, cc, a)
			}
		}
	})
}
