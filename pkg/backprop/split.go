package backprop

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// SplitTrainTest splits the provided matrix in a train and test subset.
// The test represent 'ratio' of the total number of records.
func SplitTrainTest(x *mat.Dense, ratio float64) (train, test *mat.Dense) {

	n, c := x.Dims()                          // Ttl nbr of records
	nt := int(math.Round(ratio * float64(n))) // Number of test records
	train, test = new(mat.Dense), new(mat.Dense)
	train.CloneFrom(x.Slice(0, n-nt, 0, c))
	test.CloneFrom(x.Slice(n-nt, n, 0, c))
	return train, test
}

// Shuffle input matrix rows IN PLACE
// Shufle happens based on the numer of rows of the first matrix.
func Shuffle(xx ...*mat.Dense) {

	r, _ := xx[0].Dims()

	rand.Shuffle(r, func(i, j int) {
		for _, x := range xx {
			_, c := x.Dims()
			for cc := 0; cc < c; cc++ {
				a, b := x.At(i, cc), x.At(j, cc)
				x.Set(i, cc, b)
				x.Set(j, cc, a)
			}
		}
	})
}
