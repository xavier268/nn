package backprop

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// SplitTrainTest splits the provided matrix in a train and test subset.
// ratio  = nb test records / nb ttl records
func SplitTrainTest(x *mat.Dense, ratio float64) (train, test *mat.Dense) {

	n, c := x.Dims()                          // Ttl nbr of records
	nt := int(math.Round(ratio * float64(n))) // Number of test records
	train, test = new(mat.Dense), new(mat.Dense)
	train.CloneFrom(x.Slice(0, n-nt, 0, c))
	test.CloneFrom(x.Slice(n-nt, n, 0, c))
	return train, test
}

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

// MaxCol return the indexes of the maximum columns, for each row
func MaxCol(x *mat.Dense) ([]int, []float64) {
	r, c := x.Dims()
	var (
		res  []int
		resv []float64
	)
	for i := 0; i < r; i++ {
		maxv, maxj := x.At(i, 0), 0
		for j := 1; j < c; j++ {
			if x.At(i, j) > maxv {
				maxv, maxj = x.At(i, j), j
			}
		}
		res = append(res, maxj)
		resv = append(resv, maxv)
	}
	return res, resv
}
