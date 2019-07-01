package util

import (
	"math"

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
