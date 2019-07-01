package util

import "gonum.org/v1/gonum/mat"

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
