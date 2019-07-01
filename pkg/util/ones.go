package util

import "gonum.org/v1/gonum/mat"

// Ones returns a n x 1 matrix of 1.0
type Ones struct {
	int
}

var _ mat.Matrix = new(Ones)

// NewOnes construct a size x 1 matrix of ones
func NewOnes(size int) *Ones {
	return &Ones{size}
}

// AppendOnes append a column of Ones to the matrix
func AppendOnes(x mat.Matrix) *mat.Dense {
	r, c := x.Dims()
	m := mat.NewDense(r, c+1, nil)
	m.Augment(x, NewOnes(r))
}

// At see Matrix
func (o *Ones) At(i, j int) float64 {
	return 1.0
}

// Dims see Matrix
func (o *Ones) Dims() (int, int) {
	return o.int, 1
}

// T see Matrix, transforms into a Dense
func (o *Ones) T() mat.Matrix {
	m := mat.DenseCopyOf(o).T()
	return m
}
