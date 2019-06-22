package backprop

import "gonum.org/v1/gonum/mat"

// ConstantMat is a constant value matrix implementation
type ConstantMat struct {
	v    float64
	r, c int
}

// NewConstantMat generate a new const value matrix
func NewConstantMat(r, c int, v float64) *ConstantMat {
	return &ConstantMat{v, r, c}
}

// At implements the Matrix interface
func (cm *ConstantMat) At(i, j int) float64 {
	return cm.v
}

// Dims implements the Matrix interface
func (cm *ConstantMat) Dims() (r int, c int) {
	return cm.r, cm.c
}

// T implements the Matrix interface
func (cm *ConstantMat) T() mat.Matrix {
	cm.r, cm.c = cm.c, cm.r
	return cm
}
