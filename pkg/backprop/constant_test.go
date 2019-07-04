package backprop

import (
	"fmt"
	"testing"
)

func TestConstMat(t *testing.T) {
	fmt.Println("Testing constant matrix")
	one := NewConstantMat(2, 5, 1.0)
	check(t, one.At(2, 3)-1)
	//fmt.Println(one.Dims())
	r, c := one.Dims()
	check(t, float64(r-2))
	check(t, float64(c-5))
	tone := one.T()
	check(t, one.At(2, 3)-1)
	// No boundary checks are performed ...
	check(t, one.At(20, 30)-1)

	// Checking that previous transpose was NOT applied applied to the receiver matrix.
	r, c = one.Dims()
	//fmt.Println(one.Dims())
	check(t, float64(r-2))
	check(t, float64(c-5))

	// Checking that transpose RESULT is transposed.
	r, c = tone.Dims()
	//fmt.Println(one.Dims())
	check(t, float64(r-5))
	check(t, float64(c-2))

}

func check(t *testing.T, v float64) {
	if v*v > 1e-15 {
		fmt.Println("Value ", v, " should have been 0 or at least very small")
		panic(t)
	}
}
