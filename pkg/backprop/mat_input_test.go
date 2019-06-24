package backprop

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestInputMat(t *testing.T) {
	fmt.Println("Testing matrix input entry order")
	data := []float64{
		1, 2, 3,
		4, 5, 6,
	}
	m := mat.NewDense(2, 3, data)

	// Check the oerder entry is row-first
	check(t, m.At(0, 2)-3)
	check(t, m.At(1, 1)-5)

	// Check that we can write to the slice ?
	m.Set(1, 1, 55.5)
	check(t, m.At(1, 1)-55.5)

	t.SkipNow()
	fmt.Println(mat.Formatted(m))
}
