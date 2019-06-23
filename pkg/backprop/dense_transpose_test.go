package backprop

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestTranspose(t *testing.T) {
	fmt.Println("Confirming transposition of Dense does NOT modify receiver")
	m := mat.NewDense(2, 4, nil)
	tm := m.T()
	r, c := m.Dims()
	tr, tc := tm.Dims()
	if r != 2 || c != 4 || tr != c || tc != r {
		fmt.Println(m.Dims())
		fmt.Println(tm.Dims())
		t.FailNow()
	}

}
