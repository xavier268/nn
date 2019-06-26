package backprop

import (
	"fmt"
	"testing"

	"github.com/xavier268/gonum-demo/pkg/iris"
	"gonum.org/v1/gonum/mat"
)

func TestSplit(t *testing.T) {

	fmt.Println("Testing splitting train vs. test")
	display := false // Set to true to display debug output

	x, _ := iris.GetIrisXY()
	xr, xc := x.Dims()
	tr, ts := SplitTrainTest(x, 0.3)
	rtr, ctr := tr.Dims()
	rts, cts := ts.Dims()

	if display {
		fmt.Println(mat.Formatted(x, mat.Squeeze(), mat.Excerpt(10)))
		fmt.Println(mat.Formatted(tr, mat.Squeeze(), mat.Excerpt(10)))
		fmt.Println(mat.Formatted(ts, mat.Squeeze(), mat.Excerpt(10)))
	}

	check(t, float64(xc-ctr))
	check(t, float64(xc-cts))
	check(t, float64(xr-rtr-rts))

}

func TestShuffle(t *testing.T) {
	fmt.Println("Testing shuffle")
	display := false // Set to true to print debug output
	m := mat.NewDense(5, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		13, 14, 15,
	})
	if display {
		fmt.Println("Before shuffling 1 matrix")
		fmt.Println(mat.Formatted(m, mat.Squeeze()))
	}
	Shuffle(m)
	if display {
		fmt.Println("After shuffling")
		fmt.Println(mat.Formatted(m, mat.Squeeze()))
	}
	if m.At(0, 0) == 1 || m.At(1, 0) == 4 || m.At(2, 0) == 7 || m.At(3, 0) == 10 {
		fmt.Println("Shuffling was incomplete ?!")
		t.FailNow()
	}

}

func TestShuffle2(t *testing.T) {
	fmt.Println("Testing shuffle 2 matrices")
	display := false // Set to true to print debugging output

	x := mat.NewDense(5, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		13, 14, 15,
	})
	y := mat.NewDense(5, 2, []float64{
		101, 102,
		401, 402,
		701, 702,
		1001, 1002,
		1301, 1302,
	})
	if display {
		fmt.Println("Before shuffling")
		fmt.Println(mat.Formatted(x, mat.Squeeze()))
		fmt.Println(mat.Formatted(y, mat.Squeeze()))
	}
	Shuffle(x, y)
	if display {
		fmt.Println("After shuffling")
		fmt.Println(mat.Formatted(x, mat.Squeeze()))
		fmt.Println(mat.Formatted(y, mat.Squeeze()))
	}
	r, _ := x.Dims()
	// Actual check on shuffle
	for i := 0; i < r; i++ {
		if x.At(i, 1)-x.At(i, 0) != 1 || x.At(i, 2)-x.At(i, 1) != 1 || y.At(i, 1)-y.At(i, 0) != 1 {
			t.FailNow()
		}
	}
}
