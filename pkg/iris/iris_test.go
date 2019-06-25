package iris

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestReadIris(t *testing.T) {
	//t.SkipNow()
	fmt.Println(readIrisFile())
	fmt.Println(IrisClassNames)
}

func TestDisplayXY(t *testing.T) {
	fmt.Println("Testing iris data availability")
	X, Y := GetIrisXY()
	fmt.Println("X\n", mat.Formatted(X, mat.Squeeze(), mat.Excerpt(3)))
	fmt.Println("Y\n", mat.Formatted(Y, mat.Squeeze(), mat.Excerpt(3)))
}
