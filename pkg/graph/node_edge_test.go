package graph

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestConstr(t *testing.T) {
	fmt.Println("Testing constructing Edges and Nodes")

	n1 := NewNode("n1")
	n2 := NewNode("n2")
	e := Connect(n1, n2, 5)
	if e == nil {
		panic(t)
	}
	if len(n1.To) != 1 || len(n1.From) != 0 ||
		len(n2.To) != 0 || len(n2.From) != 1 {
		fmt.Println(n1)
		fmt.Println(n2)
		panic(t)
	}
}

func TestAggregatingValues(t *testing.T) {
	fmt.Println("Testing aggregating GetVin()")

	a := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	b := mat.NewDense(2, 2, []float64{
		11, 12,
		14, 15,
	})

	var v *mat.Dense
	n1 := NewNode("n1")

	n1.AddValueSource(a)
	v = n1.GetVin()
	if r, c := v.Dims(); r != 2 || c != 3 {
		fmt.Println(mat.Formatted(v, mat.Squeeze()))
		panic(t)
	}

	n1.AddValueSource(b)
	v = n1.GetVin()
	if r, c := v.Dims(); r != 2 || c != 5 {
		fmt.Println(mat.Formatted(v, mat.Squeeze()))
		panic(t)
	}
	n1.AddValueSource(a)
	v = n1.GetVin()
	if r, c := v.Dims(); r != 2 || c != 8 {
		fmt.Println(mat.Formatted(v, mat.Squeeze()))
		panic(t)
	}

	fmt.Println("Testing aggregating GetDeltaOut")

	n1 = NewNode("n1-delta")

	n1.AddDeltaOutSource(a)
	v = n1.GetDeltaOut()
	if v == nil {
		fmt.Println(n1)
	}
	if r, c := v.Dims(); r != 2 || c != 3 {
		fmt.Println(mat.Formatted(v, mat.Squeeze()))
		panic(t)
	}

	n1.AddDeltaOutSource(b)
	v = n1.GetDeltaOut()
	if r, c := v.Dims(); r != 2 || c != 5 {
		fmt.Println(mat.Formatted(v, mat.Squeeze()))
		panic(t)
	}
	n1.AddDeltaOutSource(a)
	v = n1.GetDeltaOut()
	if r, c := v.Dims(); r != 2 || c != 8 {
		fmt.Println(mat.Formatted(v, mat.Squeeze()))
		panic(t)
	}

}

func TestDispatchingValues(t *testing.T) {
	fmt.Println("Testing dispatching SetVout()")

	a := mat.NewDense(2, 7, []float64{
		1, 2, 3, 4, 5, 6, 7,
		8, 9, 10, 11, 12, 13, 14,
	})

	n0 := NewNode("n0")

	n1 := NewNode("n1")
	n2 := NewNode("n2")
	n3 := NewNode("n3")

	e1 := Connect(n0, n1, 3)
	e2 := Connect(n0, n2, 2)
	e3 := Connect(n0, n3, 2)

	n0.SetVout(a)

	if r, c := e1.V.Dims(); r != 2 || c != 3 || e1.V.At(0, 0) != 1 {
		e1.Dump()
		fmt.Println(mat.Formatted(a, mat.Squeeze()))
		fmt.Println(mat.Formatted(e1.V, mat.Squeeze()))
		panic(t)
	}
	if r, c := e2.V.Dims(); r != 2 || c != 2 || e2.V.At(0, 0) != 4 {
		e2.Dump()
		fmt.Println(mat.Formatted(a, mat.Squeeze()))
		fmt.Println(mat.Formatted(e2.V, mat.Squeeze()))
		panic(t)
	}
	if r, c := e3.V.Dims(); r != 2 || c != 2 || e3.V.At(0, 0) != 6 {
		e3.Dump()
		fmt.Println(mat.Formatted(a, mat.Squeeze()))
		fmt.Println(mat.Formatted(e3.V, mat.Squeeze()))
		panic(t)
	}
}

func TestDispatchnigDeltaIns(t *testing.T) {
	fmt.Println("Testing dispatching SetDeltaIn()")

	a := mat.NewDense(2, 7, []float64{
		1, 2, 3, 4, 5, 6, 7,
		8, 9, 10, 11, 12, 13, 14,
	})

	n0 := NewNode("n0")

	n1 := NewNode("n1")
	n2 := NewNode("n2")
	n3 := NewNode("n3")

	e1 := Connect(n1, n0, 3)
	e2 := Connect(n2, n0, 2)
	e3 := Connect(n3, n0, 2)

	n0.SetDeltaIn(a)

	if r, c := e1.Delta.Dims(); r != 2 || c != 3 || e1.Delta.At(0, 0) != 1 {
		e1.Dump()
		fmt.Println(mat.Formatted(a, mat.Squeeze()))
		fmt.Println(mat.Formatted(e1.Delta, mat.Squeeze()))
		panic(t)
	}
	if r, c := e2.Delta.Dims(); r != 2 || c != 2 || e2.Delta.At(0, 0) != 4 {
		e2.Dump()
		fmt.Println(mat.Formatted(a, mat.Squeeze()))
		fmt.Println(mat.Formatted(e2.Delta, mat.Squeeze()))
		panic(t)
	}
	if r, c := e3.Delta.Dims(); r != 2 || c != 2 || e3.Delta.At(0, 0) != 6 {
		e3.Dump()
		fmt.Println(mat.Formatted(a, mat.Squeeze()))
		fmt.Println(mat.Formatted(e3.Delta, mat.Squeeze()))
		panic(t)
	}

}
