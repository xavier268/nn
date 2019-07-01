package graph

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Edge contain states (values and deltas)
type Edge struct {
	// Width of the neuron 'bus'
	W        int
	From, To *Node
	V        *mat.Dense
	Delta    *mat.Dense
}

// Connect construct Edge of width w and connect both nodes
// using this Edge
func Connect(f, t *Node, w int) *Edge {
	if f != nil && t != nil && f.ID == t.ID {
		// Attempting connection to self !
		panic("Attempted connection to self !?")
	}
	e := new(Edge)
	e.W = w
	e.From, e.To = f, t
	if t != nil {
		t.From = append(t.From, e)
	}
	if f != nil {
		f.To = append(f.To, e)
	}
	return e
}

// String with no carriage return before or after
func (e *Edge) String() string {

	var s string
	if e.From != nil {
		s = s + e.From.Name
	}
	s = s + fmt.Sprintf(" >%d> ", e.W)
	if e.To != nil {
		s = s + e.To.Name
	}
	return s + ", "
}

// Dump edge with current values/Delta
func (e *Edge) Dump() {
	fmt.Printf("\n *** EDGE DUMP ****\n%v\nValue : \n", e)

	if e.V == nil {
		fmt.Println("   NOT SET")
	} else {
		fmt.Println(mat.Formatted(e.V, mat.Squeeze(), mat.Excerpt(3)))
	}
	fmt.Println("Delta :")
	if e.Delta == nil {
		fmt.Println("   NOT SET")
	} else {
		fmt.Println(mat.Formatted(e.Delta, mat.Squeeze(), mat.Excerpt(3)))
	}

}
