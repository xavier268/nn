// Package graph implements node, edge and graph intefaces
// for a neural network
package graph

import (
	"fmt"

	"github.com/xavier268/nn/pkg/util"
	"gonum.org/v1/gonum/mat"
)

// Node is the interface Nodes comply with
type Node struct {
	ID   int64
	Name string
	To   []*Edge
	From []*Edge
}

// NewNode construct Node
func NewNode(name string) *Node {
	n := new(Node)
	n.ID = util.UID()
	n.Name = name
	return n
}

// String  (no carriage return at the start/end of string)
func (n *Node) String() string {
	return fmt.Sprintf("Node : %s (ID : %X)\nOutbound: %s\nInbound : %s",
		n.Name, n.ID,
		n.To,
		n.From,
	)
}

// GetVin gets aggregated input from inbound Edges
func (n *Node) GetVin() *mat.Dense {
	if len(n.From) == 0 {
		return nil
	}
	m := n.From[0].V
	if len(n.From) == 1 {
		return m
	}

	for i := 1; i < len(n.From); i++ {
		r := new(mat.Dense)
		r.Augment(m, n.From[i].V)
		m = r
	}
	return m
}

// SetVout set the outbound Edges values
// No more check/verification on dimensions at this stage
func (n *Node) SetVout(v *mat.Dense) {
	if len(n.To) == 0 {
		return
	}

	r, _ := v.Dims()
	c := 0
	for _, e := range n.To {
		e.V = mat.DenseCopyOf(v.Slice(0, r, c, c+e.W))
		c += e.W
	}
}

// GetDeltaOut get the Delta Out values from the outboud edges
func (n *Node) GetDeltaOut() *mat.Dense {
	if len(n.To) == 0 {
		return nil
	}
	m := n.To[0].Delta
	if len(n.To) == 1 {
		return m
	}

	for i := 1; i < len(n.To); i++ {
		r := new(mat.Dense)
		r.Augment(m, n.To[i].Delta)
		m = r
	}
	return m
}

// SetDeltaIn dispatches the delataIn  value to the inboud Edges
func (n *Node) SetDeltaIn(v *mat.Dense) {
	if len(n.From) == 0 {
		return
	}

	r, _ := v.Dims()
	c := 0
	for _, e := range n.From {
		e.Delta = mat.DenseCopyOf(v.Slice(0, r, c, c+e.W))
		c += e.W
	}
}

// AddValueSource attach a datasource to the node
// This is achieved by creating an Edge with the provided value
// This Edge is returned.
func (n *Node) AddValueSource(x *mat.Dense) *Edge {
	if x == nil {
		return nil
	}
	_, w := x.Dims()
	e := Connect(nil, n, w)
	e.V = x
	return e
}

// AddDeltaOutSource adds a Delata Out Source" as the target of the node
// NB : mainly used for testing/debugging
func (n *Node) AddDeltaOutSource(x *mat.Dense) *Edge {
	if x == nil {
		return nil
	}
	_, w := x.Dims()
	e := Connect(n, nil, w)
	e.Delta = x
	return e
}
