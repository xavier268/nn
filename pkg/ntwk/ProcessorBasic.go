package ntwk

import (
	"fmt"

	"github.com/xavier268/nn/pkg/util"
	"gonum.org/v1/gonum/mat"
)

// just checking compliance to interface ...
var _ Processor = new(ProcessorBasic)

//ProcessorBasic is a minimal implementation of the Processor interface
// It basically implements the identity.
type ProcessorBasic struct {

	// Define where do send the output of this processor
	// Repeat as needed
	// No duplication allowed
	To []Processor

	// From defines where we came from
	// repeat as needed
	// No duplication allowed
	From []Processor

	Name string
	ID   int64

	Nin, Nout int
}

// GetName returns the printable name
// No uniqueness assumed, only for readability and dumps
func (b *ProcessorBasic) GetName() string {
	return b.Name
}

// GetNin number of input neurons
func (b *ProcessorBasic) GetNin() int {
	return b.Nin
}

// GetNout number of output neurons
func (b *ProcessorBasic) GetNout() int {
	return b.Nout
}

// Forward computes and (recursively) return the output.
// Input is typically cached during the process
func (b *ProcessorBasic) Forward() *mat.Dense {

	if len(b.From) == 0 {
		return nil
	}
	if len(b.From) == 1 {
		return b.From[0].Forward()
	}

	m := new(mat.Dense)
	for _, p := range b.From {
		m.Augment(m, p.Forward())
	}
	return m
}

// Verify connection size
func (b *ProcessorBasic) Verify() error {
	in, out := 0, 0

	for _, i := range b.From {
		in += i.GetNout()
	}
	if in != b.Nin {
		return ErrConnectionSize
	}
	for _, i := range b.To {
		out += i.GetNin()
	}
	if out != b.Nout {
		return ErrConnectionSize
	}
	return nil
}

// GetID returns a unique ID of the Processor block
func (b *ProcessorBasic) GetID() int64 { return b.ID }

// ConnectTo establish a directed connection from receiver to p
// Duplications generate errors
func (b *ProcessorBasic) ConnectTo(p Processor) error {
	if p == nil {
		return nil
	}
	if p == b {
		panic(ErrSelfConnection)
	}
	// check duplicate
	if b.contains(b.To, p) {
		// already connected
		return nil
	}

	b.To = append(b.To, p)
	// NB : infinite recursion avoided because of previous duplicate check
	p.ConnectFrom(b)
	return nil
}

// ConnectFrom establish a directed connection from receiver to p
// Duplications generate errors
func (b *ProcessorBasic) ConnectFrom(p Processor) error {
	if p == nil {
		return nil
	}
	if p == b {
		panic(ErrSelfConnection)
	}
	// check duplicate
	if b.contains(b.From, p) {
		// already connected
		return nil
	}

	b.From = append(b.From, p)
	// NB : infinite recursion avoided because of previous duplicate check
	p.ConnectTo(b)
	return nil
}

// NewProcessorBasic constructor
func NewProcessorBasic(name string) *ProcessorBasic {
	p := new(ProcessorBasic)
	p.Name = name
	p.ID = util.UID()
	return p
}

// String - no starting nor ending carriage return
func (b *ProcessorBasic) String() string {

	var s string
	s = fmt.Sprintf("Name :\t %s (ID : %X, size %dx%d)", b.Name, b.GetID(), b.GetNin(), b.GetNout())
	s = s + fmt.Sprintf("\nConn in :")
	for _, p := range b.From {
		s = s + fmt.Sprint("\t", p.GetName())
	}
	s = s + fmt.Sprintf("\nConn out :")
	for _, p := range b.To {
		s = s + fmt.Sprint("\t", p.GetName())
	}

	return s
}

// Test if provided Processor array already contains a Processor ?
func (b *ProcessorBasic) contains(pp []Processor, p Processor) bool {
	for _, t := range pp {
		if p.GetID() == t.GetID() {
			return true
		}
	}
	return false
}
