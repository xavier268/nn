package ntwk

import (
	"fmt"
	"testing"
)

func TestCreate(t *testing.T) {

	fmt.Println("Testing creation of ProcessorBasic")
	a := NewProcessorBasic("a")
	if a == nil {
		panic(t)
	}
	if a.Verify() != nil {
		panic(t)
	}

}

func TestContains(t *testing.T) {
	fmt.Println("Testing contains utility")
	a := NewProcessorBasic("a")
	b := NewProcessorBasic("b")
	m := []Processor{a, b}
	if !a.contains(m, b) {
		panic(t)
	}

}
func TestString(t *testing.T) {
	fmt.Println("Testing String()")
	a := NewProcessorBasic("a")
	b := NewProcessorBasic("b")
	a.ConnectTo(b)
	if a.Verify() != nil || b.Verify() != nil {
		panic(t)
	}
	// t.SkipNow()
	// Testing dump
	fmt.Println(a)
	fmt.Println(b)
}

func TestConn1(t *testing.T) {

	fmt.Println("Testing ProcessorBasic connections")

	a := NewProcessorBasic("a")
	b := NewProcessorBasic("b")
	c := NewProcessorBasic("c")
	d := NewProcessorBasic("d")

	// Nil connection
	a.ConnectFrom(nil)
	if len(a.To) != 0 ||
		len(a.From) != 0 {
		fmt.Println("\n", a)
		panic(t)
	}

	// Nil connection
	a.ConnectTo(nil)
	if len(a.To) != 0 ||
		len(a.From) != 0 {
		fmt.Println("\n", a)
		panic(t)
	}
	// real connection
	a.ConnectTo(b)
	if len(a.To) != 1 ||
		len(a.From) != 0 ||
		len(b.From) != 1 ||
		len(b.To) != 0 {
		fmt.Println("\n", a, "\n", b)
		panic(t)
	}

	// Attempt duplicate
	a.ConnectTo(b)
	if len(a.To) != 1 ||
		len(a.From) != 0 ||
		len(b.From) != 1 ||
		len(b.To) != 0 {
		fmt.Println("\n", a, "\n", b)
		panic(t)
	}
	// Attempt duplicate
	b.ConnectFrom(a)
	if len(a.To) != 1 ||
		len(a.From) != 0 ||
		len(b.From) != 1 ||
		len(b.To) != 0 {
		fmt.Println("\n", a, "\n", b)
		panic(t)
	}

	c.ConnectTo(d)
	if len(a.To) != 1 ||
		len(a.From) != 0 ||
		len(b.From) != 1 ||
		len(b.To) != 0 {
		fmt.Println("\n", a, "\n", b, "\n", c, "\n", d)
		panic(t)
	}
}
