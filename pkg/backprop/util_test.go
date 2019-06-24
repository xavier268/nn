package backprop

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

const epsilon = 1e-20

func init() {
	rand.Seed(42)
}

// Check for almost null float64 value
func check(t *testing.T, v float64) {
	if math.Abs(v) >= epsilon {
		fmt.Printf("The value %f should have been nul or very small ?", v)
		t.Fail()
	}
}
