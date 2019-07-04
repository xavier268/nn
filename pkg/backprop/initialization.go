package backprop

import "math/rand"

// Initialization functions
type Initialization func(layer *Layer)

// InitializationRandom random uniform between -1 and 1
func InitializationRandom(lay *Layer) {
	lay.w.Apply(
		func(_ int, _ int, _ float64) float64 {
			return 2*rand.Float64() - 1
		}, lay.w)
}
