package util

import (
	"math/rand"
	"time"
)

func init() {
	// Initialize myrand to ensure uniqueness upon each restart
	myrand = rand.New(rand.NewSource(time.Now().Unix()))
}

var myrand *rand.Rand

// UID gerenates a unique int64 number each time it is called.
func UID() int64 {
	return myrand.Int63()
}
