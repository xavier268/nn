package mynn

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// NN defines a neural netwok interface
type NN interface {
	Train(x, y mat.Matrix)
	Evaluate(x, y mat.Matrix) float64
	Predict(x mat.Matrix) mat.Matrix
}

// MLP is a n layer perceptron network
type MLP struct {
	activation func(float64) float64 // the activation function
	weight     []mat.Matrix          // the weigh matrixes for all the layers, inc.biais
}

// NbLayers provides the total number of HIDDEN layers
func (mlp *MLP) NbLayers() int {
	return len(mlp.weight) - 1
}

// NbInput provides the number of input attributes
func (mlp *MLP) NbInput() int {
	a, _ := mlp.weight[0].Dims()
	return a - 1
}

// NbOutput provides the number of outputnodes
func (mlp *MLP) NbOutput() int {
	_, p := mlp.weight[len(mlp.weight)-1].Dims()
	return p
}

// NewMLP creates an MLP with the specified layers sizes,
// starting with input and finishing with output
func NewMLP(sizes ...int) *MLP {
	if len(sizes) < 2 {
		panic("There should be at least 2 layers - input + output")
	}
	mlp := &MLP{activation: Sigmoid}
	for i := 0; i < len(sizes)-1; i++ {
		fmt.Println("adding layer ", i)
		r := sizes[i] + 1
		c := sizes[i+1]
		w := mat.NewDense(r, c, nil)
		mlp.weight = append(mlp.weight, w)
	}
	return mlp
}

func (mlp *MLP) dump() {
	fmt.Println("Dumping a MLP Multi Layer Perceptron")
	fmt.Printf("Nb of hidden layers : %d\nNb of input %d, nb of output %d\n",
		mlp.NbLayers(), mlp.NbInput(), mlp.NbOutput())
	for i, w := range mlp.weight {
		r, c := w.Dims()
		fmt.Printf("Weights from layer %d (%d nodes) to layer %d (%d nodes) : matrix(%d x %d)\n",
			i, r-1, i+1, c, r, c)
		fmt.Println(mat.Formatted(w, mat.Squeeze()))
	}
}

// Forward on the input (n x a)
// where n is the number of records, and
// a is the number of input nodes (attributes)
func (mlp *MLP) Forward(in *mat.Dense) *mat.Dense {

	var x = in

	for _, w := range mlp.weight {
		r, _ := x.Dims()
		ones := NewConstantMat(r, 1, 1.0) // for the biais, r x 1
		y := new(mat.Dense)
		y.Augment(x, ones) // r x (c+1)
		x = new(mat.Dense)
		x.Mul(y, w) // r x (c+1) . (c+1) x p = r x p
		x.Apply(func(i, j int, v float64) float64 { return mlp.activation(v) }, x)
	}
	return x
}

// Sigmoid activation function is the default
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
