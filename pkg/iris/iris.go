package iris

import (
	"bufio"
	"encoding/csv"
	"io"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

//Iris defines a record of iris data, with class info
type Iris struct {
	sl, sw, pl, pw float64
	cl             string
}

//Names defines the class names
var Names = make(map[string]int)

func readIrisFile() []Iris {
	var iris []Iris
	const data = "../../data/iris/iris.data"
	f, err := os.Open(data)
	defer f.Close()
	if err != nil {
		panic(err)
	}

	reader := csv.NewReader(bufio.NewReader(f))
	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
		i := Iris{}
		i.sl, err = strconv.ParseFloat(line[0], 64)
		if err != nil {
			panic(err)
		}
		i.sw, err = strconv.ParseFloat(line[1], 64)
		if err != nil {
			panic(err)
		}
		i.pl, err = strconv.ParseFloat(line[2], 64)
		if err != nil {
			panic(err)
		}
		i.pw, err = strconv.ParseFloat(line[3], 64)
		if err != nil {
			panic(err)
		}
		i.cl = line[4]
		if _, ok := Names[i.cl]; !ok {
			Names[i.cl] = len(Names)
		}
		iris = append(iris, i)
	}
	return iris
}

// GetXY reads iris data in the X Y matrix format
func GetXY() (X *mat.Dense, Y *mat.Dense) {
	iris := readIrisFile()
	r := len(iris)
	X = mat.NewDense(r, 4, nil)
	Y = mat.NewDense(r, len(Names), nil)
	X.Apply(func(i, j int, _ float64) float64 {
		switch j {
		case 0:
			return iris[i].sl
		case 1:
			return iris[i].sw
		case 2:
			return iris[i].pl
		case 3:
			return iris[i].pw
		}
		return 0.
	}, X)
	Y.Apply(func(i, j int, _ float64) float64 {
		if Names[iris[i].cl] == j {
			return 1.
		}
		return 0.
	}, Y)
	return X, Y
}
