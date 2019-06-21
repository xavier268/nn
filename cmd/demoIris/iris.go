package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

func main() {
	fmt.Println(readIrisFile())
	fmt.Println(IrisClassNames)
}

//Iris defines a record of iris data, with class info
type Iris struct {
	sl, sw, pl, pw float64
	cl             string
}

//IrisClassNames defines the class names
var IrisClassNames = make(map[string]int)

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
		if _, ok := IrisClassNames[i.cl]; !ok {
			IrisClassNames[i.cl] = len(IrisClassNames)
		}
		iris = append(iris, i)
	}
	return iris

}
