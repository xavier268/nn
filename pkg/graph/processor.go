package graph

// Processor is a node capable of processing forward/backward
type Processor interface {
	Forward()
	Backward()
}
