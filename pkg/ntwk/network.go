package ntwk

// Networker defines any network based upon Processor
type Networker interface {
	Reset()
	Init()
	// To predict, the source must have been set before manually
	// The result is available from the Y Processor
	Predict(Yest Processor) error
	// Assume all sources have been set beforehand.
	Train(X, Ytrue Processor, cost Coster, params TrainParam) error
}
