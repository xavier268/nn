package ntwk

import "errors"

// ErrNotReady error to signal that Processor we depend upon are not ready for computation.
var ErrNotReady = errors.New("Processor not ready")

// ErrDuplicateConnection when building network
var ErrDuplicateConnection = errors.New("Duplicatied connections")

// ErrSelfConnection when building network
var ErrSelfConnection = errors.New("Attempt to connect to self")

// ErrConnectionSize when connection sizes do not match
var ErrConnectionSize = errors.New("Connection size do not match declared size")

// ErrNilValue when unexpected nil value
var ErrNilValue = errors.New("Unexpected nil value")
