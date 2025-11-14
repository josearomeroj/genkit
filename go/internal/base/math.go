package base

import (
	"errors"
	"math"

	"golang.org/x/exp/constraints"
)

func CosineSimilarity[T constraints.Float](a, b []T) (T, error) {
	if len(a) == 0 || len(b) == 0 {
		return 0, errors.New("vectors cannot be empty")
	}

	if len(a) != len(b) {
		return 0, errors.New("vectors must have the same dimension")
	}

	var (
		dotProduct T
		normA      T
		normB      T
	)

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0, errors.New("vectors cannot be zero vectors")
	}

	return dotProduct / T(math.Sqrt(float64(normA))*math.Sqrt(float64(normB))), nil
}
