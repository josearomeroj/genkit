package base_test

import (
	"math"
	"testing"

	"github.com/firebase/genkit/go/internal/base"
)

const epsilon = 1e-10

func almostEqual(a, b, tolerance float64) bool {
	return math.Abs(a-b) < tolerance
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name    string
		a       []float64
		b       []float64
		want    float64
		wantErr bool
		errMsg  string
	}{
		{
			name:    "identical vectors",
			a:       []float64{1, 2, 3},
			b:       []float64{1, 2, 3},
			want:    1.0,
			wantErr: false,
		},
		{
			name:    "opposite vectors",
			a:       []float64{1, 2, 3},
			b:       []float64{-1, -2, -3},
			want:    -1.0,
			wantErr: false,
		},
		{
			name:    "orthogonal vectors",
			a:       []float64{1, 0},
			b:       []float64{0, 1},
			want:    0.0,
			wantErr: false,
		},
		{
			name:    "proportional vectors (same direction)",
			a:       []float64{1, 2, 3},
			b:       []float64{2, 4, 6},
			want:    1.0,
			wantErr: false,
		},
		{
			name:    "simple 2D vectors",
			a:       []float64{3, 4},
			b:       []float64{4, 3},
			want:    0.96,
			wantErr: false,
		},
		{
			name:    "unit vectors at 45 degrees",
			a:       []float64{1, 0},
			b:       []float64{math.Sqrt(2) / 2, math.Sqrt(2) / 2},
			want:    math.Sqrt(2) / 2,
			wantErr: false,
		},
		{
			name:    "unit vectors at 60 degrees",
			a:       []float64{1, 0},
			b:       []float64{0.5, math.Sqrt(3) / 2},
			want:    0.5,
			wantErr: false,
		},

		// Edge cases with valid calculations
		{
			name:    "very small values",
			a:       []float64{0.0001, 0.0002},
			b:       []float64{0.0002, 0.0001},
			want:    0.8,
			wantErr: false,
		},
		{
			name:    "very large values",
			a:       []float64{1e10, 2e10},
			b:       []float64{2e10, 1e10},
			want:    0.8,
			wantErr: false,
		},
		{
			name:    "mixed positive and negative",
			a:       []float64{1, -2, 3, -4},
			b:       []float64{-1, 2, 3, 4},
			want:    -0.4,
			wantErr: false,
		},
		{
			name:    "single dimension vectors",
			a:       []float64{5},
			b:       []float64{3},
			want:    1.0,
			wantErr: false,
		},
		{
			name:    "single dimension opposite sign",
			a:       []float64{5},
			b:       []float64{-3},
			want:    -1.0,
			wantErr: false,
		},

		// Error cases
		{
			name:    "empty vector a",
			a:       []float64{},
			b:       []float64{1, 2, 3},
			want:    0,
			wantErr: true,
			errMsg:  "vectors cannot be empty",
		},
		{
			name:    "empty vector b",
			a:       []float64{1, 2, 3},
			b:       []float64{},
			want:    0,
			wantErr: true,
			errMsg:  "vectors cannot be empty",
		},
		{
			name:    "both vectors empty",
			a:       []float64{},
			b:       []float64{},
			want:    0,
			wantErr: true,
			errMsg:  "vectors cannot be empty",
		},
		{
			name:    "different dimensions",
			a:       []float64{1, 2, 3},
			b:       []float64{1, 2},
			want:    0,
			wantErr: true,
			errMsg:  "vectors must have the same dimension",
		},
		{
			name:    "zero vector a",
			a:       []float64{0, 0, 0},
			b:       []float64{1, 2, 3},
			want:    0,
			wantErr: true,
			errMsg:  "vectors cannot be zero vectors",
		},
		{
			name:    "zero vector b",
			a:       []float64{1, 2, 3},
			b:       []float64{0, 0, 0},
			want:    0,
			wantErr: true,
			errMsg:  "vectors cannot be zero vectors",
		},
		{
			name:    "both zero vectors",
			a:       []float64{0, 0, 0},
			b:       []float64{0, 0, 0},
			want:    0,
			wantErr: true,
			errMsg:  "vectors cannot be zero vectors",
		},

		// Numerical precision cases
		{
			name:    "near-zero but not zero vectors",
			a:       []float64{1e-100, 1e-100},
			b:       []float64{1e-100, 1e-100},
			want:    1.0,
			wantErr: false,
		},
		{
			name:    "high dimension vectors",
			a:       []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
			b:       []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
			want:    1.0,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := base.CosineSimilarity(tt.a, tt.b)

			if tt.wantErr {
				if err == nil {
					t.Errorf("CosineSimilarity() error = nil, wantErr %v", tt.wantErr)
					return
				}
				if err.Error() != tt.errMsg {
					t.Errorf("CosineSimilarity() error = %v, wantErr %v", err.Error(), tt.errMsg)
				}
				return
			}

			if err != nil {
				t.Errorf("CosineSimilarity() unexpected error = %v", err)
				return
			}

			if !almostEqual(got, tt.want, epsilon) {
				t.Errorf("CosineSimilarity() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCosineSimilarityGeneric(t *testing.T) {
	t.Run("float32 vectors", func(t *testing.T) {
		a := []float32{3, 4}
		b := []float32{4, 3}
		want := float32(0.96)

		got, err := base.CosineSimilarity(a, b)
		if err != nil {
			t.Errorf("CosineSimilarity() unexpected error = %v", err)
			return
		}

		if !almostEqual(float64(got), float64(want), epsilon) {
			t.Errorf("CosineSimilarity() = %v, want %v", got, want)
		}
	})

	t.Run("float64 vectors", func(t *testing.T) {
		a := []float64{1, 2, 3}
		b := []float64{2, 4, 6}
		want := 1.0

		got, err := base.CosineSimilarity(a, b)
		if err != nil {
			t.Errorf("CosineSimilarity() unexpected error = %v", err)
			return
		}

		if !almostEqual(got, want, epsilon) {
			t.Errorf("CosineSimilarity() = %v, want %v", got, want)
		}
	})
}
