package evaluators_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/evaluators"
)

func TestAnswerRelevancyEvaluator(t *testing.T) {
	metrics := []evaluators.MetricConfig{
		{
			MetricType: evaluators.EvaluatorAnswerRelevancy,
			JudgeLLM:   newMockJudgeLLMForRelevancy(t, true, false, nil),
			Embedder:   newDumbEmbedder(t, 2),
		},
	}

	g := genkit.Init(t.Context(), genkit.WithPlugins(&evaluators.GenkitEval{Metrics: metrics}))
	genkit.RegisterAction(g, (metrics[0].JudgeLLM.(ai.Model)))
	genkit.RegisterAction(g, (metrics[0].Embedder.(ai.Embedder)))

	t.Run("answer relevancy with valid data", func(t *testing.T) {
		dataset := []*ai.Example{
			{
				Input:      "What is the capital of France?",
				Output:     "The capital of France is Paris.",
				Context:    []any{"France is a country in Europe", "Paris is the capital city"},
				TestCaseId: "test-1",
			},
		}

		testRequest := ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_relevancy")
		if evalAction == nil {
			t.Fatal("evalAction is nil")
		}

		resp, err := evalAction.Evaluate(t.Context(), &testRequest)
		if err != nil {
			t.Fatalf("Evaluate() error = %v", err)
		}

		if len(*resp) != 1 {
			t.Fatalf("got %d responses, want 1", len(*resp))
		}

		result := (*resp)[0]
		if result.TestCaseId != "test-1" {
			t.Errorf("got TestCaseId %q, want %q", result.TestCaseId, "test-1")
		}

		if len(result.Evaluation) != 1 {
			t.Fatalf("got %d evaluations, want 1", len(result.Evaluation))
		}

		score := result.Evaluation[0]

		if scoreFloat, ok := score.Score.(float32); !ok {
			t.Errorf("score is not float32, got %T", score.Score)
		} else if scoreFloat != 1.0 {
			t.Errorf("got score %v, want 1.0", scoreFloat)
		}

		if score.Status != ai.ScoreStatusPass.String() {
			t.Errorf("got status %q, want %q", score.Status, ai.ScoreStatusPass.String())
		}
	})

	t.Run("answer relevancy with missing input", func(t *testing.T) {
		dataset := []*ai.Example{
			{
				Input:      nil,
				Output:     "Paris",
				Context:    []any{"France is a country"},
				TestCaseId: "test-missing-input",
			},
		}

		testRequest := ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_relevancy")
		if evalAction == nil {
			t.Fatal("evalAction is nil")
		}

		resp, err := evalAction.Evaluate(t.Context(), &testRequest)
		if err != nil {
			t.Fatalf("Evaluate() error = %v", err)
		}

		if len(*resp) != 1 {
			t.Fatalf("got %d responses, want 1", len(*resp))
		}

		result := (*resp)[0]
		if result.Evaluation[0].Error == "" {
			t.Error("expected error for missing input, got none")
		}
	})

	t.Run("answer relevancy with missing output", func(t *testing.T) {
		dataset := []*ai.Example{
			{
				Input:      "What is the capital of France?",
				Output:     nil,
				Context:    []any{"France is a country"},
				TestCaseId: "test-missing-output",
			},
		}

		testRequest := ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_relevancy")
		resp, err := evalAction.Evaluate(t.Context(), &testRequest)
		if err != nil {
			t.Fatalf("Evaluate() error = %v", err)
		}

		if len(*resp) != 1 {
			t.Fatalf("got %d responses, want 1", len(*resp))
		}

		result := (*resp)[0]
		if result.Evaluation[0].Error == "" {
			t.Error("expected error for missing output, got none")
		}
	})

	t.Run("answer relevancy with missing context", func(t *testing.T) {
		dataset := []*ai.Example{
			{
				Input:      "What is the capital of France?",
				Output:     "Paris",
				Context:    nil,
				TestCaseId: "test-missing-context",
			},
		}

		testRequest := ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_relevancy")
		resp, err := evalAction.Evaluate(t.Context(), &testRequest)
		if err != nil {
			t.Fatalf("Evaluate() error = %v", err)
		}

		if len(*resp) != 1 {
			t.Fatalf("got %d responses, want 1", len(*resp))
		}

		result := (*resp)[0]
		if result.Evaluation[0].Error == "" {
			t.Error("expected error for missing context, got none")
		}
	})

	t.Run("answer relevancy with empty context", func(t *testing.T) {
		dataset := []*ai.Example{
			{
				Input:      "What is the capital of France?",
				Output:     "Paris",
				Context:    []any{},
				TestCaseId: "test-empty-context",
			},
		}

		testRequest := ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_relevancy")
		resp, err := evalAction.Evaluate(t.Context(), &testRequest)
		if err != nil {
			t.Fatalf("Evaluate() error = %v", err)
		}

		if len(*resp) != 1 {
			t.Fatalf("got %d responses, want 1", len(*resp))
		}

		result := (*resp)[0]
		if result.Evaluation[0].Error == "" {
			t.Error("expected error for empty context, got none")
		}
	})
}

func TestAnswerRelevancyScoring(t *testing.T) {
	tests := []struct {
		name           string
		answered       bool
		noncommittal   bool
		similarity     float32
		wantFinalScore float32
		wantStatus     string
		wantReasoning  string
	}{
		{
			name:           "fully answered and relevant",
			answered:       true,
			noncommittal:   false,
			similarity:     1.0,
			wantFinalScore: 1.0,
			wantStatus:     ai.ScoreStatusPass.String(),
			wantReasoning:  "Cosine similarity",
		},
		{
			name:           "not answered with penalty",
			answered:       false,
			noncommittal:   false,
			similarity:     0.8,
			wantFinalScore: 0.3, // 0.8 - 0.5 penalty
			wantStatus:     ai.ScoreStatusFail.String(),
			wantReasoning:  "Cosine similarity with penalty for insufficient answer",
		},
		{
			name:           "noncommittal response",
			answered:       true,
			noncommittal:   true,
			similarity:     0.9,
			wantFinalScore: 0,
			wantStatus:     ai.ScoreStatusFail.String(),
			wantReasoning:  "Noncommittal",
		},
		{
			name:           "low similarity but answered",
			answered:       true,
			noncommittal:   false,
			similarity:     0.4,
			wantFinalScore: 0.4,
			wantStatus:     ai.ScoreStatusFail.String(),
			wantReasoning:  "Cosine similarity",
		},
		{
			name:           "not answered with penalty below zero",
			answered:       false,
			noncommittal:   false,
			similarity:     0.3,
			wantFinalScore: 0, // max(0.3 - 0.5, 0) = 0
			wantStatus:     ai.ScoreStatusFail.String(),
			wantReasoning:  "Cosine similarity with penalty for insufficient answer",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockEmbedder := createMockEmbedderWithSimilarity(t, tt.similarity)
			mockModel := newMockJudgeLLMForRelevancy(t, tt.answered, tt.noncommittal, nil)

			metrics := []evaluators.MetricConfig{
				{
					MetricType: evaluators.EvaluatorAnswerRelevancy,
					JudgeLLM:   mockModel,
					Embedder:   mockEmbedder,
				},
			}

			g := genkit.Init(t.Context(), genkit.WithPlugins(&evaluators.GenkitEval{Metrics: metrics}))
			genkit.RegisterAction(g, mockModel)
			genkit.RegisterAction(g, mockEmbedder)

			dataset := []*ai.Example{
				{
					Input:      "test question",
					Output:     "test answer",
					Context:    []any{"test context"},
					TestCaseId: "scoring-test",
				},
			}

			testRequest := ai.EvaluatorRequest{
				Dataset:      dataset,
				EvaluationId: "scoring-testrun",
			}

			evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_relevancy")
			resp, err := evalAction.Evaluate(t.Context(), &testRequest)
			if err != nil {
				t.Fatalf("Evaluate() error = %v", err)
			}

			result := (*resp)[0]
			score := result.Evaluation[0]

			if scoreFloat, ok := score.Score.(float32); !ok {
				t.Errorf("score is not float32, got %T", score.Score)
			} else if scoreFloat != tt.wantFinalScore {
				t.Errorf("got score %v, want %v", scoreFloat, tt.wantFinalScore)
			}

			if score.Status != tt.wantStatus {
				t.Errorf("got status %q, want %q", score.Status, tt.wantStatus)
			}

			if reasoning, ok := score.Details["reasoning"].(string); ok {
				if reasoning != tt.wantReasoning {
					t.Errorf("got reasoning %q, want %q", reasoning, tt.wantReasoning)
				}
			} else {
				t.Error("reasoning not found in details")
			}
		})
	}
}

func TestAnswerRelevancyWithLLMErrors(t *testing.T) {
	tests := []struct {
		name        string
		llmError    error
		expectError bool
	}{
		{
			name:        "LLM generation fails",
			llmError:    errors.New("LLM generation failed"),
			expectError: true,
		},
		{
			name:        "successful LLM generation",
			llmError:    nil,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockModel := newMockJudgeLLMForRelevancy(t, true, false, tt.llmError)

			metrics := []evaluators.MetricConfig{
				{
					MetricType: evaluators.EvaluatorAnswerRelevancy,
					JudgeLLM:   mockModel,
					Embedder:   newDumbEmbedder(t, 2),
				},
			}

			g := genkit.Init(t.Context(), genkit.WithPlugins(&evaluators.GenkitEval{Metrics: metrics}))
			genkit.RegisterAction(g, mockModel)
			genkit.RegisterAction(g, (metrics[0].Embedder.(ai.Embedder)))

			dataset := []*ai.Example{
				{
					Input:      "test input",
					Output:     "test output",
					Context:    []any{"test context"},
					TestCaseId: "error-test",
				},
			}

			testRequest := ai.EvaluatorRequest{
				Dataset:      dataset,
				EvaluationId: "error-testrun",
			}

			evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_relevancy")

			resp, err := evalAction.Evaluate(t.Context(), &testRequest)
			if err != nil {
				t.Fatalf("Evaluate() error = %v", err)
			}

			result := (*resp)[0]
			hasError := result.Evaluation[0].Error != ""

			if hasError != tt.expectError {
				t.Errorf("got error = %v, want error = %v (error message: %q)",
					hasError, tt.expectError, result.Evaluation[0].Error)
			}
		})
	}
}

func TestAnswerRelevancyWithEmbedderErrors(t *testing.T) {
	tests := []struct {
		name          string
		embedderError error
		expectError   bool
	}{
		{
			name:          "embedder fails for question",
			embedderError: errors.New("embedding generation failed"),
			expectError:   true,
		},
		{
			name:          "successful embedding generation",
			embedderError: nil,
			expectError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockModel := newMockJudgeLLMForRelevancy(t, true, false, nil)

			metrics := []evaluators.MetricConfig{
				{
					MetricType: evaluators.EvaluatorAnswerRelevancy,
					JudgeLLM:   mockModel,
					Embedder: ai.NewEmbedder(t.Name(), nil, func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
						if tt.embedderError != nil {
							return nil, tt.embedderError
						}

						return &ai.EmbedResponse{Embeddings: []*ai.Embedding{{Embedding: []float32{1}}, {Embedding: []float32{1}}}}, nil
					}),
				},
			}

			g := genkit.Init(t.Context(), genkit.WithPlugins(&evaluators.GenkitEval{Metrics: metrics}))
			genkit.RegisterAction(g, mockModel)
			genkit.RegisterAction(g, (metrics[0].Embedder.(ai.Embedder)))

			dataset := []*ai.Example{
				{
					Input:      "test input",
					Output:     "test output",
					Context:    []any{"test context"},
					TestCaseId: "embedder-error-test",
				},
			}

			testRequest := ai.EvaluatorRequest{
				Dataset:      dataset,
				EvaluationId: "embedder-error-testrun",
			}

			evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_relevancy")

			resp, err := evalAction.Evaluate(t.Context(), &testRequest)
			if err != nil {
				t.Fatalf("Evaluate() error = %v", err)
			}

			result := (*resp)[0]
			hasError := result.Evaluation[0].Error != ""

			if hasError != tt.expectError {
				t.Errorf("got error = %v, want error = %v (error message: %q)",
					hasError, tt.expectError, result.Evaluation[0].Error)
			}
		})
	}
}

func TestAnswerRelevancyEmptyEmbeddings(t *testing.T) {
	embedder := ai.NewEmbedder(t.Name(), nil, func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
		return &ai.EmbedResponse{Embeddings: []*ai.Embedding{}}, nil
	})

	mockModel := newMockJudgeLLMForRelevancy(t, true, false, nil)

	metrics := []evaluators.MetricConfig{
		{
			MetricType: evaluators.EvaluatorAnswerRelevancy,
			JudgeLLM:   mockModel,
			Embedder:   embedder,
		},
	}

	g := genkit.Init(t.Context(), genkit.WithPlugins(&evaluators.GenkitEval{Metrics: metrics}))
	genkit.RegisterAction(g, mockModel)
	genkit.RegisterAction(g, embedder)

	dataset := []*ai.Example{
		{
			Input:      "test input",
			Output:     "test output",
			Context:    []any{"test context"},
			TestCaseId: "empty-embeddings-test",
		},
	}

	testRequest := ai.EvaluatorRequest{
		Dataset:      dataset,
		EvaluationId: "empty-embeddings-testrun",
	}

	evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_relevancy")

	resp, err := evalAction.Evaluate(t.Context(), &testRequest)
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}

	result := (*resp)[0]
	if result.Evaluation[0].Error == "" {
		t.Error("expected error for empty embeddings, got none")
	}
}

func newMockJudgeLLMForRelevancy(t *testing.T, answered, noncommittal bool, err error) ai.Model {
	t.Helper()

	const jsonResponse = `{
				"question": "What is the capital of France?",
				"answered": %t,
				"noncommittal": %t
			}`

	response := fmt.Sprintf(jsonResponse, answered, noncommittal)
	return newMockJudgeLLM(t.Name(), response, err)
}

func createMockEmbedderWithSimilarity(t *testing.T, similarity float32) ai.Embedder {
	t.Helper()

	return ai.NewEmbedder(t.Name(), nil, func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
		v1, v2 := inverseCosineSimilarity(similarity, 100)

		res := []*ai.Embedding{{Embedding: v1}}
		if len(req.Input) == 2 {
			res = append(res, &ai.Embedding{Embedding: v2})
		}

		return &ai.EmbedResponse{Embeddings: res}, nil
	})
}

func inverseCosineSimilarity(similarity float32, dimensions int) ([]float32, []float32) {
	embedding1 := make([]float32, dimensions)
	embedding2 := make([]float32, dimensions)

	embedding1[0] = 1.0
	embedding2[0] = similarity

	if similarity < 1.0 {
		perpendicularMagnitude := float32(math.Sqrt(float64(1.0 - similarity*similarity)))
		embedding2[1] = perpendicularMagnitude
	}

	return embedding1, embedding2
}
