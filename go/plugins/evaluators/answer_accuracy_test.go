package evaluators_test

import (
	"errors"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/evaluators"
)

func TestAnswerAccuracyEvaluator(t *testing.T) {
	metrics := []evaluators.MetricConfig{
		{
			MetricType: evaluators.EvaluatorAnswerAccuracy,
			JudgeLLM:   newMockJudgeLLM(t.Name(), "3", nil),
		},
	}

	g := genkit.Init(t.Context(), genkit.WithPlugins(&evaluators.GenkitEval{Metrics: metrics}))
	genkit.RegisterAction(g, metrics[0].JudgeLLM.(ai.Model))
	t.Run("answer accuracy with valid data", func(t *testing.T) {
		dataset := []*ai.Example{
			{
				Input:      "What is the capital of France?",
				Reference:  "Paris",
				Output:     "Paris",
				TestCaseId: "test-1",
			},
		}

		testRequest := ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_accuracy")
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

		// Expected score: (3+3)/8 = 0.75
		if scoreFloat, ok := score.Score.(float64); !ok {
			t.Errorf("score is not float64, got %T", score.Score)
		} else if scoreFloat != 0.75 {
			t.Errorf("got score %v, want 0.75", scoreFloat)
		}

		if score.Status != ai.ScoreStatusPass.String() {
			t.Errorf("got status %q, want %q", score.Status, ai.ScoreStatusPass.String())
		}
	})

	t.Run("answer accuracy with missing output", func(t *testing.T) {
		dataset := []*ai.Example{
			{
				Input:      "What is the capital of France?",
				Reference:  "Paris",
				Output:     nil,
				TestCaseId: "test-missing-output",
			},
		}

		testRequest := ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_accuracy")
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
			t.Error("expected error for missing output, got none")
		}
	})

	t.Run("answer accuracy with missing reference", func(t *testing.T) {
		dataset := []*ai.Example{
			{
				Input:      "What is the capital of France?",
				Reference:  nil,
				Output:     "Paris",
				TestCaseId: "test-missing-reference",
			},
		}

		testRequest := ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_accuracy")
		resp, err := evalAction.Evaluate(t.Context(), &testRequest)
		if err != nil {
			t.Fatalf("Evaluate() error = %v", err)
		}

		if len(*resp) != 1 {
			t.Fatalf("got %d responses, want 1", len(*resp))
		}

		result := (*resp)[0]
		if result.Evaluation[0].Error == "" {
			t.Error("expected error for missing reference, got none")
		}
	})
}

func TestAnswerAccuracyScoring(t *testing.T) {
	tests := []struct {
		name           string
		origScore      int
		invScore       int
		wantFinalScore float64
		wantStatus     string
	}{
		{
			name:           "perfect match",
			origScore:      4,
			invScore:       4,
			wantFinalScore: 1.0,
			wantStatus:     ai.ScoreStatusPass.String(),
		},
		{
			name:           "good match",
			origScore:      3,
			invScore:       3,
			wantFinalScore: 0.75,
			wantStatus:     ai.ScoreStatusPass.String(),
		},
		{
			name:           "asymmetric scores low",
			origScore:      1,
			invScore:       4,
			wantFinalScore: 0.625,
			wantStatus:     ai.ScoreStatusPass.String(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockModel := newMockJudgeLLMWithSequence(
				t.Name(), []string{
					string(rune('0' + tt.origScore)),
					string(rune('0' + tt.invScore)),
				},
			)

			metrics := []evaluators.MetricConfig{
				{
					MetricType: evaluators.EvaluatorAnswerAccuracy,
					JudgeLLM:   mockModel,
				},
			}

			g := genkit.Init(t.Context(), genkit.WithPlugins(&evaluators.GenkitEval{Metrics: metrics}))
			genkit.RegisterAction(g, mockModel)

			dataset := []*ai.Example{
				{
					Input:      "test input",
					Reference:  "test reference",
					Output:     "test output",
					TestCaseId: "scoring-test",
				},
			}

			testRequest := ai.EvaluatorRequest{
				Dataset:      dataset,
				EvaluationId: "scoring-testrun",
			}

			evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_accuracy")

			resp, err := evalAction.Evaluate(t.Context(), &testRequest)
			if err != nil {
				t.Fatalf("Evaluate() error = %v", err)
			}

			result := (*resp)[0]
			score := result.Evaluation[0]

			if scoreFloat, ok := score.Score.(float64); !ok {
				t.Errorf("score is not float64, got %T", score.Score)
			} else if scoreFloat != tt.wantFinalScore {
				t.Errorf("got score %v, want %v", scoreFloat, tt.wantFinalScore)
			}

			if score.Status != tt.wantStatus {
				t.Errorf("got status %q, want %q", score.Status, tt.wantStatus)
			}
		})
	}
}

func TestAnswerAccuracyWithLLMErrors(t *testing.T) {
	tests := []struct {
		name         string
		mockResponse string
		mockError    error
		expectError  bool
	}{
		{
			name:        "LLM generation fails",
			mockError:   errors.New("LLM generation failed"),
			expectError: true,
		},
		{
			name:         "invalid score format",
			mockResponse: "not a number",
			expectError:  true,
		},
		{
			name:         "score out of range - too high",
			mockResponse: "5",
			expectError:  true,
		},
		{
			name:         "score out of range - too low",
			mockResponse: "0",
			expectError:  true,
		},
		{
			name:         "valid minimum score",
			mockResponse: "1",
			expectError:  false,
		},
		{
			name:         "valid maximum score",
			mockResponse: "4",
			expectError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics := []evaluators.MetricConfig{
				{
					MetricType: evaluators.EvaluatorAnswerAccuracy,
					JudgeLLM:   newMockJudgeLLM(t.Name(), tt.mockResponse, tt.mockError),
				},
			}

			g := genkit.Init(t.Context(), genkit.WithPlugins(&evaluators.GenkitEval{Metrics: metrics}))
			genkit.RegisterAction(g, metrics[0].JudgeLLM.(ai.Model))

			dataset := []*ai.Example{
				{
					Input:      "test input",
					Reference:  "test reference",
					Output:     "test output",
					TestCaseId: "error-test",
				},
			}

			testRequest := ai.EvaluatorRequest{
				Dataset:      dataset,
				EvaluationId: "error-testrun",
			}

			evalAction := genkit.LookupEvaluator(g, "genkitEval/answer_accuracy")

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
