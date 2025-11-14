package evaluators_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/evaluators"
)

func TestMaliciousnessEvaluator(t *testing.T) {
	metrics := []evaluators.MetricConfig{
		{
			MetricType: evaluators.EvaluatorMaliciousness,
			JudgeLLM:   newMockJudgeLLMForMaliciousness(t, "Content is safe and appropriate", false, nil),
		},
	}

	g := genkit.Init(t.Context(), genkit.WithPlugins(&evaluators.GenkitEval{Metrics: metrics}))
	mockModel := metrics[0].JudgeLLM.(ai.Model)
	genkit.RegisterAction(g, mockModel)

	t.Run("maliciousness with valid non-malicious data", func(t *testing.T) {
		dataset := []*ai.Example{
			{
				Input:      "How do I bake a cake?",
				Output:     "To bake a cake, you'll need flour, eggs, sugar, and butter. Mix the ingredients and bake at 350°F for 30 minutes.",
				TestCaseId: "test-1",
			},
		}

		testRequest := ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/maliciousness")
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
		} else if scoreFloat != 0.0 {
			t.Errorf("got score %v, want 0.0", scoreFloat)
		}

		if score.Status != ai.ScoreStatusPass.String() {
			t.Errorf("got status %q, want %q", score.Status, ai.ScoreStatusPass.String())
		}

		if reasoning, ok := score.Details["reasoning"].(string); ok {
			if reasoning != "Content is safe and appropriate" {
				t.Errorf("got reasoning %q, want %q", reasoning, "Content is safe and appropriate")
			}
		} else {
			t.Error("reasoning not found in details")
		}
	})

	t.Run("maliciousness with missing input", func(t *testing.T) {
		dataset := []*ai.Example{
			{
				Input:      nil,
				Output:     "Some output",
				TestCaseId: "test-missing-input",
			},
		}

		testRequest := ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/maliciousness")
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

	t.Run("maliciousness with missing output", func(t *testing.T) {
		dataset := []*ai.Example{
			{
				Input:      "How do I do something?",
				Output:     nil,
				TestCaseId: "test-missing-output",
			},
		}

		testRequest := ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/maliciousness")
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
}

func TestMaliciousnessScoring(t *testing.T) {
	tests := []struct {
		name          string
		reason        string
		verdict       bool
		wantScore     float32
		wantStatus    string
		wantReasoning string
	}{
		{
			name:          "non-malicious content",
			reason:        "Content is helpful and appropriate",
			verdict:       false,
			wantScore:     0.0,
			wantStatus:    ai.ScoreStatusPass.String(),
			wantReasoning: "Content is helpful and appropriate",
		},
		{
			name:          "malicious content detected",
			reason:        "Content contains harmful instructions",
			verdict:       true,
			wantScore:     1.0,
			wantStatus:    ai.ScoreStatusFail.String(),
			wantReasoning: "Content contains harmful instructions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockModel := newMockJudgeLLMForMaliciousness(t, tt.reason, tt.verdict, nil)

			metrics := []evaluators.MetricConfig{
				{
					MetricType: evaluators.EvaluatorMaliciousness,
					JudgeLLM:   mockModel,
				},
			}

			g := genkit.Init(t.Context(), genkit.WithPlugins(&evaluators.GenkitEval{Metrics: metrics}))
			genkit.RegisterAction(g, mockModel)

			dataset := []*ai.Example{
				{
					Input:      "test input",
					Output:     "test output",
					TestCaseId: "scoring-test",
				},
			}

			testRequest := ai.EvaluatorRequest{
				Dataset:      dataset,
				EvaluationId: "scoring-testrun",
			}

			evalAction := genkit.LookupEvaluator(g, "genkitEval/maliciousness")
			resp, err := evalAction.Evaluate(t.Context(), &testRequest)
			if err != nil {
				t.Fatalf("Evaluate() error = %v", err)
			}

			result := (*resp)[0]
			score := result.Evaluation[0]

			if scoreFloat, ok := score.Score.(float32); !ok {
				t.Errorf("score is not float32, got %T", score.Score)
			} else if scoreFloat != tt.wantScore {
				t.Errorf("got score %v, want %v", scoreFloat, tt.wantScore)
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

func TestMaliciousnessWithLLMErrors(t *testing.T) {
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
			name:        "LLM timeout error",
			llmError:    errors.New("request timeout"),
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
			mockModel := newMockJudgeLLMForMaliciousness(t, "test reason", false, tt.llmError)

			metrics := []evaluators.MetricConfig{
				{
					MetricType: evaluators.EvaluatorMaliciousness,
					JudgeLLM:   mockModel,
				},
			}

			g := genkit.Init(t.Context(), genkit.WithPlugins(&evaluators.GenkitEval{Metrics: metrics}))
			genkit.RegisterAction(g, mockModel)

			dataset := []*ai.Example{
				{
					Input:      "test input",
					Output:     "test output",
					TestCaseId: "error-test",
				},
			}

			testRequest := ai.EvaluatorRequest{
				Dataset:      dataset,
				EvaluationId: "error-testrun",
			}

			evalAction := genkit.LookupEvaluator(g, "genkitEval/maliciousness")

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

func TestMaliciousnessWithInvalidJSONResponse(t *testing.T) {
	tests := []struct {
		name         string
		jsonResponse string
		expectError  bool
	}{
		{
			name:         "invalid JSON syntax",
			jsonResponse: `{"reason": "test", "verdict": true`,
			expectError:  true,
		},
		{
			name:         "missing reason field",
			jsonResponse: `{"verdict": true}`,
			expectError:  true,
		},
		{
			name:         "empty reason field",
			jsonResponse: `{"reason": "", "verdict": false}`,
			expectError:  true,
		},
		{
			name:         "missing verdict field",
			jsonResponse: `{"reason": "test reason"}`,
			expectError:  true,
		},
		{
			name:         "extra fields in response",
			jsonResponse: `{"reason": "valid reason", "verdict": false, "confidence": 0.95}`,
			expectError:  false,
		},
		{
			name:         "valid response with whitespace",
			jsonResponse: `  {"reason": "valid reason", "verdict": true}  `,
			expectError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockModel := newMockJudgeLLM(t.Name(), tt.jsonResponse, nil)

			metrics := []evaluators.MetricConfig{
				{
					MetricType: evaluators.EvaluatorMaliciousness,
					JudgeLLM:   mockModel,
				},
			}

			g := genkit.Init(t.Context(), genkit.WithPlugins(&evaluators.GenkitEval{Metrics: metrics}))
			genkit.RegisterAction(g, mockModel)

			dataset := []*ai.Example{
				{
					Input:      "test input",
					Output:     "test output",
					TestCaseId: "json-test",
				},
			}

			testRequest := ai.EvaluatorRequest{
				Dataset:      dataset,
				EvaluationId: "json-testrun",
			}

			evalAction := genkit.LookupEvaluator(g, "genkitEval/maliciousness")

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

func newMockJudgeLLMForMaliciousness(t *testing.T, reason string, verdict bool, err error) ai.Model {
	t.Helper()

	const jsonResponse = `{
		"reason": %q,
		"verdict": %t
	}`

	response := fmt.Sprintf(jsonResponse, reason, verdict)
	return newMockJudgeLLM(t.Name(), response, err)
}
