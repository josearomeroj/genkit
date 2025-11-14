package evaluators

import (
	"context"
	_ "embed"
	"errors"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
)

type maliciousnessOutput struct {
	Reason  string `json:"reason"`
	Verdict bool   `json:"verdict"`
}

//go:embed prompts/maliciousness.prompt
var maliciousnessPrompt string

func configureMaliciousnessEvaluator(judgeLLM ai.ModelArg) (ai.Evaluator, error) {
	if judgeLLM == nil {
		return nil, errors.New("judgeLLM cannot be nil")
	}

	evalOptions := ai.EvaluatorOptions{
		DisplayName: "Maliciousness",
		Definition:  "Evaluates the maliciousness of the output",
		IsBilled:    true,
	}

	e := newRegistryEvaluator("maliciousness.prompt", maliciousnessPrompt)

	eval := ai.NewEvaluator(api.NewName(provider, "maliciousness"),
		&evalOptions,
		func(ctx context.Context, req *ai.EvaluatorCallbackRequest) (*ai.EvaluatorCallbackResponse, error) {
			dataPoint := req.Input

			prompt, err := e.Prompt()
			if err != nil {
				return nil, err
			}

			score, err := maliciousnessScore(ctx, prompt, judgeLLM, &dataPoint)
			if err != nil {
				return nil, err
			}

			return &ai.EvaluatorCallbackResponse{
				TestCaseId: dataPoint.TestCaseId,
				Evaluation: []ai.Score{score},
			}, nil
		},
	)

	e.SetEvaluator(eval)
	return e, nil
}

func maliciousnessScore(
	ctx context.Context,
	prompt ai.Prompt,
	judgeLLM ai.ModelArg,
	dataPoint *ai.Example,
) (ai.Score, error) {
	if dataPoint.Input == nil {
		return ai.Score{}, errors.New("input was not provided")
	}

	if dataPoint.Output == nil {
		return ai.Score{}, errors.New("output was not provided")
	}

	input := base.JSONString(dataPoint.Input)
	output := base.JSONString(dataPoint.Output)

	response, err := prompt.Execute(ctx, ai.WithInput(map[string]any{
		"input":      input,
		"submission": output,
	}), ai.WithModel(judgeLLM))

	if err != nil {
		return ai.Score{}, fmt.Errorf("judge LLM generation failed: %w", err)
	}

	var parsedResponse maliciousnessOutput
	if err := response.Output(&parsedResponse); err != nil {
		return ai.Score{}, fmt.Errorf("unable to parse evaluator response: %w", err)
	}

	if parsedResponse.Reason == "" {
		return ai.Score{}, errors.New("maliciousness reasoning was not provided")
	}

	var (
		score  float32
		status ai.ScoreStatus
	)

	if parsedResponse.Verdict {
		score = 1.0
		status = ai.ScoreStatusFail
	} else {
		score = 0.0
		status = ai.ScoreStatusPass
	}

	return ai.Score{
		Score:   score,
		Status:  status.String(),
		Details: map[string]any{"reasoning": parsedResponse.Reason},
	}, nil
}
