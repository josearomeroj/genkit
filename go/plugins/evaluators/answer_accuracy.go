package evaluators

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
)

//go:embed prompts/answer_accuracy.prompt
var answerAccuracyPrompt string

func configureAnswerAccuracyEvaluator(judgeLLM ai.ModelArg) (ai.Evaluator, error) {
	if judgeLLM == nil {
		return nil, errors.New("judgeLLM cannot be nil")
	}

	evalOptions := ai.EvaluatorOptions{
		DisplayName: "Answer Accuracy",
		Definition:  "Evaluates the accuracy of the output against a reference answer",
		IsBilled:    true,
	}

	e := newRegistryEvaluator("answer_accuracy.prompt", answerAccuracyPrompt)

	eval := ai.NewEvaluator(api.NewName(provider, "answer_accuracy"),
		&evalOptions,
		func(ctx context.Context, req *ai.EvaluatorCallbackRequest) (*ai.EvaluatorCallbackResponse, error) {
			dataPoint := req.Input

			prompt, err := e.Prompt()
			if err != nil {
				return nil, err
			}

			score, err := answerAccuracyScore(ctx, prompt, judgeLLM, &dataPoint)
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

func answerAccuracyScore(
	ctx context.Context,
	prompt ai.Prompt,
	judgeLLM ai.ModelArg,
	dataPoint *ai.Example,
) (ai.Score, error) {
	if dataPoint.Output == nil {
		return ai.Score{}, errors.New("output was not provided")
	}

	if dataPoint.Reference == nil {
		return ai.Score{}, errors.New("reference was not provided")
	}

	query := base.JSONString(dataPoint.Input)
	output := base.JSONString(dataPoint.Output)
	reference := base.JSONString(dataPoint.Reference)

	origScore, err := evaluateWithPrompt(ctx, prompt, judgeLLM, query, output, reference)
	if err != nil {
		return ai.Score{}, fmt.Errorf("error generating original response for answer accuracy: %w", err)
	}

	invScore, err := evaluateWithPrompt(ctx, prompt, judgeLLM, query, reference, output)
	if err != nil {
		return ai.Score{}, fmt.Errorf("error generating inverted response for answer accuracy: %w", err)
	}

	finalScore := float64(origScore+invScore) / 8.0

	return ai.Score{
		Score:  finalScore,
		Status: determineStatus(finalScore),
	}, nil
}

func evaluateWithPrompt(
	ctx context.Context,
	prompt ai.Prompt,
	judgeLLM ai.ModelArg,
	query string,
	output string,
	reference string,
) (int, error) {
	response, err := prompt.Execute(ctx, ai.WithInput(map[string]any{
		"query":     query,
		"output":    output,
		"reference": reference,
	}), ai.WithModel(judgeLLM))

	if err != nil {
		return 0, fmt.Errorf("judge LLM generation failed: %w", err)
	}

	return parseScore(response.Text())
}

func parseScore(text string) (int, error) {
	text = strings.TrimSpace(text)

	score, err := strconv.Atoi(text)
	if err != nil {
		return 0, fmt.Errorf("unable to parse score from response '%s': %w", text, err)
	}

	if score < 1 || score > 4 {
		return 0, fmt.Errorf("score %d is out of expected range (1-4)", score)
	}

	return score, nil
}
