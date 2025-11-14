package evaluators

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
)

type answerRelevancyOutput struct {
	Question     string `json:"question"`
	Answered     bool   `json:"answered"`
	Noncommittal bool   `json:"noncommittal"`
}

//go:embed prompts/answer_relevancy.prompt
var answerRelevancyPrompt string

func configureAnswerRelevancyEvaluator(judgeLLM ai.ModelArg, embedder ai.EmbedderArg) (ai.Evaluator, error) {
	if judgeLLM == nil {
		return nil, errors.New("judgeLLM cannot be nil")
	}

	if embedder == nil {
		return nil, errors.New("embedder cannot be nil")
	}

	evalOptions := ai.EvaluatorOptions{
		DisplayName: "Answer Relevancy",
		Definition:  "Evaluates the relevancy of the answer to the context",
		IsBilled:    true,
	}

	e := newRegistryEvaluator("answer_relevancy.prompt", answerRelevancyPrompt)

	eval := ai.NewEvaluator(api.NewName(provider, "answer_relevancy"),
		&evalOptions,
		func(ctx context.Context, req *ai.EvaluatorCallbackRequest) (*ai.EvaluatorCallbackResponse, error) {
			dataPoint := req.Input

			prompt, err := e.Prompt()
			if err != nil {
				return nil, err
			}

			resolvedEmbedder := ai.LookupEmbedder(e.reg, embedder.Name())
			if resolvedEmbedder == nil {
				return nil, fmt.Errorf("embedder %s not found", embedder.Name())
			}

			score, err := answerRelevancyScore(ctx, prompt, judgeLLM, resolvedEmbedder, &dataPoint)
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

func answerRelevancyScore(
	ctx context.Context,
	prompt ai.Prompt,
	judgeLLM ai.ModelArg,
	embedder ai.Embedder,
	dataPoint *ai.Example,
) (ai.Score, error) {
	if dataPoint.Input == nil {
		return ai.Score{}, errors.New("input was not provided")
	}

	if dataPoint.Output == nil {
		return ai.Score{}, errors.New("output was not provided")
	}

	if dataPoint.Context == nil || len(dataPoint.Context) == 0 {
		return ai.Score{}, errors.New("context was not provided")
	}

	input := base.JSONString(dataPoint.Input)
	output := base.JSONString(dataPoint.Output)

	var contextStrs []string
	for _, ctxItem := range dataPoint.Context {
		contextStrs = append(contextStrs, base.JSONString(ctxItem))
	}
	contextData := strings.Join(contextStrs, " ")

	response, err := prompt.Execute(ctx, ai.WithInput(map[string]any{
		"question": input,
		"answer":   output,
		"context":  contextData,
	}), ai.WithModel(judgeLLM))
	if err != nil {
		return ai.Score{}, fmt.Errorf("judge LLM generation failed: %w", err)
	}

	var relevancyResp answerRelevancyOutput
	if err := response.Output(&relevancyResp); err != nil {
		return ai.Score{}, fmt.Errorf("failed to parse answer relevancy response: %w", err)
	}

	if relevancyResp.Question == "" {
		return ai.Score{}, errors.New("error generating question for answer relevancy")
	}

	embeddings, err := generateEmbeddings(ctx, embedder, input, relevancyResp.Question)
	if err != nil {
		return ai.Score{}, fmt.Errorf("failed to generate embeddings: %w", err)
	}

	score, err := base.CosineSimilarity(embeddings[0], embeddings[1])
	if err != nil {
		return ai.Score{}, fmt.Errorf("failed to calculate cosine similarity: %w", err)
	}

	answered := relevancyResp.Answered
	isNonCommittal := relevancyResp.Noncommittal

	var answeredPenalty float32 = 0.0
	if !answered {
		answeredPenalty = 0.5
	}

	reasoning := "Cosine similarity"
	if isNonCommittal {
		reasoning = "Noncommittal"
	} else if !answered {
		reasoning = "Cosine similarity with penalty for insufficient answer"
	}

	finalScore := max(score-answeredPenalty, 0)
	if isNonCommittal {
		finalScore = 0
	}

	return ai.Score{
		Score:  finalScore,
		Status: determineStatus(finalScore),
		Details: map[string]any{
			"reasoning": reasoning,
		},
	}, nil
}

func generateEmbeddings(ctx context.Context, embedder ai.Embedder, content ...string) ([][]float32, error) {
	input := make([]*ai.Document, len(content))
	for i, c := range content {
		input[i] = ai.DocumentFromText(c, nil)
	}

	resp, err := embedder.Embed(ctx, &ai.EmbedRequest{Input: input})
	if err != nil {
		return nil, err
	}

	if len(resp.Embeddings) != len(content) {
		return nil, errors.New("unexpected number of generated embeddings")
	}

	embeddings := make([][]float32, len(content))
	for i, emb := range resp.Embeddings {
		embeddings[i] = emb.Embedding
	}

	return embeddings, nil
}
