package evaluators

import (
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"golang.org/x/exp/constraints"
)

type registryEvaluator struct {
	ai.Evaluator
	api.Action
	reg          api.Registry
	promptName   string
	promptSource string
}

func (e *registryEvaluator) Name() string {
	return e.Action.Name()
}

func (e *registryEvaluator) Register(r api.Registry) {
	e.reg = r
	if e.promptSource != "" {
		cleanName := strings.TrimSuffix(e.promptName, ".prompt")
		ai.LoadPromptFromSource(r, e.promptSource, cleanName, provider)
	}

	e.Evaluator.Register(r)
}

func (e *registryEvaluator) Prompt() (ai.Prompt, error) {
	cleanName := strings.TrimSuffix(e.promptName, ".prompt")
	key := api.NewName(provider, cleanName)

	prompt := ai.LookupPrompt(e.reg, key)
	if prompt == nil {
		return nil, fmt.Errorf("prompt not found: %s", key)
	}

	return prompt, nil
}

func newRegistryEvaluator(promptName, promptSource string) *registryEvaluator {
	return &registryEvaluator{
		promptName:   promptName,
		promptSource: promptSource,
	}
}

func (e *registryEvaluator) SetEvaluator(eval ai.Evaluator) {
	action, ok := eval.(api.Action)
	if !ok {
		panic("evaluator does not implement api.Action")
	}

	e.Evaluator = eval
	e.Action = action
}

func determineStatus[T constraints.Float](score T) string {
	if score >= 0.5 {
		return ai.ScoreStatusPass.String()
	}

	return ai.ScoreStatusFail.String()
}
