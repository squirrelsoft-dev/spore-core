// Package manifest handles task-suite manifest loading + manual promotion
// (Rules 6, 29, 31).
package manifest

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/task"
)

// LoadSuiteStr loads a TaskSuite from a JSON manifest string. It rejects a
// manifest without suite_version (Rule 6) with task.ErrMissingSuiteVersion.
func LoadSuiteStr(jsonStr string) (*task.TaskSuite, error) {
	// First check the raw JSON for the required field so a missing
	// suite_version is a precise error rather than a generic parse failure.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(jsonStr), &raw); err != nil {
		return nil, &task.ManifestParseError{Msg: err.Error()}
	}
	if _, ok := raw["suite_version"]; !ok {
		return nil, task.ErrMissingSuiteVersion
	}
	var suite task.TaskSuite
	if err := json.Unmarshal([]byte(jsonStr), &suite); err != nil {
		return nil, &task.ManifestParseError{Msg: err.Error()}
	}
	return &suite, nil
}

// LoadSuitePath loads a TaskSuite from a manifest file path.
func LoadSuitePath(path string) (*task.TaskSuite, error) {
	body, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadSuiteStr(string(body))
}

// SuiteToJSON serialises a TaskSuite back to pretty JSON.
func SuiteToJSON(suite *task.TaskSuite) (string, error) {
	b, err := json.MarshalIndent(suite, "", "  ")
	if err != nil {
		return "", &task.ManifestParseError{Msg: err.Error()}
	}
	return string(b), nil
}

// PromoteChallengeTask manually promotes a challenge task to regression,
// bumping suite_version (Rule 31). Auto-promotion is deferred. Returns an error
// if taskID is not found among the challenge tasks.
func PromoteChallengeTask(suite *task.TaskSuite, taskID string) error {
	pos := -1
	for i := range suite.Challenge {
		if string(suite.Challenge[i].ID) == taskID {
			pos = i
			break
		}
	}
	if pos < 0 {
		return &task.ManifestParseError{Msg: fmt.Sprintf("challenge task %q not found", taskID)}
	}
	t := suite.Challenge[pos]
	suite.Challenge = append(suite.Challenge[:pos], suite.Challenge[pos+1:]...)
	suite.Regression = append(suite.Regression, t)
	suite.SuiteVersion++
	return nil
}
