package task

import "encoding/json"

// Thin aliases so the tagged-union (un)marshalers read uniformly and the
// standard library json import stays in one place.

type jsonRaw = json.RawMessage

func jsonMarshal(v any) ([]byte, error)      { return json.Marshal(v) }
func jsonUnmarshal(data []byte, v any) error { return json.Unmarshal(data, v) }
