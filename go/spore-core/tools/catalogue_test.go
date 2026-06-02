package tools

import (
	"testing"
)

func names(set []StandardTool) map[string]bool {
	out := map[string]bool{}
	for _, t := range set {
		out[t.Schema.Name] = true
	}
	return out
}

func TestEveryConstructorPairsMatchingImplAndSchema(t *testing.T) {
	for _, st := range (StandardTools{}).FullSet() {
		if st.Implementation.Name() != st.Schema.Name {
			t.Fatalf("impl/schema name mismatch: impl=%q schema=%q", st.Implementation.Name(), st.Schema.Name)
		}
	}
}

func TestReadonlySetHasNoMutatingOrEscalatingTools(t *testing.T) {
	n := names((StandardTools{}).ReadonlySet())
	for _, forbidden := range []string{
		"write_file", "edit_file", "bash_command", "todo_write", "memory",
		"enter_plan_mode", "exit_plan_mode", "ask_user_question", "abort",
	} {
		if n[forbidden] {
			t.Fatalf("readonly_set leaked %q", forbidden)
		}
	}
	if !n["read_file"] || !n["grep"] {
		t.Fatalf("readonly_set missing read_file/grep: %v", n)
	}
}

func TestCodingSetReusesExistingNamesOnOverlap(t *testing.T) {
	n := names((StandardTools{}).CodingSet())
	for _, existing := range []string{"read_file", "write_file", "find_files", "grep_files", "bash_command", "task_list"} {
		if !n[existing] {
			t.Fatalf("coding_set missing existing %q", existing)
		}
	}
	if !n["edit_file"] || !n["grep"] {
		t.Fatalf("coding_set missing new edit_file/grep")
	}
	// #82: the scope-aware memory tool ships in CodingSet alongside task_list /
	// todo_write.
	if !n["memory"] {
		t.Fatalf("coding_set missing memory (#82)")
	}
	if n["abort"] {
		t.Fatalf("coding_set must not contain Tier-3 abort")
	}
}

func TestFullSetAddsTier3(t *testing.T) {
	n := names((StandardTools{}).FullSet())
	for _, tier3 := range []string{"enter_plan_mode", "exit_plan_mode", "ask_user_question", "abort"} {
		if !n[tier3] {
			t.Fatalf("full_set missing tier3 %q", tier3)
		}
	}
}

func TestWebSearchWithEndpointIsNamedWebSearch(t *testing.T) {
	st := (StandardTools{}).WebSearchWithEndpoint("http://localhost:9/search")
	if st.Implementation.Name() != "web_search" || st.Schema.Name != "web_search" {
		t.Fatalf("bundle mismatch: impl=%q schema=%q", st.Implementation.Name(), st.Schema.Name)
	}
}

func TestStandardToolBundlesImplAndSchema(t *testing.T) {
	st := (StandardTools{}).EditFile()
	if st.Implementation.Name() != "edit_file" || st.Schema.Name != "edit_file" {
		t.Fatalf("bundle mismatch: %+v", st)
	}
	if !st.Schema.Annotations.Destructive {
		t.Fatalf("edit_file schema must be destructive")
	}
}
