// Issue #115 / SC-26 — skills baked into the harness.
//
// A skill is a directory with a SKILL.md: YAML frontmatter (name + description,
// plus optional fields) followed by a markdown procedure body. This module
// discovers them and exposes them to the agent with progressive disclosure
// (the Agent Skills spec, https://agentskills.io/specification):
//
//  1. Metadata (tier 1). The name + description of every discovered skill is
//     injected every turn as a compact manifest — cheap, always present, so the
//     agent knows what exists.
//  2. Instructions (tier 2). A skill's full body is injected only once it is
//     active, and then it stays injected every turn (sticky).
//
// Unlike the pre-#115 architect-side shim (which injected skills as ad-hoc User
// messages via a wrapping context manager), the catalog feeds the ContextSources
// seam: SkillCatalog.ActiveGuides returns the manifest + active bodies as
// Guides, which the harness places in the structural System block. A skill
// becomes active when the agent calls the load_skill tool (or the host activates
// it). The active set is guarded by a mutex inside the SkillCatalog (held by
// pointer), so the load_skill tool and the per-turn ActiveGuides read the SAME
// set within a harness's lifetime — sticky for the session.

package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// LoadSkill is the registered name of the skill-activation tool.
const LoadSkill = "load_skill"

// ============================================================================
// Skill model + discovery
// ============================================================================

// SkillEntry is one parsed skill. Name is the identity (matches the skill's
// directory name per the spec); Description is the one-line manifest entry
// (where the spec puts trigger keywords); Body is the markdown procedure
// injected on load.
type SkillEntry struct {
	Name        string
	Description string
	Body        string
}

// ParseSkillDoc parses a SKILL.md: optional ---delimited YAML frontmatter
// carrying at least name and description, then the markdown body.
// Dependency-free and minimal — enough for the spec's required fields; optional
// fields are tolerated and ignored. Returns ok=false if there is no usable name
// or the body is empty.
func ParseSkillDoc(content string) (SkillEntry, bool) {
	trimmed := strings.TrimLeft(content, " \t\r\n")
	var front, body string
	if rest, found := strings.CutPrefix(trimmed, "---"); found {
		if idx := strings.Index(rest, "\n---"); idx >= 0 {
			front = rest[:idx]
			body = strings.TrimLeft(rest[idx+4:], "\n\r")
		} else {
			front = rest
			body = ""
		}
	} else {
		body = trimmed
	}

	name := strings.TrimSpace(yamlScalar(front, "name"))
	if name == "" || strings.TrimSpace(body) == "" {
		return SkillEntry{}, false
	}
	return SkillEntry{
		Name:        name,
		Description: strings.TrimSpace(yamlScalar(front, "description")),
		Body:        body,
	}, true
}

// yamlScalar pulls a single top-level `key: value` scalar from a YAML
// frontmatter block, stripping surrounding quotes. Not a general YAML parser —
// nested maps are skipped, since their indented children don't match a top-level
// `key:` line.
func yamlScalar(front, key string) string {
	for _, raw := range strings.Split(front, "\n") {
		line := strings.TrimSpace(raw)
		rest, found := strings.CutPrefix(line, key)
		if !found {
			continue
		}
		value, ok := strings.CutPrefix(strings.TrimLeft(rest, " \t"), ":")
		if !ok {
			continue
		}
		return stripQuotes(strings.TrimSpace(value))
	}
	return ""
}

// stripQuotes removes one pair of matching surrounding single/double quotes.
func stripQuotes(s string) string {
	if len(s) >= 2 {
		first, last := s[0], s[len(s)-1]
		if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
			return s[1 : len(s)-1]
		}
	}
	return s
}

// scanSkillDir scans one skills/ directory: each immediate <child>/SKILL.md is a
// candidate. A missing/unreadable directory or file is skipped silently.
func scanSkillDir(dir string) []SkillEntry {
	var out []SkillEntry
	children, err := os.ReadDir(dir)
	if err != nil {
		return out
	}
	for _, child := range children {
		text, err := os.ReadFile(filepath.Join(dir, child.Name(), "SKILL.md"))
		if err != nil {
			continue
		}
		if entry, ok := ParseSkillDoc(string(text)); ok {
			out = append(out, entry)
		}
	}
	return out
}

// upsertSkill inserts-or-replaces by Name, so a later source overrides an
// earlier one.
func upsertSkill(entries []SkillEntry, entry SkillEntry) []SkillEntry {
	for i := range entries {
		if entries[i].Name == entry.Name {
			entries[i] = entry
			return entries
		}
	}
	return append(entries, entry)
}

// ============================================================================
// Catalog
// ============================================================================

// SkillCatalog is the discovered skill catalog plus the sticky active-skill set.
// Held by POINTER and shared between the load_skill tool (which activates skills)
// and the harness (which reads ActiveGuides each turn), so both share the same
// active set for the catalog's lifetime. The active set is guarded by an
// embedded mutex — the harness is concurrent and CI runs -race.
type SkillCatalog struct {
	entries []SkillEntry

	mu     sync.Mutex
	active map[string]struct{}
}

// SkillCatalogFromEntries builds a catalog from already-parsed entries (sorted
// by name for stable output; later duplicates by name win).
func SkillCatalogFromEntries(entries ...SkillEntry) *SkillCatalog {
	var acc []SkillEntry
	for _, e := range entries {
		acc = upsertSkill(acc, e)
	}
	sort.Slice(acc, func(i, j int) bool { return acc[i].Name < acc[j].Name })
	return &SkillCatalog{entries: acc, active: make(map[string]struct{})}
}

// Discover discovers skills from extraDirs (e.g. a host's bundled skills/) plus
// the conventional <workspaceRoot>/.spore/skills and ~/.spore/skills, in that
// precedence order (last wins on a name clash). Discovery happens once.
func Discover(extraDirs []string, workspaceRoot string) *SkillCatalog {
	dirs := append([]string(nil), extraDirs...)
	dirs = append(dirs, filepath.Join(workspaceRoot, ".spore", "skills"))
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		dirs = append(dirs, filepath.Join(home, ".spore", "skills"))
	}
	var entries []SkillEntry
	for _, dir := range dirs {
		for _, entry := range scanSkillDir(dir) {
			entries = upsertSkill(entries, entry)
		}
	}
	return SkillCatalogFromEntries(entries...)
}

// Names returns the skill names, for host-driven activation (/<name>) and
// listing.
func (c *SkillCatalog) Names() []string {
	out := make([]string, len(c.entries))
	for i, e := range c.entries {
		out[i] = e.Name
	}
	return out
}

// Entries returns the full entries, for listing.
func (c *SkillCatalog) Entries() []SkillEntry {
	return append([]SkillEntry(nil), c.entries...)
}

// IsEmpty reports whether the catalog has no skills.
func (c *SkillCatalog) IsEmpty() bool {
	return len(c.entries) == 0
}

// Activate activates a skill by name so its full body is injected every turn.
// Returns false for an unknown name (no change). Sticky.
func (c *SkillCatalog) Activate(name string) bool {
	known := false
	for _, e := range c.entries {
		if e.Name == name {
			known = true
			break
		}
	}
	if !known {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.active[name] = struct{}{}
	return true
}

// ClearActive deactivates every skill (e.g. on a conversation reset). The
// manifest stays; only the sticky bodies are dropped.
func (c *SkillCatalog) ClearActive() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.active = make(map[string]struct{})
}

// Active returns the currently active skill names (sorted).
func (c *SkillCatalog) Active() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]string, 0, len(c.active))
	for name := range c.active {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// skillManifestPreamble is the fixed manifest preamble (tier 1). Copied verbatim
// from the Rust reference (skills.rs).
const skillManifestPreamble = "Reusable procedures you can load on demand. " +
	"When the user's request matches one's description, call load_skill(name) " +
	"BEFORE acting, then follow the procedure it injects. Active skills' full " +
	"bodies are included below and stay in context every turn:\n"

// ActiveGuides returns the guides injected this turn (issue #115): a single
// manifest guide listing every available skill (tier 1), followed by one guide
// per active skill carrying its full body (tier 2). Empty when the catalog has
// no skills. The harness appends these to ContextSources.Guides so they reach
// the model through the structural System block.
func (c *SkillCatalog) ActiveGuides() []Guide {
	if len(c.entries) == 0 {
		return nil
	}
	var out []Guide

	var manifest strings.Builder
	manifest.WriteString(skillManifestPreamble)
	for _, e := range c.entries {
		manifest.WriteString("- " + e.Name + ": " + e.Description + "\n")
	}
	out = append(out, Guide{ID: "AVAILABLE SKILLS", Content: manifest.String()})

	for _, name := range c.Active() {
		for _, e := range c.entries {
			if e.Name == name {
				out = append(out, Guide{ID: "ACTIVE SKILL — " + e.Name, Content: e.Body})
				break
			}
		}
	}
	return out
}

// loadSkillDescription is the fixed load_skill tool description. Copied verbatim
// from the Rust reference (skills.rs).
const loadSkillDescription = "Activate a skill by name so its full procedure " +
	"stays in your context for the rest of the session. Call this BEFORE acting " +
	"when the user's request matches a skill in the AVAILABLE SKILLS manifest."

// loadSkillParameters is the load_skill tool's JSON Schema: a single required
// string `name`.
var loadSkillParameters = json.RawMessage(`{` +
	`"type":"object",` +
	`"properties":{"name":{"type":"string","description":"The skill's name, exactly as listed in AVAILABLE SKILLS."}},` +
	`"required":["name"]` +
	`}`)

// LoadSkillTool builds the load_skill StandardTool, sharing this catalog's
// active set. Add it to the harness with HarnessBuilder.Tool — or use
// HarnessBuilder.Skills, which registers both the catalog and this tool.
func (c *SkillCatalog) LoadSkillTool() StandardTool {
	return StandardTool{
		Implementation: &loadSkillTool{catalog: c},
		Schema: RegistryToolSchema{
			Name:        LoadSkill,
			Description: loadSkillDescription,
			Parameters:  loadSkillParameters,
		},
	}
}

// ============================================================================
// load_skill tool
// ============================================================================

// loadSkillTool implements load_skill(name): activate a skill so its full
// procedure stays in context. Holds the shared SkillCatalog (to mutate the
// active set and reject unknown ids recoverably).
type loadSkillTool struct {
	catalog *SkillCatalog
}

func (t *loadSkillTool) Name() string                { return LoadSkill }
func (t *loadSkillTool) IsSubagentTool() bool        { return false }
func (t *loadSkillTool) MayProduceLargeOutput() bool { return false }

func (t *loadSkillTool) Execute(_ context.Context, call ToolCall, _ SandboxProvider, _ *ToolContext) ToolOutput {
	var input struct {
		Name string `json:"name"`
	}
	_ = json.Unmarshal(call.Input, &input)
	name := strings.TrimSpace(input.Name)
	if name == "" {
		return NewToolOutputError("invalid parameters: `name` (string) is required")
	}
	if !t.catalog.Activate(name) {
		return NewToolOutputError("unknown skill '" + name + "'. Choose one listed in AVAILABLE SKILLS.")
	}
	return NewToolOutputSuccess("Loaded skill '" + name + "' — its full procedure is now in your context. Follow it.")
}

var _ Tool = (*loadSkillTool)(nil)
