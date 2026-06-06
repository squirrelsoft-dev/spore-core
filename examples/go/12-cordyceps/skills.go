// Architect-side skill loading (zero core-harness change).
//
// # Why this lives in the example, not the harness
//
// Issue #9 added GuideTypeSkill to the GuideRegistry, and the rich
// contextmgr.StandardContextManager knows how to inject skills as a Block-3
// segment. But the LIVE harness loop does not call that rich Assemble — it calls
// StandardCompactionAdapter.Assemble, a pass-through of session.Messages (see
// issue #115 / Known Deviation #8). So today a skill reaches the model only as a
// tool-result message, never as structural injection.
//
// This file wires the chain end-to-end ARCHITECT-SIDE, exactly the pattern issue
// #115 will absorb into the library:
//
//  1. A SkillCatalog scans .spore/skills/{name}/SKILL.md (project) then
//     ~/.spore/skills/{name}/SKILL.md (user), parses YAML frontmatter
//     {name, description} + markdown body, and Registers each as a
//     GuideTypeSkill guide in a guideregistry.StandardGuideRegistry. It also
//     keeps a manifest side-list of (name, description) because Guide has no
//     dedicated description field surfaced to the model — the example owns the
//     manifest text.
//  2. The load_skill tool (see tools.go) appends a skill id to
//     run_store["active_skills"].
//  3. SkillInjectingContextManager EMBEDS the standard compaction adapter and,
//     in Assemble, prepends — ephemerally, never into session.Messages — (a) the
//     manifest of all skills, and (b) the full body of every active skill.
//     Everything else (AppendToolResult, AppendUserMessage, ShouldCompact, the
//     optional compaction/assistant/token seams) is inherited verbatim from the
//     embedded adapter.
//
// Net effect: the manifest is present every turn (progressive disclosure); a
// loaded skill's body is re-injected every turn until the session is cleared.
// Because the active set lives in run_store (not the message history), it is
// compaction-proof.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/contextmgr"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/guideregistry"
)

// activeSkillsKey is the run-store key under which the load_skill tool and the
// context manager rendezvous on the active-skill id set.
const activeSkillsKey = "active_skills"

// skillEntry is one parsed skill: its id (== frontmatter name), the one-line
// description for the manifest, and the markdown body injected when active.
type skillEntry struct {
	Name        string
	Description string
	Body        string
}

// SkillCatalog is the example's skill catalog: a StandardGuideRegistry (the real
// seam) plus the manifest side-list the example owns (because Guide carries no
// model-facing description). Bodies are resolved from the side-list, not
// re-queried from the registry, so the manifest text and the injected body
// always agree.
type SkillCatalog struct {
	registry *guideregistry.StandardGuideRegistry
	manifest []skillEntry
}

// BootstrapCatalog scans the project + user skill directories and registers the
// bundled audit skill so the example is self-contained even with an empty
// .spore/skills/. Project entries win over user entries; the bundled audit body
// seeds .spore/skills/audit/SKILL.md on first run if absent (documented in the
// README) but is also registered directly here so the example never depends on
// that seed having been written.
func BootstrapCatalog(ctx context.Context, projectRoot, bundledAudit string) *SkillCatalog {
	registry := guideregistry.NewStandardGuideRegistry()
	var manifest []skillEntry

	// 1. Bundled audit skill — always present, registered first so a
	//    project/user override of the same name supersedes it (last-wins in the
	//    manifest).
	if entry, ok := parseSkillDoc(bundledAudit); ok {
		manifest = upsertSkill(manifest, entry)
	}

	// 2. Project skills: .spore/skills/{name}/SKILL.md relative to the repo root.
	for _, entry := range scanSkillDir(filepath.Join(projectRoot, ".spore", "skills")) {
		manifest = upsertSkill(manifest, entry)
	}

	// 3. User skills: ~/.spore/skills/{name}/SKILL.md.
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		for _, entry := range scanSkillDir(filepath.Join(home, ".spore", "skills")) {
			manifest = upsertSkill(manifest, entry)
		}
	}

	// Register every manifest entry as a Skill-type guide. Empty content is
	// rejected by the registry; parseSkillDoc already guarantees a body.
	for _, entry := range manifest {
		_, _ = registry.Register(ctx, guideregistry.Guide{
			ID:        guideregistry.GuideID(entry.Name),
			Name:      entry.Name,
			Content:   entry.Body,
			GuideType: guideregistry.GuideTypeSkill,
			Source:    guideregistry.NewSourceManual(),
			Status:    guideregistry.NewStatusActive(),
			Version:   1,
		})
	}

	return &SkillCatalog{registry: registry, manifest: manifest}
}

// Registry returns the shared registry — handed to the load_skill tool.
func (c *SkillCatalog) Registry() *guideregistry.StandardGuideRegistry { return c.registry }

// Manifest returns a copy of the manifest side-list — handed to the context
// manager so it can render `name: description` lines and resolve active bodies.
func (c *SkillCatalog) Manifest() []skillEntry {
	out := make([]skillEntry, len(c.manifest))
	copy(out, c.manifest)
	return out
}

// Names returns the skill names, for the startup banner.
func (c *SkillCatalog) Names() []string {
	out := make([]string, len(c.manifest))
	for i, e := range c.manifest {
		out[i] = e.Name
	}
	return out
}

// upsertSkill inserts-or-replaces by Name so later sources override earlier ones.
func upsertSkill(manifest []skillEntry, entry skillEntry) []skillEntry {
	for i := range manifest {
		if manifest[i].Name == entry.Name {
			manifest[i] = entry
			return manifest
		}
	}
	return append(manifest, entry)
}

// scanSkillDir scans one skills/ directory: each {name}/SKILL.md is a candidate.
func scanSkillDir(dir string) []skillEntry {
	var out []skillEntry
	children, err := os.ReadDir(dir)
	if err != nil {
		return out
	}
	for _, child := range children {
		if !child.IsDir() {
			continue
		}
		content, err := os.ReadFile(filepath.Join(dir, child.Name(), "SKILL.md"))
		if err != nil {
			continue
		}
		if entry, ok := parseSkillDoc(string(content)); ok {
			out = append(out, entry)
		}
	}
	return out
}

// parseSkillDoc parses a SKILL.md: a `---`-delimited YAML frontmatter block
// carrying `name:` and `description:`, followed by the markdown body. Minimal,
// dependency-free parsing — the example owns this until #115's
// FileSystemGuideRegistry productionizes it. Returns ok=false if there is no
// usable name or the body is empty.
func parseSkillDoc(content string) (skillEntry, bool) {
	trimmed := strings.TrimLeft(content, " \t\r\n")
	var name, description, body string
	if rest, found := strings.CutPrefix(trimmed, "---"); found {
		// Split the frontmatter block off at the closing `---`.
		front, after, ok := strings.Cut(rest, "\n---")
		if !ok {
			return skillEntry{}, false
		}
		name = yamlScalar(front, "name")
		description = yamlScalar(front, "description")
		body = strings.TrimLeft(after, "\n")
	} else {
		body = trimmed
	}

	name = strings.TrimSpace(name)
	if name == "" || strings.TrimSpace(body) == "" {
		return skillEntry{}, false
	}
	return skillEntry{
		Name:        name,
		Description: strings.TrimSpace(description),
		Body:        body,
	}, true
}

// yamlScalar pulls a single `key: value` scalar out of a YAML frontmatter block,
// stripping surrounding quotes. Good enough for the {name, description} contract;
// not a general YAML parser.
func yamlScalar(front, key string) string {
	for _, line := range strings.Split(front, "\n") {
		line = strings.TrimSpace(line)
		rest, found := strings.CutPrefix(line, key)
		if !found {
			continue
		}
		rest = strings.TrimLeft(rest, " \t")
		value, ok := strings.CutPrefix(rest, ":")
		if !ok {
			continue
		}
		value = strings.TrimSpace(value)
		value = strings.Trim(value, `"'`)
		return value
	}
	return ""
}

// SkillInjectingContextManager EMBEDS the standard compaction adapter and injects
// the skill manifest + active skill bodies each turn. Only Assemble is
// overridden; every other ContextManager method (and the optional
// CompactingContextManager / AssistantMessageAppender / TokenBudgetReader seams
// the harness type-asserts for) is inherited verbatim from the embedded adapter.
type SkillInjectingContextManager struct {
	*contextmgr.StandardCompactionAdapter
	runStore sporecore.ToolRunStore
	manifest []skillEntry
}

// NewSkillInjectingContextManager wraps inner with manifest + active-skill
// injection, reading the active set from runStore.
func NewSkillInjectingContextManager(
	inner *contextmgr.StandardCompactionAdapter,
	runStore sporecore.ToolRunStore,
	manifest []skillEntry,
) *SkillInjectingContextManager {
	return &SkillInjectingContextManager{
		StandardCompactionAdapter: inner,
		runStore:                  runStore,
		manifest:                  manifest,
	}
}

// Assemble produces the inner adapter's base context, then prepends the skill
// manifest + active-skill bodies EPHEMERALLY — the returned messages are never
// written back into session.Messages, so they survive compaction and are
// re-injected every turn.
func (m *SkillInjectingContextManager) Assemble(
	ctx context.Context,
	session *sporecore.SessionState,
	task *sporecore.Task,
) sporecore.Context {
	base := m.StandardCompactionAdapter.Assemble(ctx, session, task)
	active := m.activeSkills(ctx, task.SessionID)
	injected := m.injectedMessages(active)
	base.Messages = append(injected, base.Messages...)
	return base
}

// activeSkills reads the active-skill id set from run_store["active_skills"].
// Absent / malformed ⇒ empty (the manifest is still injected).
func (m *SkillInjectingContextManager) activeSkills(ctx context.Context, sessionID sporecore.SessionID) []string {
	if m.runStore == nil {
		return nil
	}
	value, found, err := m.runStore.Get(ctx, sessionID, activeSkillsKey)
	if err != nil || !found {
		return nil
	}
	var ids []string
	if json.Unmarshal(value, &ids) != nil {
		return nil
	}
	return ids
}

// injectedMessages renders the leading injected messages: a manifest segment
// (always) plus one body segment per active skill (progressive disclosure).
// Returned as User messages so the loop still inserts the system prompt ahead of
// them at position 0.
func (m *SkillInjectingContextManager) injectedMessages(active []string) []sporecore.Message {
	var out []sporecore.Message

	// Manifest: every skill's name + one-line description.
	var b strings.Builder
	b.WriteString("AVAILABLE SKILLS (call `load_skill` with a `skill_id` to activate one; " +
		"its full procedure then stays in context):\n")
	for _, e := range m.manifest {
		fmt.Fprintf(&b, "- %s: %s\n", e.Name, e.Description)
	}
	out = append(out, sporecore.Message{
		Role:    sporecore.RoleUser,
		Content: sporecore.NewTextContent(b.String()),
	})

	// Bodies of active skills, resolved from the manifest side-list.
	for _, id := range active {
		for _, e := range m.manifest {
			if e.Name == id {
				out = append(out, sporecore.Message{
					Role:    sporecore.RoleUser,
					Content: sporecore.NewTextContent(fmt.Sprintf("ACTIVE SKILL — %s:\n\n%s", e.Name, e.Body)),
				})
				break
			}
		}
	}
	return out
}
