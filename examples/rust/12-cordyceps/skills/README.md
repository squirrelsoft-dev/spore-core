# Skills

Drop [Agent Skills](https://agentskills.io/specification) here, one directory per
skill:

```
skills/
└── security-review/
    └── SKILL.md
```

Each `SKILL.md` is YAML frontmatter followed by a markdown procedure body:

```markdown
---
name: security-review            # required; must match this directory's name
description: >
  Review changed code for security issues before committing. Use when the user
  mentions security, auth, secrets, injection, or asks for a pre-release check.
---

# Procedure

...the steps the agent should follow when this skill is active...
```

Only `name` and `description` are required (optional `license`, `compatibility`,
`metadata`, and `allowed-tools` are tolerated and ignored here). Per the spec, the
`description` is where the "when to use" keywords go — there is **no** separate
triggers field.

The agent sees every skill's `name` + `description` every turn (cheap). It loads a
skill's full body on demand — by calling the `load_skill` tool when your request
matches a skill's description, or because you ran `/<name>` in the REPL. Once
loaded, a skill stays in context for the rest of the conversation.

Skills are also discovered from `.spore/skills/<name>/SKILL.md` in the agent's
workspace and `~/.spore/skills/<name>/SKILL.md`. Discovery happens at startup —
restart to pick up new or changed skills.
