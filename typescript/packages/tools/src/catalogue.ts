/**
 * Standard Tool Catalogue (#81): the curated set of tools an architect drops
 * into a harness, plus ready-made presets.
 *
 * ## Types
 * - {@link "@spore/core".toolRegistry.StandardTool} — a tool implementation
 *   bundled with its {@link "@spore/core".toolRegistry.ToolSchema} so the two
 *   can never be separated (issue #81, Q2). `StandardToolRegistry.tool()`
 *   destructures it.
 * - {@link StandardTools} — a namespace of one constructor per catalogue tool,
 *   each returning a `StandardTool`, plus three presets:
 *   {@link StandardTools.readonlySet}, {@link StandardTools.codingSet}, and
 *   {@link StandardTools.fullSet}.
 *
 * ## Catalogue tools (constructor → registered name)
 * Tier 1 (sandbox / stateless):
 *   - `readFile` → `read_file` (EXISTING #5 tool)
 *   - `writeFile` → `write_file` (EXISTING)
 *   - `editFile` → `edit_file` (NEW)
 *   - `listDir` → `list_dir` (EXISTING)
 *   - `grepFiles` → `grep_files` (EXISTING)
 *   - `grep` → `grep` (NEW, output modes)
 *   - `findFiles` → `find_files` (EXISTING)
 *   - `bashCommand` → `bash_command` (EXISTING)
 *   - `sendMessage` → `send_message` (NEW)
 *   - `webFetch` → `web_fetch` (NEW)
 *   - `webSearch` → `web_search` (NEW)
 *
 * Tier 2 (storage via `ToolContext`):
 *   - `todoWrite` → `todo_write` (NEW, RunStore key `"todo"`)
 *   - `taskList` → `task_list` (EXISTING #71)
 *   - `memory` → `memory` (#82, scope-aware MemoryStore read/write)
 *
 * Tier 3 (escalate / clarify):
 *   - `enterPlanMode` → `enter_plan_mode` (NEW)
 *   - `exitPlanMode` → `exit_plan_mode` (NEW)
 *   - `askUserQuestion` → `ask_user_question` (NEW)
 *   - `abort` → `abort` (NEW)
 *
 * ## Q5 — overlap with the EXISTING #5 catalogue (NO renames)
 * The catalogue ships NET-NEW tools ALONGSIDE the existing #5 tools, never
 * renaming them. Where a preset needs functionality an existing tool already
 * provides, the preset REUSES the existing tool by its existing name:
 *   - `read_file`, `write_file`, `list_dir`, `find_files`, `grep_files`,
 *     `bash_command` are the EXISTING tools (their fixtures —
 *     `fixtures/tools/param_validation.json` — stay byte-identical).
 *   - `edit_file` and `grep` are NEW and live ALONGSIDE `write_file` /
 *     `grep_files`; they do NOT replace them.
 *
 * Because {@link "@spore/core".StandardToolRegistry.register} is a last-wins
 * upsert (issue #81, Q1), registering a preset and then a custom tool of the
 * same name lets the architect override a standard tool.
 *
 * ## MemoryTool (#82 — landed)
 * `MemoryTool` (`memory`) was deferred from #81 (Q3) pending the scoped
 * `MemoryStore` seam (#78). It now ships here as a Tier-2 storage tool,
 * included in {@link StandardTools.codingSet} (and {@link StandardTools.fullSet}
 * via spread) alongside `task_list` / `todo_write`.
 */

import type { toolRegistry } from "@spore/core";
type StandardTool = toolRegistry.StandardTool;

import { EditFileTool } from "./edit.js";
import { BashCommandTool } from "./exec.js";
import { ListDirTool, ReadFileTool, WriteFileTool } from "./fs.js";
import { SendMessageTool } from "./message.js";
import { FindFilesTool, GrepFilesTool, GrepTool } from "./search.js";
import { MemoryTool } from "./memory.js";
import { TaskListTool } from "./tasklist.js";
import { TodoWriteTool } from "./todo.js";
import {
  AbortTool,
  AskUserQuestionTool,
  EnterPlanModeTool,
  ExitPlanModeTool,
} from "./control.js";
import { WebFetchTool, WebSearchTool } from "./web.js";

/**
 * Namespace of catalogue-tool constructors and presets. Each constructor pairs
 * the right implementation with the right schema as a
 * {@link "@spore/core".toolRegistry.StandardTool}.
 */
export const StandardTools = {
  // ---- Tier 1 ---------------------------------------------------------

  /** `read_file` — EXISTING #5 tool (Q5 overlap: reused, not renamed). */
  readFile(): StandardTool {
    return {
      implementation: new ReadFileTool(),
      schema: ReadFileTool.schema(),
    };
  },

  /** `write_file` — EXISTING #5 tool (Q5 overlap: reused, not renamed). */
  writeFile(): StandardTool {
    return {
      implementation: new WriteFileTool(),
      schema: WriteFileTool.schema(),
    };
  },

  /** `edit_file` — NEW unique-match in-place edit (alongside `write_file`). */
  editFile(): StandardTool {
    return {
      implementation: new EditFileTool(),
      schema: EditFileTool.schema(),
    };
  },

  /** `list_dir` — EXISTING #5 tool (Q5 overlap: reused, not renamed). */
  listDir(): StandardTool {
    return { implementation: new ListDirTool(), schema: ListDirTool.schema() };
  },

  /** `grep_files` — EXISTING #5 tool (Q5 overlap: reused, not renamed). */
  grepFiles(): StandardTool {
    return {
      implementation: new GrepFilesTool(),
      schema: GrepFilesTool.schema(),
    };
  },

  /** `grep` — NEW regex search with output modes (alongside `grep_files`). */
  grep(): StandardTool {
    return { implementation: new GrepTool(), schema: GrepTool.schema() };
  },

  /** `find_files` — EXISTING #5 tool (Q5 overlap: reused, not renamed). */
  findFiles(): StandardTool {
    return {
      implementation: new FindFilesTool(),
      schema: FindFilesTool.schema(),
    };
  },

  /** `bash_command` — EXISTING #5 tool (Q5 overlap: reused, not renamed). */
  bashCommand(): StandardTool {
    return {
      implementation: new BashCommandTool(),
      schema: BashCommandTool.schema(),
    };
  },

  /** `send_message` — NEW; surfaces a `user_message` stream event via the loop. */
  sendMessage(): StandardTool {
    return {
      implementation: new SendMessageTool(),
      schema: SendMessageTool.schema(),
    };
  },

  /** `web_fetch` — NEW; GET a URL. */
  webFetch(): StandardTool {
    return {
      implementation: new WebFetchTool(),
      schema: WebFetchTool.schema(),
    };
  },

  /**
   * `web_search` — NEW; structured search over a configurable HTTP backend.
   * The default has NO backend (calls error until one is configured); pass an
   * `endpoint` (or build a {@link "@spore/core".toolRegistry.StandardTool} over
   * {@link WebSearchTool.withEndpoint}) to wire a real backend.
   */
  webSearch(endpoint?: string): StandardTool {
    return {
      implementation: new WebSearchTool(endpoint ?? null),
      schema: WebSearchTool.schema(),
    };
  },

  // ---- Tier 2 ---------------------------------------------------------

  /** `todo_write` — NEW; persists the todo list via RunStore key `"todo"`. */
  todoWrite(): StandardTool {
    return {
      implementation: new TodoWriteTool(),
      schema: TodoWriteTool.schema(),
    };
  },

  /** `task_list` — EXISTING #71 tool (Q5 overlap: reused, not renamed). */
  taskList(): StandardTool {
    return {
      implementation: new TaskListTool(),
      schema: TaskListTool.schema(),
    };
  },

  /** `memory` — NEW #82; scope-aware episodic memory read/write via MemoryStore. */
  memory(): StandardTool {
    return { implementation: new MemoryTool(), schema: MemoryTool.schema() };
  },

  // ---- Tier 3 ---------------------------------------------------------

  /** `enter_plan_mode` — NEW; escalates `enter_plan_mode`. */
  enterPlanMode(): StandardTool {
    return {
      implementation: new EnterPlanModeTool(),
      schema: EnterPlanModeTool.schema(),
    };
  },

  /** `exit_plan_mode` — NEW; escalates `exit_plan_mode { plan }`. */
  exitPlanMode(): StandardTool {
    return {
      implementation: new ExitPlanModeTool(),
      schema: ExitPlanModeTool.schema(),
    };
  },

  /** `ask_user_question` — NEW; returns `awaiting_clarification`. */
  askUserQuestion(): StandardTool {
    return {
      implementation: new AskUserQuestionTool(),
      schema: AskUserQuestionTool.schema(),
    };
  },

  /** `abort` — NEW; escalates `abort { reason }`. */
  abort(): StandardTool {
    return { implementation: new AbortTool(), schema: AbortTool.schema() };
  },

  // ---- Presets --------------------------------------------------------

  /**
   * Read-only investigation set: no mutating or escalating tools. Reuses the
   * EXISTING read-only #5 tools by name (Q5 overlap) plus the NEW `grep`.
   */
  readonlySet(): StandardTool[] {
    return [
      this.readFile(),
      this.listDir(),
      this.grepFiles(),
      this.grep(),
      this.findFiles(),
      this.webFetch(),
      this.webSearch(),
    ];
  },

  /**
   * Coding set: everything in {@link readonlySet} plus the mutating filesystem
   * tools, shell, messaging, and the storage-backed todo/task tools. Reuses
   * EXISTING tool names on overlap (Q5).
   */
  codingSet(): StandardTool[] {
    return [
      this.readFile(),
      this.writeFile(),
      this.editFile(),
      this.listDir(),
      this.grepFiles(),
      this.grep(),
      this.findFiles(),
      this.bashCommand(),
      this.sendMessage(),
      this.webFetch(),
      this.webSearch(),
      this.todoWrite(),
      this.taskList(),
      this.memory(),
    ];
  },

  /**
   * Full set: the {@link codingSet} plus every Tier-3 control tool (plan /
   * clarify / abort).
   */
  fullSet(): StandardTool[] {
    return [
      ...this.codingSet(),
      this.enterPlanMode(),
      this.exitPlanMode(),
      this.askUserQuestion(),
      this.abort(),
    ];
  },
} as const;
