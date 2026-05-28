/**
 * Per-tool input schemas (zod) and a parse helper that maps validation
 * errors to {@link ToolExecutionError} `invalid_parameters`.
 */

import { z } from "zod";
import type { ToolCall } from "@spore/core";

import type { ToolExecutionError } from "./errors.js";

export function parseParams<T>(
  schema: z.ZodType<T>,
  call: ToolCall,
): { ok: true; value: T } | { ok: false; error: ToolExecutionError } {
  const r = schema.safeParse(call.input);
  if (r.success) return { ok: true, value: r.data };
  return {
    ok: false,
    error: { kind: "invalid_parameters", reason: r.error.message },
  };
}

// ---------- Filesystem ----------

export const ReadFileParamsSchema = z.object({ path: z.string() });
export type ReadFileParams = z.infer<typeof ReadFileParamsSchema>;

export const WriteFileParamsSchema = z.object({
  path: z.string(),
  content: z.string(),
  append: z.boolean().default(false),
});
export type WriteFileParams = z.infer<typeof WriteFileParamsSchema>;

export const ListDirParamsSchema = z.object({
  path: z.string(),
  recursive: z.boolean().default(false),
});
export type ListDirParams = z.infer<typeof ListDirParamsSchema>;

export const DeleteFileParamsSchema = z.object({ path: z.string() });
export type DeleteFileParams = z.infer<typeof DeleteFileParamsSchema>;

export const MoveFileParamsSchema = z.object({
  src: z.string(),
  dst: z.string(),
});
export type MoveFileParams = z.infer<typeof MoveFileParamsSchema>;

// ---------- Exec ----------

/**
 * Parameters for the shell-free {@link import("./exec.js").ExecTool}: a program
 * name plus a verbatim argument vector. No shell is involved.
 */
export const ExecParamsSchema = z.object({
  command: z.string(),
  args: z.array(z.string()).default([]),
  /** Timeout in whole seconds. */
  timeout: z.number().int().nonnegative().nullable().optional(),
});
export type ExecParams = z.infer<typeof ExecParamsSchema>;

/**
 * Parameters for the real {@link import("./exec.js").BashCommandTool}: a single
 * shell `script` run via `/bin/sh -c`, with an optional working directory.
 */
export const ShellCommandParamsSchema = z.object({
  script: z.string(),
  working_dir: z.string().nullable().optional(),
  /** Timeout in whole seconds. */
  timeout: z.number().int().nonnegative().nullable().optional(),
});
export type ShellCommandParams = z.infer<typeof ShellCommandParamsSchema>;

export const RunTestsParamsSchema = z.object({
  command: z.string(),
  working_dir: z.string(),
  timeout: z.number().int().nonnegative().nullable().optional(),
});
export type RunTestsParams = z.infer<typeof RunTestsParamsSchema>;

// ---------- Search ----------

export const GrepFilesParamsSchema = z.object({
  pattern: z.string(),
  path: z.string(),
  recursive: z.boolean().default(false),
});
export type GrepFilesParams = z.infer<typeof GrepFilesParamsSchema>;

export const FindFilesParamsSchema = z.object({
  glob: z.string(),
  path: z.string(),
});
export type FindFilesParams = z.infer<typeof FindFilesParamsSchema>;

// ---------- Git ----------

export const GitLogParamsSchema = z.object({
  n: z.number().int().nonnegative().default(20),
  format: z.string().default("oneline"),
});
export type GitLogParams = z.infer<typeof GitLogParamsSchema>;

export const GitDiffParamsSchema = z.object({
  from: z.string().nullable().optional(),
  to: z.string().nullable().optional(),
});
export type GitDiffParams = z.infer<typeof GitDiffParamsSchema>;

export const GitCommitParamsSchema = z.object({
  message: z.string(),
  files: z.array(z.string()).default([]),
});
export type GitCommitParams = z.infer<typeof GitCommitParamsSchema>;

export const GitStatusParamsSchema = z.object({}).passthrough();
export type GitStatusParams = z.infer<typeof GitStatusParamsSchema>;

export const GitResetModeSchema = z.enum(["hard", "soft", "mixed"]);
export type GitResetMode = z.infer<typeof GitResetModeSchema>;

export const GitResetParamsSchema = z.object({
  target: z.string(),
  mode: GitResetModeSchema,
});
export type GitResetParams = z.infer<typeof GitResetParamsSchema>;

// ---------- HTTP ----------

export const HttpGetParamsSchema = z.object({
  url: z.string(),
  headers: z.record(z.string()).nullable().optional(),
});
export type HttpGetParams = z.infer<typeof HttpGetParamsSchema>;

export const HttpPostParamsSchema = z.object({
  url: z.string(),
  body: z.unknown(),
  headers: z.record(z.string()).nullable().optional(),
});
export type HttpPostParams = z.infer<typeof HttpPostParamsSchema>;

// ---------- Subagent ----------

export const SubagentParamsSchema = z.object({ instruction: z.string() });
export type SubagentParams = z.infer<typeof SubagentParamsSchema>;
