/**
 * `spore-eval` CLI — mirrors the `e2e-agent` pattern.
 *
 * Subcommands:
 *   run <suite.json> [--candidate ID ...] [--n N] [--json]
 *       Load a suite and print/serialize the comparison reports. (Candidate
 *       harness configs are not constructible from the CLI in the MVP — the
 *       command validates the suite and prints its shape; full runs are driven
 *       programmatically via `EvalHarness`.)
 *   promote <suite.json> <task_id>
 *       Manually promote a challenge task to regression, bump suite_version,
 *       rewrite the JSON in place (Rule 31). Auto-promotion is deferred.
 */

import { writeFile } from "node:fs/promises";

import { loadSuitePath, promoteChallengeTask, suiteToJson } from "../index.js";

async function main(argv: string[]): Promise<number> {
  const cmd = argv[0];
  if (cmd == null) {
    printUsage();
    process.stderr.write("no subcommand given\n");
    return 1;
  }
  switch (cmd) {
    case "run":
      return cmdRun(argv.slice(1));
    case "promote":
      return cmdPromote(argv.slice(1));
    case "-h":
    case "--help":
    case "help":
      printUsage();
      return 0;
    default:
      printUsage();
      process.stderr.write(`unknown subcommand: ${cmd}\n`);
      return 1;
  }
}

function printUsage(): void {
  process.stderr.write(
    "spore-eval — EvalHarness CLI\n\n" +
      "USAGE:\n" +
      "  spore-eval run <suite.json> [--candidate ID ...] [--n N] [--json]\n" +
      "  spore-eval promote <suite.json> <task_id>\n",
  );
}

async function cmdRun(args: string[]): Promise<number> {
  let suitePath: string | undefined;
  const candidates: string[] = [];
  let n: number | undefined;
  let json = false;
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;
    if (arg === "--candidate") {
      const v = args[++i];
      if (v == null) throw new Error("--candidate needs an ID");
      candidates.push(v);
    } else if (arg === "--n") {
      const v = args[++i];
      if (v == null) throw new Error("--n needs a number");
      n = Number.parseInt(v, 10);
      if (!Number.isInteger(n)) throw new Error("--n must be an integer");
    } else if (arg === "--json") {
      json = true;
    } else if (!arg.startsWith("--") && suitePath == null) {
      suitePath = arg;
    } else {
      throw new Error(`unexpected argument: ${arg}`);
    }
  }
  if (suitePath == null) throw new Error("missing <suite.json>");
  const suite = await loadSuitePath(suitePath);

  const summary = {
    suite_version: suite.suite_version,
    regression: suite.regression.length,
    challenge: suite.challenge.length,
    canary: suite.canary.length,
    n_runs_per_config: n ?? 3,
    candidates,
    note: "candidate harness configs are wired programmatically; this CLI validates the suite and reports its shape",
  };
  if (json) {
    process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
  } else {
    process.stdout.write(
      `loaded suite v${suite.suite_version} — regression=${suite.regression.length}, challenge=${suite.challenge.length}, canary=${suite.canary.length} (n=${n ?? 3})\n`,
    );
  }
  return 0;
}

async function cmdPromote(args: string[]): Promise<number> {
  const suitePath = args[0];
  const taskId = args[1];
  if (suitePath == null) throw new Error("missing <suite.json>");
  if (taskId == null) throw new Error("missing <task_id>");
  const suite = await loadSuitePath(suitePath);
  const before = suite.suite_version;
  promoteChallengeTask(suite, taskId);
  await writeFile(suitePath, suiteToJson(suite));
  process.stdout.write(
    `promoted ${taskId}: suite_version ${before} -> ${suite.suite_version} (challenge -> regression)\n`,
  );
  return 0;
}

main(process.argv.slice(2))
  .then((code) => {
    process.exitCode = code;
  })
  .catch((e) => {
    process.stderr.write(`${(e as Error).message}\n`);
    process.exitCode = 1;
  });
