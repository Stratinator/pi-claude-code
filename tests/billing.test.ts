import assert from "node:assert/strict";
import { chmod, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as http from "node:http";
import { join } from "node:path";
import test from "node:test";
import { isRetryableAssistantError, type AssistantMessage } from "@earendil-works/pi-ai";
import { ALLOW_USAGE_CREDITS_ENV, describeUsageCreditsRejection, usageCreditsAllowed } from "../src/billing.ts";
import { closeClaudeProcesses, runClaudeRequest } from "../src/claude-cli.ts";
import { normalizeError } from "../src/stream.ts";

function failedMessage(errorMessage: string): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: "claude-code-stream-json",
    provider: "claude-code",
    model: "sonnet",
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "error",
    errorMessage,
    timestamp: 0,
  };
}

test("rewrites usage-credit rejections into non-retryable billing errors", () => {
  for (const upstream of [
    "Claude Code API error: API Error: 400 Third-party apps now draw from your extra usage, not your plan limits.",
    'Incomplete Claude Code upstream response (status 400, upstream said: You\'re out of extra usage. Add more at claude.ai/settings/usage and keep going.)',
  ]) {
    const message = normalizeError(new Error(upstream));
    assert.match(message, /^Claude Code billing:/);
    assert.ok(message.endsWith(upstream));
    // Even with transient-looking text attached, Pi must never retry a request
    // Anthropic routed to usage credits.
    assert.equal(isRetryableAssistantError(failedMessage(`${message} 429 overloaded`)), false);
  }
  assert.equal(describeUsageCreditsRejection("Claude Code API error: overloaded"), undefined);
});

test("usage credits are opt-in", () => {
  assert.equal(usageCreditsAllowed({}), false);
  assert.equal(usageCreditsAllowed({ [ALLOW_USAGE_CREDITS_ENV]: "0" }), false);
  assert.equal(usageCreditsAllowed({ [ALLOW_USAGE_CREDITS_ENV]: "false" }), false);
  assert.equal(usageCreditsAllowed({ [ALLOW_USAGE_CREDITS_ENV]: "1" }), true);
});

test("stops when Claude Code reports a request was charged to usage credits", async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "pi-claude-code-billing-")));
  const executable = join(root, "fake-claude.mjs");
  await writeFile(
    executable,
    `#!/usr/bin/env node
import readline from "node:readline";
if (process.argv.includes("auth") && process.argv.includes("status")) {
  console.log(JSON.stringify({loggedIn: true, authMethod: "claude.ai", subscriptionType: "max"}));
  process.exit(0);
}
const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
for await (const line of input) {
  const frame = JSON.parse(line);
  if (frame.type !== "user") continue;
  const usage = {input_tokens: 3, output_tokens: 1};
  const events = [
    {type: "message_start", message: {id: "m", role: "assistant", content: [], usage}},
    {type: "content_block_start", index: 0, content_block: {type: "text", text: ""}},
    {type: "content_block_delta", index: 0, delta: {type: "text_delta", text: "ok"}},
    {type: "content_block_stop", index: 0},
    {type: "message_delta", delta: {stop_reason: "end_turn"}, usage},
    {type: "message_stop"}
  ];
  for (const event of events) console.log(JSON.stringify({type: "stream_event", event}));
  console.log(JSON.stringify({type: "assistant", message: {id: "m", role: "assistant", content: [{type: "text", text: "ok"}], stop_reason: "end_turn", usage}}));
  console.log(JSON.stringify({type: "rate_limit_event", rate_limit_info: {
    status: process.env.FAKE_STATUS, rateLimitType: "five_hour", isUsingOverage: process.env.FAKE_OVERAGE === "1",
    overageStatus: "allowed"
  }}));
  console.log(JSON.stringify({type: "result", subtype: "success", is_error: false, usage}));
}
`,
    "utf8",
  );
  await chmod(executable, 0o755);

  const request = (env: Record<string, string>) =>
    runClaudeRequest({
      model: "sonnet",
      cwd: root,
      frames: [{ type: "user", message: { role: "user", content: [{ type: "text", text: "hi" }] } }],
      systemPrompt: "system",
      manifest: [],
      extraBody: { tools: [] },
      toolNames: new Set(),
      env: { PI_CLAUDE_CODE_COMMAND: executable, ...env },
      timeoutMs: 5_000,
    });

  try {
    const plan = await request({ FAKE_STATUS: "allowed", FAKE_OVERAGE: "0" });
    assert.equal(plan.rateLimit?.status, "allowed");
    assert.equal(plan.rateLimit?.isUsingOverage, false);

    await assert.rejects(
      request({ FAKE_STATUS: "allowed_warning", FAKE_OVERAGE: "1" }),
      (error: Error) => {
        assert.match(error.message, /^Claude Code billing: .*usage credits/);
        assert.match(error.message, new RegExp(ALLOW_USAGE_CREDITS_ENV));
        assert.equal(isRetryableAssistantError(failedMessage(error.message)), false);
        return true;
      },
    );

    const allowed = await request({
      FAKE_STATUS: "allowed_warning",
      FAKE_OVERAGE: "1",
      [ALLOW_USAGE_CREDITS_ENV]: "1",
    });
    assert.equal(allowed.rateLimit?.isUsingOverage, true);
  } finally {
    await closeClaudeProcesses();
    await rm(root, { recursive: true, force: true });
  }
});

test("an upstream stream that drops before message_stop is retryable, an upstream error event decides its own", async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "pi-claude-code-drop-")));
  const executable = join(root, "fake-claude.mjs");
  await writeFile(
    executable,
    `#!/usr/bin/env node
import readline from "node:readline";
if (process.argv.includes("auth") && process.argv.includes("status")) {
  console.log(JSON.stringify({loggedIn: true, authMethod: "claude.ai", subscriptionType: "max"}));
  process.exit(0);
}
const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
for await (const line of input) {
  const frame = JSON.parse(line);
  if (frame.type !== "user") continue;
  try {
    const response = await fetch(process.env.ANTHROPIC_BASE_URL + "/v1/messages", {method: "POST", body: "{}"});
    await response.text();
  } catch {}
  console.log(JSON.stringify({type: "result", subtype: "error_during_execution", is_error: true}));
}
`,
    "utf8",
  );
  await chmod(executable, 0o755);
  const usage = { input_tokens: 2, output_tokens: 0 };
  const start = `data: ${JSON.stringify({ type: "message_start", message: { id: "m1", role: "assistant", content: [], usage } })}\n\n`;
  const upstream = http.createServer(async (request, response) => {
    for await (const _chunk of request);
    response.writeHead(200, { "content-type": "text/event-stream" });
    if (request.url?.startsWith("/drop/")) {
      response.write(start);
      setTimeout(() => response.destroy(), 20);
    } else {
      const error = { type: "error", error: { type: "invalid_request_error", message: "messages: bad shape" } };
      response.end(`${start}data: ${JSON.stringify(error)}\n\n`);
    }
  });
  await new Promise<void>((resolve) => upstream.listen(0, "127.0.0.1", resolve));
  const { port } = upstream.address() as { port: number };
  const request = (path: string) =>
    runClaudeRequest({
      model: "sonnet",
      cwd: root,
      frames: [{ type: "user", message: { role: "user", content: [{ type: "text", text: "hi" }] } }],
      systemPrompt: "system",
      manifest: [],
      extraBody: { tools: [] },
      toolNames: new Set(),
      env: { PI_CLAUDE_CODE_COMMAND: executable },
      upstreamUrl: `http://127.0.0.1:${port}${path}`,
      timeoutMs: 5_000,
    });
  const retryable = async (path: string) => {
    let message = "";
    await assert.rejects(request(path), (error: Error) => ((message = normalizeError(error)), true));
    return { message, retryable: isRetryableAssistantError(failedMessage(message)) };
  };

  try {
    const dropped = await retryable("/drop");
    assert.equal(dropped.retryable, true, dropped.message);
    const rejected = await retryable("/error");
    assert.equal(rejected.retryable, false, rejected.message);
    assert.match(rejected.message, /messages: bad shape/);
  } finally {
    await closeClaudeProcesses();
    await new Promise<void>((resolve) => upstream.close(() => resolve()));
    await rm(root, { recursive: true, force: true });
  }
});
