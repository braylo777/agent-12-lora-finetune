import { test } from "node:test";
import assert from "node:assert/strict";
import worker from "../src/index.js";
import { RESULTS } from "../src/results.js";
test("edge: accuracies are probabilities", () => { for (const m of [RESULTS.before, RESULTS.after]) assert.ok(m.accuracy >= 0 && m.accuracy <= 1 && m.correct <= m.n); });
test("edge: loss log has finite numbers", () => { for (const x of RESULTS.training.loss_log) { const v = typeof x === "number" ? x : x.loss; assert.ok(Number.isFinite(v)); } });
test("edge: unknown route does not 5xx", async () => { assert.ok((await worker.fetch(new Request("https://x/nope"), {})).status < 500); });
