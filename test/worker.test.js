import { test } from "node:test";
import assert from "node:assert/strict";
import worker from "../src/index.js";
import { RESULTS } from "../src/results.js";

test("results are measured on the same held-out split for both models", () => { assert.equal(RESULTS.before.n, RESULTS.after.n); assert.equal(RESULTS.before.n, RESULTS.split.heldout_test); });
test("accuracy fields are consistent with correct/n", () => { for (const m of [RESULTS.before, RESULTS.after]) assert.equal(m.accuracy, +(m.correct / m.n).toFixed(3)); });
test("delta equals after minus before", () => { assert.equal(RESULTS.delta_accuracy, +(RESULTS.after.accuracy - RESULTS.before.accuracy).toFixed(3)); });
test("training log records the LoRA config and a loss curve", () => { assert.ok(RESULTS.training.trainable_params < RESULTS.training.total_params / 50); assert.ok(RESULTS.training.loss_log.length >= 5); });
test("HTTP: page, results API, and honest 501 for live inference", async () => {
  const p = await worker.fetch(new Request("https://x/")); assert.match(await p.text(), /Before \/ After/);
  const r = await (await worker.fetch(new Request("https://x/api/results"))).json(); assert.equal(r.base_model, RESULTS.base_model);
  assert.equal((await worker.fetch(new Request("https://x/api/predict"))).status, 501);
});
