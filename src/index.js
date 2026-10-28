import { json } from "./core.js";
import { RESULTS } from "./results.js";
import page from "./page.html";

export default {
  async fetch(req) {
    const p = new URL(req.url).pathname;
    if (p === "/api/results") return json(RESULTS);
    if (p === "/api/predict") return json({ error: "Live inference is not served: Workers AI cannot load LoRA adapters on SmolLM2-135M. See /api/results for measured before/after outputs." }, 501);
    return new Response(page, { headers: { "content-type": "text/html;charset=utf-8" } });
  },
};
