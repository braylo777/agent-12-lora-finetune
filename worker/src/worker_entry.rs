//! Cloudflare Worker entry (wasm32 only). Same routes and JSON as the old src/index.js.
use crate::app::handle;
use crate::page::PAGE;
use worker::*;

#[event(fetch)]
async fn fetch(req: Request, _env: Env, _ctx: Context) -> Result<Response> {
    let (status, v) = handle(req.url()?.path());
    if status == 0 {
        let h = Headers::new();
        h.set("content-type", "text/html;charset=utf-8")?;
        return Ok(Response::ok(PAGE)?.with_headers(h));
    }
    Ok(Response::from_json(&v)?.with_status(status))
}
