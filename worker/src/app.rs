//! Serves the measured before/after results. Live inference is not available (see /api/predict).
use serde_json::{json, Value};

const RESULTS_JSON: &str = include_str!("../data/results.json");

pub fn results() -> Value {
    serde_json::from_str(RESULTS_JSON).expect("results.json")
}

/// Pure router: (status, json); status 0 = serve the page.
pub fn handle(path: &str) -> (u16, Value) {
    match path {
        "/api/results" => (200, results()),
        "/api/predict" => (501, json!({"error": "Live inference is not served: Workers AI cannot load LoRA adapters on SmolLM2-135M. See /api/results for measured before/after outputs."})),
        _ => (0, Value::Null),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::PAGE;

    #[test]
    fn results_are_consistent() {
        let r = results();
        assert_eq!(r["before"]["n"], r["after"]["n"]);
        assert_eq!(r["before"]["n"], r["split"]["heldout_test"]);
        for m in [&r["before"], &r["after"]] {
            let acc = (m["correct"].as_f64().unwrap() / m["n"].as_f64().unwrap() * 1000.0).round() / 1000.0;
            assert_eq!(m["accuracy"].as_f64().unwrap(), acc);
            assert!((0.0..=1.0).contains(&acc) && m["correct"].as_u64() <= m["n"].as_u64());
        }
        let d = ((r["after"]["accuracy"].as_f64().unwrap() - r["before"]["accuracy"].as_f64().unwrap()) * 1000.0).round() / 1000.0;
        assert_eq!(r["delta_accuracy"].as_f64().unwrap(), d);
        let t = &r["training"];
        assert!(t["trainable_params"].as_f64().unwrap() < t["total_params"].as_f64().unwrap() / 50.0);
        let log = t["loss_log"].as_array().unwrap();
        assert!(log.len() >= 5);
        assert!(log.iter().all(|x| x.as_f64().or_else(|| x["loss"].as_f64()).map_or(false, f64::is_finite)));
    }
    #[test]
    fn routes() {
        assert!(PAGE.contains("Before / After"));
        assert_eq!(handle("/api/results").1["base_model"], results()["base_model"]);
        assert_eq!(handle("/api/predict").0, 501);
        assert_eq!(handle("/nope").0, 0);
    }
}
