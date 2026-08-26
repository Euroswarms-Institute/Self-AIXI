//! Dev tool: print this crate's bit-token logits for an explicit token-id
//! stream — the Rust half of the llama.cpp oracle comparison
//! (scripts/oracle_check.sh). Output format matches oracle_probe.cpp:
//! one "<i> <logit0> <logit1>" line per position.

use mc_aixi::llm::model::{QGateLayout, Qwen35Model};
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut gguf: Option<PathBuf> = None;
    let mut tokens: Vec<u32> = Vec::new();
    let mut layout = QGateLayout::PerHeadInterleaved;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf = args.next().map(PathBuf::from),
            "--tokens" => {
                tokens = args
                    .next()
                    .expect("--tokens needs a value")
                    .split(',')
                    .map(|t| t.parse().expect("token id"))
                    .collect();
            }
            "--blocked-qgate" => layout = QGateLayout::Blocked,
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let (Some(gguf), false) = (gguf, tokens.is_empty()) else {
        eprintln!("usage: oracle_probe --gguf model.gguf --tokens id0,id1,... [--blocked-qgate]");
        std::process::exit(2);
    };

    let t0 = std::time::Instant::now();
    let model = match Qwen35Model::load(&gguf, layout) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("FAIL load: {e}");
            std::process::exit(1);
        }
    };
    let load_s = t0.elapsed().as_secs_f64();
    let mut state = model.new_state();
    let t1 = std::time::Instant::now();
    for (i, &t) in tokens.iter().enumerate() {
        let logits = model.advance(&mut state, t);
        println!("{i} {:.6} {:.6}", logits[0], logits[1]);
    }
    let infer_s = t1.elapsed().as_secs_f64();
    // Timing to stderr so stdout stays diffable against oracle_probe.cpp.
    eprintln!(
        "timing: load {load_s:.2}s, {} tokens in {infer_s:.2}s ({:.3}s/token)",
        tokens.len(),
        infer_s / tokens.len() as f64
    );
}
