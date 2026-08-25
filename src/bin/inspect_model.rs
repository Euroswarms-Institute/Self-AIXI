//! GGUF audit — validates a checkpoint for the dissection and prints the
//! surgical inventory: architecture, per-block layer kinds (Gated-DeltaNet
//! vs full attention, detected by tensor presence), the bit-token probe, the
//! quantization census, and exactly how many parameters the surgery keeps
//! versus discards (vocabulary rows beyond the bit alphabet, MTP head).
//!
//! Metadata-only: no tensor data is read.

use mc_aixi::llm::config::{Qwen35Config, TokenProbe};
use mc_aixi::llm::gguf::{GgufFile, GgufValue};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut path: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => path = args.next().map(PathBuf::from),
            "--help" | "-h" => {
                println!("usage: inspect-model --gguf <file.gguf>");
                return;
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let Some(path) = path else {
        eprintln!("usage: inspect-model --gguf <file.gguf>");
        std::process::exit(2);
    };

    let gguf = match GgufFile::open(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("FAIL parse: {e}");
            std::process::exit(1);
        }
    };

    println!("== container ==");
    println!(
        "gguf v{}, {} tensors, alignment {}, data at 0x{:x}",
        gguf.version,
        gguf.tensors.len(),
        gguf.alignment,
        gguf.data_offset
    );
    println!("\n== metadata ==");
    for key in &gguf.key_order {
        match &gguf.kvs[key] {
            GgufValue::StrArray(v) => println!("{key} = <{} strings>", v.len()),
            GgufValue::SkippedArray { elem_type, count } => {
                println!("{key} = <skipped array: type {elem_type} × {count}>")
            }
            GgufValue::Str(s) if s.len() > 80 => println!("{key} = <{} chars>", s.len()),
            v => println!("{key} = {v:?}"),
        }
    }

    println!("\n== architecture ==");
    let cfg = match Qwen35Config::from_gguf(&gguf) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL config: {e}");
            std::process::exit(1);
        }
    };
    println!("{cfg:#?}");

    println!("\n== layer kinds (by tensor presence) ==");
    let mut kinds = Vec::new();
    for i in 0..cfg.n_layers {
        let deltanet = gguf.tensor(&format!("blk.{i}.attn_qkv.weight")).is_some();
        let attention = gguf.tensor(&format!("blk.{i}.attn_q.weight")).is_some();
        let kind = match (deltanet, attention) {
            (true, false) => "deltanet",
            (false, true) => "attention",
            _ => "UNRECOGNIZED",
        };
        kinds.push(kind);
        println!("blk.{i:<2} {kind}");
    }
    let n_att = kinds.iter().filter(|k| **k == "attention").count();
    println!(
        "{} deltanet + {} attention blocks",
        kinds.len() - n_att,
        n_att
    );

    println!("\n== quantization census ==");
    let mut census: BTreeMap<String, (usize, u64, u64)> = BTreeMap::new();
    for t in &gguf.tensors {
        let ty = t
            .ggml_type()
            .map(|t| t.name().to_string())
            .unwrap_or_else(|_| format!("raw{}", t.raw_type));
        let bytes = t.byte_len().unwrap_or(0) as u64;
        let e = census.entry(ty).or_insert((0, 0, 0));
        e.0 += 1;
        e.1 += t.elems();
        e.2 += bytes;
    }
    for (ty, (count, elems, bytes)) in &census {
        println!(
            "{ty:<5} {count:>3} tensors  {:>7.1}M params  {:>7.1} MiB",
            *elems as f64 / 1e6,
            *bytes as f64 / (1024.0 * 1024.0)
        );
    }

    println!("\n== the carve ==");
    let probe = match TokenProbe::from_gguf(&gguf) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FAIL token probe: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "bit tokens: \"0\" → id {}, \"1\" → id {}; stream prime → id {} (eos {:?})",
        probe.bit0, probe.bit1, probe.prime, probe.eos
    );
    let total: u64 = gguf.tensors.iter().map(|t| t.elems()).sum();
    let embd: u64 = gguf
        .tensor("token_embd.weight")
        .map(|t| t.elems())
        .unwrap_or(0);
    let embd_kept = 3 * cfg.hidden as u64; // rows {bit0, bit1, prime}
    let nextn: u64 = gguf
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("nextn."))
        .map(|t| t.elems())
        .sum();
    let discarded = (embd - embd_kept) + nextn;
    println!(
        "params total {:.1}M | discarded by surgery {:.1}M ({:.1}M vocabulary rows + {:.1}M MTP head) | kept {:.1}M ({:.0}%)",
        total as f64 / 1e6,
        discarded as f64 / 1e6,
        (embd - embd_kept) as f64 / 1e6,
        nextn as f64 / 1e6,
        (total - discarded) as f64 / 1e6,
        100.0 * (total - discarded) as f64 / total as f64
    );
    println!("\nOK — checkpoint is dissectable");
}
