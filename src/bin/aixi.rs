//! The MC-AIXI agent CLI: run ρUCT over a
//! Bayesian mixture environment model on one of the JAIR §7 domains, and
//! stream per-cycle metrics — reward, running average, ξ root log-probability
//! and, the headline, the mixture posterior trajectory over the model
//! catalog (FAC-CTW depths ⊕ uniform ⊕ the dissected Qwen3.8-2B).
//!
//! Everything is finite and validated (§1.1); every run is reproducible from
//! its --seed.

use mc_aixi::agent::AixiAgent;
use mc_aixi::encoding::encode_bits_msb;
use mc_aixi::env::biased_rps::BiasedRockPaperScissors;
use mc_aixi::env::cheese_maze::CheeseMaze;
use mc_aixi::env::coin_flip::CoinFlip;
use mc_aixi::env::kuhn_poker::KuhnPoker;
use mc_aixi::env::pocman::PocMan;
use mc_aixi::env::text_bytes::TextBytes;
use mc_aixi::env::tiger::Tiger;
use mc_aixi::env::{DomainSpec, Environment};
use mc_aixi::llm::byte_model::{load_byte_carved, LlmByteModel};
use mc_aixi::llm::env_model::LlmModel;
use mc_aixi::llm::model::{QGateLayout, Qwen35Model};
use mc_aixi::models::fac_ctw::FacCtwModel;
use mc_aixi::models::mixture::BayesMixture;
use mc_aixi::models::uniform::UniformModel;
use mc_aixi::models::EnvModel;
use mc_aixi::planning::modal_byte::plan_modal_byte;
use mc_aixi::planning::rho_uct::SearchBudget;
use mc_aixi::rng::seeded;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "\
usage: aixi --env <coin_flip|biased_rps|cheese_maze|tiger|kuhn_poker|pocman|text_bytes> [options]

model options:
  --model <ctw-mix|fac-ctw|llm|full-mix|byte-llm|byte-mix>
                        catalog for the mixture xi (default ctw-mix;
                        byte-llm / byte-mix use the 256-way byte carve and
                        require an 8-bit-observation domain like text_bytes)
  --ct-depths a,b,c     FAC-CTW depths in the catalog (default 4,8,16)
  --gguf PATH           checkpoint for llm/full-mix/byte-* (default models/Qwen3.8-2B-Q4_K_M.gguf)

search budget (finite, validated; text_bytes plans by exact horizon-1
expectimax instead, so only --cycles applies there):
  --cycles N            interaction cycles (default 300)
  --mc-simulations N    rhoUCT simulations per decision (default 300)
  --horizon N           planning horizon m (default 3)
  --gamma F             discount (default 0.99)
  --uct-c F             UCB exploration constant (default 0.35)
  --root-parallel N     N independent rhoUCT searches per decision over
                        clones of xi, root statistics merged (CTW catalogs
                        only; total simulations stay --mc-simulations)

run options:
  --text-file PATH      corpus for text_bytes (default: embedded English text)
  --seed N              RNG seed (default 42)
  --report-every N      print a metrics line every N cycles (default 25)
  --csv PATH            also write per-cycle records to PATH
";

struct Args {
    env: String,
    model: String,
    ct_depths: Vec<usize>,
    gguf: PathBuf,
    text_file: Option<PathBuf>,
    cycles: usize,
    mc_simulations: u32,
    horizon: u32,
    gamma: f64,
    uct_c: f64,
    root_parallel: Option<usize>,
    seed: u64,
    report_every: usize,
    csv: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        env: String::new(),
        model: "ctw-mix".into(),
        ct_depths: vec![4, 8, 16],
        gguf: PathBuf::from("models/Qwen3.8-2B-Q4_K_M.gguf"),
        text_file: None,
        cycles: 300,
        mc_simulations: 300,
        horizon: 3,
        gamma: 0.99,
        uct_c: 0.35,
        root_parallel: None,
        seed: 42,
        report_every: 25,
        csv: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--env" => a.env = val()?,
            "--model" => a.model = val()?,
            "--ct-depths" => {
                a.ct_depths = val()?
                    .split(',')
                    .map(|d| d.parse().map_err(|_| format!("bad depth {d}")))
                    .collect::<Result<_, _>>()?;
            }
            "--gguf" => a.gguf = PathBuf::from(val()?),
            "--text-file" => a.text_file = Some(PathBuf::from(val()?)),
            "--cycles" => a.cycles = val()?.parse().map_err(|e| format!("cycles: {e}"))?,
            "--mc-simulations" => {
                a.mc_simulations = val()?.parse().map_err(|e| format!("sims: {e}"))?
            }
            "--horizon" => a.horizon = val()?.parse().map_err(|e| format!("horizon: {e}"))?,
            "--gamma" => a.gamma = val()?.parse().map_err(|e| format!("gamma: {e}"))?,
            "--uct-c" => a.uct_c = val()?.parse().map_err(|e| format!("uct-c: {e}"))?,
            "--root-parallel" => {
                let n: usize = val()?.parse().map_err(|e| format!("root-parallel: {e}"))?;
                if n == 0 {
                    return Err("--root-parallel needs at least 1 worker".into());
                }
                a.root_parallel = Some(n);
            }
            "--seed" => a.seed = val()?.parse().map_err(|e| format!("seed: {e}"))?,
            "--report-every" => {
                a.report_every = val()?.parse().map_err(|e| format!("report: {e}"))?
            }
            "--csv" => a.csv = Some(PathBuf::from(val()?)),
            "--help" | "-h" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if a.env.is_empty() {
        return Err(format!("--env is required\n{USAGE}"));
    }
    Ok(a)
}

fn build_env(args: &Args) -> Result<Box<dyn Environment>, String> {
    Ok(match args.env.as_str() {
        "coin_flip" => Box::new(CoinFlip::default()),
        "biased_rps" => Box::new(BiasedRockPaperScissors::default()),
        "cheese_maze" => Box::new(CheeseMaze::default()),
        "tiger" => Box::new(Tiger::default()),
        "kuhn_poker" => Box::new(KuhnPoker::default()),
        "pocman" => Box::new(PocMan::default()),
        "text_bytes" => match &args.text_file {
            Some(p) => {
                let corpus = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
                Box::new(TextBytes::from_bytes(corpus)?)
            }
            None => Box::new(TextBytes::embedded()),
        },
        other => return Err(format!("unknown environment {other}")),
    })
}

fn build_catalog(args: &Args, spec: &DomainSpec) -> Result<Vec<Box<dyn EnvModel>>, String> {
    let pbits = spec.percept_bits() as usize;
    let fac = |d: &usize| Box::new(FacCtwModel::new(*d, pbits)) as Box<dyn EnvModel>;
    let llm = |args: &Args| -> Result<Box<dyn EnvModel>, String> {
        eprintln!("loading {} ...", args.gguf.display());
        let t0 = Instant::now();
        let model = Qwen35Model::load(&args.gguf, QGateLayout::PerHeadInterleaved)?;
        eprintln!(
            "loaded {} ({:.1} MiB resident) in {:.1}s",
            model.cfg.n_layers,
            model.weight_bytes() as f64 / (1024.0 * 1024.0),
            t0.elapsed().as_secs_f64()
        );
        Ok(Box::new(LlmModel::new(model)))
    };
    let byte_llm = |args: &Args| -> Result<Box<dyn EnvModel>, String> {
        if spec.observation_bits != 8 {
            return Err(format!(
                "the byte carve models 8-bit observations; {} has {}",
                args.env, spec.observation_bits
            ));
        }
        eprintln!("loading {} (byte carve) ...", args.gguf.display());
        let t0 = Instant::now();
        let (model, probe, head) = load_byte_carved(&args.gguf, QGateLayout::PerHeadInterleaved)?;
        eprintln!(
            "loaded {} ({:.1} MiB resident incl. {:.1} MiB head) in {:.1}s",
            model.cfg.n_layers,
            (model.weight_bytes() + head.byte_len()) as f64 / (1024.0 * 1024.0),
            head.byte_len() as f64 / (1024.0 * 1024.0),
            t0.elapsed().as_secs_f64()
        );
        Ok(Box::new(LlmByteModel::new(
            model,
            probe,
            head,
            spec.action_bits as usize,
            spec.observation_bits as usize,
            spec.reward_bits as usize,
        )))
    };
    Ok(match args.model.as_str() {
        "fac-ctw" => vec![fac(args.ct_depths.first().ok_or("need a depth")?)],
        "ctw-mix" => {
            let mut v: Vec<Box<dyn EnvModel>> = args.ct_depths.iter().map(fac).collect();
            v.push(Box::new(UniformModel::default()));
            v
        }
        "llm" => vec![llm(args)?],
        "full-mix" => {
            let mut v: Vec<Box<dyn EnvModel>> = args.ct_depths.iter().map(fac).collect();
            v.push(Box::new(UniformModel::default()));
            v.push(llm(args)?);
            v
        }
        "byte-llm" => vec![byte_llm(args)?],
        "byte-mix" => {
            let mut v: Vec<Box<dyn EnvModel>> = args.ct_depths.iter().map(fac).collect();
            v.push(Box::new(UniformModel::default()));
            v.push(byte_llm(args)?);
            v
        }
        other => return Err(format!("unknown model catalog {other}")),
    })
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(&args) {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}

/// Render a byte for terminal-friendly per-cycle reporting.
fn show_byte(b: u64) -> String {
    match u8::try_from(b) {
        Ok(c) if (0x20..0x7F).contains(&c) => format!("'{}'", c as char),
        Ok(c) => format!("{c:#04x}"),
        Err(_) => format!("{b}"),
    }
}

/// The prediction-domain loop: exact horizon-1 expectimax (modal byte of
/// ξ's observation marginal) instead of sampled ρUCT — at m = 1 the
/// enumeration IS the search, and 256 actions would starve any sampled
/// budget anyway.
fn run_prediction(args: &Args, mut env: Box<dyn Environment>) -> Result<(), String> {
    let spec = DomainSpec::from_env(env.as_ref());
    let mut model = BayesMixture::uniform(build_catalog(args, &spec)?);
    let ids = model.component_ids();
    let mut rng = seeded(args.seed);
    env.reset(&mut rng);

    let mut csv = match &args.csv {
        Some(p) => {
            let mut f = std::fs::File::create(p).map_err(|e| format!("csv: {e}"))?;
            writeln!(
                f,
                "cycle,action,observation,reward,avg_reward,root_log_prob,{}",
                ids.join(",")
            )
            .map_err(|e| e.to_string())?;
            Some(f)
        }
        None => None,
    };

    println!(
        "env={} spec: {} actions / {}+{} percept bits | xi = mixture[{}] | planner: exact horizon-1 expectimax (modal byte) | seed {}",
        args.env,
        spec.num_actions,
        spec.observation_bits,
        spec.reward_bits,
        ids.join(" + "),
        args.seed
    );

    let mut total_reward = 0.0;
    let mut window_reward = 0.0;
    let run_start = Instant::now();
    let mut window_start = Instant::now();
    let mut abits = Vec::with_capacity(8);
    let mut pbits = Vec::new();
    let mut last_pair: (u64, u64);
    for cycle in 1..=args.cycles {
        let (action, _q) = plan_modal_byte(&mut model, &spec);
        let percept = env.step(action, &mut rng);
        abits.clear();
        encode_bits_msb(action, spec.action_bits, &mut abits);
        model.append_history_symbols(&abits);
        pbits.clear();
        percept.encode_into(spec.observation_bits, spec.reward_bits, &mut pbits);
        model.learn_symbols(&pbits);
        let reward = env.decode_reward(percept.reward_code);
        total_reward += reward;
        window_reward += reward;
        last_pair = (action, percept.observation);

        if let Some(f) = csv.as_mut() {
            let w = model.posterior_weights();
            writeln!(
                f,
                "{cycle},{action},{},{reward},{:.6},{:.3},{}",
                percept.observation,
                total_reward / cycle as f64,
                model.root_log_probability(),
                w.iter()
                    .map(|x| format!("{x:.4}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )
            .map_err(|e| e.to_string())?;
        }

        if cycle % args.report_every == 0 || cycle == args.cycles {
            let w = model.posterior_weights();
            let posterior = ids
                .iter()
                .zip(&w)
                .map(|(id, w)| format!("{id}={w:.3}"))
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "cycle {cycle:>5} | accuracy {:.4} (window {:.4}) | guess {} actual {} | ln xi = {:>9.2} | {} | {:.0} ms/cycle",
                total_reward / cycle as f64,
                window_reward / args.report_every.min(cycle) as f64,
                show_byte(last_pair.0),
                show_byte(last_pair.1),
                model.root_log_probability(),
                posterior,
                window_start.elapsed().as_millis() as f64 / args.report_every.min(cycle) as f64,
            );
            window_reward = 0.0;
            window_start = Instant::now();
        }
    }
    println!(
        "done: {} cycles in {:.1}s | prediction accuracy {:.4}",
        args.cycles,
        run_start.elapsed().as_secs_f64(),
        total_reward / args.cycles as f64
    );
    Ok(())
}

fn run(args: &Args) -> Result<(), String> {
    let mut env = build_env(args)?;
    let spec = DomainSpec::from_env(env.as_ref());
    if spec.action_bits == 8 && spec.observation_bits == 8 {
        return run_prediction(args, env);
    }
    let budget = SearchBudget::new(args.mc_simulations, args.horizon, args.uct_c, args.gamma)?;
    let model = BayesMixture::uniform(build_catalog(args, &spec)?);
    let ids = model.component_ids();
    let mut agent = AixiAgent::new(model, spec, budget);
    if let Some(workers) = args.root_parallel {
        if agent.model.try_clone_box().is_none() {
            return Err(format!(
                "--root-parallel needs a clonable model catalog; {} declines \
                 (the LLM carves keep too much state to copy per decision)",
                agent.model.model_id()
            ));
        }
        agent.root_parallel = Some((workers, args.seed ^ 0x726F_6F74));
    }
    let mut rng = seeded(args.seed);
    env.reset(&mut rng);

    let mut csv = match &args.csv {
        Some(p) => {
            let mut f = std::fs::File::create(p).map_err(|e| format!("csv: {e}"))?;
            writeln!(
                f,
                "cycle,action,observation,reward,avg_reward,root_log_prob,{}",
                ids.join(",")
            )
            .map_err(|e| e.to_string())?;
            Some(f)
        }
        None => None,
    };

    println!(
        "env={} spec: {} actions / {}+{} percept bits | xi = mixture[{}] | budget: {} sims, horizon {}, gamma {}, c {} | seed {}",
        args.env,
        spec.num_actions,
        spec.observation_bits,
        spec.reward_bits,
        ids.join(" + "),
        args.mc_simulations,
        args.horizon,
        args.gamma,
        args.uct_c,
        args.seed
    );

    let mut total_reward = 0.0;
    let mut window_reward = 0.0;
    let run_start = Instant::now();
    let mut window_start = Instant::now();
    for cycle in 1..=args.cycles {
        let t0 = Instant::now();
        let action = agent.act(&mut rng);
        let percept = env.step(action, &mut rng);
        agent.perceive(action, percept);
        let reward = env.decode_reward(percept.reward_code);
        total_reward += reward;
        window_reward += reward;
        let _cycle_time = t0.elapsed();

        if let Some(f) = csv.as_mut() {
            let w = agent.model.posterior_weights();
            writeln!(
                f,
                "{cycle},{action},{},{reward},{:.6},{:.3},{}",
                percept.observation,
                total_reward / cycle as f64,
                agent.model.root_log_probability(),
                w.iter()
                    .map(|x| format!("{x:.4}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )
            .map_err(|e| e.to_string())?;
        }

        if cycle % args.report_every == 0 || cycle == args.cycles {
            let w = agent.model.posterior_weights();
            let posterior = ids
                .iter()
                .zip(&w)
                .map(|(id, w)| format!("{id}={w:.3}"))
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "cycle {cycle:>5} | avg reward {:+.4} (window {:+.4}) | ln xi = {:>10.2} | {} | {:.0} ms/cycle",
                total_reward / cycle as f64,
                window_reward / args.report_every.min(cycle) as f64,
                agent.model.root_log_probability(),
                posterior,
                window_start.elapsed().as_millis() as f64 / args.report_every.min(cycle) as f64,
            );
            window_reward = 0.0;
            window_start = Instant::now();
        }
    }
    println!(
        "done: {} cycles in {:.1}s | average reward {:+.4}",
        args.cycles,
        run_start.elapsed().as_secs_f64(),
        total_reward / args.cycles as f64
    );
    Ok(())
}
