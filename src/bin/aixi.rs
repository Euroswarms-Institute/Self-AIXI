//! The MC-AIXI agent CLI (IMPLEMENTATION_PLAN.md §4.1/§7): run ρUCT over a
//! Bayesian mixture environment model on one of the JAIR §7 domains, and
//! stream per-cycle metrics — reward, running average, ξ root log-probability
//! and, the headline, the mixture posterior trajectory over the model
//! catalog (FAC-CTW depths ⊕ uniform ⊕ the dissected Qwen3.8-2B).
//!
//! Everything is finite and validated (§1.1); every run is reproducible from
//! its --seed.

use mc_aixi::agent::AixiAgent;
use mc_aixi::env::biased_rps::BiasedRockPaperScissors;
use mc_aixi::env::cheese_maze::CheeseMaze;
use mc_aixi::env::coin_flip::CoinFlip;
use mc_aixi::env::kuhn_poker::KuhnPoker;
use mc_aixi::env::tiger::Tiger;
use mc_aixi::env::{DomainSpec, Environment};
use mc_aixi::llm::env_model::LlmModel;
use mc_aixi::llm::model::{QGateLayout, Qwen35Model};
use mc_aixi::models::fac_ctw::FacCtwModel;
use mc_aixi::models::mixture::BayesMixture;
use mc_aixi::models::uniform::UniformModel;
use mc_aixi::models::EnvModel;
use mc_aixi::planning::rho_uct::SearchBudget;
use mc_aixi::rng::seeded;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "\
usage: aixi --env <coin_flip|biased_rps|cheese_maze|tiger|kuhn_poker> [options]

model options:
  --model <ctw-mix|fac-ctw|llm|full-mix>   catalog for the mixture xi (default ctw-mix)
  --ct-depths a,b,c     FAC-CTW depths in the catalog (default 4,8,16)
  --gguf PATH           checkpoint for llm/full-mix (default models/Qwen3.8-2B-Q4_K_M.gguf)

search budget (finite, validated):
  --cycles N            interaction cycles (default 300)
  --mc-simulations N    rhoUCT simulations per decision (default 300)
  --horizon N           planning horizon m (default 3)
  --gamma F             discount (default 0.99)
  --uct-c F             UCB exploration constant (default 0.35)

run options:
  --seed N              RNG seed (default 42)
  --report-every N      print a metrics line every N cycles (default 25)
  --csv PATH            also write per-cycle records to PATH
";

struct Args {
    env: String,
    model: String,
    ct_depths: Vec<usize>,
    gguf: PathBuf,
    cycles: usize,
    mc_simulations: u32,
    horizon: u32,
    gamma: f64,
    uct_c: f64,
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
        cycles: 300,
        mc_simulations: 300,
        horizon: 3,
        gamma: 0.99,
        uct_c: 0.35,
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
            "--cycles" => a.cycles = val()?.parse().map_err(|e| format!("cycles: {e}"))?,
            "--mc-simulations" => {
                a.mc_simulations = val()?.parse().map_err(|e| format!("sims: {e}"))?
            }
            "--horizon" => a.horizon = val()?.parse().map_err(|e| format!("horizon: {e}"))?,
            "--gamma" => a.gamma = val()?.parse().map_err(|e| format!("gamma: {e}"))?,
            "--uct-c" => a.uct_c = val()?.parse().map_err(|e| format!("uct-c: {e}"))?,
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

fn build_env(name: &str) -> Result<Box<dyn Environment>, String> {
    Ok(match name {
        "coin_flip" => Box::new(CoinFlip::default()),
        "biased_rps" => Box::new(BiasedRockPaperScissors::default()),
        "cheese_maze" => Box::new(CheeseMaze::default()),
        "tiger" => Box::new(Tiger::default()),
        "kuhn_poker" => Box::new(KuhnPoker::default()),
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

fn run(args: &Args) -> Result<(), String> {
    let mut env = build_env(&args.env)?;
    let spec = DomainSpec::from_env(env.as_ref());
    let budget = SearchBudget::new(args.mc_simulations, args.horizon, args.uct_c, args.gamma)?;
    let model = BayesMixture::uniform(build_catalog(args, &spec)?);
    let ids = model.component_ids();
    let mut agent = AixiAgent::new(model, spec, budget);
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
