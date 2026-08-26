//! Tiger (JAIR §7, after Kaelbling, Littman & Cassandra 1995).
//!
//! A tiger lurks behind one of two doors, gold behind the other. Listening
//! (−1) reveals the tiger's side with accuracy 0.85; opening a door yields
//! −100 (tiger) or +10 (gold) and re-randomizes the tiger. The reward span
//! −100..+10 makes the offset code 0..110 (7 bits) — with the 2 observation
//! bits this is the 9-percept-bit layout of the retired Python line's
//! `tiger` parity module.
//!
//! Actions: 0 = listen, 1 = open left, 2 = open right.
//! Observations: 0 = nothing, 1 = heard tiger left, 2 = heard tiger right.

use super::{Environment, Percept};
use crate::rng::AgentRng;
use rand::Rng;

const LISTEN_ACCURACY: f64 = 0.85;

pub struct Tiger {
    tiger_left: bool,
}

impl Default for Tiger {
    fn default() -> Self {
        Tiger { tiger_left: true }
    }
}

impl Environment for Tiger {
    fn name(&self) -> &'static str {
        "tiger"
    }
    fn num_actions(&self) -> u64 {
        3
    }
    fn action_bits(&self) -> u32 {
        2
    }
    fn observation_bits(&self) -> u32 {
        2
    }
    fn reward_bits(&self) -> u32 {
        7
    }
    fn reward_range(&self) -> (f64, f64) {
        (-100.0, 10.0)
    }

    fn reset(&mut self, rng: &mut AgentRng) {
        self.tiger_left = rng.random_bool(0.5);
    }

    fn step(&mut self, action: u64, rng: &mut AgentRng) -> Percept {
        assert!(action < self.num_actions());
        match action {
            0 => {
                let truthful = rng.random_bool(LISTEN_ACCURACY);
                let heard_left = self.tiger_left == truthful;
                Percept {
                    observation: if heard_left { 1 } else { 2 },
                    reward_code: 99,
                }
            }
            open_left => {
                let opened_tiger = (open_left == 1) == self.tiger_left;
                self.tiger_left = rng.random_bool(0.5);
                Percept {
                    observation: 0,
                    reward_code: if opened_tiger { 0 } else { 110 },
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seeded;

    #[test]
    fn listening_is_085_accurate() {
        let mut env = Tiger { tiger_left: true };
        let mut rng = seeded(9);
        let n = 20_000;
        let mut left_reports = 0u64;
        for _ in 0..n {
            let p = env.step(0, &mut rng);
            assert_eq!(p.reward_code, 99);
            if p.observation == 1 {
                left_reports += 1;
            }
        }
        let f = left_reports as f64 / n as f64;
        assert!((f - LISTEN_ACCURACY).abs() < 0.01, "accuracy {f}");
    }

    #[test]
    fn doors_pay_and_rerandomize() {
        let mut env = Tiger { tiger_left: true };
        let mut rng = seeded(2);
        let p = env.step(1, &mut rng); // open left onto the tiger
        assert_eq!((p.observation, p.reward_code), (0, 0));
        env.tiger_left = false;
        let p = env.step(1, &mut rng); // tiger right, left door has gold
        assert_eq!((p.observation, p.reward_code), (0, 110));
        assert_eq!(env.decode_reward(110), 10.0);
        assert_eq!(env.decode_reward(0), -100.0);
        assert_eq!(env.decode_reward(99), -1.0);
    }
}
