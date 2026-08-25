//! Biased Rock-Paper-Scissors (JAIR §7, after Farias et al. 2010).
//!
//! The opponent plays uniformly at random, except that after *winning a round
//! with rock* it deterministically plays rock again — an exploitable bias a
//! model-based agent should discover. Moves: 0 = rock, 1 = paper,
//! 2 = scissors. Reward: win +1, draw 0, loss −1 (codes 2/1/0).

use super::{Environment, Percept};
use crate::rng::AgentRng;
use rand::Rng;

/// True iff move `a` beats move `b`.
fn beats(a: u64, b: u64) -> bool {
    (b + 1) % 3 == a
}

#[derive(Default)]
pub struct BiasedRockPaperScissors {
    opponent_won_with_rock: bool,
}

impl Environment for BiasedRockPaperScissors {
    fn name(&self) -> &'static str {
        "biased_rps"
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
        2
    }
    fn reward_range(&self) -> (f64, f64) {
        (-1.0, 1.0)
    }

    fn reset(&mut self, _rng: &mut AgentRng) {
        self.opponent_won_with_rock = false;
    }

    fn step(&mut self, action: u64, rng: &mut AgentRng) -> Percept {
        assert!(action < self.num_actions());
        let opp = if self.opponent_won_with_rock {
            0
        } else {
            rng.random_range(0..3u64)
        };
        let reward_code = if beats(action, opp) {
            2 // agent wins: +1
        } else if action == opp {
            1 // draw: 0
        } else {
            0 // loss: −1
        };
        self.opponent_won_with_rock = opp == 0 && beats(opp, action);
        Percept {
            observation: opp,
            reward_code,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seeded;

    #[test]
    fn beats_table() {
        assert!(beats(1, 0) && beats(2, 1) && beats(0, 2));
        assert!(!beats(0, 1) && !beats(1, 2) && !beats(2, 0));
    }

    #[test]
    fn rock_repeats_after_rock_win() {
        let mut env = BiasedRockPaperScissors::default();
        let mut rng = seeded(3);
        env.reset(&mut rng);
        let mut checked = 0;
        let mut prev_opp_rock_win = false;
        for _ in 0..2000 {
            let p = env.step(2, &mut rng); // scissors loses to rock
            if prev_opp_rock_win {
                assert_eq!(p.observation, 0, "opponent must repeat rock");
                checked += 1;
            }
            prev_opp_rock_win = p.observation == 0 && p.reward_code == 0;
        }
        assert!(checked > 100);
    }

    #[test]
    fn exploiting_bias_beats_random() {
        // Default to scissors (letting rock wins happen), answer each
        // opponent rock win with paper. Should score clearly positive.
        let mut env = BiasedRockPaperScissors::default();
        let mut rng = seeded(11);
        env.reset(&mut rng);
        let mut total = 0i64;
        let mut next_move = 2u64;
        for _ in 0..10_000 {
            let p = env.step(next_move, &mut rng);
            total += p.reward_code as i64 - 1;
            let opp_won_with_rock = p.observation == 0 && p.reward_code == 0;
            next_move = if opp_won_with_rock { 1 } else { 2 };
        }
        assert!(total > 500, "exploit strategy only scored {total}");
    }
}
