//! Kuhn Poker (JAIR §7, after Kuhn 1950; Hoehn et al. 2005).
//!
//! Three-card poker (J=0 < Q=1 < K=2), one card each, agent moves second.
//! The opponent plays the classic first-player Nash strategy with bluff
//! parameter α = 1/3: open-bet J with probability 1/3, Q never, K always;
//! facing a bet after checking, fold J, call Q with probability 2/3, call K.
//! Exploitable head-room beyond the game value exists because the opponent
//! never adapts.
//!
//! Cycle protocol (the MC-AIXI cycle is action-then-percept, so hand t is
//! resolved by the action of cycle t, and the percept reveals hand t+1's
//! deal): percept observation = (agent card << 1) | opponent-opening-bet
//! (3 bits, per the retired Python line's `kuhn_poker` 6-percept-bit parity layout),
//! reward code = hand-t reward + 2 ∈ {0,1,3,4} (3 bits).
//!
//! Agent actions: facing a bet — 1 = call, 0 = fold; after a check —
//! 1 = bet, 0 = check (showdown for the ante).

use super::{Environment, Percept};
use crate::rng::AgentRng;
use rand::Rng;

const ALPHA: f64 = 1.0 / 3.0;

pub struct KuhnPoker {
    agent_card: u64,
    opp_card: u64,
    opp_opened: bool,
}

impl Default for KuhnPoker {
    fn default() -> Self {
        KuhnPoker {
            agent_card: 2,
            opp_card: 0,
            opp_opened: false,
        }
    }
}

impl KuhnPoker {
    fn deal(&mut self, rng: &mut AgentRng) {
        self.opp_card = rng.random_range(0..3u64);
        self.agent_card = (self.opp_card + 1 + rng.random_range(0..2u64)) % 3;
        self.opp_opened = match self.opp_card {
            0 => rng.random_bool(ALPHA),
            1 => false,
            _ => true,
        };
    }

    fn showdown(&self, stake: i64) -> i64 {
        if self.agent_card > self.opp_card {
            stake
        } else {
            -stake
        }
    }

    /// Resolve the current hand against `action`, returning the agent's
    /// signed reward in chips.
    fn resolve(&self, action: u64, rng: &mut AgentRng) -> i64 {
        if self.opp_opened {
            if action == 1 {
                self.showdown(2)
            } else {
                -1
            }
        } else if action == 1 {
            // Agent bets after the check; opponent responds per Nash.
            let opp_calls = match self.opp_card {
                0 => false,
                1 => rng.random_bool(ALPHA + 1.0 / 3.0),
                _ => true,
            };
            if opp_calls {
                self.showdown(2)
            } else {
                1
            }
        } else {
            self.showdown(1)
        }
    }

    #[cfg(test)]
    fn with_hand(agent_card: u64, opp_card: u64, opp_opened: bool) -> Self {
        KuhnPoker {
            agent_card,
            opp_card,
            opp_opened,
        }
    }
}

impl Environment for KuhnPoker {
    fn name(&self) -> &'static str {
        "kuhn_poker"
    }
    fn num_actions(&self) -> u64 {
        2
    }
    fn action_bits(&self) -> u32 {
        1
    }
    fn observation_bits(&self) -> u32 {
        3
    }
    fn reward_bits(&self) -> u32 {
        3
    }
    fn reward_range(&self) -> (f64, f64) {
        (-2.0, 2.0)
    }

    fn reset(&mut self, rng: &mut AgentRng) {
        self.deal(rng);
    }

    fn step(&mut self, action: u64, rng: &mut AgentRng) -> Percept {
        assert!(action < self.num_actions());
        let reward = self.resolve(action, rng);
        self.deal(rng);
        Percept {
            observation: (self.agent_card << 1) | u64::from(self.opp_opened),
            reward_code: (reward + 2) as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seeded;

    #[test]
    fn hand_resolutions() {
        let mut rng = seeded(1);
        // Opponent bluff-bet with J, agent calls with K: +2.
        assert_eq!(KuhnPoker::with_hand(2, 0, true).resolve(1, &mut rng), 2);
        // Opponent value-bet with K, agent folds J: −1.
        assert_eq!(KuhnPoker::with_hand(0, 2, true).resolve(0, &mut rng), -1);
        // Check-check showdown for the ante.
        assert_eq!(KuhnPoker::with_hand(1, 0, false).resolve(0, &mut rng), 1);
        assert_eq!(KuhnPoker::with_hand(0, 1, false).resolve(0, &mut rng), -1);
        // Agent bets K after a check: opponent's J always folds: +1.
        assert_eq!(KuhnPoker::with_hand(2, 0, false).resolve(1, &mut rng), 1);
    }

    #[test]
    fn opening_frequencies_by_agent_card() {
        // Marginal opponent-opening rates given the agent's own card:
        // K ⇒ opp ∈ {J,Q} ⇒ 1/6; Q ⇒ opp ∈ {J,K} ⇒ 2/3; J ⇒ opp ∈ {Q,K} ⇒ 1/2.
        let mut env = KuhnPoker::default();
        let mut rng = seeded(4);
        env.reset(&mut rng);
        let mut opened = [0u64; 3];
        let mut seen = [0u64; 3];
        for _ in 0..60_000 {
            let p = env.step(0, &mut rng);
            let card = (p.observation >> 1) as usize;
            seen[card] += 1;
            opened[card] += p.observation & 1;
        }
        let rate = |c: usize| opened[c] as f64 / seen[c] as f64;
        assert!((rate(2) - 1.0 / 6.0).abs() < 0.02, "K rate {}", rate(2));
        assert!((rate(1) - 2.0 / 3.0).abs() < 0.02, "Q rate {}", rate(1));
        assert!((rate(0) - 0.5).abs() < 0.02, "J rate {}", rate(0));
    }

    #[test]
    fn reward_codes_stay_in_range() {
        let mut env = KuhnPoker::default();
        let mut rng = seeded(6);
        env.reset(&mut rng);
        for i in 0..5_000 {
            let p = env.step(i & 1, &mut rng);
            assert!(p.reward_code <= 4);
            assert_ne!(p.reward_code, 2, "zero-chip hands are impossible in Kuhn");
            assert!(p.observation < 8 && (p.observation >> 1) < 3);
        }
    }
}
