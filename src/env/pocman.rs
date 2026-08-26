//! PocMan — partially observable PacMan (JAIR §7).
//!
//! The agent roams a 17×19 Pac-Man-style maze eating food while four ghosts
//! roam it too. It never sees the board: the observation is 16 bits of local
//! senses, which makes this the largest and most aliased of the bundled
//! domains and the JAIR paper's headline scaling result.
//!
//! Observation layout (MSB-first on the wire):
//! - 4 bits: wall adjacent in each direction (up, down, left, right);
//! - 4 bits: ghost visible by direct line of sight per direction;
//! - 4 bits: food visible by direct line of sight per direction;
//! - 3 bits: any food within Manhattan distance 2 / 3 / 4 ("smell");
//! - 1 bit: currently under a power pill.
//!
//! Rewards per cycle sum the events (JAIR's magnitudes): each move −1,
//! walking into a wall −10, food pellet +10, caught by a ghost −50 (agent
//! and ghosts respawn, food stays), eating a frightened ghost +30, clearing
//! all food +100 (fresh level). Encoded with offset 60 in 8 bits.
//!
//! Conventions this implementation pins (the paper leaves them to the
//! implementer, and published implementations each ship their own): the
//! maze drawing below; food placed per free cell with probability ½ at
//! level start; four power pills at fixed cells, 15 steps of power each;
//! ghosts chase when within Manhattan distance 5 (probability ¾ per step),
//! flee while the agent is powered, avoid reversing otherwise; the middle
//! row wraps left↔right (line of sight does not follow the wrap).

use super::{Environment, Percept};
use crate::rng::AgentRng;
use rand::Rng;

const W: usize = 17;
const H: usize = 19;

/// `#` wall, `.` free, `o` power pill (free).
const MAZE: [&[u8; W]; H] = [
    b"#################",
    b"#.......#.......#",
    b"#.##.##.#.##.##.#",
    b"#o##.##.#.##.##o#",
    b"#...............#",
    b"#.##.#.###.#.##.#",
    b"#....#..#..#....#",
    b"####.##.#.##.####",
    b"#....#.....#....#",
    b".....#.....#.....",
    b"#....#.....#....#",
    b"####.#.....#.####",
    b"#.......#.......#",
    b"#.##.##.#.##.##.#",
    b"#o.#.........#.o#",
    b"##.#.#.###.#.#.##",
    b"#..#....#....#..#",
    b"#.......#.......#",
    b"#################",
];

const AGENT_HOME: (usize, usize) = (14, 8);
const GHOST_HOMES: [(usize, usize); 4] = [(8, 8), (9, 7), (9, 9), (10, 8)];
const POWER_PILLS: [(usize, usize); 4] = [(3, 1), (3, 15), (14, 1), (14, 15)];
const POWER_STEPS: u32 = 15;
const CHASE_RANGE: i32 = 5;
const TUNNEL_ROW: usize = 9;

/// Actions: 0 = up, 1 = down, 2 = left, 3 = right (cheese-maze convention).
const MOVES: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

fn is_free(r: usize, c: usize) -> bool {
    MAZE[r][c] != b'#'
}

/// One step in direction `d`, honoring the tunnel row wrap. None = wall.
fn step_from(pos: (usize, usize), d: usize) -> Option<(usize, usize)> {
    let (dr, dc) = MOVES[d];
    let r = pos.0 as isize + dr;
    let mut c = pos.1 as isize + dc;
    if pos.0 == TUNNEL_ROW && r == TUNNEL_ROW as isize {
        c = c.rem_euclid(W as isize);
    }
    if r < 0 || r >= H as isize || c < 0 || c >= W as isize {
        return None;
    }
    let (r, c) = (r as usize, c as usize);
    is_free(r, c).then_some((r, c))
}

fn manhattan(a: (usize, usize), b: (usize, usize)) -> i32 {
    (a.0 as i32 - b.0 as i32).abs() + (a.1 as i32 - b.1 as i32).abs()
}

fn free_cells() -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for r in 0..H {
        for c in 0..W {
            if is_free(r, c) {
                v.push((r, c));
            }
        }
    }
    v
}

struct Ghost {
    pos: (usize, usize),
    /// Last direction moved (avoid reversing while roaming); 4 = none.
    dir: usize,
}

pub struct PocMan {
    pos: (usize, usize),
    ghosts: Vec<Ghost>,
    food: Vec<bool>, // indexed r * W + c
    power_left: u32,
}

impl Default for PocMan {
    fn default() -> Self {
        PocMan {
            pos: AGENT_HOME,
            ghosts: GHOST_HOMES
                .iter()
                .map(|&pos| Ghost { pos, dir: 4 })
                .collect(),
            food: vec![false; W * H],
            power_left: 0,
        }
    }
}

impl PocMan {
    fn new_level(&mut self, rng: &mut AgentRng) {
        self.pos = AGENT_HOME;
        for (g, &home) in self.ghosts.iter_mut().zip(&GHOST_HOMES) {
            g.pos = home;
            g.dir = 4;
        }
        self.power_left = 0;
        self.food = vec![false; W * H];
        for (r, c) in free_cells() {
            let special = (r, c) == AGENT_HOME
                || POWER_PILLS.contains(&(r, c))
                || GHOST_HOMES.contains(&(r, c));
            if !special && rng.random_bool(0.5) {
                self.food[r * W + c] = true;
            }
        }
        // Power pills are always present at level start.
        for &(r, c) in &POWER_PILLS {
            self.food[r * W + c] = true;
        }
    }

    fn respawn_after_capture(&mut self) {
        self.pos = AGENT_HOME;
        for (g, &home) in self.ghosts.iter_mut().zip(&GHOST_HOMES) {
            g.pos = home;
            g.dir = 4;
        }
        self.power_left = 0;
    }

    fn food_at(&self, pos: (usize, usize)) -> bool {
        self.food[pos.0 * W + pos.1]
    }

    fn food_left(&self) -> usize {
        self.food.iter().filter(|&&f| f).count()
    }

    /// Scan a direct line of sight (no tunnel wrap); true if `pred` holds
    /// anywhere before the first wall.
    fn sees(&self, d: usize, pred: impl Fn(&Self, (usize, usize)) -> bool) -> bool {
        let (dr, dc) = MOVES[d];
        let (mut r, mut c) = (self.pos.0 as isize, self.pos.1 as isize);
        loop {
            r += dr;
            c += dc;
            if r < 0 || r >= H as isize || c < 0 || c >= W as isize {
                return false;
            }
            let cell = (r as usize, c as usize);
            if !is_free(cell.0, cell.1) {
                return false;
            }
            if pred(self, cell) {
                return true;
            }
        }
    }

    fn smell(&self, dist: i32) -> bool {
        for r in 0..H {
            for c in 0..W {
                if self.food[r * W + c] && manhattan(self.pos, (r, c)) <= dist {
                    return true;
                }
            }
        }
        false
    }

    fn observe(&self) -> u64 {
        let mut obs = 0u64;
        for d in 0..4 {
            if step_from(self.pos, d).is_none() {
                obs |= 1 << (15 - d);
            }
            if self.sees(d, |s, cell| s.ghosts.iter().any(|g| g.pos == cell)) {
                obs |= 1 << (11 - d);
            }
            if self.sees(d, |s, cell| s.food_at(cell)) {
                obs |= 1 << (7 - d);
            }
        }
        for (i, dist) in [2, 3, 4].into_iter().enumerate() {
            if self.smell(dist) {
                obs |= 1 << (3 - i);
            }
        }
        if self.power_left > 0 {
            obs |= 1;
        }
        obs
    }

    fn move_ghost(&mut self, gi: usize, rng: &mut AgentRng) {
        let g = &self.ghosts[gi];
        let legal: Vec<usize> = (0..4).filter(|&d| step_from(g.pos, d).is_some()).collect();
        if legal.is_empty() {
            return;
        }
        let dist = manhattan(g.pos, self.pos);
        let toward: Vec<usize> = legal
            .iter()
            .copied()
            .filter(|&d| manhattan(step_from(g.pos, d).unwrap(), self.pos) < dist)
            .collect();
        let away: Vec<usize> = legal
            .iter()
            .copied()
            .filter(|&d| manhattan(step_from(g.pos, d).unwrap(), self.pos) > dist)
            .collect();
        let dir = if self.power_left > 0 && !away.is_empty() {
            away[rng.random_range(0..away.len())]
        } else if self.power_left == 0
            && dist <= CHASE_RANGE
            && !toward.is_empty()
            && rng.random_bool(0.75)
        {
            toward[rng.random_range(0..toward.len())]
        } else {
            // Roam: avoid reversing when another option exists.
            let reverse = [1usize, 0, 3, 2];
            let g = &self.ghosts[gi];
            let non_rev: Vec<usize> = legal
                .iter()
                .copied()
                .filter(|&d| g.dir > 3 || d != reverse[g.dir])
                .collect();
            let pool = if non_rev.is_empty() { &legal } else { &non_rev };
            pool[rng.random_range(0..pool.len())]
        };
        let g = &mut self.ghosts[gi];
        g.pos = step_from(g.pos, dir).unwrap();
        g.dir = dir;
    }

    /// Resolve agent/ghost meetings; returns the reward delta.
    fn resolve_contacts(&mut self) -> f64 {
        let mut delta = 0.0;
        let mut captured = false;
        let powered = self.power_left > 0;
        let pos = self.pos;
        for (g, &home) in self.ghosts.iter_mut().zip(GHOST_HOMES.iter()) {
            if g.pos == pos {
                if powered {
                    delta += 30.0;
                    g.pos = home;
                    g.dir = 4;
                } else {
                    delta -= 50.0;
                    captured = true;
                    break; // capture ends the cycle's interactions
                }
            }
        }
        if captured {
            self.respawn_after_capture();
        }
        delta
    }
}

impl Environment for PocMan {
    fn name(&self) -> &'static str {
        "pocman"
    }
    fn num_actions(&self) -> u64 {
        4
    }
    fn action_bits(&self) -> u32 {
        2
    }
    fn observation_bits(&self) -> u32 {
        16
    }
    fn reward_bits(&self) -> u32 {
        8
    }
    fn reward_range(&self) -> (f64, f64) {
        (-60.0, 140.0)
    }

    fn reset(&mut self, rng: &mut AgentRng) {
        self.new_level(rng);
    }

    fn step(&mut self, action: u64, rng: &mut AgentRng) -> Percept {
        assert!(action < self.num_actions());
        let mut reward = -1.0;
        match step_from(self.pos, action as usize) {
            None => reward -= 10.0,
            Some(next) => self.pos = next,
        }
        self.power_left = self.power_left.saturating_sub(1);

        // Meeting resolution happens both after the agent's move and after
        // the ghosts', so swap-throughs cannot pass through each other.
        reward += self.resolve_contacts();
        for gi in 0..self.ghosts.len() {
            self.move_ghost(gi, rng);
        }
        reward += self.resolve_contacts();

        if self.food_at(self.pos) {
            self.food[self.pos.0 * W + self.pos.1] = false;
            reward += 10.0;
            if POWER_PILLS.contains(&self.pos) {
                self.power_left = POWER_STEPS;
            }
            if self.food_left() == 0 {
                reward += 100.0;
                self.new_level(rng);
            }
        }

        let (lo, hi) = self.reward_range();
        let code = (reward.clamp(lo, hi) - lo) as u64;
        Percept {
            observation: self.observe(),
            reward_code: code,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seeded;

    #[test]
    fn maze_is_well_formed_and_fully_connected() {
        for row in MAZE {
            assert_eq!(row.len(), W);
        }
        for &(r, c) in POWER_PILLS
            .iter()
            .chain(GHOST_HOMES.iter())
            .chain([&AGENT_HOME])
        {
            assert!(is_free(r, c), "special cell ({r},{c}) is a wall");
        }
        // BFS from the agent home must reach every free cell.
        let mut seen = vec![false; W * H];
        let mut queue = vec![AGENT_HOME];
        seen[AGENT_HOME.0 * W + AGENT_HOME.1] = true;
        while let Some(p) = queue.pop() {
            for d in 0..4 {
                if let Some(n) = step_from(p, d) {
                    if !seen[n.0 * W + n.1] {
                        seen[n.0 * W + n.1] = true;
                        queue.push(n);
                    }
                }
            }
        }
        for (r, c) in free_cells() {
            assert!(seen[r * W + c], "free cell ({r},{c}) unreachable");
        }
    }

    #[test]
    fn tunnel_wraps_and_counts_as_open() {
        let left_end = (TUNNEL_ROW, 0);
        assert!(is_free(left_end.0, left_end.1));
        assert_eq!(step_from(left_end, 2), Some((TUNNEL_ROW, W - 1)));
        assert_eq!(step_from((TUNNEL_ROW, W - 1), 3), Some(left_end));
    }

    #[test]
    fn observation_bits_have_the_documented_meaning() {
        // Top-left corner pocket: walls up and left.
        let mut env = PocMan {
            pos: (1, 1),
            ..Default::default()
        };
        let obs = env.observe();
        assert_eq!(obs >> 12, 0b1010, "wall bits: up and left blocked");
        // A ghost straight down the corridor is seen (down = bit 10).
        env.ghosts[0].pos = (4, 1);
        let obs = env.observe();
        assert_ne!(obs & (1 << 10), 0, "ghost visible downward");
        // Smell bits are monotone: within-2 implies within-3 implies within-4.
        env.food[W + 3] = true; // (1,3): Manhattan distance 2
        let obs = env.observe();
        assert_eq!(obs >> 1 & 0b111, 0b111);
    }

    #[test]
    fn capture_costs_fifty_and_respawns_everyone() {
        let mut env = PocMan::default();
        let mut rng = seeded(9);
        env.new_level(&mut rng);
        env.pos = (4, 2);
        env.ghosts[0].pos = (4, 2);
        let before_food = env.food_left();
        let p = env.step(3, &mut rng); // any move; contact resolves either way
        let reward = env.decode_reward(p.reward_code);
        assert!(reward <= -50.0, "capture reward {reward}");
        assert_eq!(env.pos, AGENT_HOME);
        assert_eq!(env.food_left(), before_food, "capture must keep the food");
    }

    #[test]
    fn power_pill_arms_the_power_bit_and_ghosts_are_edible() {
        let mut env = PocMan::default();
        let mut rng = seeded(11);
        env.new_level(&mut rng);
        // Walk onto the (3,1) power pill from (4,1).
        env.pos = (4, 1);
        for g in env.ghosts.iter_mut() {
            g.pos = (9, 8); // park ghosts far away
        }
        let p = env.step(0, &mut rng);
        assert_eq!(p.observation & 1, 1, "power bit set");
        assert!(env.power_left > 0);
        // Force a meeting while powered: +30 and the ghost respawns home.
        env.ghosts[0].pos = env.pos;
        let reward = env.resolve_contacts();
        assert_eq!(reward, 30.0);
        assert_eq!(env.ghosts[0].pos, GHOST_HOMES[0]);
    }

    #[test]
    fn rewards_stay_inside_the_declared_range() {
        let mut env = PocMan::default();
        let mut rng = seeded(13);
        env.reset(&mut rng);
        let (lo, hi) = env.reward_range();
        for i in 0..2000 {
            let p = env.step(i % 4, &mut rng);
            let r = env.decode_reward(p.reward_code);
            assert!(r >= lo && r <= hi, "reward {r} outside [{lo}, {hi}]");
            assert!(p.observation < 1 << 16);
        }
    }
}
