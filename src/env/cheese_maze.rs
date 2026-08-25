//! Cheese Maze (JAIR §7, after McCallum 1996).
//!
//! A mouse wanders a fixed 7×5 maze seeking cheese. The observation is only
//! the 4-bit wall configuration around the current cell, which *aliases*
//! several cells — the classic partially-observable test that short-context
//! models cannot solve and CTW can. Rewards: bump into wall −10, ordinary
//! move −1, cheese +10 (codes 0 / 9 / 20). On eating the cheese the mouse is
//! respawned uniformly at random on a non-cheese free cell.
//!
//! Maze (`#` wall, `C` cheese):
//! ```text
//! #######
//! #     #
//! # # # #
//! #C# # #
//! #######
//! ```

use super::{Environment, Percept};
use crate::rng::AgentRng;
use rand::Rng;

const MAZE: [&[u8; 7]; 5] = [b"#######", b"#     #", b"# # # #", b"# # # #", b"#######"];
const CHEESE: (usize, usize) = (3, 1);

fn is_free(r: usize, c: usize) -> bool {
    MAZE[r][c] != b'#'
}

fn free_cells() -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for r in 0..MAZE.len() {
        for c in 0..MAZE[0].len() {
            if is_free(r, c) {
                v.push((r, c));
            }
        }
    }
    v
}

/// Actions: 0 = up, 1 = down, 2 = left, 3 = right.
const MOVES: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

pub struct CheeseMaze {
    pos: (usize, usize),
}

impl Default for CheeseMaze {
    fn default() -> Self {
        CheeseMaze { pos: (1, 1) }
    }
}

impl CheeseMaze {
    fn respawn(&mut self, rng: &mut AgentRng) {
        let cells: Vec<_> = free_cells().into_iter().filter(|&c| c != CHEESE).collect();
        self.pos = cells[rng.random_range(0..cells.len())];
    }

    /// Observation bits: up<<3 | down<<2 | left<<1 | right (1 = wall).
    fn observe(&self) -> u64 {
        let mut obs = 0u64;
        for (i, (dr, dc)) in MOVES.iter().enumerate() {
            let r = (self.pos.0 as isize + dr) as usize;
            let c = (self.pos.1 as isize + dc) as usize;
            if !is_free(r, c) {
                obs |= 1 << (3 - i);
            }
        }
        obs
    }
}

impl Environment for CheeseMaze {
    fn name(&self) -> &'static str {
        "cheese_maze"
    }
    fn num_actions(&self) -> u64 {
        4
    }
    fn action_bits(&self) -> u32 {
        2
    }
    fn observation_bits(&self) -> u32 {
        4
    }
    fn reward_bits(&self) -> u32 {
        5
    }
    fn reward_range(&self) -> (f64, f64) {
        (-10.0, 10.0)
    }

    fn reset(&mut self, rng: &mut AgentRng) {
        self.respawn(rng);
    }

    fn step(&mut self, action: u64, rng: &mut AgentRng) -> Percept {
        assert!(action < self.num_actions());
        let (dr, dc) = MOVES[action as usize];
        let target = (
            (self.pos.0 as isize + dr) as usize,
            (self.pos.1 as isize + dc) as usize,
        );
        let reward_code = if !is_free(target.0, target.1) {
            0 // −10: walked into a wall, position unchanged
        } else {
            self.pos = target;
            if self.pos == CHEESE {
                self.respawn(rng);
                20 // +10: cheese
            } else {
                9 // −1: ordinary move
            }
        };
        Percept {
            observation: self.observe(),
            reward_code,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seeded;

    #[test]
    fn maze_has_eleven_free_cells_and_cheese() {
        assert_eq!(free_cells().len(), 11);
        assert!(is_free(CHEESE.0, CHEESE.1));
    }

    #[test]
    fn wall_bump_keeps_position_and_costs_ten() {
        let mut env = CheeseMaze { pos: (1, 1) };
        let mut rng = seeded(1);
        let p = env.step(0, &mut rng); // up into the border wall
        assert_eq!(p.reward_code, 0);
        assert_eq!(env.pos, (1, 1));
    }

    #[test]
    fn observations_alias_dead_end_columns() {
        // Cells (2,3) and (2,5): corridors with walls left+right, open
        // up/down — identical observations from different states.
        let a = CheeseMaze { pos: (2, 3) }.observe();
        let b = CheeseMaze { pos: (2, 5) }.observe();
        assert_eq!(a, b);
        assert_eq!(a, 0b0011);
    }

    #[test]
    fn cheese_pays_and_respawns() {
        let mut env = CheeseMaze { pos: (2, 1) };
        let mut rng = seeded(5);
        let p = env.step(1, &mut rng); // down into the cheese at (3,1)
        assert_eq!(p.reward_code, 20);
        assert_ne!(env.pos, CHEESE);
        assert!(is_free(env.pos.0, env.pos.1));
    }
}
