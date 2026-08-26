//! Next-byte text prediction as a JAIR-style domain.
//!
//! Each cycle the agent's action is its guess for the next byte of a text
//! stream; the environment then reveals that byte as the observation and
//! pays reward 1 for a correct guess, 0 otherwise. The optimal policy is
//! exactly next-byte language modeling, which makes this the domain where
//! the dissected base model earns its place in the mixture instead of being
//! priced out by CTW.
//!
//! The stream is deterministic (a fixed corpus, wrapping); all the
//! difficulty is epistemic. Average reward equals prediction accuracy.

use super::{Environment, Percept};
use crate::rng::AgentRng;

/// A plain-English corpus embedded so the domain runs offline. Around 4 kB
/// of deliberately varied prose: a context tree memorizes a short
/// repetitive text within one lap, and the interesting comparison is
/// generalization on bytes neither model has seen. Runs longer than the
/// corpus wrap around, at which point online learners start their second,
/// much easier lap.
pub const EMBEDDED_CORPUS: &str =
    "The agent reads one byte at a time and tries to guess the next one before \
it arrives. This sounds like a small game, and it is, but it is also the \
whole of sequence prediction. A model that can compress a stream can \
predict it, and a model that can predict a stream can act on it. The \
reward here is blunt: one point for a correct guess, nothing for a miss. \
Over enough bytes the average reward converges to the accuracy of the \
predictor, and the mixture weights converge to the component that \
understands English. \
Consider what each competitor brings. A context tree sees a handful of \
recent bits and learns their statistics from nothing. It is honest and \
fast and it forgets nothing, but its window is narrow. A language model \
arrives already knowing the shape of words, the rhythm of sentences, and \
the strange habits of punctuation. It has never seen this particular \
text, yet it has read enough English to price every next letter. Bayes \
does the bookkeeping between them, and the posterior drifts toward \
whichever explanation is cheapest. \
Ordinary paragraphs make the sternest referee, so here are several about \
nothing in particular. Rain moved across the valley in the late \
afternoon, and the smell of wet dust rose from the road. A woman walked \
her bicycle up the hill because the chain had slipped again, and she was \
composing, silently, the complaint she would never send to the \
manufacturer. Farther along, two children argued about whether a magpie \
could count to five. The older one insisted that birds have no numbers, \
while the younger one had watched the magpie inspect four bright bottle \
caps and come back for a fifth. \
Kitchens are laboratories that admit no failure, only dinner. Butter \
browns through stages a chemist could chart, from foam to nut to regret. \
An onion sliced thin surrenders in minutes, while the same onion, \
quartered, holds its bite through a long simmer. People who cook every \
day develop a private physics of pans, and they trust the sound of \
frying more than any clock. \
The history of navigation is a history of guessing well. Sailors dead \
reckoned across whole oceans by speed, heading, and stubborn optimism, \
correcting themselves with a coastline when one finally appeared. The \
marine chronometer turned longitude from a philosophical embarrassment \
into an engineering problem, and the sextant made the horizon into an \
instrument. Every fix was a posterior: a belief about position, updated \
by a noisy observation, priced by its surprise. \
Cities keep their own grammar. Delivery trucks double park with the \
confidence of punctuation, and pedestrians read the pause in an engine \
the way readers read a comma. A bakery opens early, a bar closes late, \
and between them the street conjugates through every tense of a working \
day. Nobody planned this syntax, but everyone is fluent in it. \
Glaciers are patient librarians. Each winter files another layer of \
snow, each summer stamps it with melt, and the deep ice remembers \
volcanic ash from eruptions no chronicle survives to name. Drill a core \
and you read climate the way rings tell the age of a tree, except the \
book is two miles thick and the earliest pages are eight hundred \
thousand years old. \
A last word about the experiment itself. Nothing in this loop is \
trained. The network weights are frozen, the trees grow their counts, \
and the mixture simply watches who predicts better, paying each \
component in log loss and letting the cheapest explanation inherit the \
weight. That is the entire design: a fair race between compression \
learned online and knowledge distilled offline, scored one byte at a \
time. ";

pub struct TextBytes {
    corpus: Vec<u8>,
    pos: usize,
}

impl TextBytes {
    pub fn from_bytes(corpus: Vec<u8>) -> Result<Self, String> {
        if corpus.is_empty() {
            return Err("text corpus is empty".into());
        }
        Ok(TextBytes { corpus, pos: 0 })
    }

    pub fn embedded() -> Self {
        Self::from_bytes(EMBEDDED_CORPUS.as_bytes().to_vec()).unwrap()
    }

    pub fn corpus_len(&self) -> usize {
        self.corpus.len()
    }
}

impl Default for TextBytes {
    fn default() -> Self {
        Self::embedded()
    }
}

impl Environment for TextBytes {
    fn name(&self) -> &'static str {
        "text_bytes"
    }

    fn num_actions(&self) -> u64 {
        256
    }

    fn action_bits(&self) -> u32 {
        8
    }

    fn observation_bits(&self) -> u32 {
        8
    }

    fn reward_bits(&self) -> u32 {
        1
    }

    fn reward_range(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    fn reset(&mut self, _rng: &mut AgentRng) {
        self.pos = 0;
    }

    fn step(&mut self, action: u64, _rng: &mut AgentRng) -> Percept {
        let obs = self.corpus[self.pos % self.corpus.len()] as u64;
        self.pos += 1;
        Percept {
            observation: obs,
            reward_code: u64::from(action == obs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seeded;

    #[test]
    fn rewards_track_correct_guesses_and_stream_wraps() {
        let mut env = TextBytes::from_bytes(b"ab".to_vec()).unwrap();
        let mut rng = seeded(1);
        env.reset(&mut rng);
        let p = env.step(b'a' as u64, &mut rng);
        assert_eq!((p.observation, p.reward_code), (b'a' as u64, 1));
        let p = env.step(b'a' as u64, &mut rng);
        assert_eq!((p.observation, p.reward_code), (b'b' as u64, 0));
        let p = env.step(b'a' as u64, &mut rng); // wrapped
        assert_eq!((p.observation, p.reward_code), (b'a' as u64, 1));
    }
}
