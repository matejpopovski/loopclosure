//! A small seeded generator.
//!
//! Experiments need to give the same stream on every platform and across
//! dependency upgrades, and the core crate carries no dependencies, so this is
//! xoshiro256++ seeded through SplitMix64 rather than a `rand` backend.

#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 4],
    spare_normal: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut next = || {
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self {
            state: [next(), next(), next(), next()],
            spare_normal: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let [s0, s1, s2, s3] = self.state;
        let result = s0.wrapping_add(s3).rotate_left(23).wrapping_add(s0);
        let t = s1 << 17;
        let s2 = s2 ^ s0;
        let s3 = s3 ^ s1;
        let s1 = s1 ^ s2;
        let s0 = s0 ^ s3;
        let s2 = s2 ^ t;
        let s3 = s3.rotate_left(45);
        self.state = [s0, s1, s2, s3];
        result
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Uniform integer in `[0, n)`. The modulo bias is negligible for the small
    /// `n` used here.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Standard normal, by the Marsaglia polar method.
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        loop {
            let u = self.uniform(-1.0, 1.0);
            let v = self.uniform(-1.0, 1.0);
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let scale = (-2.0 * s.ln() / s).sqrt();
                self.spare_normal = Some(v * scale);
                return u * scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_the_same_stream_and_a_new_seed_a_new_one() {
        let mut rng = Rng::new(42);
        let got = [rng.next_u64(), rng.next_u64()];
        let mut again = Rng::new(42);
        assert_eq!(got, [again.next_u64(), again.next_u64()]);
        assert_ne!(Rng::new(43).next_u64(), got[0]);
    }

    #[test]
    fn normal_has_unit_moments() {
        let mut rng = Rng::new(7);
        let n = 200_000;
        let xs: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }
}
