pub mod boltzmann;
pub mod cross_sections;
pub mod thermal_cross_section;
pub mod widths;

use cyphus_integration::prelude::*;

#[derive(PartialEq)]
pub enum KineticMixingFinalStates {
    All,
    XX,
    UU,
    CC,
    TT,
    DD,
    SS,
    BB,
    EE,
    MuMu,
    TauTau,
    NueNue,
    NumuNumu,
    NutauNutau,
    HiggsZ,
}

#[derive(Clone)]
pub struct KineticMixing {
    pub mx: f64,
    pub mv: f64,
    pub gvxx: f64,
    pub eps: f64,
    pub widthv: f64,
    gk: GaussKronrodIntegrator,
}

impl KineticMixing {
    #[allow(dead_code)]
    pub fn new(mx: f64, mv: f64, gvxx: f64, eps: f64) -> KineticMixing {
        let resonance = mv / mx;
        let threshold = 2.0 * mv / mx;

        let singular_points = if threshold > 2.0 {
            if resonance > 2.0 {
                vec![resonance, threshold]
            } else {
                vec![threshold]
            }
        } else {
            vec![]
        };

        let gk = GaussKronrodIntegratorBuilder::default()
            .epsabs(0.0)
            .epsrel(1e-8)
            .singular_points(singular_points)
            .limit(1000)
            .build();

        let mut km = KineticMixing {
            mx,
            mv,
            gvxx,
            eps,
            widthv: 0.0,
            gk,
        };
        km.widthv = km.vm_decay_width();
        km
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn test_cs() {
        let km = KineticMixing::new(1e3, 1e4, 1.0, 1e-3);
        println!("{}", km.annihilation_cross_section(3e3));
    }

    #[test]
    fn test_tcs() {
        let km = KineticMixing::new(1e3, 1e4, 1.0, 1e-3);
        println!("{}", km.thermal_cross_section(1.0));
    }

    #[test]
    fn test_boltz() {
        let km = KineticMixing::new(1e3, 1e4, 1.0, 1e-1);
        let sol = km.solve_boltzmann();

        for t in sol.ts.iter() {
            println!("{},", t);
        }
        for u in sol.us.iter() {
            println!("{},", u[0]);
        }
    }
    #[test]
    fn test_rd() {
        let now = Instant::now();
        let km = KineticMixing::new(1e3, 1e2, 1.0, 1e-3);
        println!("{}, {}", km.relic_density(), now.elapsed().as_millis());

        let now = Instant::now();
        let km = KineticMixing::new(1e3, 1e3, 1.0, 1e-3);
        println!("{}, {}", km.relic_density(), now.elapsed().as_millis());

        let now = Instant::now();
        let km = KineticMixing::new(1e3, 2e3, 1.0, 1e-3);
        println!("{}, {}", km.relic_density(), now.elapsed().as_millis());

        let now = Instant::now();
        let km = KineticMixing::new(1e3, 4e3, 1.0, 1e-3);
        println!("{}, {}", km.relic_density(), now.elapsed().as_millis());

        let now = Instant::now();
        let km = KineticMixing::new(1e3, 1e4, 1.0, 1e-3);
        println!("{}, {}", km.relic_density(), now.elapsed().as_millis());
    }
}
