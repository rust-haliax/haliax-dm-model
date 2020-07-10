use crate::boltzmann::BoltzmannMethod;
use cyphus_diffeq::prelude::*;
use cyphus_integration::{GaussKronrodIntegrator, GaussKronrodIntegratorBuilder};
use cyphus_specfun::bessel::CylBesselK;
use haliax_constants::prelude::*;
use haliax_thermal_functions::prelude::*;
use ndarray::prelude::*;

pub trait AnnihilationCrossSection2To2 {
    fn dm_annihilation_cross_section(&self, cme: f64) -> f64;
}

pub trait Boltzmann {
    fn dm_thermal_cross_section(&self, x: f64) -> f64;
    /// Compute the relic density of the dark matter particle.
    fn relic_density(&self, method: BoltzmannMethod) -> f64;
    /// Solve the Boltzmann equation and return the solution containing the
    /// comoving number density.
    fn solve_boltzmann(&self, method: BoltzmannMethod) -> OdeSolution;
}

pub struct StandardDmModel<T>
where
    T: AnnihilationCrossSection2To2,
{
    /// Dark matter mass
    pub mdm: f64,
    /// Parameters of the model
    pub params: T,
    // Integrator for thermal cross section
    gk: GaussKronrodIntegrator,
}

impl<T> OdeFunction for StandardDmModel<T>
where
    T: AnnihilationCrossSection2To2,
{
    fn dudt(&mut self, mut dw: ArrayViewMut1<f64>, w: ArrayView1<f64>, logx: f64) {
        let x: f64 = logx.exp();
        let temp: f64 = self.mdm / x;
        let s: f64 = sm_entropy_density(temp);

        let weq: f64 = (neq(temp, self.mdm, 2.0, 1) / s).ln();
        let ww: f64 = w[0];

        let pf: f64 = -(std::f64::consts::PI / 45.0).sqrt() * M_PLANK * sm_sqrt_gstar(temp) * temp;
        let sigmav: f64 = self.dm_thermal_cross_section(x);

        // dW_e / dlogx
        dw[0] = pf * sigmav * (ww.exp() - (2.0 * weq - ww).exp());
    }
    fn dfdu(&mut self, mut dw: ArrayViewMut2<f64>, w: ArrayView1<f64>, logx: f64) {
        let x: f64 = logx.exp();
        let temp: f64 = self.mdm / x;
        let s: f64 = sm_entropy_density(temp);

        let weq: f64 = (neq(temp, self.mdm, 2.0, 1) / s).ln();
        let ww: f64 = w[0];

        let pf: f64 = -(std::f64::consts::PI / 45.0).sqrt() * M_PLANK * sm_sqrt_gstar(temp) * temp;
        let sigmav: f64 = self.dm_thermal_cross_section(x);

        // dW_e / dlogx
        dw[[0, 0]] = pf * sigmav * (ww.exp() + (2.0 * weq - ww).exp());
    }
}

impl<T> Boltzmann for StandardDmModel<T>
where
    T: AnnihilationCrossSection2To2,
{
    fn dm_thermal_cross_section(&self, x: f64) -> f64 {
        let m = self.mdm;
        let denom = dbg!(2.0 * x.cyl_bessel_kn_scaled(2));
        let pf = x / (denom * denom);
        let integrand = |z: f64| -> f64 {
            let z2 = z * z;
            let sig = self.params.dm_annihilation_cross_section(m * z);
            let kernal = z2 * (z2 - 4.0) * (x * z).cyl_bessel_k1_scaled() * (-x * (z - 2.0)).exp();
            sig * kernal
        };

        pf * self.gk.integrate(integrand, 2.0, f64::INFINITY).val
    }
    fn relic_density(&self, method: BoltzmannMethod) -> f64 {
        unimplemented!()
    }
    fn solve_boltzmann(&self, method: BoltzmannMethod) -> OdeSolution {}
}
