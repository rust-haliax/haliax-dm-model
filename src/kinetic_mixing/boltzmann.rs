use super::KineticMixing;
use cyphus_diffeq::prelude::*;
use cyphus_specfun::bessel::CylBesselK;
use haliax_constants::prelude::*;
use haliax_thermal_functions::prelude::*;
use ndarray::prelude::*;
use std::f64::consts::PI;

impl KineticMixing {
    fn dudt(&self, mut dw: ArrayViewMut1<f64>, w: ArrayView1<f64>, logx: f64) {
        let x: f64 = logx.exp();
        let temp: f64 = self.mx / x;
        let s: f64 = sm_entropy_density(temp);
        let n = neq(temp, self.mx, 2.0, 1);
        let weq: f64 = (n / s).ln();
        let ww: f64 = w[0];

        let pf: f64 = -(std::f64::consts::PI / 45.0).sqrt() * M_PLANK * sm_sqrt_gstar(temp) * temp;
        let sigmav: f64 = self.thermal_cross_section(x);

        // dW_e / dlogx
        dw[0] = pf * sigmav * (ww.exp() - (2.0 * weq - ww).exp());
    }
    fn dfdu(&self, mut dw: ArrayViewMut2<f64>, w: ArrayView1<f64>, logx: f64) {
        let x: f64 = logx.exp();
        let temp: f64 = self.mx / x;
        let s: f64 = sm_entropy_density(temp);

        let n = neq(temp, self.mx, 2.0, 1);
        let weq: f64 = (n / s).ln();
        let ww: f64 = w[0];

        let pf: f64 = -(std::f64::consts::PI / 45.0).sqrt() * M_PLANK * sm_sqrt_gstar(temp) * temp;
        let sigmav: f64 = self.thermal_cross_section(x);

        // dW_e / dlogx
        dw[[0, 0]] = pf * sigmav * (ww.exp() + (2.0 * weq - ww).exp());
    }
    pub fn solve_boltzmann(&self) -> OdeSolution {
        let x0: f64 = 1.0;
        let x1: f64 = 1000.0;
        let temp = self.mx / x0;
        let n = neq(temp, self.mx, 2.0, 1);
        let uinit = array![(n / sm_entropy_density(temp)).ln()];
        let tspan = (x0.ln(), x1.ln());

        let dudt = |mut dw: ArrayViewMut1<f64>,
                    w: ArrayView1<f64>,
                    logx: f64,
                    p: &KineticMixing| { p.dudt(dw.view_mut(), w.view(), logx) };
        let dfdu = |mut dw: ArrayViewMut2<f64>,
                    w: ArrayView1<f64>,
                    logx: f64,
                    p: &KineticMixing| { p.dfdu(dw.view_mut(), w.view(), logx) };

        let mut integrator =
            OdeIntegratorBuilder::default(&dudt, uinit, tspan, Radau5, self.clone())
                .dfdu(&dfdu)
                .reltol(1e-7)
                .abstol(1e-7)
                .build();

        integrator.integrate();
        integrator.sol
    }
    pub fn relic_density(&self) -> f64 {
        let sol = self.solve_boltzmann();
        let yinf = sol.us.last().unwrap()[0].exp();
        yinf * self.mx * S_TODAY / RHO_CRIT
    }
}
