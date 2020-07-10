use super::KineticMixing;
use cyphus_specfun::bessel::CylBesselK;

impl KineticMixing {
    /// Compute the thermalized annihilation cross section for
    /// chi + chibar -> anything for a given `x=mass/temperature`.
    pub fn thermal_cross_section(&self, x: f64) -> f64 {
        let m = self.mx;
        let denom = 2.0 * x.cyl_bessel_kn_scaled(2);
        let pf = x / (denom * denom);
        let integrand = |z: f64| -> f64 {
            let z2 = z * z;
            let sig = self.annihilation_cross_section(m * z);
            let kernal = z2 * (z2 - 4.0) * (x * z).cyl_bessel_k1_scaled() * (-x * (z - 2.0)).exp();
            sig * kernal
        };

        pf * self.gk.integrate(integrand, 2.0, f64::INFINITY).val
    }
}
