use super::{KineticMixing, KineticMixingFinalStates};
use haliax_constants::prelude::*;

impl KineticMixing {
    /// Compute the annihilation cross-section for dark matter to up-type quarks.
    pub fn sigma_xx_to_ququ(&self, cme: f64, mf: f64) -> f64 {
        if cme > 2.0 * self.mx && cme > 2.0 * mf {
            let temp1: f64 = self.mx.powi(2);
            let temp2: f64 = cme.powi(2);
            let temp3: f64 = mf.powi(2);
            let temp4: f64 = self.mv.powi(2);
            (ALPHA_EM
                * self.eps.powi(2)
                * self.gvxx.powi(2)
                * (2.0 * temp1 + temp2)
                * (temp2 - 4.0 * temp3).sqrt()
                * (17.0 * temp2 + 7.0 * temp3))
                / (72.0
                    * COS_THETA_WEAK.powi(2)
                    * cme.powi(2)
                    * (-4.0 * temp1 + temp2).sqrt()
                    * ((-temp2 + temp4).powi(2) + temp4 * self.widthv.powi(2)))
        } else {
            0.0
        }
    }
    /// Compute the annihilation cross-section for dark matter to down-type quarks.
    pub fn sigma_xx_to_qdqd(&self, cme: f64, mf: f64) -> f64 {
        if cme > 2.0 * self.mx && cme > 2.0 * mf {
            let temp1: f64 = cme.powi(2);
            let temp2: f64 = self.mx.powi(2);
            let temp3: f64 = mf.powi(2);
            let temp4: f64 = self.mv.powi(2);
            (ALPHA_EM
                * self.eps.powi(2)
                * self.gvxx.powi(2)
                * (temp1 + 2.0 * temp2)
                * (5.0 * temp1 - 17.0 * temp3)
                * (temp1 - 4.0 * temp3).sqrt())
                / (72.0
                    * COS_THETA_WEAK.powi(2)
                    * temp1
                    * (temp1 - 4.0 * temp2).sqrt()
                    * ((-temp1 + temp4).powi(2) + temp4 * self.widthv.powi(2)))
        } else {
            0.0
        }
    }
    /// Compute the annihilation cross-section for dark matter to leptons.
    pub fn sigma_xx_to_ll(&self, cme: f64, mf: f64) -> f64 {
        if cme > 2.0 * self.mx && cme > 2.0 * mf {
            let temp1: f64 = self.mx.powi(2);
            let temp2: f64 = cme.powi(2);
            let temp3: f64 = mf.powi(2);
            let temp4: f64 = self.mv.powi(2);
            (ALPHA_EM
                * self.eps.powi(2)
                * self.gvxx.powi(2)
                * (2.0 * temp1 + temp2)
                * (temp2 - 4.0 * temp3).sqrt()
                * (5.0 * temp2 + 7.0 * temp3))
                / (24.0
                    * COS_THETA_WEAK.powi(2)
                    * cme.powi(2)
                    * (-4.0 * temp1 + temp2).sqrt()
                    * ((-temp2 + temp4).powi(2) + temp4 * self.widthv.powi(2)))
        } else {
            0.0
        }
    }
    /// Compute the annihilation cross-section for dark matter to a vector mediators.
    pub fn sigma_xx_to_vv(&self, cme: f64) -> f64 {
        if cme > 2.0 * self.mx && cme > 2.0 * self.mv {
            let temp1: f64 = self.mx.powi(2);
            let temp2: f64 = -4.0 * temp1;
            let temp3: f64 = cme.powi(2);
            let temp4: f64 = temp2 + temp3;
            let temp5: f64 = self.mv.powi(4);
            let temp6: f64 = self.mv.powi(2);
            let temp7: f64 = -4.0 * temp6;
            let temp8: f64 = temp3 + temp7;
            let temp9: f64 = 2.0 * temp6;
            let temp10: f64 = -temp3;
            let temp11: f64 = temp10 + temp9;
            let temp12: f64 = -2.0 * temp6;
            let temp13: f64 = temp4.sqrt();
            let temp14: f64 = temp8.sqrt();
            let temp15: f64 = temp13 * temp14;
            let temp16: f64 = 2.0 * temp1;
            let temp17: f64 = temp12 + temp15 + temp3;
            (self.gvxx.powi(4)
                * ((-48.0
                    * temp13
                    * temp14
                    * (4.0 * self.mx.powi(4) + temp1 * temp3 + 2.0 * temp5))
                    / (temp5 + temp1 * temp8)
                    + (48.0
                        * (temp1 * (temp12 + temp2 + temp3) * std::f64::consts::LN_2
                            + temp3 * temp6 * 2.0 * std::f64::consts::LN_2
                            + (2.0 * temp1 * (temp16 + temp6) - temp3 * (temp1 + temp9))
                                * (-(temp17.powi(2)
                                    / (-(cme.powi(4)) + temp13 * temp14 * temp3 - 2.0 * temp5
                                        + (-2.0 * temp13 * temp14 + 4.0 * temp3) * temp6
                                        + 2.0 * temp1 * temp8)))
                                    .ln()
                            + temp11
                                * (temp12 + temp16 + temp3)
                                * (-(temp17 / (temp10 + temp15 + temp9))).ln()))
                        / temp11))
                / (384.0 * std::f64::consts::PI * cme.powi(2) * temp4)
        } else {
            0.0
        }
    }
    /// Compute the annihilation cross-section for dark matter to neutrinos.
    pub fn sigma_xx_to_nunu(&self, cme: f64) -> f64 {
        if cme > 2.0 * self.mx {
            let temp1: f64 = cme.powi(2);
            let temp2: f64 = self.mx.powi(2);
            let temp3: f64 = self.mv.powi(2);
            (ALPHA_EM * self.eps.powi(2) * self.gvxx.powi(2) * temp1.sqrt() * (temp1 + 2.0 * temp2))
                / (24.0
                    * COS_THETA_WEAK.powi(2)
                    * (temp1 - 4.0 * temp2).sqrt()
                    * ((-temp1 + temp3).powi(2) + temp3 * self.widthv.powi(2)))
        } else {
            0.0
        }
    }
    /// Compute the annihilation cross-section for dark matter to a higgs and z-boson.
    pub fn sigma_xx_to_hz(&self, cme: f64) -> f64 {
        if cme > 2.0 * self.mx && cme > HIGGS_MASS + Z_BOSON_MASS {
            let temp1: f64 = -cme;
            let temp2: f64 = -W_BOSON_MASS;
            let temp3: f64 = self.mx.powi(2);
            let temp4: f64 = cme.powi(2);
            let temp5: f64 = W_BOSON_MASS.powi(2);
            let temp6: f64 = self.mv.powi(2);
            (ALPHA_EM.powi(2)
                * self.eps.powi(2)
                * self.gvxx.powi(2)
                * std::f64::consts::PI
                * (((HIGGS_MASS + W_BOSON_MASS + cme)
                    * (HIGGS_MASS + W_BOSON_MASS + temp1)
                    * (HIGGS_MASS + cme + temp2)
                    * (HIGGS_MASS + temp1 + temp2))
                    / (-4.0 * temp3 + temp4))
                    .sqrt()
                * (2.0 * temp3 + temp4)
                * (HIGGS_MASS.powi(4) + W_BOSON_MASS.powi(4) + cme.powi(4) + 10.0 * temp4 * temp5
                    - 2.0 * HIGGS_MASS.powi(2) * (temp4 + temp5))
                * HIGGS_VEV.powi(2))
                / (48.0
                    * COS_THETA_WEAK.powi(4)
                    * W_BOSON_MASS.powi(2)
                    * cme.powi(5)
                    * SIN_THETA_WEAK_SQRD
                    * ((-temp4 + temp6).powi(2) + temp6 * self.widthv.powi(2)))
        } else {
            0.0
        }
    }
    pub fn annihilation_cross_section(&self, cme: f64) -> f64 {
        self.sigma_xx_to_ququ(cme, UP_QUARK_MASS)
            + self.sigma_xx_to_ququ(cme, CHARM_QUARK_MASS)
            + self.sigma_xx_to_ququ(cme, TOP_QUARK_MASS)
            + self.sigma_xx_to_qdqd(cme, DOWN_QUARK_MASS)
            + self.sigma_xx_to_qdqd(cme, STRANGE_QUARK_MASS)
            + self.sigma_xx_to_qdqd(cme, BOTTOM_QUARK_MASS)
            + self.sigma_xx_to_ll(cme, ELECTRON_MASS)
            + self.sigma_xx_to_ll(cme, MUON_MASS)
            + self.sigma_xx_to_ll(cme, TAU_MASS)
            + 3.0 * self.sigma_xx_to_hz(cme)
            + self.sigma_xx_to_hz(cme)
            + self.sigma_xx_to_vv(cme)
    }
}
