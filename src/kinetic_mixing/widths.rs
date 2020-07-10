use super::{KineticMixing, KineticMixingFinalStates};
use haliax_constants::prelude::*;

impl KineticMixing {
    /// Compute the partial width for V -> down-type quarks.
    pub fn width_v_to_ququ(&self, mf: f64) -> f64 {
        if self.mv > 2.0 * mf {
            return (ALPHA_EM
                * self.eps.powi(2)
                * (-4.0 * mf.powi(2) + self.mv.powi(2)).sqrt()
                * (7.0 * mf.powi(2) + 17.0 * self.mv.powi(2)))
                / (24.0 * COS_THETA_WEAK.powi(2) * self.mv.powi(2));
        } else {
            0.0
        }
    }
    /// Compute the partial width for V -> leptons.
    pub fn width_v_to_qdqd(&self, mf: f64) -> f64 {
        if self.mv > 2.0 * mf {
            return (ALPHA_EM
                * self.eps.powi(2)
                * (-4.0 * mf.powi(2) + self.mv.powi(2)).sqrt()
                * (7.0 * mf.powi(2) + 5.0 * self.mv.powi(2)))
                / (8.0 * COS_THETA_WEAK.powi(2) * self.mv.powi(2));
        } else {
            0.0
        }
    }
    /// Compute the partial width for V -> leptons.
    pub fn width_v_to_ll(&self, mf: f64) -> f64 {
        if self.mv > 2.0 * mf {
            return (ALPHA_EM
                * self.eps.powi(2)
                * (-4.0 * mf.powi(2) + self.mv.powi(2)).sqrt()
                * (7.0 * mf.powi(2) + 5.0 * self.mv.powi(2)))
                / (24.0 * COS_THETA_WEAK.powi(2) * self.mv.powi(2));
        } else {
            0.0
        }
    }
    /// Compute the partial width for V -> neutrinos.
    pub fn width_v_to_nunu(&self) -> f64 {
        ALPHA_EM * self.eps.powi(2) * self.mv / (24.0 * COS_THETA_WEAK.powi(2))
    }
    /// Compute the partial width for V -> higgs + z-boson.
    pub fn width_v_to_hz(&self) -> f64 {
        if self.mv > HIGGS_MASS + Z_BOSON_MASS {
            (ALPHA_EM.powi(2)
                * self.eps.powi(2)
                * (-(HIGGS_MASS.powi(2))
                    + (HIGGS_MASS.powi(2) + self.mv.powi(2) - W_BOSON_MASS.powi(2)).powi(2)
                        / (4.0 * self.mv.powi(2)))
                .sqrt()
                * (2.0 * self.mv.powi(2) * W_BOSON_MASS.powi(2)
                    + (-(HIGGS_MASS.powi(2)) + self.mv.powi(2) + W_BOSON_MASS.powi(2)).powi(2)
                        / 4.)
                * std::f64::consts::PI
                * HIGGS_VEV.powi(2))
                / (6.0
                    * COS_THETA_WEAK.powi(4)
                    * self.mv.powi(4)
                    * W_BOSON_MASS.powi(2)
                    * SIN_THETA_WEAK.powi(2))
        } else {
            0.0
        }
    }
    /// Compute the total width or parital of the vector meidator.
    pub fn width_v_to_xx(&self) -> f64 {
        if self.mv > 2.0 * self.mx {
            (self.gvxx.powi(2)
                * (-4.0 * self.mx.powi(2) + self.mv.powi(2)).sqrt()
                * (2.0 * self.mx.powi(2) + self.mv.powi(2)))
                / (12.0 * self.mv.powi(2) * std::f64::consts::PI)
        } else {
            0.0
        }
    }
    pub fn vm_decay_width(&self) -> f64 {
        self.width_v_to_ququ(UP_QUARK_MASS)
            + self.width_v_to_ququ(CHARM_QUARK_MASS)
            + self.width_v_to_ququ(TOP_QUARK_MASS)
            + self.width_v_to_qdqd(DOWN_QUARK_MASS)
            + self.width_v_to_qdqd(STRANGE_QUARK_MASS)
            + self.width_v_to_qdqd(BOTTOM_QUARK_MASS)
            + self.width_v_to_ll(ELECTRON_MASS)
            + self.width_v_to_ll(MUON_MASS)
            + self.width_v_to_ll(TAU_MASS)
            + 3.0 * self.width_v_to_nunu()
            + self.width_v_to_hz()
            + self.width_v_to_xx()
    }
    pub fn vm_partial_decay_width(&self, fs: KineticMixingFinalStates) -> f64 {
        match fs {
            KineticMixingFinalStates::All => {
                self.width_v_to_ququ(UP_QUARK_MASS)
                    + self.width_v_to_ququ(CHARM_QUARK_MASS)
                    + self.width_v_to_ququ(TOP_QUARK_MASS)
                    + self.width_v_to_qdqd(DOWN_QUARK_MASS)
                    + self.width_v_to_qdqd(STRANGE_QUARK_MASS)
                    + self.width_v_to_qdqd(BOTTOM_QUARK_MASS)
                    + self.width_v_to_ll(ELECTRON_MASS)
                    + self.width_v_to_ll(MUON_MASS)
                    + self.width_v_to_ll(TAU_MASS)
                    + 3.0 * self.width_v_to_nunu()
                    + self.width_v_to_hz()
                    + self.width_v_to_xx()
            }
            KineticMixingFinalStates::UU => self.width_v_to_ququ(UP_QUARK_MASS),
            KineticMixingFinalStates::CC => self.width_v_to_ququ(CHARM_QUARK_MASS),
            KineticMixingFinalStates::TT => self.width_v_to_ququ(TOP_QUARK_MASS),
            KineticMixingFinalStates::DD => self.width_v_to_ququ(DOWN_QUARK_MASS),
            KineticMixingFinalStates::SS => self.width_v_to_ququ(STRANGE_QUARK_MASS),
            KineticMixingFinalStates::BB => self.width_v_to_ququ(BOTTOM_QUARK_MASS),
            KineticMixingFinalStates::EE => self.width_v_to_ll(ELECTRON_MASS),
            KineticMixingFinalStates::MuMu => self.width_v_to_ll(MUON_MASS),
            KineticMixingFinalStates::TauTau => self.width_v_to_ll(TAU_MASS),
            KineticMixingFinalStates::NueNue => self.width_v_to_nunu(),
            KineticMixingFinalStates::NumuNumu => self.width_v_to_nunu(),
            KineticMixingFinalStates::NutauNutau => self.width_v_to_nunu(),
            KineticMixingFinalStates::HiggsZ => self.width_v_to_hz(),
            KineticMixingFinalStates::XX => self.width_v_to_xx(),
        }
    }
}
