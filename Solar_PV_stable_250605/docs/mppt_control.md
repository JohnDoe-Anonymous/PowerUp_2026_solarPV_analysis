# Control Implementation (MPPT)

We evaluate the performance of SDM_A and PPDM_A under an MPPT control scheme.
This section describes how we obtain the MPP operating point for both models.

## SDM_A: closed-form MPPT condition
At MPP:
![dP/dV condition](../figures/eq_dPdV_zero.png)

Further derivation yields the closed-form relation:
![closed-form MPPT equation](../figures/eq_mpp_closed_form.png)

where:
![chi definition](../figures/eq_chi.png)

By applying the closed-form equation, we effectively replace Z_Load with a voltage-controlled current source.

## PPDM_A: optimization-based MPP
For a non-uniform array, there is no closed-form MPPT equation.
We solve:
![optimization problem](../figures/eq_ppdm_mpp_opt.png)

where the equality constraints include (Eq. ...): KCL/KVL and per-panel diode physics (with/without bypass diodes).
