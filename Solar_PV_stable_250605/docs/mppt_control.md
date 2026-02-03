## MPPT control implementation

This section describes how the maximum power point (MPP) operating condition is
computed for the aggregated single-diode array model (SDM_A) and the per-panel
diode model (PPDM_A).

### SDM_A: closed-form MPP condition

For SDM_A, the MPP is obtained by enforcing the condition that the derivative of
array power with respect to the terminal voltage is zero. Specifically, the MPP
satisfies:

```math
\frac{\partial \left( I_{\mathrm{out}}^{a} V_{\mathrm{PV}}^{a} \right)}
{\partial V_{\mathrm{PV}}^{a}}
=
0
```
Further algebraic manipulation yields a closed-form relationship between the
array output current and voltage at the MPP:
```math
I_{\mathrm{out}}^{a}
=
\frac{
V_{\mathrm{PV}}^{a}
\left(
I_0^{a} R_{\mathrm{SH}}^{a} \chi
+
\alpha^{a}
\right)
}{
I_0^{a} R_S^{a} R_{\mathrm{SH}}^{a} \chi
+
\alpha^{a}
\left(
R_S^{a}
+
R_{\mathrm{SH}}^{a}
\right)
}
```
where the auxiliary variable $\chi$ is defined as:

```math
\chi
=
\exp\left(
\frac{V_D^{a}}{\alpha^{a}}
\right)
```

### PPDM_A: optimization-based MPP formulation

For PPDM_A, no closed-form MPP condition exists under non-uniform operating
conditions. Instead, the MPP is obtained by solving an optimization problem that
maximizes the array output power subject to the per-panel circuit constraints.

The optimization problem is formulated as:

```math
\max_{\; V_{\mathrm{PV}}^{p},\, V_D^{p},\, I_{\mathrm{out}}^{p}}
\;
I_{\mathrm{array}} V_{\mathrm{array}}
```
subject to the equality constraint:
```math
h\!\left(
V_{\mathrm{PV}}^{p},
V_D^{p},
I_{\mathrm{out}}^{p}
\right)
=
0
```
The constraint function $h(\cdot)$ enforces the per-panel electrical physics of
PPDM_A, including Kirchhoff’s current and voltage laws, panel diode equations,
array interconnection constraints, and optional bypass diode behavior.
