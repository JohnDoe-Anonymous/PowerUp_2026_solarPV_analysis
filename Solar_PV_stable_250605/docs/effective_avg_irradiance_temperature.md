### Irradiance and temperature effects on SDM parameters

Reference [osti_1048979] details how solar irradiance $`G^{p}`$ and cell
temperature $`T^{p}`$ impact various parameters of the single-diode model (SDM).
Their influence extends beyond the thermal voltage and includes changes in the
photocurrent, reverse saturation current, and shunt resistance.
In particular, variations in irradiance and temperature affect
$`I_{\mathrm{PH}}`$, $`I_0`$, and $`R_{\mathrm{SH}}`$.

The relationship between photocurrent, cell temperature, and solar irradiance is
given by:

```math
I_{\mathrm{PH}}^{p}
=
I_{\mathrm{PH,ref}}^{p}
\left( \frac{G^{p}}{G^{\mathrm{ref}}} \right)
\left[ 1 + \gamma_T \left( T^{p} - T_{\mathrm{ref}} \right) \right]
```
where $`G^{\mathrm{ref}}`$, $`T_{\mathrm{ref}}`$, and $`I_{\mathrm{PH,ref}}^{p}`$ denote the reference solar irradiance, reference cell temperature, and photocurrent at standard operating conditions, respectively.
The parameter $`\gamma_T`$ denotes the relative temperature coefficient of the short-circuit current provided by the solar panel datasheet.

Similarly, the reverse saturation current depends on the cell temperature and is
modeled as:

```math
I_0^{p}
=
I_0^{\mathrm{ref}}
\left( \frac{T^{p}}{T^{\mathrm{ref}}} \right)^3
\exp\left(
\frac{E_g^{\mathrm{ref}}}{k T^{\mathrm{ref}}}
-
\frac{E_g}{k T^{p}}
\right)
```
where $E_g$ is the semiconductor bandgap energy (eV), which is also a function of cell temperature and is defined as:
```math
E_g
=
1.16
-
7.02 \times 10^{-4}
\left(
\frac{(T^{p})^2}{T^{p} - 1108}
\right)
```

The series resistance $R_S^{p}$ is assumed to be independent of irradiance and temperature, while the shunt resistance is modeled as a function of solar irradiance:
```math
R_{\mathrm{SH}}^{p}
=
R_{\mathrm{SH,ref}}
\frac{G^{p}}{G^{\mathrm{ref}}}
```
