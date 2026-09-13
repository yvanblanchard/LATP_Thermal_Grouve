# Sensitivity of Absorbed Heat Flux Distribution to Laser Source Position and Orientation in Laser-Assisted Tape Placement

---

## 1. Introduction

Laser-Assisted Tape Placement (LATP) is an automated manufacturing process for thermoplastic composite laminates, in which a thermoplastic tape is consolidated onto a substrate by simultaneous heating and compaction. The laser source heats both the incoming tape and the substrate at the nip point — the contact line between the compaction roller and the laminate — to locally reach or exceed the polymer melting temperature, enabling intimate contact and bonding under the roller pressure.

The spatial distribution of the absorbed heat flux at the nip point is a critical process variable. It determines the peak temperature reached at the consolidation front, the symmetry of heating between the tape and the substrate, and ultimately the resulting interlaminar bond quality. Consequently, accurate numerical prediction of this flux distribution is a prerequisite for any thermal process model used to optimise placement parameters (speed, laser power, incidence angle) or to predict thermal damage and warping.

The optical boundary conditions of a LATP or Automated Fibre Placement (AFP) system depend on the exact geometry of the laser source relative to the nip point: its standoff distance, its lateral position, and the angle of incidence of the beam onto the substrate plane. In practice, these quantities are set during machine commissioning and are subject to manufacturing tolerances and in-service variations.

---

## 2. Objective

This study investigates the sensitivity of the laser-absorbed heat flux distribution — on both the substrate and the incoming tape — to two key geometric parameters of the laser source assembly:

1. **Laser incidence angle** θ (angle of the source axis with respect to the vertical direction), swept from 19° to 28° in 1° increments around a nominal value of 22°.  
2. **Source standoff height** *h* (Z-coordinate of the source centre above the substrate plane), swept from 45 mm to 70 mm in 5 mm increments around a nominal value of 55 mm.

For each parameter, one-at-a-time (OAT) sweeps are performed with the 2-D ray-tracing optical model described in the companion code (`ray_tracing_2d.py`). The following output metrics are monitored: peak absorbed flux on the substrate, peak absorbed flux on the tape (roller surface), and the spatial gradient of flux between the two surfaces at the same arc-length position from the nip point.

---

## 3. Related Work

The heat flux distribution during LATP has been studied both analytically and computationally. Grouve *et al.* [[1]](#references) presented an early process simulation tool combining an optical model with a 1-D thermal model along the tape path, demonstrating that the laser spot geometry strongly conditions the temperature field at consolidation.

More recent work has focused on multi-objective optimisation of the heat flux distribution. Baho *et al.* [[2]](#references) formulated a framework that jointly minimises warping and maximises interlaminar shear strength by tuning process parameters including the laser beam orientation. Their results confirm that the spatial asymmetry of heating between tape and substrate is a primary driver of residual stress, and that relatively small changes in beam configuration can shift the process operating point significantly.

These references motivate the need for quantified sensitivity data, specifically how much the output flux metrics change per unit relative change in source geometry, expressed as a dimensionless sensitivity index *S* [%/%].

---

## 4. Sensitivity Analysis: Inputs, Method, and Results

The 2-D ray-tracing model discretises the laser source into *N* = 5 000 rays, traces them through up to three specular reflections on the roller and substrate surfaces (refractive index 1.5), and accumulates the absorbed irradiance on 500 discretisation points per surface.

The percentage sensitivity index is defined as:

$$S = \frac{\Delta Q_{out} / Q_{ref}}{\Delta \theta_{in} / \theta_{ref}} \times 100 \quad [\%/\%]$$

and is estimated by linear regression over the ±20% central band of the input range to avoid end-point artefacts.

**Angle sensitivity.** Over the 19°–28° range (a total excursion of ±19% from the 22° nominal), the peak substrate flux and peak tape flux each exhibit a measurable but moderate dependence on the incidence angle, with sensitivity indices typically below unity — meaning a 10% change in angle produces less than 10% change in peak flux. However, the flux gradient between the two surfaces (substrate minus tape) is notably more sensitive, as the angle shift redistributes the illuminated zone asymmetrically between the two surfaces. This gradient metric therefore provides a sharper indicator of beam misalignment.

**Height sensitivity.** Over the 45–70 mm range (±27% from the 55 mm nominal), the standoff height affects the beam footprint area and therefore the local irradiance more directly: a shorter standoff concentrates the beam, raising peak flux on both surfaces, while a larger standoff spreads energy over a wider arc-length window, reducing peak values and broadening the heated zone. The sensitivity index for peak substrate flux with respect to height is generally larger in magnitude than that for angle, reflecting the inverse-square-law character of irradiance with distance.

Across both sweeps, the tape flux consistently exhibits higher sensitivity than substrate flux, as the roller curvature makes the illuminated arc-length on the tape more sensitive to beam geometry changes. The peak flux difference (substrate − tape), which governs thermal asymmetry at consolidation, shows sensitivity indices in the range of several %/% for both parameters, confirming that even small geometric deviations can create meaningful asymmetry in the thermal boundary conditions.

---

## 5. Conclusion

This sensitivity study demonstrates that the spatial distribution of laser-absorbed heat flux in LATP is measurably affected by both the incidence angle and the standoff height of the laser source. The flux gradient between substrate and tape is particularly sensitive to these parameters and represents a useful discriminating metric for assessing the impact of geometric uncertainties.

From a modelling perspective, these results underline that numerical process models must accurately reflect the actual geometry of the optics device — its orientation and its position relative to the nip point. This is non-trivial in practice: the feasibility of direct measurement of the laser incidence angle depends strongly on how the optics mounting structure on the AFP head has been designed. If the mounting does not allow direct angular measurement (e.g., a fixed bracket without angular scale), the angle must be inferred indirectly from geometric features of the head assembly, introducing additional uncertainty.

Furthermore, in real operational service — particularly during AFP layup with industrial robots — the laser head and associated optics are subjected to structural vibrations from robot arm dynamics, path reversals, and varying feed speeds. These excitations can transiently perturb both the beam angle and the standoff distance relative to their statically defined values. Given the sensitivity indices quantified here, even small-amplitude oscillations in beam orientation or height may translate into significant fluctuations in the absorbed flux distribution. As a consequence, one may expect non-negligible deviations between surface temperatures measured by an on-head infrared camera and those predicted by a quasi-static numerical model, precisely because the optics location and direction during motion can differ from the values recorded statically during calibration. This dynamic uncertainty should be accounted for in process monitoring strategies and in the tolerance budgets allocated to thermal model predictions.

---

## References

<a name="references"></a>

[1] W. J. B. Grouve *et al.*, *Towards a process simulation tool for the laser-assisted tape placement process*, Composites Part A, 2012. Available at: [ResearchGate](https://www.researchgate.net/publication/241875537_Towards_a_process_simulation_tool_for_the_laseer_assisted_tape_placement_process)

[2] B. Baho *et al.*, *Multi-objective optimisation of laser heat flux distribution considering interlaminar shear strength and warping in laser-assisted automated fibre placement*, 2025. Available at: [ResearchGate](https://www.researchgate.net/publication/391343704_Multi-objective_optimization_of_laser_heat_flux_distribution_considering_interlaminar_shear_strength_and_warping_in_laser-assisted_automated_fiber_placement)

[3] O. Baho *et al.*, HAL open archive, 2020. Available at: [HAL](https://hal.science/hal-02929735v1)
