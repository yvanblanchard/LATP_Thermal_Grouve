# Thermal LATP Thermal Simulation model

A thermal simulation project for laser-assisted tape placement process 1D-modeling heat transfer and temperature distribution in composite materials.

## Link:
https://www.researchgate.net/publication/241875537_Towards_a_process_simulation_tool_for_the_laseer_assisted_tape_placement_process

![image](https://github.com/user-attachments/assets/b3731052-0142-428a-a726-80f5bb084c55)

![image](https://github.com/user-attachments/assets/9c18515e-e223-4b0e-8b66-4e38702966d1)

## Features
- Thermal modeling for APC-2 Carbon/PEEK materials
- Variable Heat flux distribution (substrate and incoming tape)
- Temperature distribution analysis for tape and substrate
- Heat transfer calculations with boundary conditions
- Support for multiple placement velocities
- Plot heat flux and Surface temperature evolution (22 degrees laser incidence angle)

## Files
- `thermal_model_Grouve.py` - Main thermal simulation code

## Usage
```python
python thermal_model_Grouve.py
```

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- SciPy (optional, falls back to NumPy interpolation)

## Physics Model
- 1D heat equation with thermal diffusion (see W. Grouve paper for details)
- Boundary conditions for air and tooling heat transfer
- Normalized flux from optical ray tracing converted to physical heat flux
- Material motion converts spatial variation to temporal variation

## Results
- Surface temperature profiles during laser heating
- Heat flux distributions from nip-point
- Temperature rise validation against physics predictions

## Next
- Temperature through the thickness
- Compare with 2D FEA model (Calculix solver)
- Compare with analytical thermal heat exchange model (T. Weiler)
- Laser source with variable intensity distribution
- Add healing and degradation models, residual stresses (thermo-mechanical) model
