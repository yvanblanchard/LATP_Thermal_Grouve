# Thermal LATP Thermal Simulation model

A thermal simulation project for laser-assisted tape placement process 1D-modeling heat transfer and temperature distribution in composite materials.

## Link:
https://www.researchgate.net/publication/241875537_Towards_a_process_simulation_tool_for_the_laseer_assisted_tape_placement_process

![image](https://github.com/user-attachments/assets/b3731052-0142-428a-a726-80f5bb084c55)

![image](https://github.com/user-attachments/assets/9c18515e-e223-4b0e-8b66-4e38702966d1)

## Features
- Thermal modeling for APC-2 Carbon/PEEK materials
- Laser heating simulation with ray tracing integration
- Temperature distribution analysis for tape and substrate
- Heat transfer calculations with boundary conditions
- Support for multiple placement velocities

## Files
- `thermal_model_Grouve.py` - Main thermal simulation code
- Implements corrected heat flux conversion methodology
- Includes temperature initialization fixes
- Physics-based validation and timing analysis

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
- 1D heat equation with thermal diffusion
- Boundary conditions for air and tooling heat transfer
- Normalized flux from optical ray tracing converted to physical heat flux
- Material motion converts spatial variation to temporal variation

## Results
- Surface temperature profiles during laser heating
- Heat flux distributions from nip-point
- Temperature rise validation against physics predictions
