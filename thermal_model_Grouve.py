"""
Corrected Thermal Model for Laser Assisted Tape Placement Process
Combined into single file with error handling and timing

HEAT FLUX CONVERSION METHODOLOGY:
=================================

1. NORMALIZED FLUX (from optical model):
   - Values 0-1 representing local absorption efficiency
   - Accounts for reflection, beam geometry, shadowing
   - Function of distance from nip-point

2. PHYSICAL HEAT FLUX DENSITY [W/m²]:
   q"(x,t) = φ_norm(x) × q"_max
   
   Where:
   - φ_norm(x) = normalized flux from optical ray tracing
   - q"_max = P_laser / A_irradiated
   - A_irradiated = beam_width × effective_interaction_length

3. THERMAL BOUNDARY CONDITION:
   -k(∂T/∂z)|surface = h(T_surface - T_ambient) - q"(x,t)

Key Physics:
- Ray tracing provides spatial distribution of absorbed energy
- Laser power conservation: ∫ q"(x,t) dx dt ≈ P_laser × η_absorption
- Material motion converts spatial variation to temporal variation
"""

import numpy as np
import matplotlib.pyplot as plt
import time

try:
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    print("Warning: scipy not available, using linear interpolation")
    HAS_SCIPY = False

class ThermalSolver:
    def __init__(self, material_props, geometry, boundary_conditions):
        start_time = time.perf_counter()
        
        self.rho = material_props['rho']  # kg/m³
        self.cp = material_props['cp']    # J/(kg·K)
        self.kz = material_props['kz']    # W/(m·K)
        
        self.thickness = geometry['thickness']  # m
        self.N = geometry['N_points']          # Number of collocation points
        
        self.h = boundary_conditions['h']           # W/(m²·K) - air heat transfer
        self.htool = boundary_conditions['htool']   # W/(m²·K) - tooling heat transfer
        self.T_far = boundary_conditions['T_far']   # °C - ambient temperature
        self.T_initial = boundary_conditions['T_initial']  # °C - initial temperature
        
        grid_start = time.perf_counter()
        self._setup_grid()
        grid_time = time.perf_counter() - grid_start
        
        matrix_start = time.perf_counter()
        self._setup_differentiation_matrix()
        matrix_time = time.perf_counter() - matrix_start
        
        total_time = time.perf_counter() - start_time
        print(f"    Grid setup: {grid_time*1000:.2f} ms")
        print(f"    Matrix setup: {matrix_time*1000:.2f} ms")
        print(f"    Total solver initialization: {total_time*1000:.2f} ms")
    
    def _setup_grid(self):
        """Setup uniform grid (simplified from Chebyshev for robustness)"""
        self.z = np.linspace(0, self.thickness, self.N)
        self.dz = self.thickness / (self.N - 1)
        
    def _setup_differentiation_matrix(self):
        """Setup finite difference second derivative matrix"""
        N = self.N
        dz = self.dz
        
        # Second order central difference for interior points
        self.D2 = np.zeros((N, N))
        
        # Interior points: (T[i-1] - 2*T[i] + T[i+1]) / dz^2
        for i in range(1, N-1):
            self.D2[i, i-1] = 1.0 / dz**2
            self.D2[i, i] = -2.0 / dz**2  
            self.D2[i, i+1] = 1.0 / dz**2
        
        # Boundary points will be handled separately
        self.D2[0, 0] = 1.0  # Will be replaced by BC
        self.D2[-1, -1] = 1.0  # Will be replaced by BC
    
    def _apply_boundary_conditions(self, T, Q_in, dt):
        """
        Apply boundary conditions using finite difference with proper energy balance
        
        PHYSICS: 1D heat equation: ρcp(∂T/∂t) = kz(∂²T/∂z²)
        
        At heated surface (z=0): -kz(∂T/∂z) = h(T - T_far) - Q_in
        At tooling surface (z=L): kz(∂T/∂z) = htool(T - T_far)
        """
        
        # Interior points use standard discretization
        A = np.eye(self.N) - dt * (self.kz / (self.rho * self.cp)) * self.D2
        b = T.copy()
        
        dz = self.dz
        
        # BOUNDARY NODE 0 (Heated surface): Energy balance approach
        # For surface node, use half-cell volume: V = A*dz/2
        # Energy balance: ρcp*V*(∂T/∂t) = A*q_applied + A*q_conduction
        # where q_applied = Q_in - h(T[0] - T_far) and q_conduction = kz*(T[1] - T[0])/dz
        
        A[0, :] = 0  # Clear row
        
        # Coefficients for implicit scheme:
        # [ρcp*dz/(2*dt) + h + kz/dz]*T[0] - [kz/dz]*T[1] = RHS
        A[0, 0] = self.rho * self.cp * dz / (2 * dt) + self.h + self.kz / dz
        A[0, 1] = -self.kz / dz
        b[0] = (self.rho * self.cp * dz / (2 * dt)) * T[0] + self.h * self.T_far + Q_in
        
        # BOUNDARY NODE N-1 (Tooling surface): Energy balance approach  
        # Energy balance: ρcp*V*(∂T/∂t) = A*q_tooling + A*q_conduction
        # where q_tooling = htool*(T_far - T[N-1]) and q_conduction = kz*(T[N-2] - T[N-1])/dz
        
        A[-1, :] = 0  # Clear row
        
        # Coefficients for implicit scheme:
        # [ρcp*dz/(2*dt) + htool + kz/dz]*T[N-1] - [kz/dz]*T[N-2] = RHS
        A[-1, -1] = self.rho * self.cp * dz / (2 * dt) + self.htool + self.kz / dz
        A[-1, -2] = -self.kz / dz
        b[-1] = (self.rho * self.cp * dz / (2 * dt)) * T[-1] + self.htool * self.T_far
        
        return A, b
    
    def solve_transient(self, time_steps, heat_flux_func, dt):
        """Solve transient heat equation"""
        solve_start = time.perf_counter()
        
        n_steps = len(time_steps)
        T_history = np.zeros((n_steps, self.N))
        
        # Initial condition - ensure all points start at ambient temperature
        T = np.full(self.N, self.T_initial, dtype=float)
        T_history[0, :] = T
        
        # Verify initial heat flux is zero or very small
        initial_Q = heat_flux_func(time_steps[0])
        print(f"    Initial heat flux at t=0: {initial_Q:.2f} W/m² (should be ~0)")
        
        # Timing variables
        bc_total_time = 0
        solve_total_time = 0
        
        print(f"    Starting time integration: {n_steps} time steps")
        
        # Time stepping with backward Euler
        for i in range(1, n_steps):
            t = time_steps[i]
            Q_in = heat_flux_func(t)
            
            # Set up system of equations
            bc_start = time.perf_counter()
            A, b = self._apply_boundary_conditions(T, Q_in, dt)
            bc_total_time += time.perf_counter() - bc_start
            
            # Solve for next time step
            solve_step_start = time.perf_counter()
            try:
                T_new = np.linalg.solve(A, b)
                
                # Check for reasonable temperature values
                if np.any(T_new < -50) or np.any(T_new > 2000):
                    print(f"Warning: Unrealistic temperatures at step {i}: min={np.min(T_new):.1f}, max={np.max(T_new):.1f}")
                    # Use smaller time step or keep previous solution
                    T_new = T + 0.1 * (T_new - T)  # Damped update
                
                T = T_new
                
            except np.linalg.LinAlgError:
                print(f"Warning: Singular matrix at time step {i}, using previous solution")
                # Keep previous T values
                
            solve_total_time += time.perf_counter() - solve_step_start
            T_history[i, :] = T
            
            # Progress indicator for long simulations
            if i % max(1, n_steps // 10) == 0:
                progress = (i / n_steps) * 100
                elapsed = time.perf_counter() - solve_start
                surf_temp = T[0]
                print(f"    Progress: {progress:.0f}% ({elapsed:.2f}s elapsed, T_surf={surf_temp:.1f}°C)")
        
        total_solve_time = time.perf_counter() - solve_start
        
        print(f"    Boundary conditions setup: {bc_total_time*1000:.2f} ms total")
        print(f"    Linear system solving: {solve_total_time*1000:.2f} ms total")
        print(f"    Time integration completed: {total_solve_time:.3f} s")
        print(f"    Average time per step: {(total_solve_time/n_steps)*1000:.3f} ms")
        
        # Final temperature check
        final_temps = T_history[-1, :]
        print(f"    Final temperature range: {np.min(final_temps):.1f} to {np.max(final_temps):.1f}°C")
        
        return T_history
    
    def get_surface_temperature(self, T_history):
        """Get surface temperature (heated side)"""
        return T_history[:, 0]


def create_heat_flux_interpolator(distance_points, flux_values):
    """Create interpolation function for heat flux vs distance from nip-point"""
    interp_start = time.perf_counter()
    
    # Sort by distance for proper interpolation
    sort_idx = np.argsort(distance_points)
    dist_sorted = distance_points[sort_idx]
    flux_sorted = flux_values[sort_idx]
    
    # Ensure flux is exactly zero outside the heating zone
    min_dist = np.min(dist_sorted)
    max_dist = np.max(dist_sorted)
    
    print(f"    Heat flux data range: {min_dist*1000:.1f} to {max_dist*1000:.1f} mm from nip-point")
    print(f"    Max normalized flux: {np.max(flux_sorted):.3f}")
    print(f"    Min normalized flux: {np.min(flux_sorted):.3f}")
    
    if HAS_SCIPY:
        # Use scipy interpolation with explicit zero fill values
        interpolator = interp1d(dist_sorted, flux_sorted, 
                              kind='linear', 
                              bounds_error=False, 
                              fill_value=(0.0, 0.0))  # Zero outside bounds
    else:
        # Simple numpy interpolation with zero outside bounds
        def interpolator(x):
            # Return zero for values outside the data range
            if hasattr(x, '__len__'):
                result = np.zeros_like(x)
                mask = (x >= min_dist) & (x <= max_dist)
                result[mask] = np.interp(x[mask], dist_sorted, flux_sorted)
                return result
            else:
                if x < min_dist or x > max_dist:
                    return 0.0
                return np.interp(x, dist_sorted, flux_sorted)
    
    # Test interpolator at key points
    test_points = [min_dist - 0.01, min_dist, max_dist, max_dist + 0.01]
    test_results = [interpolator(pt) for pt in test_points]
    print(f"    Interpolator test: {[(pt*1000, res) for pt, res in zip(test_points, test_results)]}")
    
    interp_time = time.perf_counter() - interp_start
    print(f"    Heat flux interpolator setup: {interp_time*1000:.2f} ms")
    
    return interpolator


# Material properties for APC-2 carbon/PEEK (CORRECTED values from literature)
MATERIAL_PROPS = {
    'rho': 1600,    # kg/m³
    'cp': 1000,     # J/(kg·K) 
    'kz': 0.7       # W/(m·K) - corrected through-thickness conductivity
}

# Process parameters from paper (PHYSICS-BASED ONLY)
PROCESS_PARAMS = {
    'beam_width': 0.030,           # m (30 mm from Table 1 - EXACT from paper)
    'beam_length': 0.030,          # m (27 mm default value for process direction)
    'roller_radius': 0.040,        # m (40 mm from Table 1 - EXACT from paper)
    'tape_thickness': 130e-6,      # m (130 μm from paper)
    'substrate_thickness': 2e-3,   # m (2 mm estimated from graphs)
    'laser_power': 1500,           # W (from problem statement)
    'velocities': [0.2, 0.3],      # m/s (from paper Figure 5)
    'laser_angles': [22, 11],      # degrees (from paper)
}

# Boundary conditions (CORRECTED)
BOUNDARY_CONDITIONS = {
    'h': 25,           # W/(m²·K) - air convection
    'htool': 1000,     # W/(m²·K) - tooling convection  
    'T_far': 20,       # °C - ambient temperature
    'T_initial': 20    # °C - initial temperature
}

def extract_heat_flux_data_22deg():
    """Extract heat flux data from Figure 4 for laser angle α = 22°"""
    
    # Substrate surface heat flux
    substrate_distance = np.array([
        -0.080, -0.070, -0.060, -0.050, -0.045, -0.040,
        -0.035, -0.030, -0.025, -0.020, -0.015, -0.010,
        -0.005, 0.0
    ])

    substrate_flux = np.array([
        0.00 ,0.05, 0.05, 0.05, 0.05, 0.4,
        0.42, 0.45, 0.48, 0.52, 0.57, 0.6,
        0.0, 0.0
    ])
    
    # Tape surface heat flux
    tape_distance = np.array([
        -0.06, -0.055, 
        -0.030, -0.025, -0.020, -0.015, -0.010, -0.005, 
        -0.002, 0.000
    ])
    
    tape_flux = np.array([
        0.0, 0.8, 
        0.20, 0.30, 0.2, 0.1, 0.10, 0.0, 
        0.00, 0.00
    ])
    
    return substrate_distance, substrate_flux, tape_distance, tape_flux

def create_thermal_solver_for_component(thickness, component_name):
    """Create thermal solver for substrate or tape"""
    geometry = {
        'thickness': thickness,
        'N_points': 15  # Reduced for better stability and reasonable computation time
    }
    
    return ThermalSolver(MATERIAL_PROPS, geometry, BOUNDARY_CONDITIONS)

def simulate_component_heating(distance_points, flux_values, thickness, velocity, 
                             laser_power, simulation_time=0.4):
    """Simulate heating of substrate or tape component"""
    sim_start = time.perf_counter()
    
    print(f"  Starting simulation for {thickness*1e6:.0f}μm thick component at {velocity} m/s")
    
    # Create thermal solver
    solver_start = time.perf_counter()
    solver = create_thermal_solver_for_component(thickness, "component")
    solver_time = time.perf_counter() - solver_start
    
    # Create heat flux interpolator
    flux_start = time.perf_counter()
    flux_interpolator = create_heat_flux_interpolator(distance_points, flux_values)
    flux_time = time.perf_counter() - flux_start
    
    # USE BEAM DIMENSIONS FROM PROCESS_PARAMS
    beam_width = PROCESS_PARAMS['beam_width']    # 30 mm from Table 1
    beam_length = PROCESS_PARAMS['beam_length']  # 27 mm default (user configurable)
    
    # Total laser power distributed over beam area
    beam_area = beam_width * beam_length  # [m²]
    max_power_density = laser_power / beam_area  # [W/m²]
    
    # Calculate expected exposure time and temperature rise
    exposure_time = beam_length / velocity  # Time material spends under laser
    
    print(f"    === PHYSICS VALIDATION ===")
    print(f"    Laser power: {laser_power} W")
    print(f"    Beam dimensions: {beam_width*1000:.0f} × {beam_length*1000:.0f} mm")
    print(f"    Beam area: {beam_area*1e6:.0f} mm² = {beam_area:.6f} m²")
    print(f"    Speed: {velocity} m/s = {velocity*1000:.0f} mm/s")
    print(f"    Exposure time: {exposure_time:.3f} s")
    print(f"    Power density: {max_power_density/1e5:.2f} W/mm² = {max_power_density/1e6:.2f} MW/m²")
    print(f"    Target: ~0.1 W/mm² → Ratio: {(max_power_density/1e5)/0.1:.1f}×")
    
    # Expected temperature rise calculation
    depth_estimate = min(0.001, thickness)  # 1mm or component thickness
    mass_per_area = MATERIAL_PROPS['rho'] * depth_estimate  # kg/m²
    energy_per_area = max_power_density * exposure_time     # J/m² for full exposure
    temp_rise_estimate = energy_per_area / (mass_per_area * MATERIAL_PROPS['cp'])  # K
    
    print(f"    Expected temperature rise (physics estimate):")
    print(f"    - Penetration depth: {depth_estimate*1000:.1f} mm")
    print(f"    - Mass per area: {mass_per_area:.2f} kg/m²") 
    print(f"    - Energy per area: {energy_per_area/1000:.0f} kJ/m²")
    print(f"    - Temperature rise: {temp_rise_estimate:.0f}°C")
    print(f"    - Expected peak: {20 + temp_rise_estimate:.0f}°C")
    
    if temp_rise_estimate > 500:
        print(f"    Physics predicts {temp_rise_estimate:.0f}°C rise - still too high!")
        print(f"    This suggests the power density or exposure time calculation needs review")
    
    # CORRECTED: Proper time-distance setup
    starting_distance = min(distance_points)  # Most negative value (farthest from nip)
    ending_distance = 0.0  # Nip-point
    
    # Calculate simulation time needed to travel from start to nip-point
    travel_distance = abs(starting_distance - ending_distance)
    required_time = travel_distance / velocity
    actual_sim_time = min(simulation_time, required_time)
    
    print(f"    === SIMULATION SETUP ===")
    print(f"    Starting distance: {starting_distance*1000:.1f} mm from nip-point")
    print(f"    Travel distance: {travel_distance*1000:.1f} mm")
    print(f"    Travel time: {required_time:.3f} s")
    print(f"    Simulation time: {actual_sim_time:.3f} s")
    
    # CRITICAL: Use very small time step for stability with high heat flux
    dt = min(0.0001, actual_sim_time/2000)  # 0.1ms or smaller
    time_array = np.arange(0, actual_sim_time + dt, dt)
    n_steps = len(time_array)
    
    # Stability check: Fourier number should be < 0.5
    alpha = MATERIAL_PROPS['kz'] / (MATERIAL_PROPS['rho'] * MATERIAL_PROPS['cp'])
    dz = thickness / (solver.N - 1)
    fourier_number = alpha * dt / (dz**2)
    
    print(f"    Time step: {dt*1000:.2f} ms ({n_steps} steps)")
    print(f"    Grid spacing: {dz*1e6:.1f} μm")
    print(f"    Fourier number: {fourier_number:.4f} (should be < 0.5)")
    
    if fourier_number > 0.5:
        print(f"    ⚠️  Fourier number > 0.5 may cause instability!")
        dt = 0.4 * dz**2 / alpha  # Reduce time step
        time_array = np.arange(0, actual_sim_time + dt, dt)
        n_steps = len(time_array)
        print(f"    Reduced to dt = {dt*1000:.2f} ms ({n_steps} steps)")
    
    # Heat flux function - converts normalized flux to physical values
    def heat_flux_func(t):
        # Material starts at starting_distance and moves toward nip-point
        current_distance = starting_distance + velocity * t
        current_distance = min(current_distance, 0.0)  # Cap at nip-point
        
        # Get normalized flux from optical model (0-1)
        norm_flux = flux_interpolator(current_distance)
        
        # Handle out-of-bounds and ensure non-negative
        if hasattr(norm_flux, '__len__'):
            norm_flux = np.maximum(0, norm_flux)
        else:
            if np.isnan(norm_flux) or norm_flux < 0:
                norm_flux = 0.0
            norm_flux = max(0.0, norm_flux)
        
        # Convert to physical heat flux density [W/m²]
        physical_flux = norm_flux * max_power_density
        
        return physical_flux
    
    # Test heat flux function and validate
    test_times = [0, actual_sim_time * 0.25, actual_sim_time * 0.5, actual_sim_time * 0.75, actual_sim_time * 0.9]
    print(f"    === HEAT FLUX PROFILE ===")
    for i, t in enumerate(test_times):
        flux = heat_flux_func(t)
        distance = starting_distance + velocity * t
        print(f"    t={t:.3f}s, d={distance*1000:.1f}mm, q={flux/1e5:.3f} W/mm²")
    
    # Store both normalized and physical heat flux for plotting
    heat_flux_physical = [heat_flux_func(t) for t in time_array]
    heat_flux_normalized = []
    for t in time_array:
        current_distance = min(starting_distance + velocity * t, 0.0)
        norm_val = flux_interpolator(current_distance)
        if np.isnan(norm_val) or norm_val < 0:
            norm_val = 0.0
        heat_flux_normalized.append(max(0.0, norm_val))
    
    # Calculate actual energy input for validation
    actual_energy_input = np.trapz(heat_flux_physical, time_array)  # J/m²
    print(f"    Total energy input: {actual_energy_input/1000:.0f} kJ/m²")
    
    # Solve thermal equation using physical heat flux
    thermal_start = time.perf_counter()
    print(f"    === STARTING THERMAL SOLUTION ===")
    T_history = solver.solve_transient(time_array, heat_flux_func, dt)
    thermal_time = time.perf_counter() - thermal_start
    
    # Post-processing
    post_start = time.perf_counter()
    surface_temp = solver.get_surface_temperature(T_history)
    distance_from_nip = starting_distance + velocity * time_array
    distance_from_nip = np.minimum(distance_from_nip, 0.0)
    
    max_temp = np.max(surface_temp)
    max_temp_distance = distance_from_nip[np.argmax(surface_temp)]
    initial_temp = surface_temp[0]
    
    post_time = time.perf_counter() - post_start
    total_sim_time = time.perf_counter() - sim_start
    
    print(f"    === RESULTS VALIDATION ===")
    print(f"    Initial surface temperature: {initial_temp:.1f}°C")
    print(f"    Maximum surface temperature: {max_temp:.1f}°C at {max_temp_distance*1000:.1f} mm from nip")
    print(f"    Temperature rise: {max_temp - initial_temp:.1f}°C")
    print(f"    Physics prediction: {temp_rise_estimate:.0f}°C")
    print(f"    Ratio (actual/predicted): {(max_temp - initial_temp)/temp_rise_estimate:.2f}")    
    
    print(f"    Total component simulation: {total_sim_time:.3f} s")
    print()
    
    return time_array, surface_temp, distance_from_nip, heat_flux_normalized

def plot_surface_temperatures_22deg():
    """Plot surface temperatures and NORMALIZED heat flux for laser angle 22° with dual y-axes
    Uses updated heat flux distributions per user specifications"""
    plot_start = time.perf_counter()
    print("\n=== PLOTTING SURFACE TEMPERATURES & NORMALIZED HEAT FLUX (α = 22°) ===")
    print("=== USING UPDATED HEAT FLUX DISTRIBUTIONS ===")
    
    data_start = time.perf_counter()
    substrate_dist, substrate_flux, tape_dist, tape_flux = extract_heat_flux_data_22deg()
    data_time = time.perf_counter() - data_start
    print(f"Heat flux data extraction: {data_time*1000:.2f} ms")
    
    velocities = PROCESS_PARAMS['velocities']
    laser_power = PROCESS_PARAMS['laser_power']
    
    setup_start = time.perf_counter()
    # Create two separate plots with dual y-axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = ['red', 'green']
    setup_time = time.perf_counter() - setup_start
    print(f"Plot setup: {setup_time*1000:.2f} ms")
    
    print("\n--- SUBSTRATE SIMULATIONS ---")
    substrate_sim_start = time.perf_counter()
    
    # SUBSTRATE PLOT with dual y-axes
    ax1_flux = ax1.twinx()  # Create second y-axis for heat flux
    
    # Substrate temperature results
    for i, velocity in enumerate(velocities):
        try:
            time_array, surf_temp, distance, heat_flux_norm = simulate_component_heating(
                substrate_dist, substrate_flux, 
                PROCESS_PARAMS['substrate_thickness'], 
                velocity, laser_power
            )
            
            # Plot temperature on left y-axis
            ax1.plot(distance*1000, surf_temp, colors[i], linewidth=2, 
                    label=f'T, v = {velocity} m/s')
            
        except Exception as e:
            print(f"Error simulating substrate at velocity {velocity}: {e}")
            continue
    
    # Plot NORMALIZED heat flux on right y-axis (as in paper Figure 4)
    flux_interp = create_heat_flux_interpolator(substrate_dist, substrate_flux)
    dist_range = np.linspace(-80, 0, 200)  # mm
    
    # Get normalized flux values (0-1) directly from interpolator
    flux_normalized = [max(0, flux_interp(d/1000)) for d in dist_range]  # Convert mm to m for interpolator
    
    line_flux = ax1_flux.plot(dist_range, flux_normalized, 'k--', 
                             linewidth=2, alpha=0.8, label='Normalized heat flux')
    
    substrate_sim_time = time.perf_counter() - substrate_sim_start
    print(f"Total substrate simulations: {substrate_sim_time:.3f} s")
    
    # Format substrate plot
    ax1.set_xlabel('Distance from nip-point (mm)')
    ax1.set_ylabel('Temperature (°C)', color='blue')
    ax1.set_title('Substrate:\nTemperature & Heat Flux')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-80, 0)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax1_flux.set_ylabel('Normalized heat flux (-)', color='black')
    ax1_flux.set_ylim(0, 1.0)  # Normalized scale 0-1
    ax1_flux.tick_params(axis='y', labelcolor='black')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_flux.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    print("\n--- TAPE SIMULATIONS ---")
    tape_sim_start = time.perf_counter()
    
    # TAPE PLOT with dual y-axes
    ax2_flux = ax2.twinx()  # Create second y-axis for heat flux
    
    # Tape temperature results
    for i, velocity in enumerate(velocities):
        try:
            time_array, surf_temp, distance, heat_flux_norm = simulate_component_heating(
                tape_dist, tape_flux,
                PROCESS_PARAMS['tape_thickness'],
                velocity, laser_power
            )
            
            # Plot temperature on left y-axis
            ax2.plot(distance*1000, surf_temp, colors[i], linewidth=2,
                    label=f'T, v = {velocity} m/s')
                    
        except Exception as e:
            print(f"Error simulating tape at velocity {velocity}: {e}")
            continue
    
    # Plot NORMALIZED heat flux on right y-axis
    flux_interp_tape = create_heat_flux_interpolator(tape_dist, tape_flux)
    dist_range_tape = np.linspace(-60, 0, 200)  # mm
    
    flux_normalized_tape = [max(0, flux_interp_tape(d/1000)) for d in dist_range_tape]  # Convert mm to m
    
    ax2_flux.plot(dist_range_tape, flux_normalized_tape, 'k--', 
                 linewidth=2, alpha=0.8, label='Normalized heat flux')
    
    tape_sim_time = time.perf_counter() - tape_sim_start
    print(f"Total tape simulations: {tape_sim_time:.3f} s")
    
    # Format tape plot
    ax2.set_xlabel('Distance from nip-point (mm)')
    ax2.set_ylabel('Temperature (°C)', color='blue')
    ax2.set_title('Incoming Tape\nTemperature & Heat Flux')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-60, 0)
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax2_flux.set_ylabel('Normalized heat flux (-)', color='black')
    ax2_flux.set_ylim(0, 1.0)  # Normalized scale 0-1
    ax2_flux.tick_params(axis='y', labelcolor='black')
    
    # Combined legend for tape
    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_flux.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper left')
    
    # Final plot rendering
    render_start = time.perf_counter()
    plt.tight_layout()
    plt.suptitle('Surface Temperatures & Custom Heat Flux Distributions (α = 22°)', y=1.02, fontsize=14)
    plt.show()
    render_time = time.perf_counter() - render_start
    
    total_plot_time = time.perf_counter() - plot_start
    
    print(f"Plot rendering: {render_time*1000:.2f} ms")
    print(f"Total plotting time: {total_plot_time:.3f} s")
    print(f"Total simulation time: {(substrate_sim_time + tape_sim_time):.3f} s")
     
    # Physics validation with new beam_length
    beam_area = PROCESS_PARAMS['beam_width'] * PROCESS_PARAMS['beam_length']
    max_power_density = PROCESS_PARAMS['laser_power'] / beam_area
    actual_peak_flux = max(flux_normalized) * max_power_density / 1e5  # Convert to W/mm²
    
    print(f"\nPhysical heat flux at peak (for thermal model):")
    print(f"Beam area: {beam_area*1e6:.0f} mm² ({PROCESS_PARAMS['beam_width']*1000:.0f}×{PROCESS_PARAMS['beam_length']*1000:.0f}mm)")
    print(f"Peak power density: {max_power_density/1e5:.2f} W/mm²")
    print(f"Peak physical flux: {actual_peak_flux:.3f} W/mm²")

def main():
    """Main function to run the example"""
    main_start = time.perf_counter()
    
    print("=" * 80)
    print("  THERMAL SIMULATION FOR LASER ASSISTED TAPE PLACEMENT")
    print("  UPDATED HEAT FLUX DISTRIBUTIONS & BEAM PARAMETERS")
    print("=" * 80)
    print(f"Material: APC-2 Carbon/PEEK")
    print(f"Tape thickness: {PROCESS_PARAMS['tape_thickness']*1e6:.0f} μm")
    print(f"Substrate thickness: {PROCESS_PARAMS['substrate_thickness']*1e3:.1f} mm")
    print(f"Laser power: {PROCESS_PARAMS['laser_power']} W")
    print(f"Placement velocities: {PROCESS_PARAMS['velocities']} m/s")
    
    # BEAM CONFIGURATION FROM PROCESS_PARAMS
    beam_width = PROCESS_PARAMS['beam_width']    # 30mm
    beam_length = PROCESS_PARAMS['beam_length']  # 27mm (configurable)
    beam_area = beam_width * beam_length
    power_density = PROCESS_PARAMS['laser_power'] / beam_area
    
    # PHYSICS PREDICTION for 0.3 m/s
    velocity = 0.3  # m/s
    exposure_time = beam_length / velocity
    energy_per_area = power_density * exposure_time
    
    # Temperature rise estimate (1mm depth)
    depth = 0.001  # m
    mass_per_area = MATERIAL_PROPS['rho'] * depth
    temp_rise_physics = energy_per_area / (mass_per_area * MATERIAL_PROPS['cp'])
    expected_peak = 20 + temp_rise_physics
    
    print(f"\nBEAM CONFIGURATION:")
    print(f"Beam dimensions: {beam_width*1000:.0f} × {beam_length*1000:.0f} mm")
    print(f"Beam area: {beam_area*1e6:.0f} mm²")
    print(f"Power density: {power_density/1e5:.2f} W/mm²")
    print(f"Speed: {velocity} m/s = {velocity*1000:.0f} mm/s")
    print(f"Exposure time: {exposure_time:.3f} s")
    
    print(f"\n🧮 PHYSICS PREDICTION (v = {velocity} m/s):")
    print(f"Energy per area: {energy_per_area/1000:.0f} kJ/m²")
    print(f"Mass per area (1mm depth): {mass_per_area:.2f} kg/m²")
    print(f"Expected temp rise: {temp_rise_physics:.0f}°C")
    print(f"Expected peak temp: {expected_peak:.0f}°C")    
    
    # System info
    print(f"\nComputational Setup:")
    print(f"SciPy available: {HAS_SCIPY}")
    print(f"NumPy version: {np.__version__}")
    print(f"Start time: {time.strftime('%H:%M:%S', time.localtime())}")
    print()
    
    try:
        simulation_start = time.perf_counter()
        plot_surface_temperatures_22deg()
        simulation_time = time.perf_counter() - simulation_start
        
        main_time = time.perf_counter() - main_start
        
        print("\n" + "=" * 80)
        print("  PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Total simulation time: {simulation_time:.3f} s")
        print(f"Total execution time: {main_time:.3f} s")
        
        print(f"\nSimulation completed successfully at {time.strftime('%H:%M:%S', time.localtime())}")       
      
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        
        main_time = time.perf_counter() - main_start
        print(f"\nExecution time before error: {main_time:.3f} s")

if __name__ == "__main__":
    main()