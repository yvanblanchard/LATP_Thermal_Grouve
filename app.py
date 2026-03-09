import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from ray_tracing_2d import (
    VectorizedLaser,
    VectorizedRoller,
    VectorizedSubstrate,
    VectorizedCurvedSubstrate,
    VectorizedPolygonalSubstrate,
    VectorizedRayTracer
)

def main():
    st.title("Ray Tracing 2D - Laser Processing Simulation")
    st.sidebar.header("Laser Configuration")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'last_params' not in st.session_state:
        st.session_state.last_params = None
    
    # Sidebar controls
    st.sidebar.subheader("Laser Source Position")
    source_y_positive = st.sidebar.slider("Y Position (mm)", 0.0, 400.0, 150.0, step=0.5)
    source_z = st.sidebar.slider("Z Position (mm)", 10.0, 200.0, 55.0, step=0.5)
    
    st.sidebar.subheader("Laser Properties")
    laser_angle = st.sidebar.slider("Laser Angle (degrees)", 0.0, 45.0, 22.0, step=0.5)
    num_rays = st.sidebar.selectbox("Number of Rays", [1000, 5000, 10000], index=2)

    st.sidebar.subheader("Material Properties")
    refractive_index = st.sidebar.slider("Refractive Index", 1.0, 3.0, 1.5, step=0.1)

    st.sidebar.subheader("Substrate Shape")
    substrate_shape = st.sidebar.selectbox(
        "Substrate Type",
        ["Flat", "Curved (Arc)", "V-Groove"],
        index=0,
    )
    substrate_radius = 200.0
    groove_angle = 20.0
    if substrate_shape == "Curved (Arc)":
        substrate_radius = st.sidebar.slider("Substrate Radius (mm)", 50.0, 500.0, 200.0, step=10.0)
    elif substrate_shape == "V-Groove":
        groove_angle = st.sidebar.slider("Groove Half-Angle (degrees)", 5.0, 45.0, 20.0, step=1.0)

    st.sidebar.subheader("Simulation Parameters")
    max_reflections = st.sidebar.slider("Max Reflections", 0, 5, 3, step=1)

    # Collect current parameters
    current_params = {
        'source_y': -source_y_positive * 1e-3,
        'source_z': source_z * 1e-3,
        'laser_angle': laser_angle,
        'num_rays': num_rays,
        'max_reflections': max_reflections,
        'refractive_index': refractive_index,
        'substrate_shape': substrate_shape,
        'substrate_radius': substrate_radius,
        'groove_angle': groove_angle,
    }

    params_changed = st.session_state.last_params != current_params
    run_clicked = st.sidebar.button("Run Simulation")

    if params_changed or run_clicked:
        with st.spinner("Running ray tracing simulation..."):
            results = run_simulation(
                source_y=current_params['source_y'],
                source_z=current_params['source_z'],
                laser_angle=current_params['laser_angle'],
                num_rays=current_params['num_rays'],
                max_reflections=current_params['max_reflections'],
                refractive_index=current_params['refractive_index'],
                substrate_shape=current_params['substrate_shape'],
                substrate_radius=current_params['substrate_radius'],
                groove_angle=current_params['groove_angle'],
            )

            if results:
                st.session_state.results = results
                st.session_state.last_params = current_params
            else:
                st.session_state.results = None

    # Display results if available
    if st.session_state.results:
        display_results(st.session_state.results)
    else:
        # Display default information
        st.write("## Instructions")
        st.write("1. Adjust laser source coordinates (Y, Z) in the sidebar")
        st.write("2. Set laser angle and number of rays")
        st.write("3. Configure refractive index for materials")
        st.write("4. Select substrate shape (Flat / Curved Arc / V-Groove)")
        st.write("5. Adjust shape-specific parameters (radius or groove angle) if needed")
        st.write("6. Click 'Run Simulation' to see irradiance plots")

        st.write("## System Configuration")
        st.write("- Roller radius: 40 mm")
        st.write("- Flat substrate length: 100 mm")
        st.write("- Laser power: 1.0 W (fixed)")


def run_simulation(source_y, source_z, laser_angle, num_rays, max_reflections, refractive_index,
                   substrate_shape="Flat", substrate_radius=200.0, groove_angle=20.0):
    """Run the ray tracing simulation with given parameters"""
    try:
        # Create laser with specified parameters
        laser = VectorizedLaser(
            source_length=30e-3,  # 30 mm
            source_center=np.array([source_y, source_z]),
            source_angle=laser_angle,
            num_rays=num_rays,
            total_power=1.0  # Fixed power at 1.0 W
        )
        
        # Create surfaces with specified refractive index
        roller = VectorizedRoller(radius=40e-3, refractive_index=refractive_index)

        if substrate_shape == "Curved (Arc)":
            substrate = VectorizedCurvedSubstrate(
                radius=substrate_radius * 1e-3,
                refractive_index=refractive_index)
        elif substrate_shape == "V-Groove":
            half_length = 50e-3  # 50 mm half-width
            depth = half_length * np.tan(np.radians(groove_angle))
            substrate = VectorizedPolygonalSubstrate(
                vertices=np.array([[-2 * half_length, 0.0],
                                   [-half_length, -depth],
                                   [0.0, 0.0]]),
                refractive_index=refractive_index)
        else:  # Flat (default)
            substrate = VectorizedSubstrate(length=100e-3, refractive_index=refractive_index)
        
        # Create ray tracer
        tracer = VectorizedRayTracer(
            laser, roller, substrate,
            max_reflections=max_reflections,
            min_power_threshold_fraction=1e-10
        )
        
        # Run simulation
        tracer.trace_all_rays_vectorized()
        
        # Calculate irradiance
        substrate_dist, substrate_irradiance_gen, substrate_shadow, substrate_max_extent, substrate_total_flux = tracer.calculate_irradiance_by_generation_vectorized(substrate, 1000)
        roller_dist, roller_irradiance_gen, roller_shadow, roller_max_extent, roller_total_flux = tracer.calculate_irradiance_by_generation_vectorized(roller, 100)
        
        return {
            'laser': laser,
            'substrate_dist': substrate_dist,
            'substrate_shadow': substrate_shadow,
            'substrate_max_extent': substrate_max_extent,
            'substrate_total_flux': substrate_total_flux,
            'roller_dist': roller_dist,
            'roller_shadow': roller_shadow,
            'roller_max_extent': roller_max_extent,
            'roller_total_flux': roller_total_flux,
            'refractive_index': refractive_index
        }
        
    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")
        return None

def display_results(results):
    """Display simulation results with irradiance plots"""
    
    # Extract results
    #laser = results['laser']
    substrate_dist = results['substrate_dist']
    substrate_shadow = results['substrate_shadow']
    substrate_max_extent = results['substrate_max_extent']
    substrate_total_flux = results['substrate_total_flux']
    
    roller_dist = results['roller_dist']
    roller_shadow = results['roller_shadow']
    roller_max_extent = results['roller_max_extent']
    roller_total_flux = results['roller_total_flux']
    
    refractive_index = results['refractive_index']
    
    # Create two columns for plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Substrate Irradiance")
        fig1 = create_total_flux_plot(
            substrate_dist * 1000,  # Convert to mm
            substrate_total_flux,
            substrate_max_extent * 1000,
            "Distance from Nip Point (mm)",
            "Absorbed Power (W)",
            "Substrate Total Irradiance"
        )
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Incoming Tape Irradiance")
        fig2 = create_total_flux_plot(
            roller_dist * 1000,  # Convert to mm
            roller_total_flux,
            roller_max_extent * 1000,
            "Arc Distance from Nip Point (mm)",
            "Absorbed Power (W)",
            "Incoming Tape Total Irradiance"
        )
        st.pyplot(fig2)
    
    # Display summary statistics
    st.subheader("Simulation Summary")
    
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Substrate Shadow", f"{substrate_shadow*1000:.1f} mm")
        st.metric("Substrate Max Flux", f"{np.max(substrate_total_flux):.3f}")
        st.metric("Substrate Max Extent", f"{substrate_max_extent*1000:.1f} mm")
    
    with col2:
        st.metric("Tape Shadow", f"{roller_shadow*1000:.1f} mm")
        st.metric("Tape Max Flux", f"{np.max(roller_total_flux):.3f}")
        st.metric("Tape Max Extent", f"{roller_max_extent*1000:.1f} mm")
    
    # Display material properties
    st.subheader("Material Properties")
    st.write(f"**Refractive Index:** {refractive_index:.1f}")

def create_total_flux_plot(dist, total_flux, max_extent, xlabel, ylabel, title):
    """Create total flux irradiance plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot only total flux with black solid line
    ax.plot(dist, total_flux, 'k-', linewidth=3, label='Total Flux')
    
    # Set x-axis limit
    xlim = max_extent + 5  # Add 5mm margin
    ax.set_xlim(0, xlim)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

if __name__ == "__main__":
    main()
