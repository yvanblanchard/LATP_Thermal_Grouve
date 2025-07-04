import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


class VectorizedRayBatch:
    """
    Vectorized representation of multiple rays for high-performance computation.
    All rays are stored as arrays for simultaneous processing.
    """
    
    def __init__(self, origins: np.ndarray, directions: np.ndarray, powers: np.ndarray, 
                 generations: np.ndarray = None, active_mask: np.ndarray = None):
        """
        origins: (N, 2) array of ray origin points [y, z]
        directions: (N, 2) array of ray direction vectors [dy, dz]
        powers: (N,) array of ray powers
        generations: (N,) array of reflection generation numbers
        active_mask: (N,) boolean array indicating which rays are still active
        """
        self.origins = origins.copy()
        self.directions = directions.copy()
        self.powers = powers.copy()
        self.num_rays = len(origins)
        
        if generations is None:
            self.generations = np.zeros(self.num_rays, dtype=int)
        else:
            self.generations = generations.copy()
            
        if active_mask is None:
            self.active_mask = np.ones(self.num_rays, dtype=bool)
        else:
            self.active_mask = active_mask.copy()
            
        # Normalize directions
        norms = np.linalg.norm(self.directions, axis=1)
        self.directions = self.directions / norms[:, np.newaxis]
        
        # Storage for intersection results
        self.intersection_points = np.full((self.num_rays, 2), np.nan)
        self.intersection_distances = np.full(self.num_rays, np.inf)
        self.hit_surface_ids = np.full(self.num_rays, -1, dtype=int)  # 0=roller, 1=substrate
        self.surface_normals = np.full((self.num_rays, 2), np.nan)
        self.incidence_angles = np.full(self.num_rays, np.nan)


class VectorizedSurface(ABC):
    """Abstract base class for vectorized surface operations"""
    
    def __init__(self, surface_id: int, refractive_index: float = 1.8):
        self.surface_id = surface_id
        self.refractive_index = refractive_index
        
    @abstractmethod
    def intersect_rays_vectorized(self, ray_batch: VectorizedRayBatch) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized ray-surface intersection.
        Returns: (hit_mask, intersection_points, distances)
        """
        pass
    
    @abstractmethod
    def get_normals_vectorized(self, points: np.ndarray) -> np.ndarray:
        """Get surface normals at given points (vectorized)"""
        pass
    
    @abstractmethod
    def get_points_for_plotting(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get points for plotting the surface"""
        pass


class VectorizedRoller(VectorizedSurface):
    """Vectorized roller surface - quarter circle arc with center at (0, radius)"""
    
    def __init__(self, radius: float = 35e-3, refractive_index: float = 1.8):
        super().__init__(surface_id=0, refractive_index=refractive_index)
        self.radius = radius
        self.center = np.array([0.0, radius])  # Center at (0, radius)
        
    def intersect_rays_vectorized(self, ray_batch: VectorizedRayBatch) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized intersection with quarter circle (y <= 0, z >= 0) centered at (0, radius)"""
        origins = ray_batch.origins[ray_batch.active_mask]
        directions = ray_batch.directions[ray_batch.active_mask]
        
        if len(origins) == 0:
            return np.array([], dtype=bool), np.empty((0, 2)), np.array([])
        
        # Vectorized quadratic equation solution
        # Circle: y^2 + (z - radius)^2 = r^2, Ray: point = origin + t * direction
        # Translate origins relative to circle center
        rel_origins = origins - self.center[np.newaxis, :]
        
        a = np.sum(directions * directions, axis=1)
        b = 2 * np.sum(rel_origins * directions, axis=1)
        c = np.sum(rel_origins * rel_origins, axis=1) - self.radius**2
        
        discriminant = b**2 - 4*a*c
        valid_disc = discriminant >= 0
        
        hit_mask = np.zeros(len(origins), dtype=bool)
        intersection_points = np.full((len(origins), 2), np.nan)
        distances = np.full(len(origins), np.inf)
        
        if np.any(valid_disc):
            sqrt_disc = np.sqrt(discriminant[valid_disc])
            a_valid = a[valid_disc]
            b_valid = b[valid_disc]
            
            t1 = (-b_valid - sqrt_disc) / (2 * a_valid)
            t2 = (-b_valid + sqrt_disc) / (2 * a_valid)
            
            # Check both solutions
            for t_values in [t1, t2]:
                forward_mask = t_values > 1e-10
                if not np.any(forward_mask):
                    continue
                    
                valid_indices = np.where(valid_disc)[0][forward_mask]
                t_forward = t_values[forward_mask]
                
                points = (origins[valid_indices] + 
                         t_forward[:, np.newaxis] * directions[valid_indices])
                
                # Check quarter circle constraints (y <= 0, z >= 0)
                y_valid = points[:, 0] <= 1e-10
                z_valid = points[:, 1] >= -1e-10
                quarter_valid = y_valid & z_valid
                
                if np.any(quarter_valid):
                    final_indices = valid_indices[quarter_valid]
                    final_t = t_forward[quarter_valid]
                    final_points = points[quarter_valid]
                    
                    # Update only if closer than existing intersection
                    closer_mask = final_t < distances[final_indices]
                    update_indices = final_indices[closer_mask]
                    
                    hit_mask[update_indices] = True
                    intersection_points[update_indices] = final_points[closer_mask]
                    distances[update_indices] = final_t[closer_mask]
        
        return hit_mask, intersection_points, distances
    
    def get_normals_vectorized(self, points: np.ndarray) -> np.ndarray:
        """Vectorized normal calculation - outward from circle center at (0, radius)"""
        rel_points = points - self.center[np.newaxis, :]
        norms = np.linalg.norm(rel_points, axis=1)
        return rel_points / norms[:, np.newaxis]
    
    def get_points_for_plotting(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get quarter circle points for plotting - center at (0, radius)"""
        theta = np.linspace(np.pi, 3*np.pi/2, 100)  # From pi to 3pi/2 for left quarter
        y = self.center[0] + self.radius * np.cos(theta)
        z = self.center[1] + self.radius * np.sin(theta)
        return y, z


class VectorizedSubstrate(VectorizedSurface):
    """Vectorized substrate surface - horizontal line"""
    
    def __init__(self, length: float = 100e-3, refractive_index: float = 1.8):
        super().__init__(surface_id=1, refractive_index=refractive_index)
        self.length = length
        
    def intersect_rays_vectorized(self, ray_batch: VectorizedRayBatch) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized intersection with horizontal line z = 0, y in [-length, 0]"""
        origins = ray_batch.origins[ray_batch.active_mask]
        directions = ray_batch.directions[ray_batch.active_mask]
        
        if len(origins) == 0:
            return np.array([], dtype=bool), np.empty((0, 2)), np.array([])
        
        y0, z0 = origins[:, 0], origins[:, 1]
        dy, dz = directions[:, 0], directions[:, 1]
        
        # Avoid division by zero
        non_parallel = np.abs(dz) > 1e-10
        hit_mask = np.zeros(len(origins), dtype=bool)
        intersection_points = np.full((len(origins), 2), np.nan)
        distances = np.full(len(origins), np.inf)
        
        if np.any(non_parallel):
            # Solve for intersection with z = 0
            t = -z0[non_parallel] / dz[non_parallel]
            forward_mask = t > 1e-10
            
            if np.any(forward_mask):
                valid_indices = np.where(non_parallel)[0][forward_mask]
                t_valid = t[forward_mask]
                
                y_intersect = y0[valid_indices] + t_valid * dy[valid_indices]
                
                # Check substrate bounds
                bounds_mask = (y_intersect >= -self.length - 1e-10) & (y_intersect <= 1e-10)
                
                if np.any(bounds_mask):
                    final_indices = valid_indices[bounds_mask]
                    final_t = t_valid[bounds_mask]
                    final_y = y_intersect[bounds_mask]
                    
                    hit_mask[final_indices] = True
                    intersection_points[final_indices, 0] = final_y
                    intersection_points[final_indices, 1] = 0
                    distances[final_indices] = final_t
        
        return hit_mask, intersection_points, distances
    
    def get_normals_vectorized(self, points: np.ndarray) -> np.ndarray:
        """Vectorized normal calculation - upward direction"""
        num_points = len(points)
        normals = np.zeros((num_points, 2))
        normals[:, 1] = 1  # All normals point upward [0, 1]
        return normals
    
    def get_points_for_plotting(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get line points for plotting"""
        y = np.array([-self.length, 0])
        z = np.array([0, 0])
        return y, z

class VectorizedCurvedSubstrate(VectorizedRoller):
    """Vectorized curved substrate surface - half circle arc mirrored through Y axis"""
    
    def __init__(self, radius: float = 35e-3, refractive_index: float = 1.8):
        super().__init__(radius, refractive_index)
        self.surface_id = 1  # Override surface_id to be substrate
        self.center = np.array([0.0, -radius])  # Mirror through Y axis: center at (0, -radius)
        
    def intersect_rays_vectorized(self, ray_batch: VectorizedRayBatch) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized intersection with half circle (z <= 0) centered at (0, -radius)"""
        origins = ray_batch.origins[ray_batch.active_mask]
        directions = ray_batch.directions[ray_batch.active_mask]
        
        if len(origins) == 0:
            return np.array([], dtype=bool), np.empty((0, 2)), np.array([])
        
        # Use parent class logic but with mirrored center
        rel_origins = origins - self.center[np.newaxis, :]
        
        a = np.sum(directions * directions, axis=1)
        b = 2 * np.sum(rel_origins * directions, axis=1)
        c = np.sum(rel_origins * rel_origins, axis=1) - self.radius**2
        
        discriminant = b**2 - 4*a*c
        valid_disc = discriminant >= 0
        
        hit_mask = np.zeros(len(origins), dtype=bool)
        intersection_points = np.full((len(origins), 2), np.nan)
        distances = np.full(len(origins), np.inf)
        
        if np.any(valid_disc):
            sqrt_disc = np.sqrt(discriminant[valid_disc])
            a_valid = a[valid_disc]
            b_valid = b[valid_disc]
            
            t1 = (-b_valid - sqrt_disc) / (2 * a_valid)
            t2 = (-b_valid + sqrt_disc) / (2 * a_valid)
            
            # Check both solutions
            for t_values in [t1, t2]:
                forward_mask = t_values > 1e-10
                if not np.any(forward_mask):
                    continue
                    
                valid_indices = np.where(valid_disc)[0][forward_mask]
                t_forward = t_values[forward_mask]
                
                points = (origins[valid_indices] + 
                         t_forward[:, np.newaxis] * directions[valid_indices])
                
                # Check half circle constraints (z <= 0) - substrate in lower semicircle
                z_valid = points[:, 1] <= 1e-10  # Z <= 0 (lower half)
                
                if np.any(z_valid):
                    final_indices = valid_indices[z_valid]
                    final_t = t_forward[z_valid]
                    final_points = points[z_valid]
                    
                    # Update only if closer than existing intersection
                    closer_mask = final_t < distances[final_indices]
                    update_indices = final_indices[closer_mask]
                    
                    hit_mask[update_indices] = True
                    intersection_points[update_indices] = final_points[closer_mask]
                    distances[update_indices] = final_t[closer_mask]
        
        return hit_mask, intersection_points, distances
    
    def get_points_for_plotting(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get half circle points for plotting - center at (0, -radius) lower semicircle"""
        theta = np.linspace(0, np.pi, 100)  # From 0 to pi for full lower semicircle
        y = self.center[0] + self.radius * np.cos(theta)
        z = self.center[1] + self.radius * np.sin(theta)
        return y, z
    
class VectorizedLaser:
    """Vectorized laser source definition"""
    
    def __init__(self, source_length: float = 30e-3, source_center: np.ndarray = np.array([-20e-3, 50e-3]),
                 source_angle: float = 20.0, num_rays: int = 100, total_power: float = 1000.0):
        """
        source_length: Length of laser line source (m)
        source_center: Center coordinates [y, z] (m)
        source_angle: Angle with vertical Z axis (degrees)
        num_rays: Number of rays to generate
        total_power: Total laser power (W)
        """
        self.source_length = source_length
        self.source_center = source_center
        self.source_angle = np.radians(source_angle)
        self.num_rays = num_rays
        self.total_power = total_power
        self.power_per_ray = total_power / num_rays
        
    def generate_ray_batch(self) -> VectorizedRayBatch:
        """Generate vectorized ray batch"""
        # Source line direction (along the line)
        line_direction = np.array([np.sin(self.source_angle), np.cos(self.source_angle)])
        
        # Ray direction (perpendicular to source line, towards substrate)
        ray_direction = np.array([np.cos(self.source_angle), -np.sin(self.source_angle)])
        
        # Generate source points along the line
        t_values = np.linspace(-0.5, 0.5, self.num_rays)
        
        # Vectorized source point calculation
        origins = (self.source_center[np.newaxis, :] + 
                  t_values[:, np.newaxis] * self.source_length * line_direction[np.newaxis, :])
        
        # All rays have same direction
        directions = np.tile(ray_direction, (self.num_rays, 1))
        
        # All rays have same power
        powers = np.full(self.num_rays, self.power_per_ray)
        
        return VectorizedRayBatch(origins, directions, powers)


class VectorizedRayTracer:
    """
    High-performance vectorized ray tracing engine
    
    POWER INDEPENDENCE FIX:
    ======================
    This implementation uses a RELATIVE power threshold instead of absolute
    to ensure the distribution shape remains constant regardless of laser power.
    
    PROBLEM with absolute threshold:
    - min_power_threshold = 0.01 W (fixed)
    - 1000W case: power_per_ray = 0.1W → reflected = 0.01W → above threshold ✅
    - 1W case: power_per_ray = 0.0001W → reflected = 0.00001W → below threshold ❌
    - Result: Low power cases lose reflections → different distribution shape
    
    SOLUTION with relative threshold:
    - min_power_threshold = total_power × threshold_fraction
    - 1000W case: threshold = 1000 × 1e-6 = 0.001W
    - 1W case: threshold = 1 × 1e-6 = 0.000001W
    - Result: Same relative cutoff → same distribution shape ✅
    
    PHYSICS JUSTIFICATION:
    - Real physics: reflection behavior depends on material properties, not power
    - Fresnel coefficients are power-independent
    - Only total absorbed energy should scale with power
    - Distribution shape should be power-independent
    """
    
    def __init__(self, laser: VectorizedLaser, roller: VectorizedRoller, 
                 substrate: VectorizedSubstrate, max_reflections: int = 4,
                 min_power_threshold_fraction: float = 1e-5):
        """
        Initialize ray tracer with power-independent threshold
        
        Args:
            laser: Laser source configuration
            roller: Roller surface geometry
            substrate: Substrate surface geometry  
            max_reflections: Maximum number of reflections to trace
            min_power_threshold_fraction: Relative threshold (e.g., 1e-5 means 0.001% of total power)
        """
        self.laser = laser
        self.roller = roller
        self.substrate = substrate
        self.surfaces = [roller, substrate]
        self.max_reflections = max_reflections
        # Calculate absolute threshold based on laser power
        self.min_power_threshold = laser.total_power * min_power_threshold_fraction
        
        # Storage for all ray generations
        self.all_ray_batches = []
        
    def fresnel_reflectance_vectorized(self, theta: np.ndarray, n1: float, n2: float) -> np.ndarray:
        """Calculate Fresnel reflection coefficient for unpolarized light"""
        theta = np.asarray(theta)
        result = np.zeros_like(theta, dtype=float)
        
        # Handle total internal reflection
        tir_mask = n1/n2 * np.sin(theta) >= 1
        result[tir_mask] = 1.0
        
        # Handle regular reflection
        non_tir_mask = ~tir_mask
        if np.any(non_tir_mask):
            theta_t = np.arcsin((n1/n2) * np.sin(theta[non_tir_mask]))
            r_s = ((n1*np.cos(theta[non_tir_mask]) - n2*np.cos(theta_t)) / 
                  (n1*np.cos(theta[non_tir_mask]) + n2*np.cos(theta_t)))**2
            r_p = ((n1*np.cos(theta_t) - n2*np.cos(theta[non_tir_mask])) / 
                  (n1*np.cos(theta_t) + n2*np.cos(theta[non_tir_mask])))**2
            result[non_tir_mask] = (r_s + r_p) / 2
        
        return result
    
    def find_nearest_intersections_vectorized(self, ray_batch: VectorizedRayBatch) -> None:
        """Find nearest surface intersection for all active rays"""
        if not np.any(ray_batch.active_mask):
            return
            
        # Initialize intersection data
        ray_batch.intersection_distances[ray_batch.active_mask] = np.inf
        ray_batch.hit_surface_ids[ray_batch.active_mask] = -1
        ray_batch.intersection_points[ray_batch.active_mask] = np.nan
        
        # Get active rays only
        active_indices = np.where(ray_batch.active_mask)[0]
        num_active = len(active_indices)
        
        if num_active == 0:
            return
        
        # Track best intersections for active rays
        best_distances = np.full(num_active, np.inf)
        best_surface_ids = np.full(num_active, -1)
        best_points = np.full((num_active, 2), np.nan)
        
        # Test intersection with each surface
        for surface in self.surfaces:
            hit_mask, intersection_points, distances = surface.intersect_rays_vectorized(ray_batch)
            
            if len(hit_mask) == 0 or not np.any(hit_mask):
                continue
                
            # Find which rays had closer intersections
            hit_indices = np.where(hit_mask)[0]
            hit_distances = distances[hit_mask]
            
            # Update only if this intersection is closer
            closer_mask = hit_distances < best_distances[hit_indices]
            update_indices = hit_indices[closer_mask]
            
            if len(update_indices) > 0:
                best_distances[update_indices] = hit_distances[closer_mask]
                best_surface_ids[update_indices] = surface.surface_id
                best_points[update_indices] = intersection_points[hit_mask][closer_mask]
        
        # Update ray batch with best intersections
        ray_batch.intersection_distances[active_indices] = best_distances
        ray_batch.hit_surface_ids[active_indices] = best_surface_ids
        ray_batch.intersection_points[active_indices] = best_points
    
    def compute_reflections_vectorized(self, ray_batch: VectorizedRayBatch) -> VectorizedRayBatch:
        """Compute reflected rays for all intersections"""
        # Find rays that actually hit something and are active
        hit_mask = (ray_batch.hit_surface_ids >= 0) & ray_batch.active_mask
        
        if not np.any(hit_mask):
            return None
            
        hit_indices = np.where(hit_mask)[0]
        num_hits = len(hit_indices)
        
        if num_hits == 0:
            return None
        
        # Get surface normals for all intersection points
        roller_hits = ray_batch.hit_surface_ids[hit_indices] == 0
        substrate_hits = ray_batch.hit_surface_ids[hit_indices] == 1
        
        normals = np.zeros((num_hits, 2))
        
        if np.any(roller_hits):
            roller_points = ray_batch.intersection_points[hit_indices[roller_hits]]
            normals[roller_hits] = self.roller.get_normals_vectorized(roller_points)
            
        if np.any(substrate_hits):
            substrate_points = ray_batch.intersection_points[hit_indices[substrate_hits]]
            normals[substrate_hits] = self.substrate.get_normals_vectorized(substrate_points)
        
        # Compute incidence angles
        incident_directions = ray_batch.directions[hit_indices]
        cos_incident = np.abs(np.sum(-incident_directions * normals, axis=1))
        cos_incident = np.clip(cos_incident, 0, 1)
        
        # Compute Fresnel reflectance
        reflectances = np.zeros(num_hits)
        
        if np.any(roller_hits):
            incident_angles = np.arccos(cos_incident[roller_hits])
            roller_reflectance = self.fresnel_reflectance_vectorized(
                incident_angles, 1.0, self.roller.refractive_index)
            reflectances[roller_hits] = roller_reflectance
            
        if np.any(substrate_hits):
            incident_angles = np.arccos(cos_incident[substrate_hits])
            substrate_reflectance = self.fresnel_reflectance_vectorized(
                incident_angles, 1.0, self.substrate.refractive_index)
            reflectances[substrate_hits] = substrate_reflectance
        
        # Compute reflected powers
        incident_powers = ray_batch.powers[hit_indices]
        reflected_powers = incident_powers * reflectances
        
        # Filter rays with sufficient power
        power_mask = reflected_powers >= self.min_power_threshold
        
        if not np.any(power_mask):
            return None
            
        # Create reflected ray batch
        reflection_indices = hit_indices[power_mask]
        reflection_normals = normals[power_mask]
        reflection_directions = incident_directions[power_mask]
        reflection_powers = reflected_powers[power_mask]
        reflection_points = ray_batch.intersection_points[reflection_indices]
        
        # Compute reflected directions using vector reflection formula
        # R = I - 2(I·N)N where I is incident direction, N is normal
        dot_products = np.sum(reflection_directions * reflection_normals, axis=1)
        new_directions = (reflection_directions - 
                         2 * dot_products[:, np.newaxis] * reflection_normals)
        
        # Create new origins slightly offset from intersection points to avoid self-intersection
        offset_distance = 1e-9  # Very small offset
        new_origins = reflection_points + offset_distance * new_directions
        
        # Create new generation numbers
        new_generations = ray_batch.generations[reflection_indices] + 1
        
        # All reflected rays start as active
        new_active_mask = np.ones(len(new_origins), dtype=bool)
        
        reflected_batch = VectorizedRayBatch(
            origins=new_origins,
            directions=new_directions,
            powers=reflection_powers,
            generations=new_generations,
            active_mask=new_active_mask
        )
        
        print(f"  Created {len(new_origins)} reflected rays with powers from {np.min(reflection_powers):.3f} to {np.max(reflection_powers):.3f} W")
        
        return reflected_batch
    
    def trace_all_rays_vectorized(self) -> List[VectorizedRayBatch]:
        """Trace all rays with vectorized computation"""
        print(f"Starting vectorized ray tracing with {self.laser.num_rays} rays...")
        
        # Generate initial ray batch
        current_batch = self.laser.generate_ray_batch()
        self.all_ray_batches = [current_batch]
        
        # Debug: Check initial ray setup
        print(f"Initial laser rays: {current_batch.num_rays}")
        print(f"Ray origin range Y: [{np.min(current_batch.origins[:, 0])*1000:.1f}, {np.max(current_batch.origins[:, 0])*1000:.1f}] mm")
        print(f"Ray origin range Z: [{np.min(current_batch.origins[:, 1])*1000:.1f}, {np.max(current_batch.origins[:, 1])*1000:.1f}] mm")
        print(f"Ray direction: [{current_batch.directions[0, 0]:.3f}, {current_batch.directions[0, 1]:.3f}]")
        
        for reflection in range(self.max_reflections):
            print(f"Processing reflection {reflection + 1}/{self.max_reflections}...")
            
            # Find intersections for current batch
            self.find_nearest_intersections_vectorized(current_batch)
            
            # Debug: Check intersection statistics
            roller_hits = np.sum(current_batch.hit_surface_ids == 0)
            substrate_hits = np.sum(current_batch.hit_surface_ids == 1)
            no_hits = np.sum(current_batch.hit_surface_ids == -1)
            
            print(f"  Roller hits: {roller_hits}, Substrate hits: {substrate_hits}, No hits: {no_hits}")
            
            # Compute reflected rays
            reflected_batch = self.compute_reflections_vectorized(current_batch)
            
            if reflected_batch is None or reflected_batch.num_rays == 0:
                print(f"No more reflections after generation {reflection}")
                break
                
            self.all_ray_batches.append(reflected_batch)
            current_batch = reflected_batch
            
            print(f"Generated {reflected_batch.num_rays} reflected rays with total power: "
                  f"{np.sum(reflected_batch.powers):.1f} W")
        
        total_rays = sum(batch.num_rays for batch in self.all_ray_batches)
        print(f"Vectorized ray tracing complete. Total ray segments: {total_rays}")
        
        return self.all_ray_batches
    
    def trace_single_ray_path(self, ray_index: int = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Trace a single ray and return all its reflection segments for visualization"""
        if ray_index is None:
            ray_index = len(self.all_ray_batches[0].origins) // 2  # Middle ray
            
        ray_segments = []
        
        # Start with the initial ray
        if ray_index < len(self.all_ray_batches[0].origins):
            current_batch = self.all_ray_batches[0]
            ray_origin = current_batch.origins[ray_index]
            
            # Check if this ray hit something
            if current_batch.hit_surface_ids[ray_index] >= 0:
                ray_intersection = current_batch.intersection_points[ray_index]
                ray_segments.append((ray_origin, ray_intersection))
                
                # Follow reflections by matching intersection points
                current_intersection = ray_intersection
                
                for batch_idx in range(1, len(self.all_ray_batches)):
                    batch = self.all_ray_batches[batch_idx]
                    
                    # Find ray in this batch that started near the previous intersection
                    distances_to_intersection = np.linalg.norm(
                        batch.origins - current_intersection[np.newaxis, :], axis=1)
                    
                    closest_ray_idx = np.argmin(distances_to_intersection)
                    
                    # Check if this is actually the same ray (within tolerance)
                    if distances_to_intersection[closest_ray_idx] < 1e-8:
                        if batch.hit_surface_ids[closest_ray_idx] >= 0:
                            new_intersection = batch.intersection_points[closest_ray_idx]
                            ray_segments.append((current_intersection, new_intersection))
                            current_intersection = new_intersection
                        else:
                            break  # Ray didn't hit anything, end tracing
                    else:
                        break  # Couldn't find matching reflected ray
        
        return ray_segments
    
    def calculate_irradiance_by_generation_vectorized(self, surface: VectorizedSurface, 
                                            num_points: int = 100) -> Tuple[np.ndarray, List[np.ndarray], float, float, np.ndarray]:
        """Vectorized irradiance calculation on surface, separated by ray generation"""
        if isinstance(surface, VectorizedSubstrate):
            # For substrate (horizontal line) - distance from nip point (0,0)
            y_points = np.linspace(-surface.length, 0, num_points)
            distances_from_nip = np.abs(y_points)  # Distance from (0,0) along substrate
            z_points = np.zeros_like(y_points)
            positions = np.column_stack([y_points, z_points])
            
        elif isinstance(surface, VectorizedCurvedSubstrate):
            # For curved substrate (semicircle) - arc distance from nip point (0,0)
            theta_points = np.linspace(0, np.pi, num_points)  # From 0 to pi (full semicircle)
            y_points = surface.center[0] + surface.radius * np.cos(theta_points)
            z_points = surface.center[1] + surface.radius * np.sin(theta_points)
            positions = np.column_stack([y_points, z_points])
            # Arc distance from nip point (0,0) along the curved substrate surface
            # theta = 0 is at (radius, -radius), theta = pi/2 is nip point (0,0), theta = pi is at (-radius, -radius)
            distances_from_nip = np.abs(theta_points - np.pi/2) * surface.radius
            
        else:  # VectorizedRoller
            # For roller (quarter circle) - arc distance from nip point (0,0)
            theta_points = np.linspace(np.pi, 3*np.pi/2, num_points)  # From pi to 3pi/2
            y_points = surface.center[0] + surface.radius * np.cos(theta_points)
            z_points = surface.center[1] + surface.radius * np.sin(theta_points)
            positions = np.column_stack([y_points, z_points])
            # Arc distance from nip point (0,0) along the roller surface
            # theta = 3*pi/2 is nip point (distance 0), theta = pi is farthest point
            distances_from_nip = (3*np.pi/2 - theta_points) * surface.radius

        # Store irradiance for each generation
        irradiance_by_generation = []
        total_hits = 0
        
        # Process each ray batch (generation) separately
        for batch_idx, batch in enumerate(self.all_ray_batches):
            physical_flux = np.zeros(num_points)
            
            # Find rays that hit the target surface
            surface_hits = batch.hit_surface_ids == surface.surface_id
            num_hits = np.sum(surface_hits)
            
            if num_hits == 0:
                irradiance_by_generation.append(physical_flux)
                continue
                
            total_hits += num_hits
            hit_points = batch.intersection_points[surface_hits]
            hit_powers = batch.powers[surface_hits]
            hit_directions = batch.directions[surface_hits]
            
            # Remove any NaN intersection points
            valid_hits = ~np.isnan(hit_points).any(axis=1)
            if not np.any(valid_hits):
                irradiance_by_generation.append(physical_flux)
                continue
                
            hit_points = hit_points[valid_hits]
            hit_powers = hit_powers[valid_hits]
            hit_directions = hit_directions[valid_hits]
            
            # Calculate surface normals at hit points
            surface_normals = surface.get_normals_vectorized(hit_points)
            
            # Calculate incidence angles and Fresnel reflectance
            cos_incident = np.abs(np.sum(-hit_directions * surface_normals, axis=1))
            cos_incident = np.clip(cos_incident, 0, 1)
            
            # Calculate incident angles and Fresnel reflectance for each ray
            incident_angles = np.arccos(cos_incident)
            reflectances = self.fresnel_reflectance_vectorized(incident_angles, 1.0, surface.refractive_index)
            
            # Calculate absorption fraction for each ray (1 - reflectance)
            absorption_fractions = 1.0 - reflectances
            
            # Find nearest discretization points for each hit
            if len(hit_points) > 0:
                distances = np.linalg.norm(
                    hit_points[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
                nearest_indices = np.argmin(distances, axis=1)
                
                # Calculate absorbed power at each point
                for i, power, abs_frac in zip(nearest_indices, hit_powers, absorption_fractions):
                    absorbed_power = power * abs_frac
                    physical_flux[i] += absorbed_power

            irradiance_by_generation.append(physical_flux)
            print(f"Generation {batch_idx} on surface {surface.surface_id}: {num_hits} hits, max flux = {np.max(physical_flux):.3f} W")

        # Calculate total physical flux
        total_irradiance = sum(irradiance_by_generation)
        
        # Calculate shadow length (distance to first non-zero flux from nip point)
        shadow_length = 0.0
        flux_threshold = np.max(total_irradiance) * 0.01 if np.max(total_irradiance) > 0 else 0
        significant_flux_mask = total_irradiance > flux_threshold
        
        if np.any(significant_flux_mask):
            # Find minimum distance among points with significant flux
            significant_distances = distances_from_nip[significant_flux_mask]
            shadow_length = np.min(significant_distances)
        
        # Find maximum extent of flux (last non-null position) for plot limits
        max_extent_position = 0.0
        if np.any(significant_flux_mask):
            # Find maximum distance among points with significant flux
            significant_distances = distances_from_nip[significant_flux_mask]
            max_extent_position = np.max(significant_distances)
        
        print(f"Surface {surface.surface_id}: Total hits = {total_hits}")
        if total_hits > 0:
            # Calculate average absorption fraction for reporting
            avg_absorption = 0.0
            total_weighted_absorption = 0.0
            total_power = 0.0
            
            for batch_idx, batch in enumerate(self.all_ray_batches):
                surface_hits = batch.hit_surface_ids == surface.surface_id
                if np.any(surface_hits):
                    hit_points = batch.intersection_points[surface_hits]
                    hit_powers = batch.powers[surface_hits]
                    hit_directions = batch.directions[surface_hits]
                    
                    valid_hits = ~np.isnan(hit_points).any(axis=1)
                    if np.any(valid_hits):
                        hit_points = hit_points[valid_hits]
                        hit_powers = hit_powers[valid_hits]
                        hit_directions = hit_directions[valid_hits]
                        
                        surface_normals = surface.get_normals_vectorized(hit_points)
                        cos_incident = np.abs(np.sum(-hit_directions * surface_normals, axis=1))
                        cos_incident = np.clip(cos_incident, 0, 1)
                        
                        incident_angles = np.arccos(cos_incident)
                        reflectances = self.fresnel_reflectance_vectorized(incident_angles, 1.0, surface.refractive_index)
                        absorption_fractions = 1.0 - reflectances
                        
                        total_weighted_absorption += np.sum(absorption_fractions * hit_powers)
                        total_power += np.sum(hit_powers)
            
            if total_power > 0:
                avg_absorption = total_weighted_absorption / total_power
            
            print(f"  Average absorption fraction: {avg_absorption:.3f}")
        print(f"  Shadow length = {shadow_length*1000:.1f}mm, Max extent = {max_extent_position*1000:.1f}mm")
        print(f"  Distance range: {np.min(distances_from_nip)*1000:.1f} to {np.max(distances_from_nip)*1000:.1f}mm")
        print(f"  Flux range: {np.min(total_irradiance):.3f} to {np.max(total_irradiance):.3f} W")
        
        return distances_from_nip, irradiance_by_generation, shadow_length, max_extent_position, total_irradiance 
        
    def export_flux_data(self, substrate_dist, substrate_total_flux, substrate_extent,
                    roller_dist, roller_total_flux, roller_extent, filename="flux_data.txt"):
        """Export physical flux data to text file with 20 points resolution"""
        
        print(f"Exporting physical flux data...")
        
        # Find non-zero flux indices
        substrate_nonzero_indices = np.where(substrate_total_flux > 0)[0]
        roller_nonzero_indices = np.where(roller_total_flux > 0)[0]
        
        # Extract and sort substrate data for interpolation
        if len(substrate_nonzero_indices) > 0:
            substrate_flux_distances = substrate_dist[substrate_nonzero_indices]
            substrate_flux_values = substrate_total_flux[substrate_nonzero_indices]
            
            # Sort by distance for proper interpolation
            substrate_sort_indices = np.argsort(substrate_flux_distances)
            substrate_sorted_dist = substrate_flux_distances[substrate_sort_indices]
            substrate_sorted_flux = substrate_flux_values[substrate_sort_indices]
            
            substrate_min_dist = np.min(substrate_sorted_dist)
            substrate_max_dist = np.max(substrate_sorted_dist)
        else:
            substrate_min_dist = 0
            substrate_max_dist = substrate_extent
            substrate_sorted_dist = np.array([0, substrate_extent])
            substrate_sorted_flux = np.array([0, 0])
        
        # Extract and sort roller data for interpolation
        if len(roller_nonzero_indices) > 0:
            roller_flux_distances = roller_dist[roller_nonzero_indices]
            roller_flux_values = roller_total_flux[roller_nonzero_indices]
            
            # Sort by distance for proper interpolation
            roller_sort_indices = np.argsort(roller_flux_distances)
            roller_sorted_dist = roller_flux_distances[roller_sort_indices]
            roller_sorted_flux = roller_flux_values[roller_sort_indices]
            
            roller_min_dist = np.min(roller_sorted_dist)
            roller_max_dist = np.max(roller_sorted_dist)
        else:
            roller_min_dist = 0
            roller_max_dist = roller_extent
            roller_sorted_dist = np.array([0, roller_extent])
            roller_sorted_flux = np.array([0, 0])
        
        print(f"Debug - Substrate actual flux range: {substrate_min_dist:.6f} to {substrate_max_dist:.6f}")
        print(f"Debug - Roller actual flux range: {roller_min_dist:.6f} to {roller_max_dist:.6f}")
        
        # Create 20 points within the actual flux range
        substrate_positions = np.linspace(substrate_min_dist, substrate_max_dist, 20)
        roller_positions = np.linspace(roller_min_dist, roller_max_dist, 20)
        
        # Interpolate using sorted data
        substrate_flux = np.interp(substrate_positions, substrate_sorted_dist, substrate_sorted_flux)
        roller_flux = np.interp(roller_positions, roller_sorted_dist, roller_sorted_flux)
        
        print(f"Debug - Interpolated substrate flux range: {np.min(substrate_flux):.3f} to {np.max(substrate_flux):.3f} W")
        print(f"Debug - Interpolated roller flux range: {np.min(roller_flux):.3f} to {np.max(roller_flux):.3f} W")
        
        # Convert substrate positions to negative values (distance from nip point towards substrate)
        substrate_positions_negative = -substrate_positions
        substrate_positions_negative = substrate_positions_negative[::-1]  # Reverse order
        #substrate_flux = substrate_flux[::-1]  # Reverse flux values to match positions
  
        roller_positions_negative = -roller_positions 
        roller_positions_negative = roller_positions_negative[::-1]
        
        # Write to file
        self._write_flux_file(filename, substrate_positions_negative, substrate_flux, 
                            roller_positions_negative, roller_flux)
        
        print(f"Physical flux data exported to {filename}")
        print(f"  Substrate max flux: {np.max(substrate_flux):.3f} W")
        print(f"  Roller max flux: {np.max(roller_flux):.3f} W")

    def _write_flux_file(self, filename, substrate_positions_negative, substrate_flux,
                        roller_positions, roller_flux):
        """Write flux data to file with proper formatting"""
        with open(filename, 'w') as f:
            f.write("# Physical Heat Flux Data\n")
            f.write("# Values represent absorbed power (W)\n")
            f.write("#\n")
            f.write("# SUBSTRATE DATA\n")
            f.write("# Positions (m):\n")
            
            self._write_data_section(f, substrate_positions_negative, ".3f")
            f.write("# Flux values (W):\n")
            self._write_data_section(f, substrate_flux, ".3f")
            
            f.write("#\n")
            f.write("# INCOMING TAPE DATA\n")
            f.write("# Positions (m):\n")
            
            self._write_data_section(f, roller_positions, ".3f")
            f.write("# Flux values (W):\n")
            self._write_data_section(f, roller_flux, ".3f")

    def _write_data_section(self, file_handle, data_array, format_spec):
        """Write formatted data array to file with line breaks"""
        data_str = ", ".join([f"{val:{format_spec}}" for val in data_array])
        data_parts = data_str.split(", ")
        
        for i in range(0, len(data_parts), 6):
            line_parts = data_parts[i:i+6]
            if i == 0:
                file_handle.write(" " + ", ".join(line_parts))
            else:
                file_handle.write(",\n        " + ", ".join(line_parts))
        file_handle.write("\n")

    def verify_power_independence(self, test_power: float) -> bool:
        """
        Verify that the distribution shape is independent of laser power.
        
        Args:
            test_power: Power level to test (W)
            
        Returns:
            True if the normalized distributions match within tolerance
        """
        # Create a test laser with different power
        test_laser = VectorizedLaser(
            source_length=self.laser.source_length,
            source_center=self.laser.source_center,
            source_angle=np.degrees(self.laser.source_angle),
            num_rays=self.laser.num_rays,
            total_power=test_power
        )
        
        # Create test tracer with same relative threshold
        test_tracer = VectorizedRayTracer(
            test_laser, self.roller, self.substrate, 
            self.max_reflections, 
            min_power_threshold_fraction=self.min_power_threshold / self.laser.total_power
        )
        
        # Trace rays for both cases
        original_batches = self.trace_all_rays_vectorized()
        test_batches = test_tracer.trace_all_rays_vectorized()
        
        # Calculate normalized distributions
        _, _, _, _, original_flux = self.calculate_irradiance_by_generation_vectorized(self.substrate, 200)
        _, _, _, _, test_flux = test_tracer.calculate_irradiance_by_generation_vectorized(self.substrate, 200)
        
        # Normalize by total power
        original_normalized = original_flux / self.laser.total_power
        test_normalized = test_flux / test_power
        
        # Check if shapes match (within 5% tolerance)
        max_diff = np.max(np.abs(original_normalized - test_normalized))
        max_original = np.max(original_normalized)
        relative_error = max_diff / max_original if max_original > 0 else 0
        
        print(f"Power independence verification:")
        print(f"  Original power: {self.laser.total_power} W")
        print(f"  Test power: {test_power} W")
        print(f"  Max relative error: {relative_error:.2%}")
        print(f"  Threshold: {relative_error < 0.05}")
        
        return relative_error < 0.05
        
def run_vectorized_example():
    """Run vectorized example simulation and create plots"""
    
    # Define system parameters
    laser = VectorizedLaser(
        source_length=30e-3,           # 30 mm
        source_center=np.array([-300e-3, 97e-3]),  # 300 mm left, 97 mm up
        source_angle=20.0,             # 20 degrees
        num_rays=10000,                # 10000 rays for high resolution
        total_power=1.0             # 1000 W
    )
    
    roller = VectorizedRoller(radius=35e-3, refractive_index=1.8)        # 35 mm radius
    #substrate = VectorizedSubstrate(length=100e-3, refractive_index=1.8) # 100 mm length
    substrate = VectorizedCurvedSubstrate(radius=200e-3, refractive_index=1.8) # 200 mm radius, curved substrate
    
    # Create vectorized ray tracer with RELATIVE threshold for power independence
    tracer = VectorizedRayTracer(laser, roller, substrate, max_reflections=6, min_power_threshold_fraction=1e-6)
    
    # Print threshold information
    print(f"Laser power: {laser.total_power} W")
    print(f"Power per ray: {laser.power_per_ray:.6f} W")
    print(f"Relative threshold fraction: 1e-6")
    print(f"Absolute threshold: {tracer.min_power_threshold:.2e} W")
    print(f"This ensures same distribution shape regardless of total power!")
    
    # Trace all rays with vectorization
    import time
    start_time = time.time()
    ray_batches = tracer.trace_all_rays_vectorized()
    end_time = time.time()
    
    print(f"Vectorized computation time: {end_time - start_time:.3f} seconds")
    
    # Get single ray path for visualization
    single_ray_path = tracer.trace_single_ray_path()
    print(f"Single ray path has {len(single_ray_path)} segments")
    
    # Calculate irradiance by generation
    print("Calculating irradiance distributions by generation...")
    substrate_dist, substrate_irradiance_gen, substrate_shadow, substrate_max_extent, substrate_total_physical = tracer.calculate_irradiance_by_generation_vectorized(
        substrate, 1000)
    roller_dist, roller_irradiance_gen, roller_shadow, roller_max_extent, roller_total_physical = tracer.calculate_irradiance_by_generation_vectorized(
        roller, 100)
 
    # Create plots - main plot on top, irradiance plots below
    fig = plt.figure(figsize=(15, 12))
    
    # Main ray tracing plot (top, spanning full width)
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    
    # Irradiance plots (bottom row)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    
    # Plot 1: Ray tracing visualization with laser source line (TOP)
    ax1.set_aspect('equal')
    
    # Plot surfaces
    roller_plot_y, roller_plot_z = roller.get_points_for_plotting()
    substrate_plot_y, substrate_plot_z = substrate.get_points_for_plotting()
    
    ax1.plot(roller_plot_y * 1000, roller_plot_z * 1000, 'b-', linewidth=3, label='Incoming Tape')
    ax1.plot(substrate_plot_y * 1000, substrate_plot_z * 1000, 'g-', linewidth=3, label='Substrate')
    
    # Plot laser source line
    initial_batch = ray_batches[0]
    source_start = initial_batch.origins[0]
    source_end = initial_batch.origins[-1]
    ax1.plot([source_start[0] * 1000, source_end[0] * 1000], 
             [source_start[1] * 1000, source_end[1] * 1000], 
             'r-', linewidth=4, label='Laser Source')
    
    # Plot ray paths (sample every 50th ray to avoid clutter)
    colors = plt.cm.viridis(np.linspace(0, 1, len(ray_batches)))
    
    for batch_idx, batch in enumerate(ray_batches):  # Show all reflections
        if batch_idx == 0:
            step = 100  # Plot every 100th ray for initial batch (more selective with 5000 rays)
        else:
            step = max(1, batch.num_rays // 10)  # Plot up to 10 rays for reflections
            
        sample_indices = np.arange(0, batch.num_rays, step)
        
        for i in sample_indices:
            if batch.hit_surface_ids[i] >= 0:  # Only plot rays that hit something
                y_coords = [batch.origins[i, 0], batch.intersection_points[i, 0]]
                z_coords = [batch.origins[i, 1], batch.intersection_points[i, 1]]
                ax1.plot(np.array(y_coords) * 1000, np.array(z_coords) * 1000, 
                        color=colors[batch_idx], alpha=0.4, linewidth=0.4,
                        label=f'Generation {batch_idx}' if i == sample_indices[0] else '')
    
    # Plot single ray path with distinctive style
    if single_ray_path:
        for segment_idx, (start_point, end_point) in enumerate(single_ray_path):
            y_coords = [start_point[0], end_point[0]]
            z_coords = [start_point[1], end_point[1]]
            ax1.plot(np.array(y_coords) * 1000, np.array(z_coords) * 1000, 
                    'r-', linewidth=3, alpha=0.9, 
                    label=f'Single Ray Path' if segment_idx == 0 else '')
        
        # Mark source point of traced ray
        if len(single_ray_path) > 0:
            source_point = single_ray_path[0][0]
            ax1.plot(source_point[0] * 1000, source_point[1] * 1000, 
                    'ro', markersize=8, label='Traced Ray Source')
    
    # Mark the nip point (0,0)
    ax1.plot(0, 0, 'ko', markersize=8, label='Nip Point')
    
    # Add text with ray count, power, and shadow lengths
    text_str = (f'Rays: {laser.num_rays:,}\nPower: {laser.total_power:.0f} W\n'
                f'Reflections: {len(ray_batches)-1}\n'
                f'Flux: Physical (W)\n'
                f'Substrate Shadow: {substrate_shadow*1000:.1f} mm\n'
                f'Incoming Tape Shadow: {roller_shadow*1000:.1f} mm')
    ax1.text(0.02, 0.98, text_str, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Y Position (mm)')
    ax1.set_ylabel('Z Position (mm)')
    ax1.set_title('Vectorized Ray Tracing Visualization')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Define specific colors for generations
    generation_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']
    
    # Plot 2: Substrate irradiance by generation
    total_substrate = np.zeros_like(substrate_dist)
    
    for gen_idx, irradiance in enumerate(substrate_irradiance_gen):
        if np.max(irradiance) > 0:  # Only plot non-zero irradiance
            if gen_idx == 0:
                label = 'Direct'
                color = generation_colors[0]  # red
            elif gen_idx == 1:
                label = '1st Reflection'
                color = generation_colors[1]  # green
            elif gen_idx == 2:
                label = '2nd Reflection'
                color = generation_colors[2]  # blue
            else:
                label = f'Reflection {gen_idx}'
                color = generation_colors[min(gen_idx, len(generation_colors)-1)]
                
            ax2.plot(substrate_dist * 1000, irradiance, 
                    color=color, linewidth=2, label=label)
            total_substrate += irradiance
    
    # Plot total with black solid line
    ax2.plot(substrate_dist * 1000, total_substrate, 'k-', linewidth=2, label='Total')
    
    # Set x-axis limit to max extent position + 5mm
    substrate_xlim = (substrate_max_extent + 5e-3) * 1000  # Convert to mm
    ax2.set_xlim(0, substrate_xlim)
    
    ax2.set_xlabel('Distance from Nip Point (mm)')
    ax2.set_ylabel('Absorbed Power (W)')
    ax2.set_title('Substrate Irradiance by Generation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Incoming Tape irradiance by generation
    total_roller = np.zeros_like(roller_dist)
    
    for gen_idx, irradiance in enumerate(roller_irradiance_gen):
        if np.max(irradiance) > 0:  # Only plot non-zero irradiance
            if gen_idx == 0:
                label = 'Direct'
                color = generation_colors[0]  # red
            elif gen_idx == 1:
                label = '1st Reflection'
                color = generation_colors[1]  # green
            elif gen_idx == 2:
                label = '2nd Reflection'
                color = generation_colors[2]  # blue
            else:
                label = f'Reflection {gen_idx}'
                color = generation_colors[min(gen_idx, len(generation_colors)-1)]
                
            ax3.plot(roller_dist * 1000, irradiance, 
                    color=color, linewidth=2, label=label)
            total_roller += irradiance
    
    # Plot total with black solid line
    ax3.plot(roller_dist * 1000, total_roller, 'k-', linewidth=2, label='Total')
    
    # Set x-axis limit to max extent position + 5mm
    roller_xlim = (roller_max_extent + 5e-3) * 1000  # Convert to mm
    ax3.set_xlim(0, roller_xlim)
    
    ax3.set_xlabel('Arc Distance from Nip Point (mm)')
    ax3.set_ylabel('Absorbed Power (W)')
    ax3.set_title('Incoming Tape Irradiance by Generation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print performance and summary statistics
    print(f"\nVectorized Simulation Summary:")
    print(f"Laser source position: Y={laser.source_center[0]*1000:.1f}mm, Z={laser.source_center[1]*1000:.1f}mm")
    print(f"Computation time: {end_time - start_time:.3f} seconds")
    print(f"Total rays: {laser.num_rays}")
    print(f"Total ray segments across all reflections: {sum(batch.num_rays for batch in ray_batches)}")
    print(f"Total laser power: {laser.total_power:.1f} W")
    print(f"Power per ray: {laser.power_per_ray:.3f} W")
    print(f"Single ray traced through {len(single_ray_path)} reflections")
    print(f"Flux type: Physical (W)")
    print(f"Shadow lengths: Substrate = {substrate_shadow*1000:.1f}mm, Incoming Tape = {roller_shadow*1000:.1f}mm")
    print(f"Max extent positions: Substrate = {substrate_max_extent*1000:.1f}mm, Incoming Tape = {roller_max_extent*1000:.1f}mm")
    
    # Print irradiance statistics by generation
    print(f"\nSubstrate Irradiance by Generation:")
    substrate_total_irr = sum(substrate_irradiance_gen)
    for gen_idx, irradiance in enumerate(substrate_irradiance_gen):
        max_irr = np.max(irradiance)
        if max_irr > 0:
            gen_name = 'Direct' if gen_idx == 0 else f'Reflection {gen_idx}'
            total_power = np.sum(irradiance)
            print(f"  {gen_name}: Max = {max_irr:.1f} W, Total = {total_power:.1f} W")
    
    print(f"  Total Max = {np.max(substrate_total_irr):.1f} W")
    
    print(f"\nIncoming Tape Irradiance by Generation:")
    roller_total_irr = sum(roller_irradiance_gen)
    for gen_idx, irradiance in enumerate(roller_irradiance_gen):
        max_irr = np.max(irradiance)
        if max_irr > 0:
            gen_name = 'Direct' if gen_idx == 0 else f'Reflection {gen_idx}'
            total_power = np.sum(irradiance)
            print(f"  {gen_name}: Max = {max_irr:.1f} W, Total = {total_power:.1f} W")
    
    print(f"  Total Max = {np.max(roller_total_irr):.1f} W")
    
    # Power conservation check
    total_initial_power = np.sum(ray_batches[0].powers)
    total_final_power = sum(np.sum(batch.powers) for batch in ray_batches[1:])
    print(f"Power conservation: Initial={total_initial_power:.1f}W, "
          f"Reflected={total_final_power:.1f}W")
    
    # Export flux data to text file
    tracer.export_flux_data(
        substrate_dist, total_substrate, substrate_max_extent,
        roller_dist, total_roller, roller_max_extent,
        "flux_data.txt"
    )

if __name__ == "__main__":

    # Run main example with power-independent ray tracing
    run_vectorized_example()