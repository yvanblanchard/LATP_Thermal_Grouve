"""Tests for VectorizedPolygonalSubstrate and related ray tracing components."""
import numpy as np
import pytest
from ray_tracing_2d import (
    VectorizedRayBatch,
    VectorizedSubstrate,
    VectorizedPolygonalSubstrate,
    VectorizedRoller,
    VectorizedLaser,
    VectorizedRayTracer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ray_batch(origins, directions):
    """Convenience wrapper that creates a VectorizedRayBatch from plain arrays."""
    return VectorizedRayBatch(
        origins=np.asarray(origins, dtype=float),
        directions=np.asarray(directions, dtype=float),
        powers=np.ones(len(origins)),
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestPolygonalSubstrateConstruction:
    def test_requires_at_least_two_vertices(self):
        with pytest.raises(ValueError):
            VectorizedPolygonalSubstrate(vertices=np.array([[0.0, 0.0]]))

    def test_flat_polygon_matches_flat_substrate_attributes(self):
        """A single-segment horizontal polygon should mirror VectorizedSubstrate."""
        length = 100e-3
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-length, 0.0], [0.0, 0.0]]))
        assert poly.num_segments == 1
        assert poly.surface_id == 1
        np.testing.assert_allclose(poly.total_length, length, rtol=1e-10)

    def test_vgroove_has_two_segments(self):
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [-0.05, -0.02], [0.0, 0.0]]))
        assert poly.num_segments == 2

    def test_cumulative_lengths_are_monotone(self):
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [-0.05, -0.02], [0.0, 0.0]]))
        assert np.all(np.diff(poly.cumulative_lengths) > 0)


# ---------------------------------------------------------------------------
# Normal tests
# ---------------------------------------------------------------------------

class TestPolygonalSubstrateNormals:
    def test_flat_horizontal_normal_points_up(self):
        """Horizontal flat polygon → normal must be [0, 1]."""
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [0.0, 0.0]]))
        points = np.array([[-0.05, 0.0]])
        normals = poly.get_normals_vectorized(points)
        np.testing.assert_allclose(normals[0], [0.0, 1.0], atol=1e-10)

    def test_normal_has_unit_length(self):
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [-0.05, -0.02], [0.0, 0.0]]))
        points = np.array([[-0.08, -0.004], [-0.02, -0.008]])
        normals = poly.get_normals_vectorized(points)
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(lengths, np.ones(len(points)), atol=1e-10)

    def test_normals_z_component_nonnegative(self):
        """All normals on a convex (non-overhanging) substrate must have z ≥ 0."""
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [-0.05, -0.02], [0.0, 0.0]]))
        # Sample a few points along the surface
        positions, _ = poly.get_arc_positions(20)
        normals = poly.get_normals_vectorized(positions)
        assert np.all(normals[:, 1] >= -1e-12), "Some normals point downward."


# ---------------------------------------------------------------------------
# Intersection tests
# ---------------------------------------------------------------------------

class TestPolygonalSubstrateIntersection:
    def test_vertical_ray_hits_flat_horizontal_polygon(self):
        """A ray shooting straight down should hit z=0 substrate."""
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [0.0, 0.0]]))
        batch = make_ray_batch(
            origins=[[-0.05, 0.05]],
            directions=[[0.0, -1.0]],
        )
        hit_mask, pts, dists = poly.intersect_rays_vectorized(batch)
        assert hit_mask[0], "Expected a hit on horizontal polygon."
        np.testing.assert_allclose(pts[0], [-0.05, 0.0], atol=1e-10)
        np.testing.assert_allclose(dists[0], 0.05, atol=1e-10)

    def test_ray_outside_polygon_does_not_hit(self):
        """A ray shooting down outside the substrate extent must not hit."""
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [0.0, 0.0]]))
        batch = make_ray_batch(
            origins=[[0.05, 0.05]],   # y = 0.05, outside substrate y-extent [-0.1, 0]
            directions=[[0.0, -1.0]],
        )
        hit_mask, _, _ = poly.intersect_rays_vectorized(batch)
        assert not hit_mask[0], "Ray outside substrate should not hit."

    def test_ray_hits_first_segment_of_vgroove(self):
        """Ray aimed at the left slope of a V-groove must hit that segment."""
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [-0.05, -0.02], [0.0, 0.0]]))
        # Midpoint of first segment: (-0.075, -0.01)
        batch = make_ray_batch(
            origins=[[-0.075, 0.05]],
            directions=[[0.0, -1.0]],
        )
        hit_mask, pts, dists = poly.intersect_rays_vectorized(batch)
        assert hit_mask[0], "Expected a hit on the first V-groove slope."
        np.testing.assert_allclose(pts[0, 0], -0.075, atol=1e-8)
        np.testing.assert_allclose(pts[0, 1], -0.01, atol=1e-8)

    def test_closest_hit_returned_for_two_parallel_segments(self):
        """When a ray could hit two segments, the nearer one is chosen."""
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [-0.05, -0.02], [0.0, 0.0]]))
        # Vertical ray at y = 0.0: passes through nip point (end vertex)
        batch = make_ray_batch(
            origins=[[0.0, 0.05]],
            directions=[[0.0, -1.0]],
        )
        hit_mask, pts, dists = poly.intersect_rays_vectorized(batch)
        # Should hit the last segment (second slope) at (0, 0)
        assert hit_mask[0]
        np.testing.assert_allclose(pts[0], [0.0, 0.0], atol=1e-8)

    def test_backward_ray_does_not_hit(self):
        """A ray pointing away from the substrate must not register a hit."""
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [0.0, 0.0]]))
        batch = make_ray_batch(
            origins=[[-0.05, -0.05]],   # below substrate
            directions=[[0.0, -1.0]],   # going further down
        )
        hit_mask, _, _ = poly.intersect_rays_vectorized(batch)
        assert not hit_mask[0]

    def test_empty_ray_batch_returns_empty(self):
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [0.0, 0.0]]))
        batch = make_ray_batch(
            origins=np.empty((0, 2)),
            directions=np.empty((0, 2)),
        )
        hit_mask, pts, dists = poly.intersect_rays_vectorized(batch)
        assert len(hit_mask) == 0
        assert pts.shape == (0, 2)
        assert len(dists) == 0


# ---------------------------------------------------------------------------
# Arc position tests
# ---------------------------------------------------------------------------

class TestPolygonalSubstrateArcPositions:
    def test_arc_positions_start_and_end(self):
        """First position should be near first vertex; last near nip point."""
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [0.0, 0.0]]))
        positions, distances = poly.get_arc_positions(50)
        np.testing.assert_allclose(positions[0], [-0.1, 0.0], atol=1e-10)
        np.testing.assert_allclose(positions[-1], [0.0, 0.0], atol=1e-10)

    def test_distance_from_nip_decreases(self):
        """Arc distances from the nip point should decrease along the polygon."""
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [0.0, 0.0]]))
        _, distances = poly.get_arc_positions(50)
        assert np.all(np.diff(distances) <= 1e-12)

    def test_distance_at_nip_is_zero(self):
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [0.0, 0.0]]))
        _, distances = poly.get_arc_positions(50)
        np.testing.assert_allclose(distances[-1], 0.0, atol=1e-12)

    def test_correct_number_of_positions(self):
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [-0.05, -0.02], [0.0, 0.0]]))
        positions, distances = poly.get_arc_positions(100)
        assert len(positions) == 100
        assert len(distances) == 100


# ---------------------------------------------------------------------------
# Flat-polygon vs VectorizedSubstrate equivalence
# ---------------------------------------------------------------------------

class TestFlatPolygonEquivalence:
    """A single-segment horizontal polygon must behave identically to VectorizedSubstrate."""

    def setup_method(self):
        length = 100e-3
        self.flat = VectorizedSubstrate(length=length, refractive_index=1.5)
        self.poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-length, 0.0], [0.0, 0.0]]),
            refractive_index=1.5)

    def test_same_hits_for_vertical_rays(self):
        y_origins = np.linspace(-0.09, -0.01, 20)
        origins    = np.column_stack([y_origins, np.full(20, 0.05)])
        directions = np.tile([0.0, -1.0], (20, 1))
        batch_flat = make_ray_batch(origins, directions)
        batch_poly = make_ray_batch(origins, directions)

        hit_flat, pts_flat, _ = self.flat.intersect_rays_vectorized(batch_flat)
        hit_poly, pts_poly, _ = self.poly.intersect_rays_vectorized(batch_poly)

        np.testing.assert_array_equal(hit_flat, hit_poly)
        np.testing.assert_allclose(pts_flat[hit_flat], pts_poly[hit_poly], atol=1e-10)

    def test_same_normals_for_surface_points(self):
        points = np.column_stack([np.linspace(-0.09, -0.01, 10),
                                   np.zeros(10)])
        n_flat = self.flat.get_normals_vectorized(points)
        n_poly = self.poly.get_normals_vectorized(points)
        np.testing.assert_allclose(n_flat, n_poly, atol=1e-10)


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------

class TestPolygonalSubstrateIntegration:
    """Run a small full simulation with a V-groove polygonal substrate."""

    def test_vgroove_simulation_runs_without_error(self):
        laser = VectorizedLaser(
            source_length=30e-3,
            source_center=np.array([-150e-3, 60e-3]),
            source_angle=22.0,
            num_rays=500,
            total_power=1.0,
        )
        roller = VectorizedRoller(radius=40e-3, refractive_index=1.8)
        half = 50e-3
        depth = half * np.tan(np.radians(20))
        substrate = VectorizedPolygonalSubstrate(
            vertices=np.array([[-2 * half, 0.0], [-half, -depth], [0.0, 0.0]]),
            refractive_index=1.5,
        )
        tracer = VectorizedRayTracer(laser, roller, substrate,
                                     max_reflections=2,
                                     min_power_threshold_fraction=1e-6)
        tracer.trace_all_rays_vectorized()

        dist, irr_gen, shadow, extent, total = tracer.calculate_irradiance_by_generation_vectorized(substrate, 200)

        assert len(dist) == 200
        assert len(total) == 200
        assert np.any(total > 0), "No flux recorded on V-groove substrate."
        assert extent >= 0

    def test_polygonal_substrate_surface_id_is_1(self):
        poly = VectorizedPolygonalSubstrate(
            vertices=np.array([[-0.1, 0.0], [0.0, 0.0]]))
        assert poly.surface_id == 1
