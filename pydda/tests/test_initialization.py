#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:50:58 2018

@author: rjackson
"""

import pydda
import pyart
import numpy as np
import pytest
from datetime import datetime


def test_make_const_wind_field():
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 10000), (-10000, 10000), (-10000, 10000))
    )

    # a zero field
    fdata3 = np.zeros((20, 20, 20))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)

    Grid = pydda.initialization.make_constant_wind_field(
        Grid, wind=(2.0, 3.0, 4.0), vel_field="zero_field"
    )

    assert np.all(Grid["u"].values == 2.0)
    assert np.all(Grid["v"].values == 3.0)
    assert np.all(Grid["w"].values == 4.0)


def test_make_wind_field_from_profile():
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 10000), (-10000, 10000), (-10000, 10000))
    )

    # a zero field
    fdata3 = np.zeros((20, 20, 20))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)
    height = np.arange(0, 10000, 100)
    u_sound = np.ones(height.shape)
    v_sound = np.ones(height.shape)

    profile = pyart.core.HorizontalWindProfile.from_u_and_v(height, u_sound, v_sound)

    Grid = pydda.initialization.make_wind_field_from_profile(
        Grid, profile, vel_field="zero_field"
    )

    assert np.all(np.round(Grid["u"].values) == 1)
    assert np.all(np.round(Grid["v"].values) == 1)
    assert np.all(Grid["w"].values == 0.0)


def _make_small_grid():
    """5x5x5 Cartesian grid for fast IDW tests."""
    Grid = pyart.testing.make_empty_grid(
        (5, 5, 5), ((0, 10000), (-5000, 5000), (-5000, 5000))
    )
    Grid.add_field("zero_field", {"data": np.zeros((5, 5, 5)), "_FillValue": -9999.0})
    return pydda.io.read_from_pyart_grid(Grid)


def test_iem_idw_single_station_no_sounding():
    """With one station and no sounding, all grid points inherit the station wind."""
    Grid = _make_small_grid()
    station_obs = [{"x": 0.0, "y": 0.0, "z": 0.0, "u": 5.0, "v": 3.0, "w": 0.0}]
    Grid = pydda.initialization.make_initialization_from_iem_obs(Grid, station_obs)
    # One station → all normalised IDW weights = 1 → uniform result everywhere
    assert np.allclose(Grid["u"].values, 5.0)
    assert np.allclose(Grid["v"].values, 3.0)
    assert np.all(Grid["w"].values == 0.0)


def test_iem_idw_sounding_only():
    """Profile only, no stations: output should match sounding at each z level."""
    Grid = _make_small_grid()
    grid_z = Grid["z"].values
    u_snd = grid_z / 1000.0  # linearly increasing with height
    v_snd = np.full_like(grid_z, 2.0)
    profile = pyart.core.HorizontalWindProfile.from_u_and_v(grid_z, u_snd, v_snd)

    Grid = pydda.initialization.make_initialization_from_iem_obs(
        Grid, [], profile=profile
    )

    for iz in range(len(grid_z)):
        assert np.allclose(Grid["u"].values[0, iz, :, :], u_snd[iz])
        assert np.allclose(Grid["v"].values[0, iz, :, :], v_snd[iz])
    assert np.all(Grid["w"].values == 0.0)


def test_iem_idw_sounding_plus_station_anomaly():
    """Station anomaly is added on top of the sounding background at every level."""
    Grid = _make_small_grid()
    grid_z = Grid["z"].values
    u_back = 3.0
    profile = pyart.core.HorizontalWindProfile.from_u_and_v(
        grid_z, np.full_like(grid_z, u_back), np.zeros_like(grid_z)
    )
    # Station u=7 at z=0; sounding at z=0 is 3; anomaly = +4
    station_obs = [{"x": 0.0, "y": 0.0, "z": 0.0, "u": 7.0, "v": 0.0, "w": 0.0}]
    Grid = pydda.initialization.make_initialization_from_iem_obs(
        Grid, station_obs, profile=profile
    )
    # Single station → anomaly spread is uniform → u = 3 + 4 = 7 everywhere
    assert np.allclose(Grid["u"].values, u_back + 4.0)
    assert np.all(Grid["w"].values == 0.0)


def test_iem_idw_midpoint_average():
    """Two equidistant stations: the midpoint should receive their average."""
    Grid = _make_small_grid()
    grid_x = Grid["x"].values  # e.g. [-5000, -2500, 0, 2500, 5000]
    xa, xb = float(grid_x[0]), float(grid_x[-1])
    assert np.isclose(abs(xa), abs(xb)), "grid must be symmetric about x=0"
    ny2, nx2 = len(Grid["y"].values) // 2, len(Grid["x"].values) // 2

    station_obs = [
        {"x": xa, "y": 0.0, "z": 0.0, "u": 10.0, "v": 0.0, "w": 0.0},
        {"x": xb, "y": 0.0, "z": 0.0, "u": 0.0, "v": 0.0, "w": 0.0},
    ]
    Grid = pydda.initialization.make_initialization_from_iem_obs(Grid, station_obs)
    # At (x=0, y=0): equal distances to both stations → average u = 5
    u_mid = Grid["u"].values[0, :, ny2, nx2]
    assert np.allclose(u_mid, 5.0, atol=1e-6)


def test_iem_idw_higher_power_weights_nearest():
    """Higher IDW power concentrates weight on the nearest station."""
    Grid = _make_small_grid()
    grid_x = Grid["x"].values
    xa, xb = float(grid_x[0]), float(grid_x[-1])
    ny2 = len(Grid["y"].values) // 2

    station_obs = [
        {"x": xa, "y": 0.0, "z": 0.0, "u": 10.0, "v": 0.0, "w": 0.0},
        {"x": xb, "y": 0.0, "z": 0.0, "u": 0.0, "v": 0.0, "w": 0.0},
    ]
    G1 = pydda.initialization.make_initialization_from_iem_obs(
        Grid.copy(deep=True), station_obs, power=1
    )
    G2 = pydda.initialization.make_initialization_from_iem_obs(
        Grid.copy(deep=True), station_obs, power=2
    )
    # At x[1] (closer to station A than to B), power=2 should give higher u
    u_p1 = G1["u"].values[0, 0, ny2, 1]
    u_p2 = G2["u"].values[0, 0, ny2, 1]
    assert u_p2 > u_p1


def test_iem_idw_output_shape_and_dims():
    """u, v, w must have shape (1, nz, ny, nx) with dims (time, z, y, x)."""
    Grid = _make_small_grid()
    nz = len(Grid["z"].values)
    ny = len(Grid["y"].values)
    nx = len(Grid["x"].values)
    station_obs = [{"x": 0.0, "y": 0.0, "z": 0.0, "u": 2.0, "v": 1.0, "w": 0.0}]
    Grid = pydda.initialization.make_initialization_from_iem_obs(Grid, station_obs)
    expected_shape = (1, nz, ny, nx)
    for field in ("u", "v", "w"):
        assert Grid[field].values.shape == expected_shape
        assert Grid[field].dims == ("time", "z", "y", "x")


def test_iem_idw_raises_when_no_data():
    """Empty station_obs with no profile must raise ValueError."""
    Grid = _make_small_grid()
    with pytest.raises(ValueError):
        pydda.initialization.make_initialization_from_iem_obs(Grid, [])


def test_get_iem_data():
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 100000.0), (-100000.0, 100000.0), (-100000.0, 100000.0))
    )
    fdata3 = np.zeros((20, 20, 20))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)
    station_obs = pydda.constraints.get_iem_obs(Grid)
    names = [x["site_id"] for x in station_obs]
    assert "P28" in names
    assert "WLD" in names
    assert "WDG" in names
    assert "SWO" in names
    assert "END" in names


def test_hrrr_data():
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 100000.0), (-100000.0, 100000.0), (-100000.0, 100000.0))
    )
    fdata3 = np.zeros((20, 20, 20))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)
    Grid = pydda.constraints.add_hrrr_constraint_to_grid(
        Grid, pydda.tests.get_sample_file("ruc2anl_130_20110520_0800_001.grb2")
    )

    assert Grid["U_hrrr"].max() > 15
    assert Grid["V_hrrr"].max() > 15
    assert Grid["W_hrrr"].max() > 0


def test_hrrr_uv_rotated_to_true_north():
    # HRRR's u and v are relative to its Lambert Conformal Conic grid, not
    # true north. add_hrrr_constraint_to_grid must rotate them before they
    # are interpolated onto the analysis grid.
    grid_shape = (4, 4, 4)
    grid_limits = ((0, 5000.0), (-50000.0, 50000.0), (-50000.0, 50000.0))
    file_path = pydda.tests.get_sample_file("ruc2anl_130_20110520_0800_001.grb2")

    def make_grid(origin_lat, origin_lon):
        Grid = pyart.testing.make_empty_grid(grid_shape, grid_limits)
        for field in ("origin_latitude", "radar_latitude"):
            getattr(Grid, field)["data"] = np.array([origin_lat])
        for field in ("origin_longitude", "radar_longitude"):
            getattr(Grid, field)["data"] = np.array([origin_lon])
        Grid.init_point_longitude_latitude()
        fdata3 = np.zeros(grid_shape)
        Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
        return pydda.io.read_from_pyart_grid(Grid)

    # Place the analysis domain well away from HRRR's -97.5 degree central
    # meridian, where the grid-rotation correction has a large, easily
    # measurable effect.
    Grid = make_grid(38.5, -85.0)
    Grid = pydda.constraints.add_hrrr_constraint_to_grid(Grid, file_path)
    u_true_north = Grid["U_hrrr"].values
    v_true_north = Grid["V_hrrr"].values

    # Setting both true latitudes to the equator collapses the Lambert
    # Conformal cone factor to zero, which makes the rotation a no-op.
    # This gives an otherwise identical baseline of the raw, grid-relative
    # winds to compare against.
    Grid_grid_relative = make_grid(38.5, -85.0)
    Grid_grid_relative = pydda.constraints.add_hrrr_constraint_to_grid(
        Grid_grid_relative, file_path, truelat1=0.0, truelat2=0.0
    )
    u_grid_relative = Grid_grid_relative["U_hrrr"].values
    v_grid_relative = Grid_grid_relative["V_hrrr"].values

    # The rotation should meaningfully change the wind components this far
    # from the central meridian...
    assert np.abs(u_true_north - u_grid_relative).max() > 0.5
    assert np.abs(v_true_north - v_grid_relative).max() > 0.5

    # ...while preserving wind speed, since a coordinate rotation cannot
    # change the magnitude of the wind vector.
    speed_true_north = np.sqrt(u_true_north**2 + v_true_north**2)
    speed_grid_relative = np.sqrt(u_grid_relative**2 + v_grid_relative**2)
    np.testing.assert_allclose(speed_true_north, speed_grid_relative, rtol=1e-4)


def _linear_wind_grid(
    U0=7.0,
    V0=-3.0,
    Ux=1.2e-4,
    Uy=3.0e-5,
    Vy=-4.0e-4,
    fall_speed=0.0,
    shift_radar=False,
):
    """
    Builds a Grid whose Doppler velocity is synthesized from an exactly linear
    horizontal wind field, so the VVAD fit of Protat et al. (2024) must
    recover the wind exactly. The flow is irrotational by construction
    (Vx == Uy), matching the closure the reconstruction assumes.

    Returns the Grid together with the true u and v fields and the true
    control variables.
    """
    from pydda.constraints.vad_data import _geometry

    Grid = pyart.testing.make_empty_grid(
        (12, 40, 40), ((500.0, 8000.0), (-40000.0, 40000.0), (-40000.0, 40000.0))
    )
    Grid.add_field("velocity", {"data": np.zeros((12, 40, 40)), "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)

    if shift_radar:
        # Move the radar off the grid origin so the reconstruction origin is
        # genuinely exercised.
        Grid["radar_latitude"].values[0] += 0.12
        Grid["radar_longitude"].values[0] += -0.10
        Grid = pydda.retrieval.angles.add_elevation_as_field(Grid)

    Vx = Uy  # irrotational
    dx, dy, rh, az, el = _geometry(Grid)
    u_true = U0 + Ux * dx + Uy * dy
    v_true = V0 + Vx * dx + Vy * dy

    el = np.asarray(el)
    vr = (
        np.cos(el) * np.sin(az) * u_true
        + np.cos(el) * np.cos(az) * v_true
        - np.sin(el) * fall_speed
    )
    Grid["velocity"] = Grid["velocity"].copy(data=np.expand_dims(vr, 0))

    truth = {
        "U0": U0,
        "V0": V0,
        "DIV": Ux + Vy,
        "DET": Ux - Vy,
        "DES": Uy + Vx,
        "fall_speed": fall_speed,
    }
    return Grid, u_true, v_true, truth


@pytest.mark.parametrize("shift_radar", [False, True])
def test_vvad_recovers_linear_wind(shift_radar):
    """A wind field that exactly satisfies the VAD linearity assumption must
    be recovered to machine precision at every level."""
    Grid, u_true, v_true, truth = _linear_wind_grid(shift_radar=shift_radar)
    profiles = pydda.constraints.vvad_retrieval(Grid, vel_field="velocity")

    assert profiles["valid"].values.all()
    ok = profiles["valid"].values
    for name, expected in truth.items():
        np.testing.assert_allclose(
            profiles[name].values[ok], expected, atol=1e-9, rtol=0
        )


def test_vvad_separates_fall_speed_from_divergence():
    """A uniform fall speed must be recovered on its own rather than aliasing
    into the divergence."""
    Grid, _, _, truth = _linear_wind_grid(fall_speed=2.5)
    profiles = pydda.constraints.vvad_retrieval(Grid, vel_field="velocity")

    ok = profiles["valid"].values
    np.testing.assert_allclose(profiles["fall_speed"].values[ok], 2.5, atol=1e-9)
    np.testing.assert_allclose(
        profiles["DIV"].values[ok], truth["DIV"], atol=1e-12, rtol=0
    )


def test_vvad_horizontal_wind_round_trip():
    """Reconstructing from the fitted profiles must return the linear wind
    field it was fitted to, and NaN beyond max_range."""
    Grid, u_true, v_true, _ = _linear_wind_grid()
    profiles = pydda.constraints.vvad_retrieval(Grid, vel_field="velocity")

    u_vad, v_vad = pydda.constraints.vvad_horizontal_wind(profiles, Grid)
    np.testing.assert_allclose(u_vad, u_true, atol=1e-7)
    np.testing.assert_allclose(v_vad, v_true, atol=1e-7)

    rh = np.sqrt(Grid["point_x"].values ** 2 + Grid["point_y"].values ** 2)
    u_clip, _ = pydda.constraints.vvad_horizontal_wind(
        profiles, Grid, max_range=20000.0
    )
    assert np.all(np.isnan(u_clip[rh > 20000.0]))
    assert np.all(np.isfinite(u_clip[rh <= 20000.0]))


def test_vvad_rejects_undersampled_levels():
    """Each acceptance criterion must reject a level it is meant to, and an
    invalid level must reconstruct as NaN rather than as a wild extrapolation.
    """
    Grid, _, _, _ = _linear_wind_grid()

    for kwargs in (
        {"min_points": 10**9},
        {"min_azimuth_coverage": 1.01},
        {"max_condition": 1e-6},
    ):
        profiles = pydda.constraints.vvad_retrieval(
            Grid, vel_field="velocity", **kwargs
        )
        assert not profiles["valid"].values.any(), kwargs
        assert np.all(np.isnan(profiles["U0"].values)), kwargs

        u_vad, v_vad = pydda.constraints.vvad_horizontal_wind(profiles, Grid)
        assert np.all(np.isnan(u_vad)) and np.all(np.isnan(v_vad)), kwargs


def test_make_constraint_from_vvad_adds_fields():
    """The constraint builder writes U_vvad/V_vvad into every Grid, and the
    two combine modes cover the same points by different means."""
    Grid, u_true, v_true, _ = _linear_wind_grid()
    Grids = pydda.constraints.make_constraint_from_vvad(
        [Grid, Grid.copy(deep=True)], vel_field="velocity"
    )

    for g in Grids:
        assert "U_vvad" in g.variables and "V_vvad" in g.variables
        assert g["U_vvad"].dims == ("time", "z", "y", "x")
        # No W_vvad: the VVAD supplies horizontal components only.
        assert "W_vvad" not in g.variables

    np.testing.assert_allclose(Grids[0]["U_vvad"].values.squeeze(), u_true, atol=1e-7)

    nearest = pydda.constraints.make_constraint_from_vvad(
        [Grid.copy(deep=True)], vel_field="velocity", combine="nearest"
    )
    np.testing.assert_allclose(nearest[0]["U_vvad"].values.squeeze(), u_true, atol=1e-7)

    with pytest.raises(ValueError):
        pydda.constraints.make_constraint_from_vvad(
            [Grid], vel_field="velocity", combine="not_a_mode"
        )
