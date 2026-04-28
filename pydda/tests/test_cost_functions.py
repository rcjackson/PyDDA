"""Nosetests for continuous integration"""

import pydda
import pyart
import numpy as np
import pytest
import xarray as xr

try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def _make_radvel_inputs():
    """Build shared test inputs for radial velocity cost/gradient tests."""
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 10000), (-10000, 10000), (-10000, 10000))
    )
    fdata3 = np.zeros((20, 20, 20))
    Grid.fields["zero_field"] = {"data": fdata3, "units": "m/s"}
    Grid = pydda.io.read_from_pyart_grid(Grid)
    vrs = [np.ma.array(Grid["zero_field"].values)]
    azs = [Grid["AZ"].values]
    els = [Grid["EL"].values]
    wts = [np.ma.zeros((20, 20, 20))]
    weights = np.ones((1, 20, 20, 20))
    return vrs, azs, els, wts, weights


def test_calculate_rad_velocity_cost():
    vrs, azs, els, wts, weights = _make_radvel_inputs()
    u = np.zeros((20, 20, 20))
    v = np.zeros((20, 20, 20))
    w = np.zeros((20, 20, 20))
    rmsVr = 1.0
    cost = pydda.cost_functions.calculate_radial_vel_cost_function(
        vrs, azs, els, u, v, w, wts, rmsVr, weights
    )
    grad = pydda.cost_functions.calculate_grad_radial_vel(
        vrs, els, azs, u, v, w, wts, weights, rmsVr
    )

    assert cost == 0
    assert np.all(grad == 0)


def test_calculate_rad_velocity_cost_parallel():
    """Vectorized (parallel=True) radial velocity cost and gradient match serial results."""
    vrs, azs, els, wts, weights = _make_radvel_inputs()
    rng = np.random.default_rng(42)
    u = rng.random((20, 20, 20))
    v = rng.random((20, 20, 20))
    w = rng.random((20, 20, 20))
    rmsVr = 1.0

    serial_cost = pydda.cost_functions.calculate_radial_vel_cost_function(
        vrs, azs, els, u, v, w, wts, rmsVr, weights, parallel=False
    )
    parallel_cost = pydda.cost_functions.calculate_radial_vel_cost_function(
        vrs, azs, els, u, v, w, wts, rmsVr, weights, parallel=True
    )
    np.testing.assert_allclose(parallel_cost, serial_cost)

    serial_grad = pydda.cost_functions.calculate_grad_radial_vel(
        vrs, els, azs, u, v, w, wts, weights, rmsVr, parallel=False
    )
    parallel_grad = pydda.cost_functions.calculate_grad_radial_vel(
        vrs, els, azs, u, v, w, wts, weights, rmsVr, parallel=True
    )
    np.testing.assert_allclose(parallel_grad, serial_grad)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="Jax not installed")
def test_calculate_rad_velocity_cost_jax():
    """Test with a zero velocity field radar"""
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 10000), (-10000, 10000), (-10000, 10000))
    )

    # a zero field
    fdata3 = np.zeros((20, 20, 20))
    Grid.fields["zero_field"] = {"data": fdata3, "units": "m/s"}
    Grid = pydda.io.read_from_pyart_grid(Grid)
    vel_field = "zero_field"
    vrs = [np.array(Grid[vel_field].values)]
    azs = [Grid["AZ"].values]
    els = [Grid["EL"].values]
    u = jnp.zeros((20, 20, 20))
    v = jnp.zeros((20, 20, 20))
    w = jnp.zeros((20, 20, 20))
    rmsVr = 1.0
    wts = [jnp.zeros((20, 20, 20))]
    weights = [jnp.ones((20, 20, 20))]
    cost = pydda.cost_functions.jax.calculate_radial_vel_cost_function(
        vrs, azs, els, u, v, w, wts, rmsVr, weights
    )
    grad = pydda.cost_functions.jax.calculate_grad_radial_vel(
        vrs, azs, els, u, v, w, wts, weights, rmsVr
    )

    assert cost == 0
    assert np.all(grad == 0)


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
def test_calculate_rad_velocity_cost_tf():
    """Test with a zero velocity field radar"""
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 10000), (-10000, 10000), (-10000, 10000))
    )

    # a zero field
    fdata3 = np.zeros((20, 20, 20))
    Grid.fields["zero_field"] = {"data": fdata3, "units": "m/s"}
    Grid = pydda.io.read_from_pyart_grid(Grid)
    vel_field = "zero_field"
    vrs = [tf.constant(Grid[vel_field].values.squeeze(), dtype=tf.float32)]
    azs = [tf.constant(Grid["AZ"].values.squeeze(), dtype=tf.float32)]
    els = [tf.constant(Grid["EL"].values.squeeze(), dtype=tf.float32)]
    u = tf.zeros((20, 20, 20), dtype=tf.float32)
    v = tf.zeros((20, 20, 20), dtype=tf.float32)
    w = tf.zeros((20, 20, 20), dtype=tf.float32)
    rmsVr = 1.0
    wts = [tf.zeros((20, 20, 20), dtype=tf.float32)]
    weights = [tf.ones((20, 20, 20), dtype=tf.float32)]
    cost = pydda.cost_functions.tf.calculate_radial_vel_cost_function(
        vrs, azs, els, u, v, w, wts, rmsVr, weights
    )
    grad = pydda.cost_functions.tf.calculate_grad_radial_vel(
        vrs, azs, els, u, v, w, wts, weights, rmsVr
    )

    assert cost.numpy() == 0
    assert np.all(grad.numpy() == 0)


def test_calculate_fall_speed():
    """Check to see if fall speeds are realistic"""
    ref_field = 10 * np.ones((10, 100, 100))
    grid_shape = (10, 100, 100)
    grid_limits = ((0, 10000), (-100000, 100000), (-100000, 100000))
    grid = pyart.testing.make_empty_grid(grid_shape, grid_limits)
    field_dic = {"data": ref_field, "long_name": "reflectivity", "units": "dBZ"}
    grid.fields = {"reflectivity": field_dic}
    grid.radar_latitude = {"data": np.array([0])}
    grid.radar_longitude = {"data": np.array([0])}
    grid.radar_altitude = {"data": np.array([0])}
    fall_speed = pydda.cost_functions.calculate_fall_speed(
        pydda.io.read_from_pyart_grid(grid), refl_field="reflectivity"
    )
    assert fall_speed.shape == (1, 10, 100, 100)
    assert fall_speed[0, 1, 1, 1] < -3


def test_calculate_mass_continuity():
    """In a constant wind field, div * V = 0, so we should get zero for mass
    continuity cost function and gradient"""
    u = 10 * np.ones((10, 10, 10))
    v = 10 * np.ones((10, 10, 10))
    w = 0 * np.ones((10, 10, 10))
    dx = 100.0
    dy = 100.0
    dz = 100.0
    z = np.arange(0, 1000.0, 100)
    cost = pydda.cost_functions.calculate_mass_continuity(u, v, w, z, dx, dy, dz)
    cost_grad = pydda.cost_functions.calculate_mass_continuity_gradient(
        u, v, w, z, dx, dy, dz
    )

    assert cost == 0
    assert np.all(cost_grad == 0)

    v = np.zeros((10, 10, 10))
    for i in np.arange(10, 0, -1):
        u[:, :, 10 - i] = i
    cost = pydda.cost_functions.calculate_mass_continuity(u, v, w, z, dx, dy, dz)
    assert cost > 0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="Jax not installed")
def test_calculate_mass_continuity_jax():
    """In a constant wind field, div * V = 0, so we should get zero for mass
    continuity cost function and gradient"""
    u = 10 * np.ones((10, 10, 10))
    v = 10 * np.ones((10, 10, 10))
    w = 0 * np.ones((10, 10, 10))
    dx = 100.0
    dy = 100.0
    dz = 100.0
    z = np.arange(0, 1000.0, 100)
    cost = pydda.cost_functions.jax.calculate_mass_continuity(u, v, w, z, dx, dy, dz)
    cost_grad = pydda.cost_functions.jax.calculate_mass_continuity_gradient(
        u, v, w, z, dx, dy, dz
    )

    assert cost == 0
    assert jnp.all(cost_grad == 0)

    v = np.zeros((10, 10, 10))
    for i in np.arange(10, 0, -1):
        u[:, :, 10 - i] = i
    cost = pydda.cost_functions.jax.calculate_mass_continuity(u, v, w, z, dx, dy, dz)
    assert cost > 0


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
def test_calculate_mass_continuity_tf():
    """In a constant wind field, div * V = 0, so we should get zero for mass
    continuity cost function and gradient"""
    u = 10 * tf.ones((10, 10, 10), dtype=tf.float32)
    v = 10 * tf.ones((10, 10, 10), dtype=tf.float32)
    w = 0 * tf.ones((10, 10, 10), dtype=tf.float32)
    dx = 100.0
    dy = 100.0
    dz = 100.0
    z = np.tile(np.arange(0, 1000.0, 100), (10, 10, 1))
    z = tf.constant(z, dtype=tf.float32)
    cost = pydda.cost_functions.tf.calculate_mass_continuity(u, v, w, z, dx, dy, dz)
    cost_grad = pydda.cost_functions.tf.calculate_mass_continuity_gradient(
        u, v, w, z, dx, dy, dz
    )

    assert cost.numpy() == 0
    assert tf.math.reduce_all(cost_grad == 0)

    v = tf.zeros((10, 10, 10), dtype=tf.float32)
    u = u.numpy()
    for i in jnp.arange(10, 0, -1):
        u[:, :, 10 - i] = i
    u = tf.constant(u, dtype=tf.float32)
    cost = pydda.cost_functions.tf.calculate_mass_continuity(u, v, w, z, dx, dy, dz)
    assert cost.numpy() > 0


def test_calculate_smoothness_cost():
    """The Laplacian of a constant field is zero"""
    u = 10 * np.ones((10, 10, 10))
    v = 10 * np.ones((10, 10, 10))
    w = 0 * np.ones((10, 10, 10))

    dx = 100.0
    dy = 100.0
    dz = 100.0
    np.arange(0, 1000.0, 100)

    cost = pydda.cost_functions.calculate_smoothness_cost(u, v, w, dx, dy, dz)
    cost_grad = pydda.cost_functions.calculate_smoothness_gradient(
        u, v, w, dx, dy, dz, Cx=1e-4, Cy=1e-4, Cz=1e-4
    )

    assert cost == 0
    assert np.all(cost_grad == 0)

    """ Now, put in a discontinuity """
    u[:, :, 5] = -10
    cost = pydda.cost_functions.calculate_smoothness_cost(u, v, w, dx, dy, dz)
    assert cost > 0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="Jax not installed")
def test_calculate_smoothness_cost_jax():
    """The Laplacian of a constant field is zero"""
    u = 10 * np.ones((10, 10, 10))
    v = 10 * np.ones((10, 10, 10))
    w = 0 * np.ones((10, 10, 10))

    dx = 100.0
    dy = 100.0
    dz = 100.0
    np.arange(0, 1000.0, 100)

    cost = pydda.cost_functions.jax.calculate_smoothness_cost(u, v, w, dx, dy, dz)
    cost_grad = pydda.cost_functions.jax.calculate_smoothness_gradient(
        u, v, w, dx, dy, dz, Cx=1e-4, Cy=1e-4, Cz=1e-4
    )

    assert cost == 0
    assert jnp.all(cost_grad == 0)

    """ Now, put in a discontinuity """
    u[:, :, 5] = -10
    cost = pydda.cost_functions.jax.calculate_smoothness_cost(u, v, w, dx, dy, dz)
    assert cost > 0


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
def test_calculate_smoothness_cost_tensorflow():
    """The Laplacian of a constant field is zero"""
    u = 10 * tf.ones((10, 10, 10))
    v = 10 * tf.ones((10, 10, 10))
    w = 0 * tf.ones((10, 10, 10))

    dx = 100.0
    dy = 100.0
    dz = 100.0
    tf.range(0, 1000.0, 100)

    cost = pydda.cost_functions.tf.calculate_smoothness_cost(u, v, w, dx, dy, dz)
    cost_grad = pydda.cost_functions.tf.calculate_smoothness_gradient(
        u, v, w, dx, dy, dz, Cx=1e-4, Cy=1e-4, Cz=1e-4
    )

    assert cost.numpy() == 0
    assert np.all(cost_grad.numpy() == 0)

    """ Now, put in a discontinuity """
    u = u.numpy()
    u[:, :, 5] = -10
    u = tf.constant(u)
    cost = pydda.cost_functions.tf.calculate_smoothness_cost(u, v, w, dx, dy, dz)
    assert cost.numpy() > 0


def test_background_cost():
    """Zero cost when background matches wind field, nonzero otherwise"""
    u = 10 * np.ones((10, 10, 10))
    v = 10 * np.ones((10, 10, 10))
    w = 0 * np.ones((10, 10, 10))
    weights = np.ones((10, 10, 10))

    u_back = 10 * np.ones(10)
    v_back = 10 * np.ones(10)
    cost = pydda.cost_functions.calculate_background_cost(
        u, v, w, weights, u_back, v_back
    )
    grad = pydda.cost_functions.calculate_background_gradient(
        u, v, w, weights, u_back, v_back
    )

    assert cost == 0
    assert np.all(grad == 0)

    u_back = 4 * np.ones(10)
    v_back = 4 * np.ones(10)

    cost = pydda.cost_functions.calculate_background_cost(
        u, v, w, weights, u_back, v_back
    )
    grad = pydda.cost_functions.calculate_background_gradient(
        u, v, w, weights, u_back, v_back
    )

    assert cost > 0
    assert np.any(grad > 0)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
def test_background_cost_jax():
    """Zero cost when background matches wind field, nonzero otherwise"""
    u = 10 * jnp.ones((10, 10, 10))
    v = 10 * jnp.ones((10, 10, 10))
    w = 0 * jnp.ones((10, 10, 10))
    weights = jnp.ones((10, 10, 10))

    u_back = 10 * jnp.ones(10)
    v_back = 10 * jnp.ones(10)
    cost = pydda.cost_functions.jax.calculate_background_cost(
        u, v, w, weights, u_back, v_back
    )
    grad = pydda.cost_functions.jax.calculate_background_gradient(
        u, v, w, weights, u_back, v_back
    )

    assert cost == 0
    assert jnp.all(grad == 0)

    u_back = 4 * jnp.ones(10)
    v_back = 4 * jnp.ones(10)

    cost = pydda.cost_functions.jax.calculate_background_cost(
        u, v, w, weights, u_back, v_back
    )
    grad = pydda.cost_functions.jax.calculate_background_gradient(
        u, v, w, weights, u_back, v_back
    )

    assert cost > 0
    assert jnp.any(grad > 0)


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
def test_background_cost_tensorflow():
    """Zero cost when background matches wind field, nonzero otherwise"""
    u = 10 * tf.ones((10, 10, 10), dtype=tf.float32)
    v = 10 * tf.ones((10, 10, 10), dtype=tf.float32)
    0 * tf.ones((10, 10, 10), dtype=tf.float32)
    weights = tf.ones((10, 10, 10), dtype=tf.float32)

    u_back = 10 * tf.ones(10, dtype=tf.float32)
    v_back = 10 * tf.ones(10, dtype=tf.float32)
    cost = pydda.cost_functions.tf.calculate_background_cost(
        u, v, weights, u_back, v_back
    )
    grad = pydda.cost_functions.tf.calculate_background_gradient(
        u, v, weights, u_back, v_back
    )

    assert cost.numpy() == 0
    assert tf.math.reduce_all(grad == 0)

    u_back = 4 * tf.ones(10, dtype=tf.float32)
    v_back = 4 * tf.ones(10, dtype=tf.float32)

    cost = pydda.cost_functions.tf.calculate_background_cost(
        u, v, weights, u_back, v_back
    )
    grad = pydda.cost_functions.tf.calculate_background_gradient(
        u, v, weights, u_back, v_back
    )

    assert cost.numpy() > 0
    assert tf.math.reduce_any(grad > 0)


def test_vert_vorticity():
    u = 10 * np.ones((10, 10, 10))
    v = 10 * np.ones((10, 10, 10))
    w = 0 * np.ones((10, 10, 10))

    dx = 100.0
    dy = 100.0
    dz = 100.0

    z = np.arange(0, 1000.0, 100)
    cost = pydda.cost_functions.calculate_vertical_vorticity_cost(
        u, v, w, dx, dy, dz, 10, 10
    )
    cost_grad = pydda.cost_functions.calculate_vertical_vorticity_gradient(
        u, v, w, dx, dy, dz, 10, 10
    )

    assert cost == 0
    assert np.all(cost_grad == 0)

    """ Put in a convergent wind field (u decreases from 10 to 0 m/s) """
    v = np.zeros((10, 10, 10))
    for i in np.arange(10, 0, -1):
        u[:, :, 10 - i] = i
        v[:, :, 10 - i] = i
    cost = pydda.cost_functions.calculate_vertical_vorticity_cost(
        u, v, w, z, dx, dy, dz, 10, 10
    )
    assert cost > 0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="Jax not installed")
def test_vert_vorticity_jax():
    u = 10 * np.ones((10, 10, 10))
    v = 10 * np.ones((10, 10, 10))
    w = 0 * np.ones((10, 10, 10))

    dx = 100.0
    dy = 100.0
    dz = 100.0

    np.arange(0, 1000.0, 100)
    cost = pydda.cost_functions.jax.calculate_vertical_vorticity_cost(
        u, v, w, dx, dy, dz, 10, 10
    )
    cost_grad = pydda.cost_functions.jax.calculate_vertical_vorticity_gradient(
        u, v, w, dx, dy, dz, 10, 10
    )

    assert cost == 0
    assert jnp.all(cost_grad == 0)

    """ Put in a convergent wind field (u decreases from 10 to 0 m/s) """
    u = np.asanyarray(u)
    w = np.asanyarray(w)
    v = np.zeros((10, 10, 10))
    for i in np.arange(10, 0, -1):
        u[:, :, 10 - i] = i
        v[:, :, 10 - i] = i
    cost = pydda.cost_functions.jax.calculate_vertical_vorticity_cost(
        u, v, w, dx, dy, dz, 10, 10
    )
    assert cost > 0


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
def test_vert_vorticity_tf():
    u = 10 * tf.ones((10, 10, 10), dtype=tf.float32)
    v = 10 * tf.ones((10, 10, 10), dtype=tf.float32)
    w = 0 * tf.ones((10, 10, 10), dtype=tf.float32)

    dx = 100.0
    dy = 100.0
    dz = 100.0

    tf.range(0, 1000.0, 100, dtype=tf.float32)
    cost = pydda.cost_functions.tf.calculate_vertical_vorticity_cost(
        u, v, w, dx, dy, dz, 10, 10
    )
    cost_grad = pydda.cost_functions.tf.calculate_vertical_vorticity_gradient(
        u, v, w, dx, dy, dz, 10, 10
    )

    assert cost.numpy() == 0
    assert tf.math.reduce_all(cost_grad == 0)

    """ Put in a convergent wind field (u decreases from 10 to 0 m/s) """
    v = np.zeros((10, 10, 10))
    u = u.numpy()
    for i in np.arange(10, 0, -1):
        u[:, :, 10 - i] = float(i)
        v[:, :, 10 - i] = float(i)
    cost = pydda.cost_functions.tf.calculate_vertical_vorticity_cost(
        tf.constant(u, dtype=tf.float32),
        tf.constant(v, dtype=tf.float32),
        w,
        dx,
        dy,
        dz,
        10,
        10,
    )
    assert cost.numpy() > 0


def test_point_cost():
    u = 1 * np.ones((10, 10, 10))
    v = 1 * np.ones((10, 10, 10))
    x = np.linspace(-10, 10, 10)
    y = np.linspace(-10, 10, 10)
    z = np.linspace(-10, 10, 10)
    x, y, z = np.meshgrid(x, y, z)

    my_point1 = {"x": 0, "y": 0, "z": 0, "u": 2.0, "v": 2.0, "w": 0.0}
    cost = pydda.cost_functions.calculate_point_cost(
        u,
        v,
        x,
        y,
        z,
        [my_point1],
    )
    grad = pydda.cost_functions.calculate_point_gradient(
        u,
        v,
        x,
        y,
        z,
        [my_point1],
    )

    assert cost > 0
    assert np.all(grad <= 0)

    my_point1 = {"x": 0, "y": 0, "z": 0, "u": -2.0, "v": -2.0, "w": 0.0}
    my_point2 = {"x": 3, "y": 3, "z": 0, "u": 2.0, "v": 2.0, "w": 0.0}

    cost = pydda.cost_functions.calculate_point_cost(
        u,
        v,
        x,
        y,
        z,
        [my_point1],
    )
    grad = pydda.cost_functions.calculate_point_gradient(
        u,
        v,
        x,
        y,
        z,
        [my_point1],
    )
    assert cost > 0
    assert np.all(grad >= 0)

    cost = pydda.cost_functions.calculate_point_cost(
        u,
        v,
        x,
        y,
        z,
        [my_point1, my_point2],
    )
    grad = pydda.cost_functions.calculate_point_gradient(
        u,
        v,
        x,
        y,
        z,
        [my_point1, my_point2],
    )
    assert cost > 0
    assert ~np.all(grad >= 0)

    my_point1 = {"x": 0, "y": 0, "z": 0, "u": 1.0, "v": 1.0, "w": 0.0}
    my_point2 = {"x": 3, "y": 3, "z": 0, "u": 1.0, "v": 1.0, "w": 0.0}
    cost = pydda.cost_functions.calculate_point_cost(
        u,
        v,
        x,
        y,
        z,
        [my_point1, my_point2],
    )
    grad = pydda.cost_functions.calculate_point_gradient(
        u,
        v,
        x,
        y,
        z,
        [my_point1, my_point2],
    )
    assert cost == 0
    assert np.all(grad == 0)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="Jax not installed")
def test_point_cost_jax():
    u = 1 * np.ones((10, 10, 10))
    v = 1 * np.ones((10, 10, 10))
    x = np.linspace(-10, 10, 10)
    y = np.linspace(-10, 10, 10)
    z = np.linspace(-10, 10, 10)
    x, y, z = np.meshgrid(x, y, z)

    my_point1 = {"x": 0, "y": 0, "z": 0, "u": 2.0, "v": 2.0, "w": 0.0}
    cost = pydda.cost_functions.jax.calculate_point_cost(
        u,
        v,
        x,
        y,
        z,
        [my_point1],
    )
    grad = pydda.cost_functions.jax.calculate_point_gradient(
        u,
        v,
        x,
        y,
        z,
        [my_point1],
    )

    assert cost > 0
    assert np.all(grad <= 0)

    my_point1 = {"x": 0, "y": 0, "z": 0, "u": -2.0, "v": -2.0, "w": 0.0}
    my_point2 = {"x": 3, "y": 3, "z": 0, "u": 2.0, "v": 2.0, "w": 0.0}

    cost = pydda.cost_functions.jax.calculate_point_cost(
        u,
        v,
        x,
        y,
        z,
        [my_point1],
    )
    grad = pydda.cost_functions.jax.calculate_point_gradient(
        u,
        v,
        x,
        y,
        z,
        [my_point1],
    )
    assert cost > 0
    assert np.all(grad >= 0)

    cost = pydda.cost_functions.jax.calculate_point_cost(
        u,
        v,
        x,
        y,
        z,
        [my_point1, my_point2],
    )
    grad = pydda.cost_functions.jax.calculate_point_gradient(
        u,
        v,
        x,
        y,
        z,
        [my_point1, my_point2],
    )
    assert cost > 0
    assert ~np.all(grad >= 0)

    my_point1 = {"x": 0, "y": 0, "z": 0, "u": 1.0, "v": 1.0, "w": 0.0}
    my_point2 = {"x": 3, "y": 3, "z": 0, "u": 1.0, "v": 1.0, "w": 0.0}
    cost = pydda.cost_functions.jax.calculate_point_cost(
        u,
        v,
        x,
        y,
        z,
        [my_point1, my_point2],
    )
    grad = pydda.cost_functions.jax.calculate_point_gradient(
        u,
        v,
        x,
        y,
        z,
        [my_point1, my_point2],
    )
    assert cost == 0
    assert np.all(grad == 0)


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
def test_point_cost_tf():
    u = 1 * tf.ones((10, 10, 10), dtype=tf.float32)
    v = 1 * tf.ones((10, 10, 10), dtype=tf.float32)
    0 * tf.ones((10, 10, 10), dtype=tf.float32)

    x = tf.constant(np.linspace(-10, 10, 10), dtype=tf.float32)
    y = tf.constant(np.linspace(-10, 10, 10), dtype=tf.float32)
    z = tf.constant(np.linspace(-10, 10, 10), dtype=tf.float32)
    x, y, z = tf.meshgrid(x, y, z)

    my_point1 = {"x": 0, "y": 0, "z": 0, "u": 2.0, "v": 2.0, "w": 0.0}
    cost = pydda.cost_functions.tf.calculate_point_cost(
        u,
        v,
        x,
        y,
        z,
        [my_point1],
    )
    grad = pydda.cost_functions.tf.calculate_point_gradient(
        u,
        v,
        x,
        y,
        z,
        [my_point1],
    )

    assert cost.numpy() > 0
    assert tf.math.reduce_all(grad <= 0)

    my_point1 = {"x": 0, "y": 0, "z": 0, "u": -2.0, "v": -2.0, "w": 0.0}
    my_point2 = {"x": 3, "y": 3, "z": 0, "u": 2.0, "v": 2.0, "w": 0.0}

    cost = pydda.cost_functions.tf.calculate_point_cost(
        u,
        v,
        x,
        y,
        z,
        [my_point1],
    )
    grad = pydda.cost_functions.tf.calculate_point_gradient(
        u,
        v,
        x,
        y,
        z,
        [my_point1],
    )
    assert cost.numpy() > 0
    assert tf.math.reduce_all(grad >= 0)

    cost = pydda.cost_functions.tf.calculate_point_cost(
        u,
        v,
        x,
        y,
        z,
        [my_point1, my_point2],
    )
    grad = pydda.cost_functions.tf.calculate_point_gradient(
        u,
        v,
        x,
        y,
        z,
        [my_point1, my_point2],
    )
    assert cost.numpy() > 0
    assert ~tf.math.reduce_all(grad >= 0)

    my_point1 = {"x": 0, "y": 0, "z": 0, "u": 1.0, "v": 1.0, "w": 0.0}
    my_point2 = {"x": 3, "y": 3, "z": 0, "u": 1.0, "v": 1.0, "w": 0.0}
    cost = pydda.cost_functions.tf.calculate_point_cost(
        u,
        v,
        x,
        y,
        z,
        [my_point1, my_point2],
    )
    grad = pydda.cost_functions.tf.calculate_point_gradient(
        u,
        v,
        x,
        y,
        z,
        [my_point1, my_point2],
    )
    assert cost.numpy() == 0
    assert tf.math.reduce_all(grad == 0)


def test_model_cost():
    u = 10 * np.ones((10, 10, 10))
    v = 10 * np.ones((10, 10, 10))
    w = 0 * np.ones((10, 10, 10))
    weights = np.ones((10, 10, 10))

    cost = pydda.cost_functions.calculate_model_cost(u, v, w, weights, u, v, w)
    cost_grad = pydda.cost_functions.calculate_model_gradient(u, v, w, weights, u, v, w)

    # If model == observations, then cost is zero
    assert cost == 0
    assert np.all(cost_grad == 0)

    # If model is further from obs, cost is greater
    cost1 = pydda.cost_functions.calculate_model_cost(u, v, w, weights, u - 1, v, w)
    cost2 = pydda.cost_functions.calculate_model_cost(u, v, w, weights, u - 1, v - 1, w)
    assert cost2 > cost1


@pytest.mark.skipif(not JAX_AVAILABLE, reason="Jax not installed")
def test_model_cost_jax():
    u = 10 * np.ones((10, 10, 10))
    v = 10 * np.ones((10, 10, 10))
    w = 0 * np.ones((10, 10, 10))
    weights = np.ones((10, 10, 10))

    cost = pydda.cost_functions.jax.calculate_model_cost(u, v, w, weights, u, v, w)
    cost_grad = pydda.cost_functions.jax.calculate_model_gradient(
        u, v, w, weights, u, v, w
    )

    # If model == observations, then cost is zero
    assert cost == 0
    assert jnp.allclose(cost_grad, 0)

    # If model is further from obs, cost is greater
    cost1 = pydda.cost_functions.jax.calculate_model_cost(u, v, w, weights, u - 1, v, w)
    cost2 = pydda.cost_functions.jax.calculate_model_cost(
        u, v, w, weights, u - 1, v - 1, w
    )
    assert cost2 > cost1


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
def test_model_cost_tf():
    u = 10 * tf.ones((10, 10, 10))
    v = 10 * tf.ones((10, 10, 10))
    w = 0 * tf.ones((10, 10, 10))
    weights = tf.ones((10, 10, 10))

    cost = pydda.cost_functions.tf.calculate_model_cost(u, v, w, weights, u, v, w)
    cost_grad = pydda.cost_functions.tf.calculate_model_gradient(
        u, v, w, weights, u, v, w
    )

    # If model == observations, then cost is zero
    assert cost == 0
    assert tf.math.reduce_all(cost_grad == 0)

    # If model is further from obs, cost is greater
    cost1 = pydda.cost_functions.tf.calculate_model_cost(u, v, w, weights, u - 1, v, w)
    cost2 = pydda.cost_functions.tf.calculate_model_cost(
        u, v, w, weights, u - 1, v - 1, w
    )
    assert cost2 > cost1
