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


def _make_upper_bc_inputs():
    """A synthetic wind field and a single radar whose echo top is at level 5."""
    nz, ny, nx = 10, 8, 8
    rng = np.random.default_rng(42)
    u = rng.random((nz, ny, nx))
    v = rng.random((nz, ny, nx))
    w = rng.random((nz, ny, nx))
    z = np.tile(np.linspace(0.0, 9000.0, nz)[:, None, None], (1, ny, nx))

    # One radar reporting valid velocities only in the lowest six levels of one
    # quadrant of the domain. Everything else is filled the way PyDDA fills
    # points without an observation.
    vr = np.full((nz, ny, nx), -9999.0)
    vr[:6, :4, :4] = 1.0
    return u, v, w, z, [vr]


def _w_component(grad, shape):
    """Pull the w block out of a flattened (3, nz, ny, nx) gradient."""
    return np.asarray(grad).reshape((3,) + shape)[2]


def test_calculate_echo_top_mask():
    u, v, w, z, vrs = _make_upper_bc_inputs()
    mask = pydda.cost_functions.calculate_echo_top_mask(vrs, z, above=2.0)

    # The condition never applies below the "above" level...
    assert not mask[z <= 2000.0].any()
    # ...nor where the radar sees an echo...
    assert not mask[np.logical_and(vrs[0] > -1000, z > 2000.0)].any()
    # ...but it does apply in the echo-free air aloft.
    assert mask[6:, 6:, 6:].all()

    # Raising "above" can only shrink the masked volume.
    higher = pydda.cost_functions.calculate_echo_top_mask(vrs, z, above=5.0)
    assert higher.sum() < mask.sum()
    assert not np.logical_and(higher, ~mask).any()


def test_mass_continuity_gradient_upper_bc():
    """upper_bc selects between no condition, a grid top condition and an
    echo top condition."""
    u, v, w, z, vrs = _make_upper_bc_inputs()
    shape = u.shape
    mask = pydda.cost_functions.calculate_echo_top_mask(vrs, z, above=2.0)
    args = (u, v, w, z, 1000.0, 1000.0, 1000.0)

    none_bc = _w_component(
        pydda.cost_functions.calculate_mass_continuity_gradient(*args, upper_bc=0),
        shape,
    )
    grid_top = _w_component(
        pydda.cost_functions.calculate_mass_continuity_gradient(*args, upper_bc=1),
        shape,
    )
    echo_top = _w_component(
        pydda.cost_functions.calculate_mass_continuity_gradient(
            *args, upper_bc=2, upper_bc_mask=mask
        ),
        shape,
    )

    # The surface is always impermeable, the domain top only for upper_bc=1.
    assert np.all(none_bc[0] == 0)
    assert not np.all(none_bc[-1] == 0)
    assert np.all(grid_top[-1] == 0)
    np.testing.assert_allclose(grid_top[:-1], none_bc[:-1])

    # upper_bc=2 pins w exactly on the mask and leaves everything else alone.
    assert np.all(echo_top[mask] == 0)
    np.testing.assert_allclose(echo_top[~mask], none_bc[~mask])
    assert np.any(echo_top != grid_top)

    # The legacy booleans still mean what they used to. This is the regression
    # guard for `upper_bc is True`, which silently skipped the condition for
    # the integer modes.
    np.testing.assert_allclose(
        _w_component(
            pydda.cost_functions.calculate_mass_continuity_gradient(
                *args, upper_bc=True
            ),
            shape,
        ),
        grid_top,
    )
    np.testing.assert_allclose(
        _w_component(
            pydda.cost_functions.calculate_mass_continuity_gradient(
                *args, upper_bc=False
            ),
            shape,
        ),
        none_bc,
    )


def test_all_gradients_honor_echo_top_upper_bc():
    """Every gradient term that takes upper_bc applies the echo top mask."""
    u, v, w, z, vrs = _make_upper_bc_inputs()
    shape = u.shape
    mask = pydda.cost_functions.calculate_echo_top_mask(vrs, z, above=2.0)
    els = [np.ones(shape) * 0.5]
    azs = [np.ones(shape) * 0.5]
    wts = [np.zeros(shape)]
    weights = np.ones((1,) + shape)

    grads = {
        "radial_vel": pydda.cost_functions.calculate_grad_radial_vel(
            vrs, els, azs, u, v, w, wts, weights, 1.0, upper_bc=2, upper_bc_mask=mask
        ),
        "smoothness": pydda.cost_functions.calculate_smoothness_gradient(
            u,
            v,
            w,
            1000.0,
            1000.0,
            1000.0,
            Cx=1e-4,
            Cy=1e-4,
            Cz=1e-4,
            upper_bc=2,
            upper_bc_mask=mask,
        ),
        "mass_continuity": pydda.cost_functions.calculate_mass_continuity_gradient(
            u, v, w, z, 1000.0, 1000.0, 1000.0, upper_bc=2, upper_bc_mask=mask
        ),
        "vorticity": pydda.cost_functions.calculate_vertical_vorticity_gradient(
            u,
            v,
            w,
            1000.0,
            1000.0,
            1000.0,
            1.0,
            1.0,
            coeff=1e-5,
            upper_bc=2,
            upper_bc_mask=mask,
        ),
    }
    for name, grad in grads.items():
        grad_w = _w_component(grad, shape)
        assert np.all(grad_w[mask] == 0), "%s ignored the echo top mask" % name
        assert np.all(grad_w[0] == 0), "%s ignored the surface condition" % name


@pytest.mark.skipif(not JAX_AVAILABLE, reason="Jax not installed")
def test_mass_continuity_gradient_upper_bc_jax():
    u, v, w, z, vrs = _make_upper_bc_inputs()
    shape = u.shape
    mask = pydda.cost_functions.calculate_echo_top_mask(vrs, z, above=2.0)
    args = (
        jnp.array(u),
        jnp.array(v),
        jnp.array(w),
        jnp.array(z),
        1000.0,
        1000.0,
        1000.0,
    )

    none_bc = _w_component(
        pydda.cost_functions.jax.calculate_mass_continuity_gradient(*args, upper_bc=0),
        shape,
    )
    grid_top = _w_component(
        pydda.cost_functions.jax.calculate_mass_continuity_gradient(*args, upper_bc=1),
        shape,
    )
    echo_top = _w_component(
        pydda.cost_functions.jax.calculate_mass_continuity_gradient(
            *args, upper_bc=2, upper_bc_mask=mask
        ),
        shape,
    )

    assert np.all(none_bc[0] == 0)
    assert not np.all(none_bc[-1] == 0)
    assert np.all(grid_top[-1] == 0)
    assert np.all(echo_top[mask] == 0)
    np.testing.assert_allclose(echo_top[~mask], none_bc[~mask])


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
def test_mass_continuity_gradient_upper_bc_tf():
    u, v, w, z, vrs = _make_upper_bc_inputs()
    shape = u.shape
    mask = pydda.cost_functions.calculate_echo_top_mask(vrs, z, above=2.0)
    args = (
        tf.constant(u, dtype=tf.float32),
        tf.constant(v, dtype=tf.float32),
        tf.constant(w, dtype=tf.float32),
        tf.constant(z, dtype=tf.float32),
        1000.0,
        1000.0,
        1000.0,
    )
    tf_mask = tf.constant(mask, dtype=tf.bool)

    none_bc = _w_component(
        pydda.cost_functions.tf.calculate_mass_continuity_gradient(*args, upper_bc=0),
        shape,
    )
    grid_top = _w_component(
        pydda.cost_functions.tf.calculate_mass_continuity_gradient(*args, upper_bc=1),
        shape,
    )
    echo_top = _w_component(
        pydda.cost_functions.tf.calculate_mass_continuity_gradient(
            *args, upper_bc=2, upper_bc_mask=tf_mask
        ),
        shape,
    )

    assert np.all(none_bc[0] == 0)
    assert not np.all(none_bc[-1] == 0)
    assert np.all(grid_top[-1] == 0)
    assert np.all(echo_top[mask] == 0)
    np.testing.assert_allclose(echo_top[~mask], none_bc[~mask])


from scipy.optimize import approx_fprime


def _make_vad_inputs(seed=0, shape=(5, 6, 7)):
    """Random winds, VVAD winds, and a partially active i_vad mask."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=shape)
    v = rng.normal(size=shape)
    w = rng.normal(size=shape)
    u_vad = rng.normal(size=shape)
    v_vad = rng.normal(size=shape)
    vad_weights = (rng.random(shape) > 0.4).astype(float)
    return u, v, w, u_vad, v_vad, vad_weights


def test_vad_cost():
    """The VVAD cost is the i_vad-weighted sum of squared horizontal
    departures, and is zero when the analysis equals the VVAD."""
    u, v, w, u_vad, v_vad, vad_weights = _make_vad_inputs()

    expected = 2.5 * np.sum((np.square(u - u_vad) + np.square(v - v_vad)) * vad_weights)
    actual = pydda.cost_functions.np.calculate_vad_cost(
        u, v, vad_weights, u_vad, v_vad, coeff=2.5
    )
    np.testing.assert_allclose(actual, expected)

    # No cost where the analysis already matches the VVAD.
    assert (
        pydda.cost_functions.np.calculate_vad_cost(
            u_vad, v_vad, vad_weights, u_vad, v_vad, coeff=2.5
        )
        == 0.0
    )

    # No cost at all where i_vad is zero everywhere.
    assert (
        pydda.cost_functions.np.calculate_vad_cost(
            u, v, np.zeros_like(vad_weights), u_vad, v_vad, coeff=2.5
        )
        == 0.0
    )


def test_vad_gradient():
    """The analytic VVAD gradient matches finite differences, leaves w
    untouched, and vanishes wherever i_vad is zero."""
    u, v, w, u_vad, v_vad, vad_weights = _make_vad_inputs()
    shape = u.shape
    n = int(np.prod(shape))
    coeff = 1.7

    def J(x):
        x = x.reshape((3,) + shape)
        return pydda.cost_functions.np.calculate_vad_cost(
            x[0], x[1], vad_weights, u_vad, v_vad, coeff=coeff
        )

    x0 = np.concatenate([u.ravel(), v.ravel(), w.ravel()])
    finite_diff = approx_fprime(x0, J, 1e-6)
    analytic = pydda.cost_functions.np.calculate_vad_gradient(
        u, v, vad_weights, u_vad, v_vad, coeff=coeff, upper_bc=0
    )
    np.testing.assert_allclose(analytic, finite_diff, atol=1e-4)

    # The VVAD constrains the horizontal wind only.
    assert np.all(analytic[2 * n :] == 0)

    # Eq. (19) of Protat et al. (2024) switches the term off entirely where
    # multi-Doppler information is available, so it must contribute no
    # gradient at those points.
    grad = analytic.reshape((3,) + shape)
    assert np.all(grad[0][vad_weights == 0] == 0)
    assert np.all(grad[1][vad_weights == 0] == 0)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="Jax not installed")
def test_vad_cost_jax():
    """The jax VVAD cost and gradient agree with the numpy ones."""
    u, v, w, u_vad, v_vad, vad_weights = _make_vad_inputs()

    np_cost = pydda.cost_functions.np.calculate_vad_cost(
        u, v, vad_weights, u_vad, v_vad, coeff=2.5
    )
    jax_cost = pydda.cost_functions.jax.calculate_vad_cost(
        u, v, vad_weights, u_vad, v_vad, coeff=2.5
    )
    np.testing.assert_allclose(float(jax_cost), np_cost, rtol=1e-5)

    np_grad = pydda.cost_functions.np.calculate_vad_gradient(
        u, v, vad_weights, u_vad, v_vad, coeff=2.5
    )
    jax_grad = pydda.cost_functions.jax.calculate_vad_gradient(
        u, v, vad_weights, u_vad, v_vad, coeff=2.5
    )
    np.testing.assert_allclose(np.asarray(jax_grad), np_grad, atol=1e-5)


def test_vad_gradient_upper_bc():
    """The VVAD gradient honors the upper boundary condition modes even
    though its own w gradient is identically zero."""
    u, v, w, u_vad, v_vad, vad_weights = _make_vad_inputs()
    mask = np.zeros(u.shape, dtype=bool)
    mask[-2:] = True

    for upper_bc in (0, 1, 2):
        grad = pydda.cost_functions.np.calculate_vad_gradient(
            u,
            v,
            vad_weights,
            u_vad,
            v_vad,
            coeff=1.0,
            upper_bc=upper_bc,
            upper_bc_mask=mask,
        )
        assert np.all(_w_component(grad, u.shape) == 0)
