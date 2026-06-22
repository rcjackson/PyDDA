import numpy as np

try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def calculate_radial_vel_cost_function(
    vrs, azs, els, u, v, w, wts, rmsVr, weights, coeff=1.0
):
    """
    Calculates the cost function due to difference of the wind field from
    radar radial velocities. For more information on this cost function, see
    Potvin et al. (2012) and Shapiro et al. (2009).

    All arrays in the given lists must have the same dimensions and represent
    the same spatial coordinates.

    Parameters
    ----------
    vrs: List of float arrays
        List of radial velocities from each radar
    els: List of float arrays
        List of elevations from each radar
    azs: List of float arrays
        List of azimuths from each radar
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    wts: List of float arrays
        Float array containing fall speed from radar.
    rmsVr: float
        The sum of squares of velocity/num_points. Use for normalization
        of data weighting coefficient
    weights: n_radars x_bins x y_bins float array
        Data weights for each pair of radars
    coeff: float
        Constant for cost function

    Returns
    -------
    J_o: float
         Observational cost function

    References
    -----------

    Potvin, C.K., A. Shapiro, and M. Xue, 2012: Impact of a Vertical Vorticity
    Constraint in Variational Dual-Doppler Wind Analysis: Tests with Real and
    Simulated Supercell Data. J. Atmos. Oceanic Technol., 29, 32–49,
    https://doi.org/10.1175/JTECH-D-11-00019.1

    Shapiro, A., C.K. Potvin, and J. Gao, 2009: Use of a Vertical Vorticity
    Equation in Variational Dual-Doppler Wind Analysis. J. Atmos. Oceanic
    Technol., 26, 2089–2106, https://doi.org/10.1175/2009JTECHA1256.1
    """

    J_o = tf.Variable(0.0, dtype=tf.float32)
    lambda_o = coeff / (rmsVr * rmsVr)

    for i in range(len(vrs)):
        v_ar = (
            tf.math.cos(els[i]) * tf.math.sin(azs[i]) * u
            + tf.math.cos(els[i]) * tf.math.cos(azs[i]) * v
            + tf.math.sin(els[i]) * (w - tf.math.abs(wts[i]))
        )
        J_o.assign_add(
            lambda_o * tf.reduce_sum(tf.math.square(vrs[i] - v_ar) * weights[i])
        )
    return J_o


def calculate_grad_radial_vel(
    vrs, els, azs, u, v, w, wts, weights, rmsVr, coeff=1.0, upper_bc=True, lower_bc=True
):
    """
    Calculates the gradient of the cost function due to difference of wind
    field from radar radial velocities.

    All arrays in the given lists must have the same dimensions and represent
    the same spatial coordinates.

    Parameters
    ----------
    vrs: List of float arrays
        List of radial velocities from each radar
    els: List of float arrays
        List of elevations from each radar
    azs: List of azimuths
        List of azimuths from each radar
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    coeff: float
        Constant for cost function
    vel_name: str
        Background velocity field name
    weights: n_radars x_bins x y_bins float array
        Data weights for each pair of radars
    upper_bc: bool
        Set to true to impose w=0 at top of domain.
    lower_bc: bool
        Set to true to impose w=0 at bottom of domain.
    Returns
    -------
    y: 1-D float array
         Gradient vector of observational cost function.

    More information
    ----------------
    The gradient is calculated by taking the functional derivative of the
    cost function. For more information on functional derivatives, see the
    Euler-Lagrange Equation:

    https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation
    """

    p_x1 = tf.Variable(tf.zeros(vrs[0].shape))
    p_y1 = tf.Variable(tf.zeros(vrs[0].shape))
    p_z1 = tf.Variable(tf.zeros(vrs[0].shape))
    lambda_o = coeff / (rmsVr * rmsVr)

    for i in range(len(vrs)):
        v_ar = (
            tf.math.cos(els[i]) * tf.math.sin(azs[i]) * u
            + tf.math.cos(els[i]) * tf.math.cos(azs[i]) * v
            + tf.math.sin(els[i]) * (w - tf.math.abs(wts[i]))
        )

        x_grad = (
            2 * (v_ar - vrs[i]) * tf.math.cos(els[i]) * tf.math.sin(azs[i]) * weights[i]
        ) * lambda_o
        y_grad = (
            2 * (v_ar - vrs[i]) * tf.math.cos(els[i]) * tf.math.cos(azs[i]) * weights[i]
        ) * lambda_o
        z_grad = (2 * (v_ar - vrs[i]) * tf.math.sin(els[i]) * weights[i]) * lambda_o

        p_x1.assign_add(x_grad)
        p_y1.assign_add(y_grad)
        p_z1.assign_add(z_grad)

    # Impermeability condition
    if lower_bc is True:
        p_z1 = tf.concat(
            [tf.zeros((1, u.shape[1], u.shape[2]), dtype=tf.float32), p_z1[1:, :, :]],
            axis=0,
        )
    if upper_bc is True:
        p_z1 = tf.concat(
            [p_z1[:-1, :, :], tf.zeros((1, u.shape[1], u.shape[2]), dtype=tf.float32)],
            axis=0,
        )
    y = tf.stack((p_x1, p_y1, p_z1), axis=0)
    return tf.reshape(y, (3 * np.prod(u.shape),))


def calculate_smoothness_cost(u, v, w, dx, dy, dz, Cx=1e-5, Cy=1e-5, Cz=1e-5):
    """
    Calculates the smoothness cost function by taking the Laplacian of the
    wind field.

    All arrays in the given lists must have the same dimensions and represent
    the same spatial coordinates.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    Cx: float
        Constant controlling smoothness in x-direction
    Cy: float
        Constant controlling smoothness in y-direction
    Cz: float
        Constant controlling smoothness in z-direction

    Returns
    -------
    Js: float
        value of smoothness cost function
    """
    dudx = _tf_gradient(u, dx, axis=2)
    dudy = _tf_gradient(u, dy, axis=1)
    dudz = _tf_gradient(u, dz, axis=0)
    dvdx = _tf_gradient(v, dx, axis=2)
    dvdy = _tf_gradient(v, dy, axis=1)
    dvdz = _tf_gradient(v, dz, axis=0)
    dwdx = _tf_gradient(w, dx, axis=2)
    dwdy = _tf_gradient(w, dy, axis=1)
    dwdz = _tf_gradient(w, dz, axis=0)

    x_term = (
        Cx
        * (
            _tf_gradient(dudx, dx, axis=2)
            + _tf_gradient(dvdx, dx, axis=2)
            + _tf_gradient(dwdx, dx, axis=2)
        )
        ** 2
    )
    y_term = (
        Cy
        * (
            _tf_gradient(dudy, dy, axis=1)
            + _tf_gradient(dvdy, dy, axis=1)
            + _tf_gradient(dwdy, dy, axis=1)
        )
        ** 2
    )
    z_term = (
        Cz
        * (
            _tf_gradient(dudz, dz, axis=0)
            + _tf_gradient(dvdz, dz, axis=0)
            + _tf_gradient(dwdz, dz, axis=0)
        )
        ** 2
    )
    return tf.math.reduce_sum(x_term + y_term + z_term)


def calculate_smoothness_gradient(
    u, v, w, dx, dy, dz, Cx=1e-5, Cy=1e-5, Cz=1e-5, upper_bc=True, lower_bc=True
):
    """
    Calculates the gradient of the smoothness cost function
    by taking the Laplacian of the Laplacian of the wind field.

    All arrays in the given lists must have the same dimensions and represent
    the same spatial coordinates.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    Cx: float
        Constant controlling smoothness in x-direction
    Cy: float
        Constant controlling smoothness in y-direction
    Cz: float
        Constant controlling smoothness in z-direction
    upper_bc: bool
        Set to true to impose w=0 at top of domain.
    lower_bc: bool
        Set to true to impose w=0 at bottom of domain.
    Returns
    -------
    y: float array
        value of gradient of smoothness cost function
    """

    with tf.GradientTape() as tape:
        tape.watch(u)
        tape.watch(v)
        tape.watch(w)
        loss = calculate_smoothness_cost(u, v, w, dx, dy, dz, Cx=Cx, Cy=Cy, Cz=Cz)

    vars = {"u": u, "v": v, "w": w}
    grad = tape.gradient(loss, vars)
    p_x1 = grad["u"]
    p_y1 = grad["v"]
    p_z1 = grad["w"]

    # Impermeability condition
    if lower_bc is True:
        p_z1 = tf.concat(
            [tf.zeros((1, u.shape[1], u.shape[2]), dtype=tf.float32), p_z1[1:, :, :]],
            axis=0,
        )
    if upper_bc is True:
        p_z1 = tf.concat(
            [p_z1[:-1, :, :], tf.zeros((1, u.shape[1], u.shape[2]), dtype=tf.float32)],
            axis=0,
        )
    y = tf.stack((p_x1, p_y1, p_z1), axis=0)
    return tf.reshape(y, (3 * np.prod(u.shape),))


def calculate_point_cost(u, v, x, y, z, point_list, Cp=1e-3, roi=500.0):
    """
    Calculates the cost function related to point observations. A mean square error cost
    function term is applied to points that are within the sphere of influence
    whose radius is determined by *roi*.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    x:  Float array
        X coordinates of grid centers
    y:  Float array
        Y coordinates of grid centers
    z:  Float array
        Z coordinated of grid centers
    point_list: list of dicts
        List of point constraints.
        Each member is a dict with keys of "u", "v", to correspond
        to each component of the wind field and "x", "y", "z"
        to correspond to the location of the point observation.

        In addition, "site_id" gives the METAR code (or name) to the station.
    Cp: float
        The weighting coefficient of the point cost function.
    roi: float
        Radius of influence of observations

    Returns
    -------
    J: float
        The cost function related to the difference between wind field and points.

    """
    J = tf.Variable(0.0)

    for the_point in point_list:
        # Instead of worrying about whole domain, just find points in radius of influence
        # Since we know that the weight will be zero outside the sphere of influence anyways
        xp = tf.ones_like(x) * the_point["x"]
        yp = tf.ones_like(y) * the_point["y"]
        zp = tf.ones_like(z) * the_point["z"]
        up = tf.ones_like(u) * the_point["u"]
        vp = tf.ones_like(v) * the_point["v"]

        the_box = tf.where(
            tf.math.logical_and(
                tf.math.logical_and(
                    tf.math.abs(x - xp) < roi, tf.math.abs(y - yp) < roi
                ),
                tf.math.abs(z - zp) < roi,
            ),
            1.0,
            0.0,
        )
        J.assign_add(tf.math.reduce_sum(((u - up) ** 2 + (v - vp) ** 2) * the_box))

    return J * Cp


def calculate_point_gradient(u, v, x, y, z, point_list, Cp=1e-3, roi=500.0):
    """
    Calculates the gradient of the cost function related to point observations.
    A mean square error cost function term is applied to points that are within the sphere of influence
    whose radius is determined by *roi*.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    x: Float array
        X coordinates of grid centers
    y: Float array
        Y coordinates of grid centers
    z: Float array
        Z coordinated of grid centers
    point_list: list of dicts
        List of point constraints. Each member is a dict with keys of "u", "v",
        to correspond to each component of the wind field and "x", "y", "z"
        to correspond to the location of the point observation.

        In addition, "site_id" gives the METAR code (or name) to the station.
    Cp: float
        The weighting coefficient of the point cost function.
    roi: float
        Radius of influence of observations
    Returns
    -------
    gradJ: float array
        The gradient of the cost function related to the difference between wind field and points.

    """
    gradJ_u = tf.Variable(tf.zeros(u.shape, dtype=tf.float32))
    gradJ_v = tf.Variable(tf.zeros(v.shape, dtype=tf.float32))
    gradJ_w = tf.Variable(tf.zeros(u.shape, dtype=tf.float32))
    for the_point in point_list:
        # Instead of worrying about whole domain, just find points in radius of influence
        # Since we know that the weight will be zero outside the sphere of influence anyways
        xp = tf.ones_like(x, dtype=tf.float32) * the_point["x"]
        yp = tf.ones_like(y, dtype=tf.float32) * the_point["y"]
        zp = tf.ones_like(z, dtype=tf.float32) * the_point["z"]
        up = tf.ones_like(u, dtype=tf.float32) * the_point["u"]
        vp = tf.ones_like(v, dtype=tf.float32) * the_point["v"]

        the_box = tf.where(
            tf.math.logical_and(
                tf.math.logical_and(
                    tf.math.abs(x - xp) < roi, tf.math.abs(y - yp) < roi
                ),
                tf.math.abs(z - zp) < roi,
            ),
            1.0,
            0.0,
        )
        gradJ_u.assign_add((2 * (u - up) * the_box))
        gradJ_v.assign_add((2 * (v - vp) * the_box))
    gradJ = tf.stack([gradJ_u, gradJ_v, gradJ_w], axis=0)
    gradJ = tf.reshape(gradJ, (3 * np.prod(u.shape),))
    return gradJ * Cp


def _tf_diff(x, axis):
    if axis == 0:
        return x[1:, :, :] - x[:-1, :, :]
    elif axis == 1:
        return x[:, 1:, :] - x[:, :-1, :]
    elif axis == 2:
        return x[:, :, 1:] - x[:, :, :-1]


def _tf_gradient(x, dx, axis):
    if axis == 0:
        fd = (
            _tf_diff(tf.concat([x, tf.expand_dims(x[-1, :, :], 0)], axis=0), axis=0)
            / dx
        )
        bd = (
            _tf_diff(tf.concat([tf.expand_dims(x[0, :, :], 0), x], axis=0), axis=0) / dx
        )
    elif axis == 1:
        fd = (
            _tf_diff(tf.concat([x, tf.expand_dims(x[:, -1, :], 1)], axis=1), axis=1)
            / dx
        )
        bd = (
            _tf_diff(tf.concat([tf.expand_dims(x[:, 0, :], 1), x], axis=1), axis=1) / dx
        )
    elif axis == 2:
        fd = (
            _tf_diff(tf.concat([x, tf.expand_dims(x[:, :, -1], 2)], axis=2), axis=2)
            / dx
        )
        bd = (
            _tf_diff(tf.concat([tf.expand_dims(x[:, :, 0], 2), x], axis=2), axis=2) / dx
        )

    cd = (fd + bd) / 2

    if axis == 0:
        return tf.concat(
            [
                tf.expand_dims(fd[0, :, :], 0),
                cd[1:-1, :, :],
                tf.expand_dims(bd[-1, :, :], 0),
            ],
            axis=0,
        )
    elif axis == 1:
        return tf.concat(
            [
                tf.expand_dims(fd[:, 0, :], 1),
                cd[:, 1:-1, :],
                tf.expand_dims(bd[:, -1, :], 1),
            ],
            axis=1,
        )
    elif axis == 2:
        return tf.concat(
            [
                tf.expand_dims(fd[:, :, 0], 2),
                cd[:, :, 1:-1],
                tf.expand_dims(bd[:, :, -1], 2),
            ],
            axis=2,
        )


def calculate_mass_continuity(u, v, w, z, dx, dy, dz, coeff=1500.0, anel=1):
    """
    Calculates the mass continuity cost function by taking the divergence
    of the wind field.

    All arrays in the given lists must have the same dimensions and represent
    the same spatial coordinates.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    dx: float
        Grid spacing in x direction.
    dy: float
        Grid spacing in y direction.
    dz: float
        Grid spacing in z direction.
    z: Float array (1D)
        1D Float array with heights of grid
    coeff: float
        Constant controlling contribution of mass continuity to cost function
    anel: int
        = 1 use anelastic approximation, 0=don't

    Returns
    -------
    J: float
        value of mass continuity cost function

    """
    # Jax version of the cost function
    rho = tf.math.exp(-z / 10000.0)
    dudx = _tf_gradient(u, dx, axis=2)
    dvdy = _tf_gradient(v, dy, axis=1)
    dwdz = _tf_gradient(w, dz, axis=0)

    if anel == 1:
        drho_dz = _tf_gradient(rho, dz, axis=0)
        anel_term = w / rho * drho_dz
    else:
        anel_term = tf.zeros(w.shape)

    return (
        coeff * tf.math.reduce_sum(tf.math.square(dudx + dvdy + dwdz + anel_term)) / 2.0
    )


def calculate_mass_continuity_gradient(
    u, v, w, z, dx, dy, dz, coeff=1500.0, anel=1, upper_bc=True, lower_bc=True
):
    """
    Calculates the gradient of mass continuity cost function. This is done by
    taking the negative gradient of the divergence of the wind field.

    All grids must have the same grid specification.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    z: Float array (1D)
        1D Float array with heights of grid
    dx: float
        Grid spacing in x direction.
    dy: float
        Grid spacing in y direction.
    dz: float
        Grid spacing in z direction.
    coeff: float
        Constant controlling contribution of mass continuity to cost function
    anel: int
        = 1 use anelastic approximation, 0=don't
    upper_bc: bool
        Set to true to impose w=0 at top of domain.

    Returns
    -------
    y: float array
        value of gradient of mass continuity cost function

    """

    rho = tf.math.exp(-z / 10000.0)
    dudx = _tf_gradient(u, dx, axis=2)
    dvdy = _tf_gradient(v, dy, axis=1)
    dwdz = _tf_gradient(w, dz, axis=0)

    if anel == 1:
        drho_dz = _tf_gradient(rho, dz, axis=0)
        anel_term = w / rho * drho_dz
    else:
        anel_term = tf.zeros(w.shape)
    div = dudx + dvdy + dwdz + anel_term

    p_x1 = -_tf_gradient(div, dx, axis=2) * coeff
    p_y1 = -_tf_gradient(div, dy, axis=1) * coeff
    p_z1 = -_tf_gradient(div, dz, axis=0) * coeff

    # Impermeability condition
    if lower_bc is True:
        p_z1 = tf.concat(
            [tf.zeros((1, u.shape[1], u.shape[2]), dtype=tf.float32), p_z1[1:, :, :]],
            axis=0,
        )
    if upper_bc is True:
        p_z1 = tf.concat(
            [p_z1[:-1, :, :], tf.zeros((1, u.shape[1], u.shape[2]), dtype=tf.float32)],
            axis=0,
        )
    y = tf.stack((p_x1, p_y1, p_z1), axis=0)
    return tf.reshape(y, (3 * np.prod(u.shape),))


def calculate_background_cost(u, v, weights, u_back, v_back, Cb=0.01):
    """
    Calculates the background cost function. The background cost function is
    simply the sum of the squared differences between the wind field and the
    background wind field multiplied by the weighting coefficient.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    weights: Float array
        Weights for each point to consider into cost function
    u_back: 1D float array
        Zonal winds vs height from sounding
    w_back: 1D float array
        Meridional winds vs height from sounding
    Cb: float
        Weight of background constraint to total cost function

    Returns
    -------
    cost: float
        value of background cost function
    """
    the_shape = u.shape
    cost = tf.Variable(0.0, dtype=tf.float32)

    for i in range(the_shape[0]):
        cost.assign_add(
            tf.math.reduce_sum(
                Cb
                * (
                    tf.math.square(u[i] - u_back[i]) * weights[i]
                    + tf.math.square(v[i] - v_back[i]) * weights[i]
                )
            )
        )

    return cost


def calculate_background_gradient(u, v, weights, u_back, v_back, Cb=0.01):
    """
    Calculates the gradient of the background cost function. For each u, v
    this is given as 2*coefficent*(analysis wind - background wind).

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    weights: Float array
        Weights for each point to consider into cost function
    u_back: 1D float array
        Zonal winds vs height from sounding
    w_back: 1D float array
        Meridional winds vs height from sounding
    Cb: float
        Weight of background constraint to total cost function
    upper_bc: bool
        Set to true to impose w=0 at top of domain.

    Returns
    -------
    y: float array
        value of gradient of background cost function
    """

    the_shape = u.shape
    u_grad = tf.Variable(tf.zeros(the_shape, dtype=tf.float32))
    v_grad = tf.Variable(tf.zeros(the_shape, dtype=tf.float32))
    w_grad = tf.Variable(tf.zeros(the_shape, dtype=tf.float32))

    for i in range(the_shape[0]):
        add_array = tf.Variable(
            Cb * 2 * (u[i] - u_back[i]) * (weights[i]), dtype=tf.float32
        )
        add_array = tf.concat(
            [
                tf.zeros((i, the_shape[1], the_shape[2]), dtype=tf.float32),
                tf.expand_dims(add_array, 0),
                tf.zeros(
                    (the_shape[0] - i - 1, the_shape[1], the_shape[2]), dtype=tf.float32
                ),
            ],
            axis=0,
        )
        u_grad.assign_add(add_array)
        add_array = tf.Variable(
            Cb * 2 * (v[i] - v_back[i]) * (weights[i]), dtype=tf.float32
        )
        add_array = tf.concat(
            [
                tf.zeros((i, the_shape[1], the_shape[2]), dtype=tf.float32),
                tf.expand_dims(add_array, 0),
                tf.zeros(
                    (the_shape[0] - i - 1, the_shape[1], the_shape[2]), dtype=tf.float32
                ),
            ],
            axis=0,
        )
        v_grad.assign_add(add_array)
    y = np.stack([u_grad, v_grad, w_grad], axis=0)
    return tf.reshape(y, (3 * np.prod(u.shape),))


def calculate_vertical_vorticity_cost(u, v, w, dx, dy, dz, Ut, Vt, coeff=1):
    """
    Calculates the cost function due to deviance from vertical vorticity
    equation. For more information of the vertical vorticity cost function,
    see Potvin et al. (2012) and Shapiro et al. (2009).

    Parameters
    ----------
    u: 3D array
        Float array with u component of wind field
    v: 3D array
        Float array with v component of wind field
    w: 3D array
        Float array with w component of wind field
    dx: float array
        Spacing in x grid
    dy: float array
        Spacing in y grid
    dz: float array
        Spacing in z grid
    coeff: float
        Weighting coefficient
    Ut: float
        U component of storm motion
    Vt: float
        V component of storm motion

    Returns
    -------
    Jv: float
        Value of vertical vorticity cost function.

    References
    ----------

    Potvin, C.K., A. Shapiro, and M. Xue, 2012: Impact of a Vertical Vorticity
    Constraint in Variational Dual-Doppler Wind Analysis: Tests with Real and
    Simulated Supercell Data. J. Atmos. Oceanic Technol., 29, 32–49,
    https://doi.org/10.1175/JTECH-D-11-00019.1

    Shapiro, A., C.K. Potvin, and J. Gao, 2009: Use of a Vertical Vorticity
    Equation in Variational Dual-Doppler Wind Analysis. J. Atmos. Oceanic
    Technol., 26, 2089–2106, https://doi.org/10.1175/2009JTECHA1256.1

    """

    dvdz = _tf_gradient(v, dz, axis=0)
    dudz = _tf_gradient(u, dz, axis=0)
    dvdx = _tf_gradient(v, dx, axis=2)
    dwdy = _tf_gradient(w, dy, axis=1)
    dwdx = _tf_gradient(w, dx, axis=2)
    dudx = _tf_gradient(u, dx, axis=2)
    dvdy = _tf_gradient(v, dy, axis=1)
    dudy = _tf_gradient(u, dy, axis=1)
    # print(dvdx, np.gradient(v.numpy(), dx, axis=2))
    zeta = dvdx - dudy

    dzeta_dx = _tf_gradient(zeta, dx, axis=2)
    dzeta_dy = _tf_gradient(zeta, dy, axis=1)
    dzeta_dz = _tf_gradient(zeta, dz, axis=0)

    jv_array = (
        (u - Ut) * dzeta_dx
        + (v - Vt) * dzeta_dy
        + w * dzeta_dz
        + (dvdz * dwdx - dudz * dwdy)
        + zeta * (dudx + dvdy)
    )

    return tf.math.reduce_sum(coeff * tf.math.square(jv_array))


# Using Jax version of function
def calculate_vertical_vorticity_gradient(
    u, v, w, dx, dy, dz, Ut, Vt, coeff=1e-5, upper_bc=True, lower_bc=True
):
    """
    Calculates the gradient of the cost function due to deviance from vertical
    vorticity equation. This is done by taking the functional derivative of
    the vertical vorticity cost function.

    Parameters
    ----------
    u: 3D array
        Float array with u component of wind field
    v: 3D array
        Float array with v component of wind field
    w: 3D array
        Float array with w component of wind field
    dx: float array
        Spacing in x grid
    dy: float array
        Spacing in y grid
    dz: float array
        Spacing in z grid
    Ut: float
        U component of storm motion
    Vt: float
        V component of storm motion
    coeff: float
        Weighting coefficient
    upper_bc: bool
        Set to true to impose w=0 at top of domain.

    Returns
    -------
    Jv: 1D float array
        Value of the gradient of the vertical vorticity cost function.

    References
    ----------

    Potvin, C.K., A. Shapiro, and M. Xue, 2012: Impact of a Vertical Vorticity
    Constraint in Variational Dual-Doppler Wind Analysis: Tests with Real and
    Simulated Supercell Data. J. Atmos. Oceanic Technol., 29, 32–49,
    https://doi.org/10.1175/JTECH-D-11-00019.1

    Shapiro, A., C.K. Potvin, and J. Gao, 2009: Use of a Vertical Vorticity
    Equation in Variational Dual-Doppler Wind Analysis. J. Atmos. Oceanic
    Technol., 26, 2089–2106, https://doi.org/10.1175/2009JTECHA1256.1

    """

    with tf.GradientTape() as tape:
        tape.watch(u)
        tape.watch(v)
        tape.watch(w)
        loss = calculate_vertical_vorticity_cost(u, v, w, dx, dy, dz, Ut, Vt, coeff)
    vars = {"u": u, "v": v, "w": w}
    grad = tape.gradient(loss, vars)
    p_x1 = grad["u"]
    p_y1 = grad["v"]
    p_z1 = grad["w"]

    # Impermeability condition
    if lower_bc is True:
        p_z1 = tf.concat(
            [tf.zeros((1, u.shape[1], u.shape[2]), dtype=tf.float32), p_z1[1:, :, :]],
            axis=0,
        )
    if upper_bc is True:
        p_z1 = tf.concat(
            [p_z1[:-1, :, :], tf.zeros((1, u.shape[1], u.shape[2]), dtype=tf.float32)],
            axis=0,
        )
    y = tf.stack((p_x1, p_y1, p_z1), axis=0)
    return tf.reshape(y, (3 * np.prod(u.shape),))


def calculate_model_cost(u, v, w, weights, u_model, v_model, w_model, coeff=1.0):
    """
    Calculates the cost function for the model constraint.
    This is calculated simply as the sum of squares of the differences
    between the model wind field and the analysis wind field. Vertical
    velocities are not factored into this cost function as there is typically
    a high amount of uncertainty in model derived vertical velocities.

    Parameters
    ----------
    u: 3D array
        Float array with u component of wind field
    v: 3D array
        Float array with v component of wind field
    w: 3D array
        Float array with w component of wind field
    weights: list of 3D arrays
        Float array showing how much each point from model weighs into
        constraint.
    u_model: list of 3D arrays
        Float array with u component of wind field from model
    v_model: list of 3D arrays
        Float array with v component of wind field from model
    w_model: list of 3D arrays
        Float array with w component of wind field from model
    coeff: float
        Weighting coefficient

    Returns
    -------
    Jv: float
        Value of model cost function
    """

    cost = 0
    for i in range(len(u_model)):
        cost += coeff * tf.math.reduce_sum(
            tf.math.square(u - u_model[i]) * weights[i]
            + tf.math.square(v - v_model[i]) * weights[i]
        )
    return cost


def calculate_model_gradient(
    u, v, w, weights, u_model, v_model, w_model, coeff=1.0, upper_bc=True, lower_bc=True
):
    """
    Calculates the cost function for the model constraint.
    This is calculated simply as twice the differences
    between the model wind field and the analysis wind field for each u, v.
    Vertical velocities are not factored into this cost function as there is
    typically a high amount of uncertainty in model derived vertical
    velocities. Therefore, the gradient for all of the w's will be 0.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    weights: list of 3D float arrays
        Weights for each point to consider into cost function
    u_model: list of 3D float arrays
        Zonal wind field from model
    v_model: list of 3D float arrays
        Meridional wind field from model
    w_model: list of 3D float arrays
        Vertical wind field from model
    coeff: float
        Weight of background constraint to total cost function

    Returns
    -------
    y: float array
        value of gradient of background cost function
    """

    with tf.GradientTape() as tape:
        tape.watch(u)
        tape.watch(v)
        tape.watch(w)
        loss = calculate_model_cost(
            u, v, w, weights, u_model, v_model, w_model, coeff=coeff
        )

    vars = {"u": u, "v": v, "w": w}
    grad = tape.gradient(loss, vars)
    p_x1 = grad["u"]
    p_y1 = grad["v"]
    p_z1 = tf.zeros(p_x1.shape)

    # Impermeability condition
    if lower_bc is True:
        p_z1 = tf.concat(
            [tf.zeros((1, u.shape[1], u.shape[2]), dtype=tf.float32), p_z1[1:, :, :]],
            axis=0,
        )
    if upper_bc is True:
        p_z1 = tf.concat(
            [p_z1[:-1, :, :], tf.zeros((1, u.shape[1], u.shape[2]), dtype=tf.float32)],
            axis=0,
        )
    y = tf.stack((p_x1, p_y1, p_z1), axis=0)
    return tf.reshape(y, (3 * np.prod(u.shape),))
