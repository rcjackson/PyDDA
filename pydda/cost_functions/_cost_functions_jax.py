import numpy as np
import scipy

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


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
    # Jax version of the cost function
    J_o = 0
    lambda_o = coeff / (rmsVr * rmsVr)
    for i in range(len(vrs)):
        v_ar = (
            jnp.cos(els[i]) * jnp.sin(azs[i]) * u
            + jnp.cos(els[i]) * jnp.cos(azs[i]) * v
            + jnp.sin(els[i]) * (w - jnp.abs(wts[i]))
        )
        the_weight = jnp.asarray(weights[i])
        J_o += lambda_o * jnp.sum(jnp.square(vrs[i] - v_ar) * the_weight)
    return J_o


def calculate_grad_radial_vel(
    vrs, els, azs, u, v, w, wts, weights, rmsVr, coeff=1.0, upper_bc=True
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

    Returns
    -------
    y: 1-D float array
         Gradient vector of observational cost function.

    More information
    ----------------
    The gradient is calculated using Jax's vector Jacobian product.

    # Use zero for all masked values since we don't want to add them into
    # the cost function
    """
    primals, fun_vjp = jax.vjp(
        calculate_radial_vel_cost_function,
        vrs,
        azs,
        els,
        u,
        v,
        w,
        wts,
        rmsVr,
        weights,
        coeff,
    )
    _, _, _, p_x1, p_y1, p_z1, _, _, _, _ = fun_vjp(1.0)

    # Impermeability condition
    p_z1 = p_z1.at[0, :, :].set(0)
    if upper_bc is True:
        p_z1 = p_z1.at[-1, :, :].set(0)
    y = jnp.stack((p_x1, p_y1, p_z1), axis=0)
    return y.flatten()


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
    dx: float
        Grid spacing in x-direction
    dy: float
        Grid spacing in in y-direction
    dz: float
        Grid spacing in in z-direction
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
    dudx = jnp.gradient(u, dx, axis=2)
    dudy = jnp.gradient(u, dy, axis=1)
    dudz = jnp.gradient(u, dz, axis=0)
    dvdx = jnp.gradient(v, dx, axis=2)
    dvdy = jnp.gradient(v, dy, axis=1)
    dvdz = jnp.gradient(v, dz, axis=0)
    dwdx = jnp.gradient(w, dx, axis=2)
    dwdy = jnp.gradient(w, dy, axis=1)
    dwdz = jnp.gradient(w, dz, axis=0)

    x_term = (
        Cx
        * (
            jnp.gradient(dudx, dx, axis=2)
            + jnp.gradient(dvdx, dx, axis=2)
            + jnp.gradient(dwdx, dx, axis=2)
        )
        ** 2
    )
    y_term = (
        Cy
        * (
            jnp.gradient(dudy, dy, axis=1)
            + jnp.gradient(dvdy, dy, axis=1)
            + jnp.gradient(dwdy, dy, axis=1)
        )
        ** 2
    )
    z_term = (
        Cz
        * (
            jnp.gradient(dudz, dz, axis=0)
            + jnp.gradient(dvdz, dz, axis=0)
            + jnp.gradient(dwdz, dz, axis=0)
        )
        ** 2
    )
    return jnp.sum(x_term + y_term + z_term)


def calculate_smoothness_gradient(
    u, v, w, dx, dy, dz, Cx=1e-5, Cy=1e-5, Cz=1e-5, upper_bc=True
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
    dx: float
        Grid spacing in x-direction
    dy: float
        Grid spacing in in y-direction
    dz: float
        Grid spacing in in z-direction
    Cx: float
        Constant controlling smoothness in x-direction
    Cy: float
        Constant controlling smoothness in y-direction
    Cz: float
        Constant controlling smoothness in z-direction

    Returns
    -------
    y: float array
        value of gradient of smoothness cost function
    """
    primals, fun_vjp = jax.vjp(
        calculate_smoothness_cost, u, v, w, dx, dy, dz, Cx, Cy, Cz
    )
    grad_u, grad_v, grad_w, _, _, _, _, _, _ = fun_vjp(1.0)

    # Impermeability condition
    grad_w = grad_w.at[0, :, :].set(0)
    if upper_bc is True:
        grad_w = grad_w.at[-1, :, :].set(0)
    y = jnp.stack([grad_u, grad_v, grad_w], axis=0)

    return y.flatten()


def calculate_point_cost(u, v, x, y, z, point_list, Cp=1e-3):
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
        The cost function related to the difference between
        wind field and points.
    """
    J = 0.0
    for the_point in point_list:
        dist = jnp.sqrt(
            (x - the_point["x"]) ** 2
            + (y - the_point["y"]) ** 2
            + (z - the_point["z"]) ** 2
        )
        dist = jnp.maximum(dist, 1.0)
        weight = 1 / dist**2
        weight = weight / jnp.sum(weight)
        J += jnp.sum(weight * ((u - the_point["u"]) ** 2 + (v - the_point["v"]) ** 2))

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

    gradJ_u = jnp.zeros_like(u)
    gradJ_v = jnp.zeros_like(v)
    gradJ_w = jnp.zeros_like(u)

    for the_point in point_list:
        dist = jnp.sqrt(
            (x - the_point["x"]) ** 2
            + (y - the_point["y"]) ** 2
            + (z - the_point["z"]) ** 2
        )
        dist = jnp.maximum(dist, 1.0)
        weight = 1 / dist**2
        weight = weight / jnp.sum(weight)
        gradJ_u += 2 * (u - the_point["u"]) * weight
        gradJ_v += 2 * (v - the_point["v"]) * weight

    gradJ = jnp.stack([gradJ_u, gradJ_v, gradJ_w], axis=0).flatten()
    return gradJ * Cp


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
    dudx = jnp.gradient(u, dx, axis=2)
    dvdy = jnp.gradient(v, dy, axis=1)
    dwdz = jnp.gradient(w, dz, axis=0)

    if anel == 1:
        if not isinstance(z, np.ma.MaskedArray):
            rho = jnp.exp(-z / 10000.0)
        else:
            rho = jnp.exp(-z.filled() / 10000.0)
        drho_dz = jnp.gradient(rho, dz, axis=0)
        anel_term = w / rho * drho_dz
    else:
        anel_term = jnp.zeros(w.shape)
    return coeff * jnp.sum(jnp.square(dudx + dvdy + dwdz + anel_term)) / 2.0


def calculate_mass_continuity_gradient(
    u, v, w, z, dx, dy, dz, coeff=1500.0, anel=1, upper_bc=True
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

    Returns
    -------
    y: float array
        value of gradient of mass continuity cost function
    """
    # Z should not be masked, but just in case it is
    if isinstance(z, np.ma.MaskedArray):
        z_in = z.filled(-9999.0)
    else:
        z_in = z
    primals, fun_vjp = jax.vjp(
        calculate_mass_continuity, u, v, w, z_in, dx, dy, dz, coeff, anel
    )
    grad_u, grad_v, grad_w, _, _, _, _, _, _ = fun_vjp(1.0)

    # Impermeability condition
    grad_w = grad_w.at[0, :, :].set(0)
    if upper_bc is True:
        grad_w = grad_w.at[-1, :, :].set(0)
    y = jnp.stack([grad_u, grad_v, grad_w], axis=0)
    return y.flatten()


def calculate_background_cost(u, v, w, weights, u_back, v_back, Cb=0.01):
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
    cost = 0
    for i in range(the_shape[0]):
        cost += Cb * jnp.sum(
            jnp.square(u[i] - u_back[i]) * (weights[i])
            + jnp.square(v[i] - v_back[i]) * (weights[i])
        )
    return cost


def calculate_background_gradient(u, v, w, weights, u_back, v_back, Cb=0.01):
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

    Returns
    -------
    y: float array
        value of gradient of background cost function
    """
    primals, fun_vjp = jax.vjp(
        calculate_background_cost, u, v, w, weights, u_back, v_back, Cb
    )
    u_grad, v_grad, w_grad, _, _, _, _ = fun_vjp(1.0)
    y = jnp.stack([u_grad, v_grad, w_grad], axis=0)
    return y.flatten().copy()


def calculate_vertical_vorticity_cost(u, v, w, dx, dy, dz, Ut, Vt, coeff=1e-5):
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
    # Jax version of the cost function
    dvdz = jnp.gradient(v, dz, axis=0)
    dudz = jnp.gradient(u, dz, axis=0)
    jnp.gradient(w, dz, axis=0)
    dvdx = jnp.gradient(v, dx, axis=2)
    dwdy = jnp.gradient(w, dy, axis=1)
    dwdx = jnp.gradient(w, dx, axis=2)
    dudx = jnp.gradient(u, dx, axis=2)
    dvdy = jnp.gradient(v, dy, axis=1)
    dudy = jnp.gradient(u, dy, axis=1)
    zeta = dvdx - dudy
    dzeta_dx = jnp.gradient(zeta, dx, axis=2)
    dzeta_dy = jnp.gradient(zeta, dy, axis=1)
    dzeta_dz = jnp.gradient(zeta, dz, axis=0)
    jv_array = (
        (u - Ut) * dzeta_dx
        + (v - Vt) * dzeta_dy
        + w * dzeta_dz
        + (dvdz * dwdx - dudz * dwdy)
        + zeta * (dudx + dvdy)
    )

    return jnp.sum(coeff * jv_array**2)


def calculate_vertical_vorticity_gradient(
    u, v, w, dx, dy, dz, Ut, Vt, coeff=1e-5, upper_bc=True
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
    # Jax version of the gradient cost function
    primals, fun_vjp = jax.vjp(
        calculate_vertical_vorticity_cost, u, v, w, dx, dy, dz, Ut, Vt, coeff
    )
    u_grad, v_grad, w_grad, _, _, _, _, _, _ = fun_vjp(1.0)
    # Impermeability condition
    w_grad.at[0, :, :].set(0)
    if upper_bc is True:
        w_grad.at[-1, :, :].set(0)
    y = jnp.stack([u_grad, v_grad, w_grad], axis=0)
    return y.flatten().copy()


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
        cost += coeff * jnp.sum(
            jnp.square(u - u_model[i]) * weights[i]
            + jnp.square(v - v_model[i]) * weights[i]
        )
    return cost


def calculate_model_gradient(u, v, w, weights, u_model, v_model, w_model, coeff=1.0):
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
        value of gradient of model cost function
    """
    primals, fun_vjp = jax.vjp(
        calculate_model_cost, u, v, w, weights, u_model, v_model, w_model, coeff
    )
    u_grad, v_grad, w_grad, _, _, _, _, _ = fun_vjp(1.0)
    y = jnp.stack([u_grad, v_grad, w_grad], axis=0)
    return y.flatten().copy()
