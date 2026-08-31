"""
Variational velocity azimuth display (VVAD) retrieval and constraint.

This module implements the VVAD component of the SWIRL wind analysis system
described in Protat et al. (2024), https://doi.org/10.1175/JTECH-D-23-0155.1.
"""

import numpy as np
import pyart
import xarray as xr

#: Names of the linear wind model profiles retrieved by :func:`vvad_retrieval`.
_PROFILE_NAMES = ["U0", "V0", "DIV", "DET", "DES"]


def _radar_xy(Grid):
    """
    Returns the (x, y) position of the Grid's radar in the Grid's own
    Cartesian coordinate system, in m.
    """
    x0, y0 = pyart.core.geographic_to_cartesian(
        Grid["radar_longitude"].values[0],
        Grid["radar_latitude"].values[0],
        Grid["projection"].attrs,
    )
    return float(np.asarray(x0).squeeze()), float(np.asarray(y0).squeeze())


def _geometry(Grid):
    """
    Returns (dx, dy, rh, az, el) for every grid point relative to the Grid's
    radar. *dx*/*dy* are the Cartesian offsets from the radar in m, *rh* is the
    horizontal distance from the radar in m, and *az*/*el* are in radians.

    The azimuth is taken from the Cartesian offsets rather than from the
    Grid's great-circle ``AZ`` field, because the VAD linear wind model of
    Eqs. (1) and (2) of Protat et al. (2024) is defined in the Cartesian
    analysis frame; using the Cartesian bearing keeps the forward model
    algebraically exact. The elevation is taken from the Grid's ``EL`` field so
    that the 4/3 earth beam propagation already computed by
    :func:`pydda.retrieval.angles.add_elevation_as_field` is respected.
    """
    x0, y0 = _radar_xy(Grid)
    dx = Grid["point_x"].values - x0
    dy = Grid["point_y"].values - y0
    rh = np.sqrt(dx**2 + dy**2)
    az = np.arctan2(dx, dy)
    el = np.deg2rad(np.ma.masked_invalid(Grid["EL"].values.squeeze()))
    return dx, dy, rh, az, el


def _design_matrix(rh, az, el, fit_fall_speed):
    """
    Builds the VVAD forward model of Eqs. (3)-(8) of Protat et al. (2024),
    expressed directly in terms of the control variables rather than the
    Fourier coefficients. Substituting Eqs. (4)-(8) into Eq. (3) gives

        Vr = cos(el) sin(az)          U0
           + cos(el) cos(az)          V0
           + 0.5 rh cos(el)           DIV
           - 0.5 rh cos(el) cos(2 az) DET
           + 0.5 rh cos(el) sin(2 az) DES
           - sin(el)                  V_T

    where *rh* is the horizontal distance from the radar. Note that Eq. (4) of
    the paper writes this coefficient as ``0.5 R cos(el)`` for a range *R*
    "along the radar beam"; that is exact when *R* is the horizontal range, as
    used here, and picks up a second factor of cos(el) if *R* is read as the
    slant range. The horizontal range is used because it is what the linear
    wind model of Eqs. (1) and (2) is actually a function of.

    Returns an (npoints, ncoeff) array whose columns are ordered as
    ``U0, V0, DIV, DET, DES`` and, if *fit_fall_speed*, ``V_T``.
    """
    cos_el = np.cos(el)
    half = 0.5 * rh * cos_el
    columns = [
        cos_el * np.sin(az),
        cos_el * np.cos(az),
        half,
        -half * np.cos(2 * az),
        half * np.sin(2 * az),
    ]
    if fit_fall_speed:
        columns.append(-np.sin(el))
    return np.stack(columns, axis=-1)


def _azimuth_coverage(az, n_sectors=12):
    """
    Returns the fraction of the *n_sectors* equal azimuth sectors that contain
    at least one point. A second-order Fourier series in azimuth cannot be
    determined from a narrow sector of echo, so this is the physical
    requirement that :func:`vvad_retrieval` enforces alongside the point count.
    """
    if az.size == 0:
        return 0.0
    sectors = np.floor((np.rad2deg(az) % 360.0) / (360.0 / n_sectors)).astype(int)
    return len(np.unique(sectors)) / n_sectors


def vvad_retrieval(
    Grid,
    vel_field=None,
    min_points=100,
    min_azimuth_coverage=0.5,
    max_condition=50.0,
    max_range=150000.0,
    fit_fall_speed=True,
):
    """
    Retrieves the vertical profiles of the linear wind model from a single
    radar using the variational velocity azimuth display (VVAD) technique of
    Protat et al. (2024).

    For each height level, the horizontal wind is assumed to be linear,

    .. math::

        U(X, Y) = U_0 + U_X (X - X_0) + U_Y (Y - Y_0)

        V(X, Y) = V_0 + V_X (X - X_0) + V_Y (Y - Y_0)

    so that the Doppler velocity is a second-order Fourier series in azimuth
    [Eqs. (3)-(8) of Protat et al. (2024)]. The control variables retrieved at
    each level are :math:`U_0`, :math:`V_0`, the horizontal divergence
    :math:`\\mathrm{DIV} = U_X + V_Y`, the stretching deformation
    :math:`\\mathrm{DET} = U_X - V_Y`, and the shearing deformation
    :math:`\\mathrm{DES} = U_Y + V_X`.

    Parameters
    ----------
    Grid: xarray Dataset
        A PyDDA Grid for a single radar, as returned by
        :func:`pydda.io.read_grid` or :func:`pydda.io.read_from_pyart_grid`.
    vel_field: str or None
        Name of the Doppler velocity field in *Grid*. Uses the Py-ART default
        for corrected velocity if None.
    min_points: int
        Minimum number of valid Doppler velocities required at a height level
        for the retrieval at that level to be considered valid. Protat et al.
        (2024) use 100, determined empirically to filter out noisy retrievals.
    min_azimuth_coverage: float
        Minimum fraction of the twelve 30 degree azimuth sectors around the
        radar that must contain valid data for a level to be retrieved. See
        the Notes on why this is needed in addition to *min_points*.
    max_condition: float
        Maximum condition number of the column-equilibrated forward model
        matrix for a level to be retrieved. See the Notes.
    max_range: float
        Maximum horizontal distance from the radar in m over which Doppler
        velocities are used. Protat et al. (2024) use 150 km, beyond which the
        linearity assumption is not considered valid.
    fit_fall_speed: bool
        If True, the hydrometeor terminal fall speed :math:`V_T` is retrieved
        as an additional control variable at each level. Protat et al. (2024)
        instead separate the fall speed and divergence contributions to the
        Fourier coefficient :math:`a_0` by a linear regression following Protat
        et al. (1997); solving for it as an additional unknown of the same
        linear system is equivalent and is done here for simplicity. If False,
        the fall speed is assumed to be zero, which will alias into DIV.

    Returns
    -------
    profiles: xarray Dataset
        Dataset on the Grid's ``z`` coordinate containing ``U0``, ``V0``,
        ``DIV``, ``DET``, ``DES``, the retrieved ``fall_speed``, the number of
        points used at each level (``npoints``), and a boolean ``valid`` flag.
        Levels that failed the retrieval are NaN and flagged invalid. The
        radar position is recorded in the ``radar_x`` and ``radar_y``
        attributes for use by :func:`vvad_horizontal_wind`.

    Notes
    -----
    The linearity assumption underpinning every VAD technique is reasonable in
    stratiform precipitation but not in convection, so these profiles should be
    regarded as valid in stratiform conditions only.

    Protat et al. (2024) state a single acceptance criterion, the point count
    *min_points*. On a PyDDA analysis grid that criterion alone is not
    sufficient: near the echo top a level can hold several hundred valid
    Doppler velocities that are confined to a narrow sector of azimuth, which
    cannot determine a second-order Fourier series. Such levels return
    deformation terms two orders of magnitude too large, and reconstructing
    them over the domain produces horizontal winds of hundreds of m/s. This
    implementation therefore adds two further criteria, *min_azimuth_coverage*
    and *max_condition*, either of which alone removes those levels on the
    PyDDA sample data. They are exposed as parameters so the paper's behavior
    can be recovered by setting *min_azimuth_coverage* to 0 and
    *max_condition* to :data:`numpy.inf`.

    Protat et al. (2024) minimize their Eq. (9) with a conjugate-gradient
    method. That cost function is quadratic in the control variables and the
    height levels are independent of one another, so its global minimum is the
    linear least-squares solution computed level by level; this
    implementation solves for it directly with :func:`numpy.linalg.lstsq`,
    which returns the identical minimum without needing a convergence
    criterion.

    References
    ----------
    Protat, A., V. Louf, and J. P. Brook, 2024: SWIRL: The First Australian
    Operational Radar-Based 3D Wind Analysis System. *J. Atmos. Oceanic
    Technol.*, **41**, 891-910, https://doi.org/10.1175/JTECH-D-23-0155.1.

    Examples
    --------
    >>> profiles = pydda.constraints.vvad_retrieval(Grid)
    >>> profiles["U0"].where(profiles["valid"])
    """
    if vel_field is None:
        vel_field = pyart.config.get_field_name("corrected_velocity")
    if vel_field not in Grid.variables:
        raise ValueError("%s is not a field in the input Grid!" % vel_field)

    _, _, rh, az, el = _geometry(Grid)
    vr = np.ma.masked_invalid(Grid[vel_field].values.squeeze())

    z = Grid["z"].values
    nz = len(z)
    ncoeff = 6 if fit_fall_speed else 5

    solution = np.full((nz, ncoeff), np.nan)
    npoints = np.zeros(nz, dtype=int)
    coverage = np.zeros(nz)
    condition = np.full(nz, np.nan)
    valid = np.zeros(nz, dtype=bool)

    in_range = rh <= max_range
    usable = np.logical_and(
        np.logical_and(~np.ma.getmaskarray(vr), ~np.ma.getmaskarray(el)),
        in_range,
    )

    for k in range(nz):
        level = usable[k]
        npoints[k] = int(np.count_nonzero(level))
        if npoints[k] < min_points:
            continue
        az_level = az[k][level]
        coverage[k] = _azimuth_coverage(az_level)
        if coverage[k] < min_azimuth_coverage:
            continue

        A = _design_matrix(
            rh[k][level], az_level, np.asarray(el[k])[level], fit_fall_speed
        )
        b = np.asarray(vr[k])[level]

        # Equilibrate the columns before solving. They differ by five orders
        # of magnitude, since the U0/V0 columns are O(1) while the
        # DIV/DET/DES columns carry a factor of the range in m, and that
        # scaling alone would otherwise dominate the conditioning.
        scale = np.linalg.norm(A, axis=0)
        if np.any(scale == 0):
            continue
        condition[k] = np.linalg.cond(A / scale)
        if not np.isfinite(condition[k]) or condition[k] > max_condition:
            continue

        coeffs, _, rank, _ = np.linalg.lstsq(A / scale, b, rcond=None)
        if rank < ncoeff:
            continue
        solution[k] = coeffs / scale
        valid[k] = True

    profiles = xr.Dataset(coords={"z": Grid["z"]})
    for i, name in enumerate(_PROFILE_NAMES):
        profiles[name] = xr.DataArray(solution[:, i], dims=("z",))
    profiles["fall_speed"] = xr.DataArray(
        solution[:, 5] if fit_fall_speed else np.zeros(nz), dims=("z",)
    )
    profiles["npoints"] = xr.DataArray(npoints, dims=("z",))
    profiles["azimuth_coverage"] = xr.DataArray(coverage, dims=("z",))
    profiles["condition"] = xr.DataArray(condition, dims=("z",))
    profiles["valid"] = xr.DataArray(valid, dims=("z",))

    units = {
        "U0": "m/s",
        "V0": "m/s",
        "DIV": "1/s",
        "DET": "1/s",
        "DES": "1/s",
        "fall_speed": "m/s",
    }
    long_names = {
        "U0": "Zonal wind at the radar location",
        "V0": "Meridional wind at the radar location",
        "DIV": "Horizontal divergence",
        "DET": "Stretching deformation",
        "DES": "Shearing deformation",
        "fall_speed": "Hydrometeor terminal fall speed",
        "npoints": "Number of Doppler velocities used in the fit",
        "azimuth_coverage": "Fraction of azimuth sectors containing data",
        "condition": "Condition number of the equilibrated forward model",
        "valid": "Whether the VVAD retrieval succeeded at this level",
    }
    for name, long_name in long_names.items():
        profiles[name].attrs["long_name"] = long_name
        if name in units:
            profiles[name].attrs["units"] = units[name]

    x0, y0 = _radar_xy(Grid)
    profiles.attrs["radar_x"] = x0
    profiles.attrs["radar_y"] = y0
    profiles.attrs["max_range"] = max_range
    profiles.attrs["min_points"] = min_points
    profiles.attrs["min_azimuth_coverage"] = min_azimuth_coverage
    profiles.attrs["max_condition"] = max_condition

    return profiles


def vvad_horizontal_wind(profiles, Grid, max_range=None):
    """
    Reconstructs horizontal maps of the horizontal wind from VVAD profiles
    using Eqs. (1) and (2) of Protat et al. (2024).

    Following the paper, the flow is assumed to be irrotational, i.e. the
    vertical component of vorticity :math:`\\mathrm{ROT} = V_X - U_Y` is zero,
    which closes the system as

    .. math::

        U_X = (\\mathrm{DIV} + \\mathrm{DET}) / 2

        V_Y = (\\mathrm{DIV} - \\mathrm{DET}) / 2

        U_Y = V_X = \\mathrm{DES} / 2

    Because the reconstruction only needs the fitted profiles, it supplies
    winds at grid points where the radar has no observation at all, which is
    what allows the VVAD constraint to fill data voids.

    Parameters
    ----------
    profiles: xarray Dataset
        VVAD profiles from :func:`vvad_retrieval`. The radar position is read
        from its ``radar_x`` and ``radar_y`` attributes.
    Grid: xarray Dataset
        The Grid to reconstruct the wind onto. This need not be the Grid the
        profiles were retrieved from, but it must share its coordinate system.
    max_range: float or None
        Maximum horizontal distance from the radar in m to reconstruct over.
        Defaults to the ``max_range`` the profiles were retrieved with.

    Returns
    -------
    u_vad, v_vad: 3D float arrays
        Zonal and meridional wind of shape ``(nz, ny, nx)``. Points beyond
        *max_range* of the radar, and all points at levels where the VVAD
        retrieval failed, are NaN.

    Notes
    -----
    The irrotational assumption means this reconstruction cannot represent a
    rotating mesoscale flow; Protat et al. (2024) relax it with a double VAD
    (DVAD) when two suitably spaced radars are available. DVAD is not
    implemented in PyDDA.
    """
    if max_range is None:
        max_range = profiles.attrs.get("max_range", 150000.0)

    dx = Grid["point_x"].values - profiles.attrs["radar_x"]
    dy = Grid["point_y"].values - profiles.attrs["radar_y"]
    rh = np.sqrt(dx**2 + dy**2)

    div = profiles["DIV"].values[:, np.newaxis, np.newaxis]
    det = profiles["DET"].values[:, np.newaxis, np.newaxis]
    des = profiles["DES"].values[:, np.newaxis, np.newaxis]
    u_x = 0.5 * (div + det)
    v_y = 0.5 * (div - det)
    u_y = v_x = 0.5 * des

    u_vad = profiles["U0"].values[:, np.newaxis, np.newaxis] + u_x * dx + u_y * dy
    v_vad = profiles["V0"].values[:, np.newaxis, np.newaxis] + v_x * dx + v_y * dy

    outside = np.logical_or(
        rh > max_range, ~profiles["valid"].values[:, np.newaxis, np.newaxis]
    )
    u_vad = np.where(outside, np.nan, u_vad)
    v_vad = np.where(outside, np.nan, v_vad)
    return u_vad, v_vad


def make_constraint_from_vvad(
    Grids,
    vel_field=None,
    min_points=100,
    min_azimuth_coverage=0.5,
    max_condition=50.0,
    max_range=150000.0,
    fit_fall_speed=True,
    combine="consensus",
):
    """
    Adds the VVAD horizontal wind of Protat et al. (2024) to a list of Grids
    as the ``U_vvad`` and ``V_vvad`` fields, for use with the *Cvad* constraint
    of :func:`pydda.retrieval.get_dd_wind_field`.

    A VVAD is retrieved for each radar in *Grids* with
    :func:`vvad_retrieval`, each is reconstructed onto the analysis grid with
    :func:`vvad_horizontal_wind`, and the results are combined.

    Parameters
    ----------
    Grids: list of xarray Datasets
        The PyDDA Grids to retrieve VVADs from. The constraint fields are
        written into every Grid in the list, since PyDDA's retrieval reads
        constraint fields from the first Grid.
    vel_field: str or None
        Name of the Doppler velocity field. Uses the Py-ART default for
        corrected velocity if None.
    min_points: int
        Minimum number of valid Doppler velocities per height level, see
        :func:`vvad_retrieval`.
    min_azimuth_coverage: float
        Minimum fraction of azimuth sectors that must contain data, see
        :func:`vvad_retrieval`.
    max_condition: float
        Maximum condition number of the equilibrated forward model, see
        :func:`vvad_retrieval`.
    max_range: float
        Maximum horizontal distance from each radar in m, see
        :func:`vvad_retrieval`.
    fit_fall_speed: bool
        Whether to retrieve the terminal fall speed, see
        :func:`vvad_retrieval`.
    combine: str
        How to combine the reconstructions from multiple radars.

        *'consensus'* - Average the reconstructions from every radar whose
        *max_range* covers the point. This is the behavior of Protat et al.
        (2024).

        *'nearest'* - Use the reconstruction from the closest radar that
        covers the point, which keeps the linearity assumption as local as
        possible.

    Returns
    -------
    Grids: list of xarray Datasets
        The input Grids with ``U_vvad`` and ``V_vvad`` added.

    Notes
    -----
    No ``W_vvad`` is produced. Protat et al. (2024) integrate the continuity
    equation to obtain a VVAD vertical velocity but use only the horizontal
    components for blending, since a stratiform vertical motion that is
    constant over the whole domain has little value as a constraint.

    Where two or more radars overlap, Protat et al. (2024) use a double VAD
    (DVAD) rather than a consensus of single-radar VVADs, which additionally
    retrieves the vertical vorticity and so can represent a rotating flow.
    DVAD is not implemented in PyDDA, so ``combine='consensus'`` averages
    VVADs, each of which individually assumes irrotational flow.

    References
    ----------
    Protat, A., V. Louf, and J. P. Brook, 2024: SWIRL: The First Australian
    Operational Radar-Based 3D Wind Analysis System. *J. Atmos. Oceanic
    Technol.*, **41**, 891-910, https://doi.org/10.1175/JTECH-D-23-0155.1.

    Examples
    --------
    >>> Grids = pydda.constraints.make_constraint_from_vvad(Grids)
    >>> Grids, params = pydda.retrieval.get_dd_wind_field(Grids, Cvad=1.0)
    """
    if combine not in ("consensus", "nearest"):
        raise ValueError(
            "combine must be one of 'consensus' or 'nearest', got '%s'" % combine
        )

    u_all = []
    v_all = []
    dist_all = []
    for Grid in Grids:
        profiles = vvad_retrieval(
            Grid,
            vel_field=vel_field,
            min_points=min_points,
            min_azimuth_coverage=min_azimuth_coverage,
            max_condition=max_condition,
            max_range=max_range,
            fit_fall_speed=fit_fall_speed,
        )
        u_vad, v_vad = vvad_horizontal_wind(profiles, Grids[0], max_range=max_range)
        u_all.append(u_vad)
        v_all.append(v_vad)
        dist_all.append(
            np.sqrt(
                (Grids[0]["point_x"].values - profiles.attrs["radar_x"]) ** 2
                + (Grids[0]["point_y"].values - profiles.attrs["radar_y"]) ** 2
            )
        )

    u_all = np.stack(u_all)
    v_all = np.stack(v_all)

    with np.errstate(invalid="ignore"):
        if combine == "consensus":
            u_out = np.nanmean(u_all, axis=0)
            v_out = np.nanmean(v_all, axis=0)
        else:
            # Rank the radars by distance, but only where they have a
            # reconstruction, so a nearer radar with a failed level does not
            # veto a farther one that succeeded.
            dist = np.where(np.isfinite(u_all), np.stack(dist_all), np.inf)
            nearest = np.argmin(dist, axis=0)
            no_data = np.all(~np.isfinite(u_all), axis=0)
            u_out = np.take_along_axis(u_all, nearest[np.newaxis], axis=0)[0]
            v_out = np.take_along_axis(v_all, nearest[np.newaxis], axis=0)[0]
            u_out = np.where(no_data, np.nan, u_out)
            v_out = np.where(no_data, np.nan, v_out)

    for i in range(len(Grids)):
        Grids[i]["U_vvad"] = xr.DataArray(
            np.expand_dims(u_out, 0),
            dims=("time", "z", "y", "x"),
            attrs={
                "long_name": "Zonal component of wind velocity from VVAD",
                "units": "m/s",
                "combine": combine,
                "max_range": max_range,
            },
        )
        Grids[i]["V_vvad"] = xr.DataArray(
            np.expand_dims(v_out, 0),
            dims=("time", "z", "y", "x"),
            attrs={
                "long_name": "Meridional component of wind velocity from VVAD",
                "units": "m/s",
                "combine": combine,
                "max_range": max_range,
            },
        )

    return Grids
