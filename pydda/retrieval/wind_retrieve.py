"""
Created on Mon Aug  7 09:17:40 2017

@author: rjackson
"""

import pyart
import numpy as np
import time
import math

from .. import cost_functions
from ..cost_functions import J_function, grad_J
from scipy.optimize import fmin_l_bfgs_b
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from copy import deepcopy
from .angles import add_azimuth_as_field, add_elevation_as_field


def get_dd_wind_field(Grids, u_init, v_init, w_init, vel_name=None,
                      refl_field=None, u_back=None, v_back=None, z_back=None,
                      frz=4500.0, Co=1.0, Cm=1500.0, Cx=0.0,
                      Cy=0.0, Cz=0.0, Cb=0.0, Cv=0.0, Cmod=0.0,
                      Ut=None, Vt=None, filt_iterations=2,
                      mask_outside_opt=False, weights_obs=None,
                      weights_model=None, weights_bg=None,
                      max_iterations=200, mask_w_outside_opt=True,
                      filter_window=9, filter_order=4, min_bca=30.0,
                      max_bca=150.0, upper_bc=True, model_fields=None,
                      output_cost_functions=True):
    """
    This function takes in a list of Py-ART Grid objects and derives a
    wind field. Every Py-ART Grid in Grids must have the same grid
    specification. 

    In order for the model data constraint to be used,
    the model data must be added as a field to at least one of the
    grids in Grids. This involves interpolating the model data to the
    Grids' coordinates. There are helper functions for this for WRF
    and HRRR data in :py:func:`pydda.constraints`:

    :py:func:`make_constraint_from_wrf`

    :py:func:`add_hrrr_constraint_to_grid`

    Parameters
    ==========

    Grids: list of Py-ART Grids
        The list of Py-ART grids to take in corresponding to each radar.
        All grids must have the same shape, x coordinates, y coordinates
        and z coordinates.
    u_init: 3D ndarray
        The intial guess for the zonal wind field, input as a 3D array
        with the same shape as the fields in Grids.
    v_init: 3D ndarray
        The intial guess for the meridional wind field, input as a 3D array
        with the same shape as the fields in Grids.
    w_init: 3D ndarray
        The intial guess for the vertical wind field, input as a 3D array
        with the same shape as the fields in Grids.
    vel_name: string
        Name of radial velocity field. Setting to None will have PyDDA attempt
        to automatically detect the velocity field name.
    refl_field: string
        Name of reflectivity field. Setting to None will have PyDDA attempt 
        to automatically detect the reflectivity field name.
    u_back: 1D array
        Background zonal wind field from a sounding as a function of height.
        This should be given in the sounding's vertical coordinates.
    v_back: 1D array
        Background meridional wind field from a sounding as a function of
        height. This should be given in the sounding's vertical coordinates.
    z_back: 1D array
        Heights corresponding to background wind field levels in meters. This
        is given in the sounding's original coordinates.
    frz: float
        Freezing level used for fall speed calculation in meters.
    Co: float
        Weight for cost function related to observed radial velocities.
    Cm: float
        Weight for cost function related to the mass continuity equation.
    Cx: float
        Weight for cost function related to smoothness in x direction
    Cy: float
        Weight for cost function related to smoothness in y direction
    Cz: float
        Weight for cost function related to smoothness in z direction
    Cv: float
        Weight for cost function related to vertical vorticity equation.
    Cmod: float
        Weight for cost function related to custom constraints.
    weights_obs: list of floating point arrays or None
        List of weights for each point in grid from each radar in Grids.
        Set to None to let PyDDA determine this automatically.
    weights_model: list of floating point arrays or None
        List of weights for each point in grid from each custom field in
        model_fields. Set to None to let PyDDA determine this automatically.
    weights_bg: list of floating point arrays or None
        List of weights for each point in grid from the sounding. Set to None
        to let PyDDA determine this automatically.
    Ut: float
        Prescribed storm motion in zonal direction.
        This is only needed if Cv is not zero.
    Vt: float
        Prescribed storm motion in meridional direction.
        This is only needed if Cv is not zero.
    filt_iterations: int
        If this is greater than 0, PyDDA will run a low pass filter on
        the retrieved wind field and then do the optimization step for
        filt_iterations iterations. Set to 0 to disable the low pass filter.
    max_outside_opt: bool
        If set to true, wind values outside the multiple doppler lobes will
        be masked, i.e. if less than 2 radars provide coverage for a given
        point.
    max_iterations: int
        The maximum number of iterations to run the optimization loop for.
    max_w_outside_opt: bool
        If set to true, vertical winds outside the multiple doppler lobes will
        be masked, i.e. if less than 2 radars provide coverage for a given
        point.
    filter_window: int
        Window size to use for the low pass filter. A larger window will
        increase the number of points factored into the polynomial fit for
        the filter, and hence will increase the smoothness.
    filter_order: int
        The order of the polynomial to use for the low pass filter. Higher
        order polynomials allow for the retention of smaller scale features
        but may also not remove enough noise.
    min_bca: float
        Minimum beam crossing angle in degrees between two radars. 30.0 is the
        typical value used in many publications.
    max_bca: float
        Minimum beam crossing angle in degrees between two radars. 150.0 is the
        typical value used in many publications.
    upper_bc: bool
        Set this to true to enforce w = 0 at the top of the atmosphere. This is
        commonly called the impermeability condition.
    model_fields: list of strings
        The list of fields in the first grid in Grids that contain the custom
        data interpolated to the Grid's grid specification. Helper functions
        to create such gridded fields for HRRR and NetCDF WRF data exist
        in ::pydda.constraints::. PyDDA will look for fields named U_(model
        field name), V_(model field name), and W_(model field name). For
        example, if you have U_hrrr, V_hrrr, and W_hrrr, then specify ["hrrr"]
        into model_fields.
    output_cost_functions: bool
        Set to True to output the value of each cost function every
        10 iterations.

    Returns
    =======
    new_grid_list: list
        A list of Py-ART grids containing the derived wind fields. These fields
        are displayable by the visualization module.
    """

    # We have to have a prescribed storm motion for vorticity constraint
    if(Ut is None or Vt is None):
        if(Cv != 0.0):
            raise ValueError(('Ut and Vt cannot be None if vertical ' +
                              'vorticity constraint is enabled!'))

    if not isinstance(Grids, list):
        raise ValueError('Grids has to be a list!')

    
    # Ensure that all Grids are on the same coordinate system
    prev_grid = Grids[0]
    for g in Grids:
        if not np.allclose(
            g.x['data'], prev_grid.x['data'], atol=10):
            raise ValueError('Grids do not have equal x coordinates!')

        if not np.allclose(
            g.y['data'], prev_grid.y['data'], atol=10):
            raise ValueError('Grids do not have equal y coordinates!')

        if not np.allclose(
            g.z['data'], prev_grid.z['data'], atol=10):
            raise ValueError('Grids do not have equal z coordinates!')

        if not g.origin_latitude['data'] == prev_grid.origin_latitude['data']:
            raise ValueError(("Grids have unequal origin lat/lons!"))

        prev_grid = g

    # Disable background constraint if none provided
    if(u_back is None or v_back is None):
        u_back2 = np.zeros(u_init.shape[0])
        v_back2 = np.zeros(v_init.shape[0])
        C8 = 0.0
    else:
        # Interpolate sounding to radar grid
        print('Interpolating sounding to radar grid')
        u_interp = interp1d(z_back, u_back, bounds_error=False)
        v_interp = interp1d(z_back, v_back, bounds_error=False)
        u_back2 = u_interp(Grids[0].z['data'])
        v_back2 = v_interp(Grids[0].z['data'])
        print('Interpolated U field:')
        print(u_back2)
        print('Interpolated V field:')
        print(v_back2)
        print('Grid levels:')
        print(Grids[0].z['data'])

    # Parse names of velocity field
    if refl_field is None:
        refl_field = pyart.config.get_field_name('reflectivity')

    # Parse names of velocity field
    if vel_name is None:
        vel_name = pyart.config.get_field_name('corrected_velocity')
    winds = np.stack([u_init, v_init, w_init])
    wts = []
    vrs = []
    azs = []
    els = []

    # Set up wind fields and weights from each radar
    weights = np.zeros(
        (len(Grids), u_init.shape[0], u_init.shape[1], u_init.shape[2]))

    bg_weights = np.zeros(v_init.shape)
    if(model_fields is not None):
        mod_weights = np.ones(
            (len(model_fields), u_init.shape[0], u_init.shape[1],
             u_init.shape[2]))
    else:
        mod_weights = np.zeros(
            (1, u_init.shape[0], u_init.shape[1], u_init.shape[2]))

    if(model_fields is None):
        if(Cmod != 0.0):
            raise ValueError(
                 'Cmod must be zero if model fields are not specified!')

    bca = np.zeros(
        (len(Grids), len(Grids), u_init.shape[1], u_init.shape[2]))
    M = np.zeros(len(Grids))
    sum_Vr = np.zeros(len(Grids))

    for i in range(len(Grids)):
        wts.append(cost_functions.calculate_fall_speed(Grids[i],
                                                       refl_field=refl_field))
        add_azimuth_as_field(Grids[i], dz_name=refl_field)
        add_elevation_as_field(Grids[i], dz_name=refl_field)
        vrs.append(Grids[i].fields[vel_name]['data'])
        azs.append(Grids[i].fields['AZ']['data']*np.pi/180)
        els.append(Grids[i].fields['EL']['data']*np.pi/180)

    if(len(Grids) > 1):
        for i in range(len(Grids)):
            for j in range(i+1, len(Grids)):
                print(("Calculating weights for radars " + str(i) +
                       " and " + str(j)))
                bca[i, j] = get_bca(Grids[i].radar_longitude['data'],
                                    Grids[i].radar_latitude['data'],
                                    Grids[j].radar_longitude['data'],
                                    Grids[j].radar_latitude['data'],
                                    Grids[i].point_x['data'][0],
                                    Grids[i].point_y['data'][0],
                                    Grids[i].get_projparams())

                for k in range(vrs[i].shape[0]):
                    if(weights_obs is None):
                        cur_array = weights[i, k]
                        cur_array[np.logical_and(
                            ~vrs[i][k].mask,
                            np.logical_and(
                                bca[i, j] >= math.radians(min_bca),
                                bca[i, j] <= math.radians(max_bca)))] += 1
                        weights[i, k] = cur_array
                    else:
                        weights[i, k] = weights_obs[i][k, :, :]

                    if(weights_obs is None):
                        cur_array = weights[j, k]
                        cur_array[np.logical_and(
                            ~vrs[j][k].mask,
                            np.logical_and(
                                bca[i, j] >= math.radians(min_bca),
                                bca[i, j] <= math.radians(max_bca)))] += 1
                        weights[j, k] = cur_array
                    else:
                        weights[j, k] = weights_obs[j][k, :, :]

                    if(weights_bg is None):
                        cur_array = bg_weights[k]
                        cur_array[np.logical_or(
                            bca[i, j] >= math.radians(min_bca),
                            bca[i, j] <= math.radians(max_bca))] = 1
                        cur_array[vrs[i][k].mask] = 0
                        bg_weights[i] = cur_array
                    else:
                        bg_weights[i] = weights_bg[i]

        print("Calculating weights for models...")
        coverage_grade = weights.sum(axis=0)
        coverage_grade = coverage_grade/coverage_grade.max()

        # Weigh in model input more when we have no coverage
        # Model only weighs 1/(# of grids + 1) when there is full
        # Coverage
        if(model_fields is not None):
            if(weights_model is None):
                for i in range(len(model_fields)):
                    mod_weights[i] = 1 - (coverage_grade/(len(Grids)+1))
            else:
                for i in range(len(model_fields)):
                    mod_weights[i] = weights_model[i]
    else:
        weights[0] = np.where(~vrs[0].mask, 1, 0)
        bg_weights = np.where(~vrs[0].mask, 0, 1)

    weights[weights > 0] = 1
    sum_Vr = np.sum(np.square(vrs*weights))
    rmsVr = np.sum(sum_Vr)/np.sum(weights)

    del bca
    grid_shape = u_init.shape
    # Parse names of velocity field

    winds = winds.flatten()
    ndims = len(winds)

    print(("Starting solver "))
    dx = np.diff(Grids[0].x['data'], axis=0)[0]
    dy = np.diff(Grids[0].y['data'], axis=0)[0]
    dz = np.diff(Grids[0].z['data'], axis=0)[0]
    print('rmsVR = ' + str(rmsVr))
    print('Total points:' + str(weights.sum()))
    z = Grids[0].point_z['data']

    the_time = time.time()
    bt = time.time()

    # First pass - no filter
    wcurr = w_init
    wprev = 100*np.ones(w_init.shape)
    wprevmax = 99
    wcurrmax = w_init.max()
    iterations = 0
    warnflag = 99999
    coeff_max = np.max([Co, Cb, Cm, Cx, Cy, Cz, Cb])
    bounds = [(-x, x) for x in 100*np.ones(winds.shape)]

    u_model = []
    v_model = []
    w_model = []
    if(model_fields is not None):
        for the_field in model_fields:
            u_field = ("U_" + the_field)
            v_field = ("V_" + the_field)
            w_field = ("W_" + the_field)
            u_model.append(Grids[0].fields[u_field]["data"])
            v_model.append(Grids[0].fields[v_field]["data"])
            w_model.append(Grids[0].fields[w_field]["data"])

    while(iterations < max_iterations and
          (abs(wprevmax-wcurrmax) > 0.02)):
        wprevmax = wcurrmax
        winds = fmin_l_bfgs_b(J_function, winds, args=(vrs, azs, els,
                                                       wts, u_back, v_back,
                                                       u_model, v_model,
                                                       w_model,
                                                       Co, Cm, Cx, Cy, Cz, Cb,
                                                       Cv, Cmod, Ut, Vt,
                                                       grid_shape,
                                                       dx, dy, dz, z, rmsVr,
                                                       weights, bg_weights,
                                                       mod_weights,
                                                       upper_bc,
                                                       False),
                              maxiter=10, pgtol=1e-3, bounds=bounds,
                              fprime=grad_J, disp=0, iprint=-1)
        if(output_cost_functions is True):
            J_function(winds[0], vrs, azs, els, wts, u_back, v_back,
                       u_model, v_model, w_model,
                       Co, Cm, Cx, Cy, Cz, Cb, Cv, Cmod, Ut, Vt,
                       grid_shape, dx, dy, dz, z, rmsVr,
                       weights, bg_weights, mod_weights,
                       upper_bc, True)
            grad_J(winds[0], vrs, azs, els, wts, u_back, v_back,
                   u_model, v_model, w_model,
                   Co, Cm, Cx, Cy, Cz, Cb, Cv, Cmod, Ut, Vt,
                   grid_shape, dx, dy, dz, z, rmsVr,
                   weights, bg_weights, mod_weights,
                   upper_bc, True)
        warnflag = winds[2]['warnflag']
        winds = np.reshape(winds[0], (3, grid_shape[0], grid_shape[1],
                                      grid_shape[2]))
        iterations = iterations+10
        print('Iterations before filter: ' + str(iterations))
        wcurrmax = winds[2].max()
        winds = np.stack([winds[0], winds[1], winds[2]])
        winds = winds.flatten()

    if(filt_iterations > 0):
        print('Applying low pass filter to wind field...')
        winds = np.reshape(winds, (3, grid_shape[0], grid_shape[1],
                                   grid_shape[2]))
        winds[0] = savgol_filter(winds[0], 9, 3, axis=0)
        winds[0] = savgol_filter(winds[0], 9, 3, axis=1)
        winds[0] = savgol_filter(winds[0], 9, 3, axis=2)
        winds[1] = savgol_filter(winds[1], 9, 3, axis=0)
        winds[1] = savgol_filter(winds[1], 9, 3, axis=1)
        winds[1] = savgol_filter(winds[1], 9, 3, axis=2)
        winds[2] = savgol_filter(winds[2], 9, 3, axis=0)
        winds[2] = savgol_filter(winds[2], 9, 3, axis=1)
        winds[2] = savgol_filter(winds[2], 9, 3, axis=2)
        winds = np.stack([winds[0], winds[1], winds[2]])
        winds = winds.flatten()
        iterations = 0
        while(iterations < filt_iterations):
            winds = fmin_l_bfgs_b(
                J_function, winds, args=(vrs, azs, els,
                                         wts, u_back, v_back,
                                         u_model, v_model, w_model,
                                         Co, Cm, Cx, Cy, Cz, Cb,
                                         Cv, Cmod, Ut, Vt,
                                         grid_shape,
                                         dx, dy, dz, z, rmsVr,
                                         weights, bg_weights,
                                         mod_weights,
                                         upper_bc,
                                         False),
                maxiter=10, pgtol=1e-3, bounds=bounds,
                fprime=grad_J, disp=0, iprint=-1)

            warnflag = winds[2]['warnflag']
            winds = np.reshape(winds[0], (3, grid_shape[0], grid_shape[1],
                                          grid_shape[2]))
            iterations = iterations+1
            print('Iterations after filter: ' + str(iterations))
            winds = np.stack([winds[0], winds[1], winds[2]])
            winds = winds.flatten()
    print("Done! Time = " + "{:2.1f}".format(time.time() - bt))

    # First pass - no filter
    the_winds = np.reshape(winds, (3, grid_shape[0], grid_shape[1],
                                   grid_shape[2]))
    u = the_winds[0]
    v = the_winds[1]
    w = the_winds[2]
    where_mask = np.sum(weights, axis=0) + np.sum(mod_weights, axis=0)
    
    u = np.ma.array(u)
    w = np.ma.array(w)
    v = np.ma.array(v)

    if(mask_outside_opt is True):
        u = np.ma.masked_where(where_mask < 1, u)
        v = np.ma.masked_where(where_mask < 1, v)
        w = np.ma.masked_where(where_mask < 1, w)

    if(mask_w_outside_opt is True):
        w = np.ma.masked_where(where_mask < 1, w)

    u_field = deepcopy(Grids[0].fields[vel_name])
    u_field['data'] = u
    u_field['standard_name'] = 'u_wind'
    u_field['long_name'] = 'meridional component of wind velocity'
    u_field['min_bca'] = min_bca
    u_field['max_bca'] = max_bca
    v_field = deepcopy(Grids[0].fields[vel_name])
    v_field['data'] = v
    v_field['standard_name'] = 'v_wind'
    v_field['long_name'] = 'zonal component of wind velocity'
    v_field['min_bca'] = min_bca
    v_field['max_bca'] = max_bca
    w_field = deepcopy(Grids[0].fields[vel_name])
    w_field['data'] = w
    w_field['standard_name'] = 'w_wind'
    w_field['long_name'] = 'vertical component of wind velocity'
    w_field['min_bca'] = min_bca
    w_field['max_bca'] = max_bca

    new_grid_list = []

    for grid in Grids:
        temp_grid = deepcopy(grid)
        temp_grid.add_field('u', u_field, replace_existing=True)
        temp_grid.add_field('v', v_field, replace_existing=True)
        temp_grid.add_field('w', w_field, replace_existing=True)
        new_grid_list.append(temp_grid)

    return new_grid_list


def get_bca(rad1_lon, rad1_lat, rad2_lon, rad2_lat, x, y, projparams):
    """
    This function gets the beam crossing angle between two lat/lon pairs.

    Parameters
    ==========
    rad1_lon: float
        The longitude of the first radar.
    rad1_lat: float
        The latitude of the first radar.
    rad2_lon: float
        The longitude of the second radar.
    rad2_lat: float
        The latitude of the second radar.
    x: nD float array
        The Cartesian x coordinates of the grid
    y: nD float array
        The Cartesian y corrdinates of the grid
    projparams: Py-ART projparams
        The projection parameters of the Grid

    Returns
    =======
    bca: nD float array
        The beam crossing angle between the two radars in radians.

    """

    rad1 = pyart.core.geographic_to_cartesian(rad1_lon, rad1_lat, projparams)
    rad2 = pyart.core.geographic_to_cartesian(rad2_lon, rad2_lat, projparams)
    # Create grid with Radar 1 in center

    x = x-rad1[0]
    y = y-rad1[1]
    rad2 = np.array(rad2) - np.array(rad1)
    a = np.sqrt(np.multiply(x, x) + np.multiply(y, y))
    b = np.sqrt(pow(x-rad2[0], 2) + pow(y-rad2[1], 2))
    c = np.sqrt(rad2[0]*rad2[0] + rad2[1]*rad2[1])
    theta_1 = np.arccos(x/a)
    theta_2 = np.arccos((x-rad2[1])/b)
    return np.arccos((a*a+b*b-c*c)/(2*a*b))
