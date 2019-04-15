import numpy as np
import math
import time

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import fmin_l_bfgs_b
from .angles import add_azimuth_as_field, add_elevation_as_field
from .. import cost_functions
from .wind_retrieve import get_bca
from copy import deepcopy

def _J_coarse(winds, residual, vrs, azs, els, wts, u_back, v_back, u_model,
              v_model, w_model, Co, Cm, Cx, Cy, Cz, Cb, Cv, Cmod,
              Ut, Vt, grid_shape, dx, dy, dz, z, rmsVr, weights,
              bg_weights, model_weights, upper_bc, print_out=False):
    cost = cost_functions.J_function(
        winds, vrs, azs, els, wts, u_back, v_back, u_model, v_model, w_model,
        Co, Cm, Cx, Cy, Cz, Cb, Cv, Cmod, Ut, Vt, grid_shape, dx, dy, dz, z,
        rmsVr, weights, bg_weights, model_weights, upper_bc, print_out)

    finites = np.logical_and(np.isfinite(winds, residual)
    cost -= np.dot(winds_nonan[finites], residual[finites])
    if print_out:
        print("Total cost function - residual: " + str(cost))
    return cost


def _grad_coarse(winds, residual, vrs, azs, els, wts, u_back, v_back, u_model,
                 v_model, w_model, Co, Cm, Cx, Cy, Cz, Cb, Cv, Cmod,
                 Ut, Vt, grid_shape, dx, dy, dz, z, rmsVr, weights,
                 bg_weights, model_weights, upper_bc, print_out=False):

    gradJ = cost_functions.grad_J(
        winds, vrs, azs, els, wts, u_back, v_back, u_model, v_model, w_model,
        Co, Cm, Cx, Cy, Cz, Cb, Cv, Cmod, Ut, Vt, grid_shape, dx, dy, dz, z,
        rmsVr, weights, bg_weights, model_weights, upper_bc, False)
    finites = np.logical_and(np.isfinite(winds, residual)
    if print_out:
        print(("Norm of gradient - residual:" +
              str(np.linalg.norm(gradJ[finites] + residual))))
    return gradJ - residual


def get_dd_wind_field_multigrid(Grids, u_init, v_init, w_init, vel_name=None,
                                refl_field=None, u_back=None, v_back=None,
                                z_back=None, frz=4500.0, Co=1.0, Cm=1500.0,
                                Cx=0.0, Cy=0.0, Cz=0.0, Cb=0.0, Cv=0.0,
                                Cmod=0.0, Ut=None, Vt=None,
                                filt_iterations=2, mask_outside_opt=False,
                                weights_obs=None, weights_model=None,
                                weights_bg=None, max_iterations=200,
                                mask_w_outside_opt=True,
                                filter_window=9, filter_order=4,
                                min_bca=30.0, max_bca=150.0,
                                upper_bc=True, model_fields=None,
                                output_cost_functions=True):

    # We have to have a prescribed storm motion for vorticity constraint
    if (Ut is None or Vt is None):
        if (Cv != 0.0):
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
    if (u_back is None or v_back is None):
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
    u_back_coarse = (u_back2[:-2:-2] +
                     u_back2[1:-1:2])/2.0
    v_back_coarse = (v_back2[:-2:-2] +
                     u_back2[1:-1:2]) / 2.0

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
    if (model_fields is not None):
        mod_weights = np.ones(
            (len(model_fields), u_init.shape[0], u_init.shape[1],
             u_init.shape[2]))
    else:
        mod_weights = np.zeros(
            (1, u_init.shape[0], u_init.shape[1], u_init.shape[2]))

    if (model_fields is None):
        if (Cmod != 0.0):
            raise ValueError(
                'Cmod must be zero if model fields are not specified!')

    bca = np.zeros(
        (len(Grids), len(Grids), u_init.shape[1], u_init.shape[2]))
    M = np.zeros(len(Grids))
    sum_Vr = np.zeros(len(Grids))
    rad_no = 0
    vrs_coarse = []
    azs_coarse = []
    els_coarse = []
    wts_coarse = []

    for i in range(len(Grids)):
        wts.append(cost_functions.calculate_fall_speed(Grids[i],
                                                       refl_field=refl_field))
        add_azimuth_as_field(Grids[i], dz_name=refl_field)
        add_elevation_as_field(Grids[i], dz_name=refl_field)
        vrs.append(Grids[i].fields[vel_name]['data'])
        azs.append(Grids[i].fields['AZ']['data'] * np.pi / 180)
        els.append(Grids[i].fields['EL']['data'] * np.pi / 180)
        vrs_coarse.append((vrs[i][:-2:2, :-2:2, :-2:2] +
                           vrs[i][1:-1:2, 1:-1:2, 1:-1:2]) / 2.0)
        wts_coarse.append((wts[i][:-2:2, :-2:2, :-2:2] +
                           wts[i][1:-1:2, 1:-1:2, 1:-1:2]) / 2.0)
        azs_coarse.append((vrs[i][:-2:2, :-2:2, :-2:2] +
                           vrs[i][1:-1:2, 1:-1:2, 1:-1:2]) / 2.0)
        els_coarse.append((vrs[i][:-2:2, :-2:2, :-2:2] +
                           vrs[i][1:-1:2, 1:-1:2, 1:-1:2]) / 2.0)

    if (len(Grids) > 1):
        for i in range(len(Grids)):
            for j in range(i + 1, len(Grids)):
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
                    if (weights_obs is None):
                        cur_array = weights[i, k]
                        cur_array[np.logical_and(
                            ~vrs[i][k].mask,
                            np.logical_and(
                                bca[i, j] >= math.radians(min_bca),
                                bca[i, j] <= math.radians(max_bca)))] += 1
                        weights[i, k] = cur_array
                    else:
                        weights[i, k] = weights_obs[i][k, :, :]

                    if (weights_obs is None):
                        cur_array = weights[j, k]
                        cur_array[np.logical_and(
                            ~vrs[j][k].mask,
                            np.logical_and(
                                bca[i, j] >= math.radians(min_bca),
                                bca[i, j] <= math.radians(max_bca)))] += 1
                        weights[j, k] = cur_array
                    else:
                        weights[j, k] = weights_obs[j][k, :, :]

                    if (weights_bg is None):
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
        coverage_grade = coverage_grade / coverage_grade.max()

        # Weigh in model input more when we have no coverage
        # Model only weighs 1/(# of grids + 1) when there is full
        # Coverage
        if model_fields is not None:
            if weights_model is None:
                for i in range(len(model_fields)):
                    mod_weights[i] = 1 - (coverage_grade / (len(Grids) + 1))
            else:
                for i in range(len(model_fields)):
                    mod_weights[i] = weights_model[i]
    else:
        weights[0] = np.where(~vrs[0].mask, 1, 0)
        bg_weights = np.where(~vrs[0].mask, 0, 1)

    weights[weights > 0] = 1
    weights_coarse = (weights[:, :-2:2, :-2:2, :-2:2] +
                      weights[:, 1:-1:2, 1:-1:2, 1:-1:2]) / 2.0
    mod_weights_coarse = (mod_weights[:, :-2:2, :-2:2, :-2:2] +
                          mod_weights[:, 1:-1:2, 1:-1:2, 1:-1:2]) / 2.0
    bg_weights_coarse = (bg_weights[:-2:2, :-2:2, :-2:2] +
                         bg_weights[1:-1:2, 1:-1:2, 1:-1:2]) / 2.0

    sum_Vr = np.sum(np.square(vrs * weights))
    rmsVr = np.sum(sum_Vr) / np.sum(weights)

    del bca
    grid_shape = u_init.shape
    grid_shape_coarse = vrs_coarse[0].shape

    # Parse names of velocity field

    winds = winds.flatten()
    ndims = len(winds)

    print(("Starting solver "))
    dx = np.diff(Grids[0].x['data'], axis=0)[0]
    dy = np.diff(Grids[0].y['data'], axis=0)[0]
    dz = np.diff(Grids[0].z['data'], axis=0)[0]
    print('rmsVR = ' + str(rmsVr))
    print('Total points:' + str(weights_coarse.sum()))
    z = Grids[0].point_z['data']
    y = Grids[0].point_y['data']
    x = Grids[0].point_x['data']
    zc = (z[:-2:2, :-2:2, :-2:2] +
          z[1:-1:2, 1:-1:2, 1:-1:2])/2.0
    z_coarse = Grids[0].z['data']
    x_coarse = Grids[0].x['data']
    y_coarse = Grids[0].y['data']
    z_coarse = (z_coarse[:-2:2] +
                z_coarse[1:-1:2]) / 2.0
    x_coarse = (x_coarse[:-2:2] +
                x_coarse[1:-1:2]) / 2.0
    y_coarse = (y_coarse[:-2:2] +
                y_coarse[1:-1:2]) / 2.0


    the_time = time.time()
    bt = time.time()

    # First pass - no filter
    wcurr = w_init
    wprev = 100 * np.ones(w_init.shape)
    wprevmax = 99
    wcurrmax = w_init.max()
    iterations = 0
    warnflag = 99999
    coeff_max = np.max([Co, Cb, Cm, Cx, Cy, Cz, Cb])


    u_model = []
    v_model = []
    w_model = []
    u_model_coarse = []
    v_model_coarse = []
    w_model_coarse = []
    if (model_fields is not None):
        mod_no = 0
        for the_field in model_fields:
            u_field = ("U_" + the_field)
            v_field = ("V_" + the_field)
            w_field = ("W_" + the_field)
            u_model.append(Grids[0].fields[u_field]["data"])
            v_model.append(Grids[0].fields[v_field]["data"])
            w_model.append(Grids[0].fields[w_field]["data"])
            u_model_coarse.append((u_model[mod_no][:-2:2, :-2:2, :-2:2] +
                                   u_model[mod_no][1:-1:2, 1:-1:2, 1:-1:2]) / 2.0)
            v_model_coarse.append((v_model[mod_no][:-2:2, :-2:2, :-2:2] +
                                   v_model[mod_no][1:-1:2, 1:-1:2, 1:-1:2]) / 2.0)
            w_model_coarse.append((w_model[mod_no][:-2:2, :-2:2, :-2:2] +
                                   w_model[mod_no][1:-1:2, 1:-1:2, 1:-1:2]) / 2.0)
            mod_no += 1
    warnflag = 0
    while (iterations < max_iterations and
           np.abs(wprevmax - wcurrmax) > 0.02):
        wprevmax = wcurrmax
        winds_new = winds.copy()
        winds = np.reshape(winds,
                          (3, grid_shape[0], grid_shape[1], grid_shape[2]))
        # relaxation step - do 5 iterations of gradient descent
        for i in range(5):
            fine_gradient = cost_functions.grad_J(
                winds_new, vrs, azs, els, wts, u_back, v_back, u_model, v_model,
                w_model, Co, Cm, Cx, Cy, Cz, Cb, Cv, Cmod, Ut, Vt, grid_shape,
                dx, dy, dz, z, rmsVr, weights, bg_weights, mod_weights,
                upper_bc, False)
            step_size = 0.001
            winds_new = winds_new - step_size*fine_gradient

        # Now we solve the coarse grid problem, simply using the residual
        # We have to handle the u, v, w, components of the winds separately
        # when interpolating/extrapolating!
        winds_new = np.reshape(
            winds_new, (3, grid_shape[0], grid_shape[1], grid_shape[2]))
        fine_gradient = np.reshape(
            fine_gradient, (3, grid_shape[0], grid_shape[1], grid_shape[2]))
        residual = -1/fine_gradient

        # Linear interpolation

        u_coarse = (winds_new[0, :-2:2, :-2:2, :-2:2] +
                    winds_new[0, 1:-1:2, 1:-1:2, 1:-1:2]) / 2.0
        v_coarse = (winds_new[1, :-2:2, :-2:2, :-2:2] +
                    winds_new[1, 1:-1:2, 1:-1:2, 1:-1:2]) / 2.0
        w_coarse = (winds_new[2, :-2:2, :-2:2, :-2:2] +
                    winds_new[2, 1:-1:2, 1:-1:2, 1:-1:2]) / 2.0
        u_gcoarse = (residual[0, :-2:2, :-2:2, :-2:2] +
                     residual[0, 1:-1:2, 1:-1:2, 1:-1:2]) / 2.0
        v_gcoarse = (residual[1, :-2:2, :-2:2, :-2:2] +
                     residual[1, 1:-1:2, 1:-1:2, 1:-1:2]) / 2.0
        w_gcoarse = (residual[2, :-2:2, :-2:2, :-2:2] +
                     residual[2, 1:-1:2, 1:-1:2, 1:-1:2]) / 2.0

        winds_coarse = np.stack([u_coarse, v_coarse, w_coarse]).flatten()
        residual = np.stack([u_gcoarse, v_gcoarse, w_gcoarse]).flatten()

        bounds = [(-x, x) for x in 100 * np.ones(winds_coarse.shape)]

        w = fmin_l_bfgs_b(_J_coarse, winds_coarse, args=(
            residual, vrs_coarse, azs_coarse, els_coarse,
            wts_coarse, u_back_coarse, v_back_coarse,
            u_model_coarse, v_model_coarse,
            w_model_coarse, Co, Cm, Cx, Cy, Cz, Cb,
            Cv, Cmod, Ut, Vt, grid_shape_coarse,
            dx, dy, dz, zc, rmsVr, weights_coarse,
            bg_weights_coarse, mod_weights_coarse, upper_bc, False),
                          maxiter=1, pgtol=1e-3, bounds=bounds,
                          fprime=_grad_coarse, disp=0, iprint=-1)
        winds_coarse = w[0]
        warnflag = w[2]["warnflag"]

        if output_cost_functions is True and iterations % 1 == 0:
            _J_coarse(
                winds_coarse, residual, vrs_coarse, azs_coarse, els_coarse,
                wts_coarse, u_back_coarse, v_back_coarse, u_model_coarse,
                v_model_coarse, w_model_coarse, Co, Cm, Cx, Cy, Cz, Cb, Cv,
                Cmod, Ut, Vt, grid_shape_coarse, dx, dy, dz, zc,
                rmsVr, weights_coarse, bg_weights_coarse, mod_weights_coarse,
                upper_bc, True)
            _grad_coarse(
                winds_coarse, residual, vrs_coarse, azs_coarse, els_coarse,
                wts_coarse, u_back_coarse, v_back_coarse, u_model_coarse,
                v_model_coarse, w_model_coarse, Co, Cm, Cx, Cy, Cz, Cb, Cv,
                Cmod, Ut, Vt, grid_shape_coarse, dx, dy, dz, zc, rmsVr,
                weights_coarse, bg_weights_coarse, mod_weights_coarse,
                upper_bc, True)
        winds_coarse = np.reshape(winds_coarse, (3, grid_shape_coarse[0],
                                                 grid_shape_coarse[1],
                                                 grid_shape_coarse[2]))
        iterations = iterations + 1

        # Now we extrapolate to finer grid using linear interpolation

        u_fine_interp = RegularGridInterpolator(
            (z_coarse, y_coarse, x_coarse), winds_coarse[0],
            bounds_error=False, fill_value=None)
        v_fine_interp = RegularGridInterpolator(
            (z_coarse, y_coarse, x_coarse), winds_coarse[1],
            bounds_error=False, fill_value=None)
        w_fine_interp = RegularGridInterpolator(
            (z_coarse, y_coarse, x_coarse), winds_coarse[2],
            bounds_error=False, fill_value=None)
        winds[0] += u_fine_interp((z, y, x))
        winds[1] += v_fine_interp((z, y, x))
        winds[2] += w_fine_interp((z, y, x))
        wcurrmax = winds[2].max()
        winds = winds.flatten()

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

    if (mask_outside_opt is True):
        u = np.ma.masked_where(where_mask < 1, u)
        v = np.ma.masked_where(where_mask < 1, v)
        w = np.ma.masked_where(where_mask < 1, w)

    if (mask_w_outside_opt is True):
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