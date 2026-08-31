"""
Filling single-Doppler data voids with the VVAD constraint
----------------------------------------------------------

A dual-Doppler retrieval is only well posed where two radars see the same
point from sufficiently different directions. Everywhere else - beyond the
dual-Doppler lens, and at the low levels that sit below the lowest radar gate -
the horizontal wind is underdetermined.

Protat et al. (2024) address this in the SWIRL system by adding a variational
velocity azimuth display (VVAD) term to the cost function. A VVAD fits vertical
profiles of a linear wind model to each radar's Doppler velocities, and those
profiles reconstruct a horizontal wind everywhere within range of the radar,
including at points with no observation at all. The term is switched on only
where at most one radar contributes, so it never competes with genuine
multi-Doppler information.

This example retrieves the TWP-ICE case with and without the constraint.

Protat, A., V. Louf, and J. P. Brook, 2024: SWIRL: The First Australian
Operational Radar-Based 3D Wind Analysis System. J. Atmos. Oceanic Technol.,
41, 891-910, https://doi.org/10.1175/JTECH-D-23-0155.1

Author: Robert C. Jackson

"""

from copy import deepcopy

import numpy as np
import pydda
from matplotlib import pyplot as plt

berr_grid = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
cpol_grid = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR1)

berr_grid = pydda.initialization.make_constant_wind_field(
    berr_grid, (0.0, 0.0, 0.0), vel_field="corrected_velocity"
)

# Retrieve a VVAD from each radar and reconstruct a consensus horizontal wind.
Grids = pydda.constraints.make_constraint_from_vvad(
    [berr_grid, cpol_grid], vel_field="corrected_velocity"
)

# The VVAD profiles themselves are worth inspecting: only the levels with
# enough well distributed data are retrieved.
profiles = pydda.constraints.vvad_retrieval(Grids[1], vel_field="corrected_velocity")
valid = profiles["valid"].values

fig, ax = plt.subplots(1, 3, figsize=(11, 5), sharey=True)
ax[0].plot(profiles["U0"].values[valid], profiles["z"].values[valid] / 1e3, "o-")
ax[0].plot(profiles["V0"].values[valid], profiles["z"].values[valid] / 1e3, "s-")
ax[0].legend(["$U_0$", "$V_0$"])
ax[0].set_xlabel("Wind at the radar (m/s)")
ax[0].set_ylabel("Height (km)")
ax[1].plot(profiles["DIV"].values[valid] * 1e4, profiles["z"].values[valid] / 1e3, "o-")
ax[1].set_xlabel("Divergence ($10^{-4}$ s$^{-1}$)")
ax[2].plot(profiles["npoints"].values, profiles["z"].values / 1e3, "k-")
ax[2].plot(
    profiles["npoints"].values[~valid],
    profiles["z"].values[~valid] / 1e3,
    "rx",
    label="rejected",
)
ax[2].set_xlabel("Points in fit")
ax[2].legend()
fig.suptitle("VVAD profiles from the CPOL radar")
plt.show()

# Now retrieve with and without the constraint. Protat et al. (2024) weight
# the VVAD term equally with the radial velocity term.
common = dict(
    Co=1.0,
    Cm=256.0,
    Cx=1e-2,
    Cy=1e-2,
    Cz=1e-2,
    max_iterations=100,
    vel_name="corrected_velocity",
    refl_field="reflectivity",
    mask_outside_opt=True,
)

no_vvad, _ = pydda.retrieval.get_dd_wind_field(deepcopy(Grids), Cvad=0.0, **common)
with_vvad, parameters = pydda.retrieval.get_dd_wind_field(
    deepcopy(Grids), Cvad=1.0, **common
)

print(
    "The VVAD constraint was active at %d of %d grid points."
    % (int((parameters.vad_weights > 0).sum()), parameters.vad_weights.size)
)

# Where the constraint acts, i.e. where multi-Doppler information is absent.
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
ax.pcolormesh(
    Grids[0]["point_x"].values[3] / 1e3,
    Grids[0]["point_y"].values[3] / 1e3,
    parameters.vad_weights[3],
    cmap="Greys",
    vmin=0,
    vmax=1.5,
)
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_title("$i_{vad}$ at z = 1.5 km: dark is where the VVAD constrains the wind")
plt.show()

# The retrieved winds, without and with the constraint.
for Grids_out, label in ((no_vvad, "without VVAD"), (with_vvad, "with VVAD")):
    plt.figure(figsize=(9, 9))
    pydda.vis.plot_horiz_xsection_barbs(
        Grids_out,
        None,
        "reflectivity",
        level=3,
        w_vel_contours=[3, 6, 9],
        barb_spacing_x_km=10.0,
        barb_spacing_y_km=10.0,
    )
    plt.title("Horizontal winds at z = 1.5 km, %s" % label)
    plt.show()
