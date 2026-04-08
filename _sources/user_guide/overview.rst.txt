Overview
==========================

PyDDA (Pythonic Direct Data Assimilation) is a Python package for retrieving
three-dimensional wind fields from one or more Doppler weather radars. This
technique is commonly called *multi-Doppler* or *dual-Doppler* wind retrieval
(though PyDDA supports an arbitrary number of radars).

Why multi-Doppler retrieval?
----------------------------

A single Doppler radar measures only the component of wind motion *along* the
radar beam (the radial velocity). By combining observations from two or more
radars viewing the same storm volume from different angles, it becomes possible
to decompose the full three-dimensional wind vector (u, v, w) at each grid point.

The quality of the retrieved vertical velocity **w** depends strongly on the
*beam-crossing angle* (BCA) — the angle between the two radar beams at each
grid point. PyDDA uses a BCA threshold (default: 30°–150°) to mask grid points
where the geometry is too unfavorable for reliable multi-Doppler synthesis.

How PyDDA works
---------------

PyDDA formulates wind retrieval as a variational problem. It minimizes a cost
function **J** that penalizes deviations from:

* observed radial velocities from each radar (*J*\ :sub:`o`)
* the anelastic mass continuity equation (*J*\ :sub:`m`)
* smoothness of the wind field (*J*\ :sub:`s`)
* optional background/sounding constraints (*J*\ :sub:`b`)
* optional model constraints, e.g., HRRR or WRF (*J*\ :sub:`mod`)

The minimization uses the L-BFGS-B algorithm (via SciPy, Jax, or TensorFlow)
to find the wind field that best satisfies all active constraints simultaneously.
Each constraint is weighted by a user-supplied coefficient, allowing the
relative importance of each term to be tuned for a given radar network geometry
and scientific application.

Workflow overview
-----------------

The full multi-Doppler retrieval workflow consists of the following steps,
each covered in a dedicated section of this user guide:

1. **Read radar data** — Load raw radar files using Py-ART
   (:ref:`reading-radar-data`).
2. **Quality control** — Remove noise, ground clutter, and second-trip echoes,
   then dealias the Doppler velocities (:ref:`dealiasing-velocities`).
3. **Grid to Cartesian coordinates** — Project the radar data from native
   antenna coordinates onto a common Cartesian grid (:ref:`gridding`).
4. **Retrieve the wind field** — Run PyDDA's variational solver
   (:ref:`retrieving_winds`).
5. **Optimize parameters** — Tune the cost function weights to balance
   accuracy and smoothness (:ref:`optimizing-wind-retrieval`).
6. **Visualize** — Plot horizontal and vertical cross-sections of the
   retrieved wind field (:ref:`visualizing-winds`).

If you would like to contribute a new section to this user guide,
please use the ``template.rst`` file in the documentation source directory.
