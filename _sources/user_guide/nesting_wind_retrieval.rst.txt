.. _nesting-wind-retrieval:

Nested wind retrievals
======================

PyDDA supports nested wind retrievals via :func:`pydda.retrieval.get_dd_wind_field_nested`.
A nested retrieval performs the optimization on a coarse outer grid first, then uses
the result to initialize progressively finer inner grids. This approach can reduce
computational cost while still resolving fine-scale features in a region of interest.

-----------------------------
Building the nested grid tree
-----------------------------

Nested grids are organized in an :class:`xarray.DataTree`. Each node of the tree
represents one nesting level and contains one child dataset per radar
(named ``radar_0``, ``radar_1``, …). Per-level retrieval parameters can be stored
as node attributes to override the top-level keyword arguments for that nesting level.

.. code-block:: python

    import xarray as xr
    from xarray import DataTree
    import pydda

    # Assume grid_ktlx_coarse, grid_kict_coarse, grid_ktlx_fine, grid_kict_fine
    # have already been created with pyart.map.grid_from_radars and converted
    # with pydda.io.read_from_pyart_grid.

    tree = DataTree.from_dict(
        {
            "/nest_0/radar_0": grid_ktlx_coarse,
            "/nest_0/radar_1": grid_kict_coarse,
            "/nest_1/radar_0": grid_ktlx_fine,
            "/nest_1/radar_1": grid_kict_fine,
        }
    )

    output_tree = pydda.retrieval.get_dd_wind_field_nested(
        tree,
        Cm=256.0,
        Co=1e-2,
        Cx=100,
        Cy=100,
        Cz=1,
        model_fields=["hrrr"],
        refl_field="DBZ",
        engine="scipy",
    )

The function returns the same :class:`xarray.DataTree` with an ``output_grids``
entry added to each node containing the retrieved wind fields for that nesting level.

.. note::

   All grids within the same nesting level must share the same grid shape,
   origin latitude/longitude, and coordinate arrays. The fine grid does not need
   to match the coarse grid.

For full API details see :func:`pydda.retrieval.get_dd_wind_field_nested`.
