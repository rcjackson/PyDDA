.. PyDDA documentation master file, created by
   sphinx-quickstart on Tue May 15 12:19:06 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the PyDDA documentation!
===================================

.. image:: logo.png
    :width: 200px
    :align: center
    :height: 100px
    :alt: alternate text

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3942686.svg
   :target: https://doi.org/10.5281/zenodo.3942686

This is the main page for the documentation of PyDDA. Below are links that
provide documentation on the installation and use of PyDDA as well as
description of each of PyDDA's subroutines.

.. grid:: 1 2 2 2
    :gutter: 2

    .. grid-item-card:: :octicon:`book;10em`
        :link: user_guide/index
        :link-type: doc
        :text-align: center

        **User Guide**

        The user guide provides in-depth information on how
        to use PyDDA, including how to get started.
        This is where to look for general conceptual descriptions on how
        to use parts of PyDDA, including how to make your first wind retrieval and
        the required data preprocessing to do so.

    .. grid-item-card:: :octicon:`list-unordered;10em`
        :link: dev_reference/index
        :link-type: doc
        :text-align: center

        **Reference Guide**

        The reference guide contains detailed descriptions on
        every function and class within PyDDA. This is where to turn to understand
        how to use a particular feature or where to search for a specific tool

    .. grid-item-card:: :octicon:`terminal;10em`
        :link: contributors_guide/index
        :link-type: doc
        :text-align: center

        **Developer Guide**

        Want to help make PyDDA better? Found something
        that's not working quite right? You can find instructions on how to
        contribute to PyDDA here. You can also find detailed descriptions on
        tools useful for developing PyDDA.

    .. grid-item-card:: :octicon:`graph;10em`
        :link: source/auto_examples/index
        :link-type: doc
        :text-align: center

        **Example Gallery**

        Check out PyDDA's gallery of examples which contains
        sample code demonstrating various parts of PyDDA's functionality.

=========================
System Requirements
=========================

This works on any modern version of Linux, Mac OS X, and Windows. For Windows,
HRRR data integration is not supported. In addition, since PyDDA takes advantage
of parallelism we recommend:
::
    An Intel machine with at least 4 cores
    8 GB RAM
    1 GB hard drive space
::

While PyDDA will work on less than this, you may run into performance issues.
In addition, we do not support Python versions less than 3.9.
If you have an older version installed, PyDDA may work just fine but we will
not provide support for any issues unless you are using at least Python 3.9.

=========================
Installation instructions
=========================

The GitHub repository for PyDDA is available at:

`<https://github.com/openradar/PyDDA>`_

We do not support Python versions less than 3.9. If you use earlier Python versions,
be aware, there is no testing done for these versions so problems may arise!

In order to use the HRRR data constraint, cfgrib needs to be installed. `cfgrib
<http://github.com/ecmwf/cfgrib>`_ currently only works on Mac OS and Linux, so
this is an optional dependency of PyDDA so that Windows users can still use PyDDA.
In order to install cfgrib, simply do:

.. _cfgrib: https://github.com/ecmwf/cfgrib

::

 pip install cfgrib
::

There are multiple ways to install PyDDA. The recommended way to install PyDDA is
through the use of the `Anaconda <http://anaconda.org>`_ package manager. If you
have anaconda installed simply type:
::

    conda install -c conda-forge pydda
::

This will install PyDDA and all of the required dependencies. You still need to
install `cfgrib <http://github.com/ecmwf/cfgrib>`_ if you wish to read HRRR data.
If you do not have anaconda, you can still install PyDDA using pip. Running this command
will install PyDDA and the required dependencies:
::

    pip install pydda
::


If you wish to contribute to PyDDA, you should install PyDDA from source. To do this,
just type in the following commands assuming you have the above dependencies installed.

::

 git clone https://github.com/openradar/PyDDA
 cd PyDDA
 pip install -e .
::

Finally, PyDDA now supports using `Jax <https://jax.readthedocs.io>`_ and `TensorFlow <https://tensorflow.org>`_
for solving the three dimensional wind field. PyDDA requires TensorFlow 2.x and the
tensorflow-probability package for TensorFlow to be enabled.
In addition, both of these packages can utilize CUDA-enabled GPUs for much faster processing. These two
dependencies are optional as the user can still use PyDDA with the SciPy ecosystem.
The Jax optimizer uses the same optimizer as SciPy's (`L-BFGS-B <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_).


Known issues
============

The TensorFlow engine uses the unbounded version of this optimizer which removes the constraint that
wind magnitudes must be less than 100 m/s. The removal of this constraint can sometimes
result in numerical instability, so it is recommended that the user test out both Jax and TensorFlow
if they desire GPU-accelerated retrievals.

Contents:

.. toctree::
   :maxdepth: 3

   user_guide/index
   contributors_guide/index
   source/auto_examples/index
   dev_reference/index

Further support
===============


We are now requesting that all questions related to PyDDA that are not potential software issues to be
relegated to the `openradar Discourse group <https://openradar.discourse.group>`_ with a 'pydda' tag on it. This
enables the entire open radar science community to answer questions related to PyDDA so that both the maintainer
and users can answer questions people may have.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
