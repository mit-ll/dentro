# Plot Module

There are two different modules listed: `bokeh` and `matplotlib`.  Each is designed to utilize a different backend for plotting.  The `bokeh` backend is built on Java and designed to utilize modern web browsers for displaying plots.  `matplotlib` is designed to adhere to MATLAB's API.

The `bokeh` backend should be used when interactive features are needed.  It provides more features designed to interact with the data (i.e. zooming, tapclick, etc).  The `matplotlib` backend should be used when you need more fine grain control over how specific plots are rendered.  However the trade-off is that `matplotlib` provides less user interaction.

The `utils` module is a set of common functions for the plotting module.  Both `bokeh` and `matplotlib` depends on common functionality found in `utils`.
