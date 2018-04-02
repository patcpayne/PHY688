This directory contains a one dimensional hydrodynamics code.
The code solves the Euler Equations of hydrodynamics and outputs
the results of the code as compared to the exact solution to
four different shock tube problems. The simulation is run using
64, 128, and 256 cells. In addition, it is run using flux limited
slopes to obtain second-order spatial accuracy and is run with
second-order runga-kutta for time accuracy. The limiting can be changed
by modifying the last line of the file euler1d.py.