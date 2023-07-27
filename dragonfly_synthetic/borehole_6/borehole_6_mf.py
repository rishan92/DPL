"""
  Borehole function in 8 dimensions (6 input) with a 2D (2 input) fidelity space.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=bad-whitespace

try:
  from borehole_6 import borehole_6_z
except ImportError:
  from .borehole_6 import borehole_6_z

def borehole_6_mf(z, x):
  """ Computes the Bore Hole function. """
  f0 = z[0] / 0.25
  f1 = z[1]
  return borehole_6_z(x, [f0, f1])


# Write a function like this called obj.
def objective(z, x):
  """ Objective. """
  return borehole_6_mf(z, x)

def cost(z):
  """ Cost function. """
  return 0.1 + 0.4 * (4 * z[0])**2 + 0.5 * z[1]

def main(z, x):
  """ Main function. """
  return borehole_6_mf(z, x), cost(z)

