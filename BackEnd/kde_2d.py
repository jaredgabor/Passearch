import datetime
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import numpy as np
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt


"""
OUT OF DATE!!!
Stuff for kernel density estimation in 2d.
"""

##############################
###############
# Some stuff for kernel density estimation
#
from sklearn.neighbors import KernelDensity

def kde_2d(xcoords, ycoords, xgrid, ygrid, bandwidth=None,
           doplot=0):
    """Do a kernel-density estimate in 2 dimensions.
    
    xgrid: a 1D array giving the x-positions where we want to
      estimate the density.  This is analogous to the bin locations
      in a histogram.
    """

    if bandwidth is None:
        bandwidth = 1.0

    # Rescale data because bandwidth must be equal for all dimensions
    # **Not necesarry as long as ygrid and xgrid bin sizes are equal
    xcoords_scaled = xcoords  ##rescale_dat(xcoords)
    ycoords_scaled = ycoords  ## rescale_dat(ycoords)

    # Arrange the inputs into 2 columns to give them to KernelDensity
    data = np.column_stack((xcoords_scaled, ycoords_scaled))

    # Run the kernel density estimate.
    kde = KernelDensity(bandwidth=bandwidth, rtol=1e-5, atol=1e-8, 
                        kernel='gaussian')
    print "Done with setup.  Now fitting"
    myfit = kde.fit(data)

    print "Done fitting. Next we apply the new grid"

    # Create a grid of points given the input x and ygrids.
    # We have to rearrange into a 2-column by npixels array.
    points_mygrid = grid_arrange(xgrid, ygrid)
    
    # Find the density at each desired grid point.
    log_pdf = kde.score_samples(points_mygrid)
    pdf = np.exp(log_pdf)

    # Rearrange the density array so that it's like an image
    pdf_im = np.reshape(pdf, (len(ygrid),len(xgrid)) )
    
    if doplot > 0:
        plt.imshow(pdf_im, origin='low', cmap='Greens', interpolation = 'none')

    return pdf_im


def rescale_dat(dat, n_std=1.0):
    """Re-scale data to be close to zero
    
    dat: a 1-D array (e.g. y-positions on a map)
    n_std: Width of output distribution in terms of 
      number of standard deviations 
    """
    newdat = (dat - np.mean(dat)) / np.std(dat) * n_std
    return newdat
    
def grid_arrange(xgrid, ygrid):
    nx = len(xgrid)
    ny = len(ygrid)

    # Create a 2D grid giving the x-value for each pixel
    xgrid2d = np.tile(xgrid,(ny,1))

    # Same for y-values.
    ygrid2d = np.tile(ygrid, (nx,1)).transpose()

    # Put the 2 grids together in one big array.
    mygrid = np.vstack((xgrid2d, ygrid2d))

    # Rearrange the values to make a 2 by (nx*ny) array,
    # and make it so there are 2 columns (like a 2-column matrix).
    points_mygrid = mygrid.reshape((2,nx*ny))
    points_mygrid = points_mygrid.transpose()
    return points_mygrid

