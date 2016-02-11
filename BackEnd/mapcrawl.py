import numpy as np
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt

import passearch_model
import diff_maps

def one_step(im, startpos, bestpos, oldvec):
    """Take one step towards the destination position.
    
    oldvec: the vector from the previous position to the current
        position (i.e. the vector that brought us here
        to startpos).
    """
    vec1 = bestpos-startpos
    
    trial_ind = np.mgrid[startpos[0]-1:startpos[0]+2,
                        startpos[1]-1:startpos[1]+2]
    nx = 3
    ny = 3
    weight = np.zeros((ny,nx))
    for ii in range(nx):
        for jj in range(ny):
            if ii == 1 and jj == 1:
                weight[jj,ii] = 0
                break
            vec2 = trial_ind[:,jj, ii] - startpos
            
            # Prevent moving back to old steps, and the immediate
            # neighbors of old steps.
            if np.dot(vec2, oldvec) < 0:
                weight[jj,ii] = 0
                break
                
            # Find the angle between this trial move and
            # the final destination.
            cos_theta = np.dot(vec1,vec2) / \
                (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            theta = np.arccos(cos_theta)
            factor = 1.5
            weight[jj,ii] = np.cos(theta / factor)
            
            
    
    subim = im[startpos[0]-1:startpos[0]+2,
              startpos[1]-1:startpos[1]+2]
    flux = subim * weight

    # If new_ind == startpos, then just move to the pixel closest 
    # to the destination
    if np.nanmax(flux) <= 0:
        flux = weight

#     print weight
#     print ""
#     print 'flux', flux
    ipix_max = np.argmax(flux)
    ij_max = np.array(np.unravel_index(ipix_max, subim.shape))
#     print ij_max, ipix_max
    new_ind = [trial_ind[0,ij_max[0],ij_max[0]], 
               trial_ind[1,ij_max[1],ij_max[1]]]
    new_ind = np.array(new_ind)

    if (new_ind - startpos).sum() == 0:
        print "BAH", subim, flux, weight
    return new_ind
    


def test():
    """Test out the map crawling algorithm 
    """

    # make up a starting position
    startpos = np.array([500,300])

    print "reading pickup file"
    # make a map that shows allowed streets.
    with np.load("/Volumes/MYPASSPORT/Data/pickup_ims_3.npz") as dat:
        ims = dat['arr_0']
    im = ims[:,:,:,:].sum(axis=2).sum(axis=2)
    wstreet = im > 5.0  # arbitrary limit

    print " reading timediff file"
    # Get a timediff map
    with np.load("/Volumes/MYPASSPORT/Data/timediff_ims_3.npz") as dat:
        ims = dat['arr_0']
    im = ims[:,:,5,19]

    # downgrade image resolution to find the best final destination
    npix_x_lowres, npix_y_lowres = (16,26)
    im_low = diff_maps.reduce_resolution(im, npix_y_lowres, npix_x_lowres)
    im_low[~np.isfinite(im_low)] = np.nanmax(im_low) * 1.1

    # Convert startpos to the lower resolution scale
    reduction_factor = 50
    start_low = startpos / reduction_factor

    # Find the best spot on the lowres image
    rsearch_pix = 4.0
    hipoint_low = \
        passearch_model.search_hipoint(im_low, start_low, rsearch_pix)
    
    # Translate back to hires 
    hipoint = hipoint_low * reduction_factor
    print "Highpoint", hipoint

    # Get smoothed timediff image
    smoothim = get_smoothed(im_low, wstreet)

    # Invert timediff to get a "demand index"
    demandim = 1.0 / smoothim
    demandim[smoothim <= 0] = 0.0

    # Loop until we've reached the high point
    newpoint = startpos
    oldpoint = newpoint
    step_vector = newpoint - oldpoint
    while (newpoint - hipoint).sum() > 0:
        oldpoint = newpoint
        newpoint = one_step(demandim, newpoint, hipoint, step_vector)
        step_vector = newpoint - oldpoint
        print "NEWPOINT", newpoint
        pass
    
    
def get_smoothed(im_lowres, streetmap_image):
    """

    streetmap_image: a boolean image array indicating which pixels are
      actually streets (and not buildings)
    """
    # zoom in on the lowres image to create a big image
    reduction_factor = 50
    zoomim = ndimage.interpolation.zoom(im_lowres, reduction_factor,order=0)
    ny_zoom, nx_zoom = zoomim.shape

    smoothim = streetmap_image.astype('float') * 0.0
    smoothim[0:ny_zoom, 0:nx_zoom] = zoomim
    smoothim[~streetmap_image] = 0.0
    return smoothim
