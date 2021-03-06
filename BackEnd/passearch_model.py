import datetime
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
##import psycopg2
import pandas as pd
import numpy as np
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt

data_dir = "../../Data/"

coord_bottom_left = [-74.021759, 40.698861]
coord_top_right = [-73.927689, 40.816927]

#-----------Define convenient defaults for make_2dhist
# Length of 1 degree longitude divided by length
# of 1 degree latitude (at latitude of NYC)
aspect_geo = 52.52 / 69.0 
X_LENGTH_FACTOR = 52.52
Y_LENGTH_FACTOR = 69.0
coord_bottom_left_default = [-74.021759, 40.698861]
coord_top_right_default = [-73.927689, 40.816927]
npix_default = 800

lngmin = coord_bottom_left_default[0]
lngmax = coord_top_right_default[0]
latmin = coord_bottom_left_default[1]
latmax = coord_top_right_default[1]

# Default image is 6.2 miles = 10km across.  100 small pix = 0.775 miles.
# One lowres pixel = 0.194 miles.

def make_2dhist(lng, lat, npix_base=npix_default, 
                coord_bottom_left = coord_bottom_left_default,
                coord_top_right = coord_top_right_default,
                center_width_height = None, weights=None):
    """Make a 2d histogram from lat and long data

    lng, lat: arrays of longitude and latitude
    npix_base: # of pixels for x-axis (i.e. longitude axis).  The 
       # of pixels for the y-axis will be determined so that pixels
       are square.
    center_width_height: Define image boundaries by a rectangle with 
       a given (Lng,Lat) center and given width and height in MILES.
       This should be a 4-element vector where first 2 give Lng, Lat,
       and second 2 give width, height in miles.
    """
    xmin = lng.min()
    xmax = lng.max()
    ymin = lat.min()
    ymax = lat.max()
##    print "Range of Longitude, Latitude", xmin, xmax, ymin, ymax
    
    if center_width_height is not None:
        # Calculate Lng/Lat coords of image.
        imwidth_miles = center_width_height[1]
        imwidth_miles = center_width_height[2]
        dlng = imwidth_miles / X_LENGTH_FACTOR
        dlat = imheight_miles / Y_LENGTH_FACTOR
        coord_bottom_left = [center[0] - dlng/2.0, center[1] - dlat/2.0]
        coord_bottom_right = [center[1] + dlng/2.0, center[1] + dlng/2.0]
    else:
        imwidth_miles = (coord_top_right[0] - coord_bottom_left[0]) *\
            X_LENGTH_FACTOR
        imheight_miles = (coord_top_right[1] - coord_bottom_left[1]) *\
            Y_LENGTH_FACTOR
        
    # Determine # of pixels on y-axis
    npix = [npix_base, round(npix_base * imheight_miles / imwidth_miles)]
    
    # Determine Lng/Lat image boundaries
    imrange = [[coord_bottom_left[0], coord_top_right[0]],
              [coord_bottom_left[1], coord_top_right[1]] ]
    im,xedge,yedge = np.histogram2d(lng,lat,bins=npix, range=imrange,
                                    weights=weights)

    ##print "im shape", im.shape

    # Transpose the image so that it will look right when we plot it.
    im = im.transpose()
    return im


# Functions to convert from Long, Lat coordinates to 
# pixels and back.
# NB: when plotting, x and y axes are flipped!  Here 
# we use the convention that longitude is x-axis, and
# latitude is y-axis.
latrange = [latmin, latmax]
lngrange = [lngmin, lngmax]
###nbins = [im.shape[1], im.shape[0]]
imwidth_miles = (coord_top_right[0] - coord_bottom_left[0]) * X_LENGTH_FACTOR
imheight_miles = (coord_top_right[1] - coord_bottom_left[1]) * Y_LENGTH_FACTOR
npix = [npix_default, round(npix_default * imheight_miles / imwidth_miles)]
nbins = npix

def pix_to_coord(pixel, lngrange=lngrange, latrange=latrange,
                 nbins = nbins):
    """Convert pixel (ix,iy) to (long, lat) coordinate
    
    pixel: 2-element array
    lngrange: 2-element array giving min and max longitude
    latrange: 2-element array giving min and max latitude
    nbins: 2-element array giving the number of bins in x
       x and y directions
    """
    
    # Check whether given pixel indexes are within bounds
    if (pixel[0] < 0) or (pixel[0] > nbins[0]) or \
        (pixel[1] < 0) or (pixel[1] > nbins[1]):
            raise Exception("Pixel error")
            
    pix_size_x = (lngrange[1] - lngrange[0]) / nbins[0]
    pix_size_y = (latrange[1] - latrange[0]) / nbins[1]

    coord_lng = pixel[0] * pix_size_x + lngrange[0]
    coord_lat = pixel[1] * pix_size_y + latrange[0]    
    return np.array([coord_lng, coord_lat])
    
def pix_to_coord_all(im, lngrange=lngrange, latrange=latrange,
                 nbins = nbins):
    """Convert all pixel positions in a standard image to lng, lat coords.
    """
    npix_y, npix_x = im.shape
    nbins = [npix_x, npix_y]
    pix_size_x = (lngrange[1] - lngrange[0]) / nbins[0]
    pix_size_y = (latrange[1] - latrange[0]) / nbins[1]
    
    coords = np.mgrid[0:npix_y, 0:npix_x]
    xcoords = coords[1]
    ycoords = coords[0]

    coord_lng = xcoords * pix_size_x + lngrange[0]
    coord_lat = ycoords * pix_size_y + latrange[0]
    
    return coord_lng, coord_lat


def coord_to_pix(coord, lngrange = lngrange, latrange=latrange):
    """Convert coordinate (long, lat) to a pixel index (ix,iy)"""

    # Check whether given coord is within bounds
    if (coord[0] < lngrange[0]) or (coord[0] > lngrange[1]) or \
            (coord[1] < latrange[0]) or (coord[1] > latrange[1]):
        print "BAD COORDINATE", coord[0], coord[1], type(coord[0])
##        raise Exception("coordinate error ")
        return None
    
    # Figure out the pixel sizes
    pix_size_x = (lngrange[1] - lngrange[0]) / nbins[0]
    pix_size_y = (latrange[1] - latrange[0]) / nbins[1]
    
    # convert to pixel indexes
    pix_x = (coord[0] - lngrange[0]) / pix_size_x
    pix_y = (coord[1] - latrange[0]) / pix_size_y
    return np.array([pix_x, pix_y])


def coord_to_pix_all(lat, lng, lngrange=lngrange, latrange=latrange):
    """Convert many lat/lng values to pixel indexes
    """
    # Figure out the pixel sizes in units of 
    pix_size_x = (lngrange[1] - lngrange[0]) / nbins[0]
    pix_size_y = (latrange[1] - latrange[0]) / nbins[1]
    
    pix_x = (lng - lngrange[0]) / pix_size_x
    pix_y = (lat - latrange[0]) / pix_size_y
    return pix_x, pix_y

def get_time_index(timestamp=None):
    """Fetch the current time index
    
    The "time index" here refers to the time axis of our binned
    pickup time data.  If we split it into 24 hours, then 
    we have 24 time bins.  For a given time, we want to know
    which bin to use.  In this function, the "given time" is
    the present time in NYC.
    
    ***Requires datetime module
    """
    if timestamp is not None:
        pass

##    time_now = datetime.datetime.now()
    time_now = pd.Timestamp.now('US/Eastern')

    hour_now = time_now.hour
    day_now = time_now.weekday()

    # Approximate fractional month
    month_now = time_now.day / 30.0 + (time_now.month - 1)

    return day_now, hour_now, month_now, time_now
    
# Function to calculate the distance between one pixel
# in an image and all other pixels
def imdist(image, pixel):
    """Find distance between pixel and other image pixels"""
    # Create array of index values.  igrid[0] is an array
    # of the same size as "image" which gives the x-pixel
    # indexes of each pixel
    pix_x = pixel[0]
    pix_y = pixel[1]

##    npix_x, npix_y = image.shape
    npix_y, npix_x = image.shape
##    igrid = np.mgrid[0:npix_x, 0:npix_y].astype(float)
    igrid = np.mgrid[0:npix_y, 0:npix_x].astype(float)
    dist = np.sqrt( (igrid[0] - pix_y)**2.0 + (igrid[1]-pix_x)**2.0)
    return dist


def penalty_func(distance_image):
    # NB: This is operating on low-resolution image
    scale_radius = 10.0  # in pixels
##    return 1.0 / distance_image
##    newim = np.ones_like(distance_image) 
##    newim = distance_image.copy()

    # Here, the weighting decreases with distance.  This works is
    # we're trying to find the maximum demand (but not if we're
    # doing the minimum wait time!
##    newim = np.ones_like(distance_image) - distance_image / scale_radius

    newim = np.ones_like(distance_image) + distance_image / 50.

    newim[distance_image > scale_radius] = np.nan
    return newim
    pass

# def model1(im, lng, lat):
#     """Given longitude and latitude, choose the best destination"""

#     # Find the pixel corresponding to this lng and lat
#     coord = [lng, lat]
#     pixel = coord_to_pix(coord)

#     # Compute the distance image.
#     dist_im = imdist(im, pixel)

#     # Find the locally-weighted best destination
#     print "TYPE IM", type(im), type(dist_im)
#     local = im * penalty_func(dist_im)

#     # Find the maximum pixel
#     ipix_max = np.argmax(local)
#     ij_max = np.array(np.unravel_index(ipix_max, im.shape))

#     # Convert the pixel coords back to lat and long
#     LngLat = pix_to_coord(ij_max)

#     return LngLat
#     pass

def convert_pix_to_lowres(pixel, im):
    """Convert a 2-dimensional standard pixel position to a low-res version.

    im: a low=resolution image whose shape we'll use to determine 
       the reduction factor (aka zoom factor)
       
    """
    imshape = np.array(im.shape)
    
    reduction_factor = npix_default / np.min(imshape)

##    reduction_factor = 25
    pixel = pixel / reduction_factor
    return pixel

def model2(im, lng, lat):
    """Given the longitude, latitude, choose the best destination"""
    """Given longitude and latitude, choose the best destination"""

    # Find the pixel corresponding to this lng and lat
    coord = [lng, lat]
    pixel = coord_to_pix(coord)
    hipix = pixel

##    im = im.transpose()

    # Convert this pixel to the lowres-version
    pixel = convert_pix_to_lowres(pixel, im)

    # Compute the distance image.
    dist_im = imdist(im, pixel)

    print "DISTIM", np.nanmax(dist_im.ravel()), im.shape, pixel, hipix

    # Find the locally-weighted best destination
    local = im * penalty_func(dist_im)

#    np.save("junk.npy", dist_im)
#    np.save("local.npy", local)
#    np.save("base_im.npy", im)
#    print "MAX", np.nanmin(local)

    # Find the MINIMUM pixel
    ipix_max = np.nanargmin(local)
    ij_max = np.array(np.unravel_index(ipix_max, im.shape))

    # Convert this pixel back to hires version
##    ij_max = ij_max * 50
    ij_max = ij_max * 25

    # ij_max here is given as [pix_y, pix_x] as for images.
    # We must reverse it.
    
    # Convert the pixel coords back to lat and long
    LngLat = pix_to_coord(ij_max[::-1])

    return LngLat


def search_hipoint(im, pixel, rsearch_pix):
    """Find the peak in the image within rsearch_pix of pixel
    """
    dist_im = imdist(im, pixel)
    local = im.copy()
    local[dist_im > rsearch_pix] = 0.0

    # Find the maximum pixel
    ipix_max = np.argmax(local)
    ij_max = np.array(np.unravel_index(ipix_max, im.shape))

    return ij_max




def mask_lowres_to_hires(im_low, maskim = None):
    """ 
    Create a high-resolution map based on a low-resolution map,
    where the gaps between streets are nan.
    """

    # Read the high-resolution mask image.
    if maskim is None:
##        file = "/Volumes/MYPASSPORT/Data/maskim_hires.npz"
##        file = "/Users/jgabor/NEW_WORKDIR/INSIGHT/DATA/maskim_hires.npz"
        file = data_dir + 'maskim_hires.npz'
        with np.load(file) as dat:
            # This should be a float array
            maskim = dat['arr_0']

    print "TESTME PASSEARCH_MODEL", np.nansum(maskim)

    # Zoom in on the lowres image to make it highres
    reduction_factor = 25
    order = 0  # interpolation order for zoom in
    zoomim = ndimage.interpolation.zoom(im_low, reduction_factor, order=order)
    ny_zoom, nx_zoom = zoomim.shape

    # Make an image to store the result with the correct shape.
    # (NB: because the zoomed im may have the incorrect shape)
    maskedim = maskim.astype('float') * 0.0
        
    print "HEEEEEY", maskedim.shape, zoomim.shape, ny_zoom, nx_zoom
    maskedim[0:ny_zoom, 0:nx_zoom] = zoomim
#     maskedim[maskim <= 0.0] = np.nan
#     maskedim[~np.isfinite(maskim)] = np.nan
    return maskedim


def expand_features(xx, order=9):
    """Expand features xx into a polynomial in xx                               
                                                                                
    xx: a 1-D array                                                             
    """
    features = xx
    for index in np.arange(2, order+1):
        features = np.column_stack([features, xx**index])

    return features


def norm_weights(weights, do_log=False):
    """ Normalize the map weights for nice visualization.
    """
    weights = 1.0 / weights

    if do_log:
        weights = np.log10(weights)

    print "WWWWW", weights.min(), weights.max()
    weights = weights - weights.min()
    print "WWWWW", weights.min(), weights.max()

    weights = weights / weights.sum() * len(weights)
    
##    np.save("junk2.npy", weights)

    return weights
