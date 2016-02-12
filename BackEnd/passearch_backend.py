import numpy as np

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
##import psycopg2

import passearch_model

DATA_DIR = passearch_model.data_dir

def get_results(lng, lat, model_ims, fit_arr, 
                timestamp = None,
                connection=None):
    """Figure out the best lat and lng to go to!
    """

    # Get current time index
    day_index, hour_index, month_frac, time_now = \
        passearch_model.get_time_index(timestamp)

##    hour_index = 14

    print "DAY INDEX", day_index, hour_index, month_frac

    # Check whether the inputs are OK.
    check = check_coords(lng,lat, day_index, hour_index, model_ims)
    if check < 1:
        return check, [-1,-1], -1, -1

    # Fetch the image
#     im = fetch_image2(day_index, hour_index)
    im = fetch_image3(day_index, hour_index, model_ims, fit_arr, month_frac)

    LngLat = passearch_model.model2(im, lng, lat)
    
    return_time = time_now.strftime("%c")

    return check, LngLat, im, return_time


def fetch_image(index, connection=None):
    """Get an image from the database"""
    if connection is None:
        dbname = 'taxi'
        username = 'jgabor'
        con = None
        con = psycopg2.connect(database = dbname, user = username)
    else:
        con = connection

    # query:
    sql_query = "SELECT im FROM images WHERE id = %d ;" % index
    subdat = pd.read_sql_query(sql_query,con)

    # Extract just the image array from the data frame
    im = np.array( subdat['im'].values[0] )

    # If we opened a connection just for this, we should now close it
    if connection is None:
        con.close()

    return im


def fetch_image2(iday, ihour):

    dir = DATA_DIR

    model_file = dir + 'model_lowres.npz'
    with np.load(model_file) as dat:
        ims = dat['arr_0']


    return ims[:,:,iday,ihour]


def fetch_image3(iday, ihour, model_ims, fit_arr, month_frac):
    """Get image from already-loaded model arrays. 

    month_frac: The month of the year, expressed as a fraction, where 
    0.5 would represent January 15th.
    """
    
    mean_im = model_ims[:,:,iday,ihour]

    if fit_arr is None:
        print "No Fit array provided"
        return mean_im


    bestim = mean_im.copy()
    fit_arr_thishour = fit_arr[:,:,iday,ihour]
    
    wgood = np.isfinite(mean_im)
    
    npix_y, npix_x = mean_im.shape
    count=0
    for jy in range(npix_y):
        for ix in range(npix_x):
            if wgood[jy, ix] is False:
                continue
            clf = fit_arr[jy, ix, iday, ihour]
            if clf is None:
                continue
            xx = passearch_model.expand_features(month_frac)
            yy = clf.predict(xx)
            bestim[jy,ix] = mean_im[jy,ix] + yy
            count+=1

    print "Fetched image pixels:", count, wgood.sum(), bestim.size
    return bestim
    
def check_coords(lng, lat, day_index, hour_index, model_ims):
    """ Check whether latitude and longitude are within bounds.
    """

    im = model_ims[:,:,day_index, hour_index]

    coord = [lng, lat]
    pixel = passearch_model.coord_to_pix(coord)

    # If this pixel is off the map, return check=0
    if pixel is None:
        return 0

    pixel_lores = passearch_model.convert_pix_to_lowres(pixel, im)
    
    print "PIXEL", pixel, pixel_lores

    if np.isfinite(im[pixel_lores[1], pixel_lores[0]]):
        return 1
    else:
        return 0

