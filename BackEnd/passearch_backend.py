import numpy as np

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
##import psycopg2

import passearch_model

DATA_DIR = passearch_model.data_dir

def get_results(lng, lat, connection=None):
##    con = inputs
##    con = connection

    # Get current time index
    day_index, hour_index = passearch_model.get_time_index()

    # Fetch the image
    im = fetch_image2(day_index, hour_index)

    LngLat = passearch_model.model2(im, lng, lat)
    
    return LngLat, im


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
##    dir = "/Volumes/MYPASSPORT/Data/"
#    dir = "/Users/jgabor/NEW_WORKDIR/INSIGHT/DATA/"
##    fit_file = dir + "difference_fits.npy"

    dir = DATA_DIR

    model_file = dir + 'model_lowres.npz'
    with np.load(model_file) as dat:
        ims = dat['arr_0']


    return ims[:,:,iday,ihour]


#     mean_ims_file = dir + "mean_ims_timediff.npz"
#     with np.load(means_ims_file) as dat:
#         diff_ims = dat['arr_0']
        
#     hour = 20
#     day = 3
#     xpos = 300
#     ypos = 700
#     month = 8
#     imonth = month - 1
    
#     month_ims_file = dir + "timediff_ims_%d.npz" % month
#     with np.load(months_ims_file) as dat:
#         month_ims = dat['arr_0']

#     map = month_ims[:, :, day, hour]
