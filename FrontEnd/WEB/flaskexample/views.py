# from flask import render_template
# from flaskexample import app

# @app.route('/')
# @app.route('/index')
# def index():
# ###   return "Hello, World!"
#    user = { 'nickname': 'Miguel' } 
#    return render_template("index.html",
#                           title='Home',
#                           user=user)
import os
filename = os.environ.get('PYTHONSTARTUP')
if filename and os.path.isfile(filename):
    with open(filename) as fobj:
       startup_file = fobj.read()
    exec(startup_file)


from flask import render_template
from flask import request
from flask import jsonify
from flaskexample import app
##import requests

# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import numpy as np
# import psycopg2

##from a_Model import ModelIt
import passearch_backend
import passearch_model

user = 'jgabor' #add your username here (same as previous postgreSQL)                      
host = 'localhost'
dbname = 'taxi'
# db = create_engine('postgresql://%s%s/%s'%(user,host,dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)

## Load the models into memory for fast lookup.
DATA_DIR = passearch_model.data_dir
model_ims_file = DATA_DIR + 'model_ims_lowres.npz'
with np.load(model_ims_file) as dat:
    model_ims = dat['arr_0']

fit_file = DATA_DIR + 'fit_arr.npz'
# with np.load(fit_file) as dat:
#     fit_arr = dat['arr_0']
fit_arr = None

print "Done reading model data"

class ModelData:
    def __init__(self):
        self.bestmap = None
        pass

    def set_map(self, map):
        self.bestmap = map


model_data = ModelData()

@app.route('/')
@app.route('/index')
# def index():
#     return render_template("index.html",
#        title = 'Home', user = { 'nickname': 'Miguel' },
#        )

@app.route('/input')
def passearch_input():
    return render_template("input.html")


##@app.route('/output')
@app.route('/output')
def passearch_output():
  #pull 'birth_month' from input field and store it
  lat = float(request.args.get('latbox'))
  lng = float(request.args.get('lngbox'))

  # Careful w/ lat/lng order here!
  try:
      check, result, result_map, return_time = \
          passearch_backend.get_results(lng, lat, model_ims, fit_arr)
  except:
      print "UNKNOWN ERROR!!!"
      return render_template("unknown.html")

#   check, result, result_map, return_time = \
#       passearch_backend.get_results(lng, lat, model_ims, fit_arr)

  print "CHECKING", check
  if check < 1:
      return render_template("badcoords.html")

  # if all checks were OK, then continue
  return_lat = result[1]
  return_lng = result[0]

  np.save('resultmap.npy', result_map)

  model_data.set_map(result_map)

  return render_template("output.html", return_lat = return_lat, 
                         return_lng = return_lng,
                         input_lat = lat, input_lng = lng, 
                         return_time=return_time)


@app.route('/badcoords')
def bad_coords():
    return render_template("badcoords.html")

@app.route('/unknown')
def unknown():
    return render_template("unknown.html")

@app.route('/_fetchcoords')
def fetch_coords():
    
    # Turn the model image (low-resolution) into a high-resolution image.
##    modelim_hires = passearch_model.mask_lowres_to_hires(model_data.bestmap)
    modelim_hires = model_data.bestmap

    print "BLAHBLAHBLAH"

##    np.save('testim.npy', modelim_hires)

    # Subsample a small region?

    # flatten 

    # convert each pixel position into lat/lng coordinates
    lng, lat = passearch_model.pix_to_coord_all(modelim_hires)

    # Find "good" points
    print "GOOG", np.nansum(modelim_hires > 0.0)
    wgood = modelim_hires > 0

    lng = lng[wgood]
    lat = lat[wgood]
    weights = modelim_hires[wgood]

    # flatten arrays
    lng = lng.ravel()
    lat = lat.ravel()
    weights = weights.ravel()    

    # Renormalize the weights.  They must be inverted so that 
    # higher weight == more demand.
#    weights = weights - 0.9 * weights.min()
    weights = passearch_model.norm_weights(weights, do_log=False)
#    weights = 1.0 / weights
#    weights = weights - weights.min()
#    print 'WEIGHTS MINMAX', weights.min(), weights.max()
#    weights = weights / weights.sum() * len(weights)
#    print 'WEIGHTS MINMAX', weights.min(), weights.max()
#    weights = weights - weights.min()
#    weights = weights / weights.max() * 100
#    weights += 1

    # Create a data table of lat, lng, and weights based on each pixel.
    df = pd.DataFrame(lat, columns=['latitude'])
    df['longitude'] = lng
    df['weights'] = weights

    # take a small subset
#    ii = 3
#    df = df[np.array(df.index) % ii == 0]
#    ww = np.random.choice(np.arange(len(df)), size = len(df)/3, replace=False)
#    df = df[ww]
##    df = df.sample(frac=0.5)


#    df['weights'] = (df['weights'] - 1.0)*100. + 1.0

    print "UNIQUE POINTS:", \
        weights.min(),weights.max(), len(df), wgood.sum()

    # Rename columns for easier handling on the website.
    df.columns = [0,1,2]

#     arr = [[40.771,-73.97,1.0],  
#            [40.772,-73.97, 3.0]]
#     df = pd.DataFrame(arr)

#     data = df.to_dict(orient='index')
    data = df.to_dict(orient = 'record')

##    print data
##    print "MY DATA", df[0:10]
##    print "MY DATA", data

    return jsonify(result=data)
