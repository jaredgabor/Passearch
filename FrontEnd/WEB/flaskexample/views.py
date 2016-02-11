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

print "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM"


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
##  patient = request.args.get('birth_month')
##  patient = request.args.get('latspan')
  lat = float(request.args.get('latbox'))
  lng = float(request.args.get('lngbox'))

  print "AAAAAARRRRRRGGGGGHHHHH", lat, lng

  
  # Careful w/ lat/lng order here!
  result, result_map = passearch_backend.get_results(lng, lat)  # con)
  return_lat = result[1]
  return_lng = result[0]

  model_data.set_map(result_map)

  return render_template("output.html", return_lat = return_lat, 
                         return_lng = return_lng,
                         input_lat = lat, input_lng = lng)
      


# @app.route('/_submitcoords')
# def process_coords():
#     testval = request.args.get('data',0,type=float)
    
#     answer = testval*2.0
#     print "HEYEYEYEYEYEYE"
    
#     return jsonify(result=answer)


@app.route('/_fetchcoords')
def fetch_coords():
    
    # Turn the model image (low-resolution) into a high-resolution image.
    modelim_hires = passearch_model.mask_lowres_to_hires(model_data.bestmap)

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
    weights = weights - 0.9 * weights.min()
    weights = 1.0 / weights
    weights = weights - weights.min()
    weights = weights / weights.max() * 100
    weights += 1

    # Create a data table of lat, lng, and weights based on each pixel.
    df = pd.DataFrame(lat, columns=['latitude'])
    df['longitude'] = lng
    df['weights'] = weights

    # take a small subset
    ii = 3
    df = df[np.array(df.index) % ii == 0]
#    ww = np.random.choice(np.arange(len(df)), size = len(df)/3, replace=False)
#    df = df[ww]
##    df = df.sample(frac=0.5)


    df['weights'] = (df['weights'] - 1.0)*100. + 1.0

    print "UNIQUE POINTS:", \
        weights.min(),weights.max(), len(df), wgood.sum()

    # Rename columns for easier handling on the website.
    df.columns = [0,1,2]

#     arr = [[40.771,-73.97,1.0],  
#            [40.772,-73.97, 3.0]]
#     df = pd.DataFrame(arr)

#     data = df.to_dict(orient='index')
    data = df.to_dict(orient = 'record')

##    print "MY DATA", df[0:10]
##    print "MY DATA", data

    return jsonify(result=data)

