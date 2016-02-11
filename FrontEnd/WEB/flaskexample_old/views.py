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


from flask import render_template
from flask import request
from flaskexample import app
##import requests

# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
# import pandas as pd
# import psycopg2

##from a_Model import ModelIt
import passearch_backend

user = 'jgabor' #add your username here (same as previous postgreSQL)                      
host = 'localhost'
dbname = 'taxi'
# db = create_engine('postgresql://%s%s/%s'%(user,host,dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)

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
  result = passearch_backend.get_results(lng, lat)  # con)
  return_lat = result[1]
  return_lng = result[0]

  return render_template("output.html", return_lat = return_lat, 
                         return_lng = return_lng,
                         input_lat = lat, input_lng = lng)
      
