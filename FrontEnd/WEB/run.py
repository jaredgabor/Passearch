#!/usr/bin/env python

# Retrieve and execute startup file to get the right
# paths
#import os
#filename = os.environ.get('PYTHONSTARTUP')
#if filename and os.path.isfile(filename):
#    with open(filename) as fobj:
#       startup_file = fobj.read()
#    exec(startup_file)



from flaskexample import app
app.run(debug = True, host='0.0.0.0', port=5000)

