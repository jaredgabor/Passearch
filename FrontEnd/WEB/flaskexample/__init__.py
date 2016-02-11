import os
import sys
filename = os.environ.get('PYTHONSTARTUP')
if filename and os.path.isfile(filename):
    with open(filename) as fobj:
       startup_file = fobj.read()
    exec(startup_file)


other_pythondir1 = "/home/ubuntu/Passearch"
for root, dirs, files in os.walk(other_pythondir1):
    sys.path.append(root)

print "THIS IS A MESSAGE!"

from flask import Flask
app = Flask(__name__)
from flaskexample import views

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
