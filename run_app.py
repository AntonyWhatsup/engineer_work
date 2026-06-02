import os
import sys

# Add the web_app directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'web_app')))

from web_app.app import app

if __name__ == '__main__':
    print("Starting Credit Risk AI web application from root...")
    app.run(debug=True, port=5000)
