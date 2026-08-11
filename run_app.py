import os

from web_app.app import create_app

if __name__ == "__main__":
    print("Starting Credit Default Risk DSS prototype...")
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    create_app().run(debug=debug, port=int(os.environ.get("PORT", "5000")))
