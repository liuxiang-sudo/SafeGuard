"""Development entry point: python app.py."""
from multiguard.web.app import app

if __name__ == "__main__":
    app.run(debug=True)
