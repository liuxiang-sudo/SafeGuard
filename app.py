"""Development entry point: python app.py."""
from safeguard.web.app import app

if __name__ == "__main__":
    app.run(debug=True)
