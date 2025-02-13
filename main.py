import os
from fracticody_engine import app  # Ensure fracticody_engine.py exists and has the app object

print("✅ main.py has started...")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets a PORT environment variable
    print(f"✅ Starting FractiCody server on port {port}...")  # Debugging output
    app.run(host="0.0.0.0", port=port)
