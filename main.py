import os
from fracticody_engine.fracticody_engine import app  # Adjusted for subfolder

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets a PORT environment variable
    print(f"✅ Starting FractiCody server on port {port}...")  # Debugging Output
    app.run(host="0.0.0.0", port=port)
