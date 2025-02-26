from fracticody_engine import FractiCognition
from admin_dashboard.admin_ui import AdminInterface
from config import load_config

def test_fractiverse_system():
    try:
        config = load_config()
        cognitive_engine = FractiCognition()
        admin_ui = AdminInterface(cognitive_engine)
        print("✓ Systems initialized")
        test_input = "Hello FractiVerse"
        response = cognitive_engine.process(test_input)
        print(f"Cognitive Response: {response}")
        system_status = admin_ui.get_system_status()
        print(f"System Status: {system_status}")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_fractiverse_system()
