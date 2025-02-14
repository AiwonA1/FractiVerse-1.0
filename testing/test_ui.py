"""
🧪 Test - FractiGator UI Components
Unit tests for user interface modules.
"""
import unittest
from user_interface.fractigator_navigation import RealityNavigator
from user_interface.active_projects import ActiveProjects
from user_interface.daily_suggestions import DailySuggestions

class TestUI(unittest.TestCase):
    def test_reality_navigation(self):
        navigator = RealityNavigator()
        navigator.switch_reality("FractiVerse")
        self.assertEqual(navigator.get_current_reality(), "🌀 Current Reality: FractiVerse")

    def test_active_projects(self):
        projects = ActiveProjects()
        self.assertTrue("Active Projects" in projects.list_active_projects())

    def test_daily_suggestions(self):
        suggestions = DailySuggestions()
        self.assertTrue("🔹" in suggestions.get_daily_suggestion())

if __name__ == "__main__":
    unittest.main()
