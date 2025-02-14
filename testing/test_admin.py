"""
🧪 Test - FractiCody Admin Dashboard
Unit tests for system monitoring, auditing, and broadcast messaging.
"""
import unittest
from admin_dashboard.admin_monitor import AdminMonitor
from admin_dashboard.cognition_flow import CognitionFlow
from admin_dashboard.fracti_auditing import FractiAuditing
from admin_dashboard.broadcast_messaging import BroadcastMessaging

class TestAdmin(unittest.TestCase):
    def test_admin_monitor(self):
        monitor = AdminMonitor()
        self.assertTrue("CPU Usage" in monitor.get_system_status())

    def test_cognition_flow(self):
        cognition = CognitionFlow()
        self.assertTrue("🧠" in cognition.get_current_state())

    def test_auditing_system(self):
        auditing = FractiAuditing()
        auditing.log_event("Security Check Passed")
        self.assertTrue("Security Check Passed" in auditing.list_events())

    def test_broadcast_messaging(self):
        messaging = BroadcastMessaging()
        messaging.send_message("System Update")
        self.assertTrue("System Update" in messaging.list_messages())

if __name__ == "__main__":
    unittest.main()
