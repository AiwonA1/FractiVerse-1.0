"""
📊 FractiAdmin 1.0 - System Monitoring
Monitors system performance, AI processing, and FractiChain transactions.
"""

import os
import psutil
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Initialize FastAPI app
app = FastAPI()

# Setup templates for UI rendering
templates = Jinja2Templates(directory="admin_dashboard/templates")


class FractiAdminMonitor:
    def __init__(self):
        self.system_metrics = {
            "CPU Usage": "N/A",
            "Memory Usage": "N/A",
            "Active AI Nodes": 0,
            "FractiChain Transactions": 0
        }

    def update_system_metrics(self):
        """Fetches and updates system performance metrics."""
        self.system_metrics["CPU Usage"] = f"{psutil.cpu_percent()}%"
        self.system_metrics["Memory Usage"] = f"{round(psutil.virtual_memory().total / (1024 ** 3), 2)}GB"
        self.system_metrics["Active AI Nodes"] = self.get_active_nodes()
        self.system_metrics["FractiChain Transactions"] = self.get_transaction_count()

    def get_active_nodes(self):
        """Mock function to return active AI nodes."""
        return 9  # Placeholder value

    def get_transaction_count(self):
        """Mock function to return FractiChain transactions."""
        return 0  # Placeholder value

    def get_system_metrics(self):
        """Returns current system metrics."""
        self.update_system_metrics()
        return self.system_metrics


# Initialize monitoring instance
admin = FractiAdminMonitor()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serves the Admin UI dashboard."""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "system_metrics": admin.get_system_metrics()}
    )


# If running locally
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))  # Default to 8080
    uvicorn.run(app, host="0.0.0.0", port=port)
