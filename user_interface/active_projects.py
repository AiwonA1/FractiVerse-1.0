"""
📋 Active Projects - AI-Driven Project Management
Manages active user projects and dynamically prioritizes tasks.
"""

class ActiveProjects:
    def __init__(self):
        self.projects = [
            {"name": "Unipixel Research", "status": "🔬 In Progress", "priority": "High"},
            {"name": "FractiChain AI Transactions", "status": "📈 Scaling", "priority": "Medium"},
            {"name": "Quantum AI Experiments", "status": "🚀 Prototype Phase", "priority": "High"},
        ]

    def list_active_projects(self):
        """Returns a formatted list of all active projects."""
        return [
            f"📌 {p['name']} - {p['status']} (Priority: {p['priority']})"
            for p in self.projects
        ]

    def add_project(self, name, status="🛠️ New", priority="Medium"):
        """Adds a new project to the list."""
        self.projects.append({"name": name, "status": status, "priority": priority})
        return f"✅ Project '{name}' added successfully."

    def update_project_status(self, name, status):
        """Updates the status of an existing project."""
        for project in self.projects:
            if project["name"] == name:
                project["status"] = status
                return f"🔄 Updated '{name}' to status: {status}"
        return "❌ Project not found."

    def prioritize_project(self, name, priority):
        """Adjusts the priority of an active project."""
        for project in self.projects:
            if project["name"] == name:
                project["priority"] = priority
                return f"⭐ Updated priority for '{name}' to {priority}"
        return "❌ Project not found."

if __name__ == "__main__":
    projects = ActiveProjects()
    print("\n".join(projects.list_active_projects()))
    print(projects.add_project("AI Ethics & Governance", "Planning", "High"))
    print(projects.update_project_status("Unipixel Research", "🧠 Advanced Testing"))
    print(projects.prioritize_project("FractiChain AI Transactions", "High"))
