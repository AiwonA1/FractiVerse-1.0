"""
📂 Active Projects Tracker
Displays user-selected projects and ongoing AI-driven tasks.
"""
class ActiveProjects:
    def __init__(self):
        self.projects = ["Unipixel Research", "FractiChain AI Transactions", "Quantum AI Experiments"]

    def list_active_projects(self):
        """Returns the list of active user projects."""
        return f"📋 Active Projects: {', '.join(self.projects)}"

    def add_project(self, project_name):
        """Adds a new project to the active list."""
        self.projects.append(project_name)
        return f"✅ Project Added: {project_name}"

if __name__ == "__main__":
    projects = ActiveProjects()
    print(projects.list_active_projects())
    print(projects.add_project("AI Ethics Simulation"))
