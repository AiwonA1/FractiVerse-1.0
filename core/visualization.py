"""Visualization functions for FractiVerse"""
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

def save_visualization_with_timestamp(name: str, data: Dict[str, Any], viz_type: str, output_dir: Path, config: Dict[str, Any]):
    """Save visualization with timestamp and proper directory structure"""
    try:
        # Set style for better visualizations
        sns.set_style("darkgrid")
        plt.style.use(config["visualization"]["style"])
        
        timestamp = datetime.now(timezone.utc).strftime("%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = output_dir / viz_type / filename
        
        # Create visualization based on type
        plt.figure(figsize=(12, 8))
        if viz_type == "gauge":
            sns.barplot(x=list(data["metrics"].keys()), y=list(data["metrics"].values()))
            plt.ylim(0, 1)
        elif viz_type == "status":
            status_colors = {"healthy": "green", "warning": "yellow", "error": "red"}
            plt.pie([1], labels=[data["status"]], colors=[status_colors.get(data["status"], "gray")])
        elif viz_type == "timeline":
            sns.lineplot(data=data)
            
        plt.title(f"{name} - {viz_type}")
        plt.tight_layout()
        plt.savefig(filepath, dpi=config["visualization"]["dpi"])
        plt.close()
        
        print(f"📊 Saved visualization: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Failed to save visualization: {str(e)}")
        return False 