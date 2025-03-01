"""
FractiCody Basic Components Initialization Test
---------------------------------------------

This test verifies the initialization and connectivity of FractiCody's core components.
It serves as a fundamental health check of the system's critical components.

Components Tested:
1. FractiCore: The core cognitive system
   - Required method: activate()
   - Primary cognitive processing unit

2. MemoryManager: Memory handling system
   - Required method: load_memory()
   - Manages memory storage and retrieval

3. DecisionEngine: Decision making system
   - Required method: process_decision()
   - Handles logical decision processes

4. ProcessingUnit: Data processing system
   - Required method: process()
   - Handles core data processing tasks

Test Process:
1. Verifies component imports
2. Checks component initialization
3. Validates required methods
4. Generates detailed status report

Output:
- Terminal status updates
- HTML report with component status
- Overall system health assessment
"""

import os
import sys
import time
import webbrowser
from core.fracticody_engine import FractiCodyEngine
from core.memory_manager import MemoryManager
from core.fracti_decision_engine import FractiDecisionEngine
from core.fracti_fpu import FractiProcessingUnit

def test_component_imports():
    """Test if all components can be imported and instantiated"""
    print("\n🔍 Testing Component Imports...")
    
    results = {}
    
    # Test MemoryManager
    try:
        m = MemoryManager()
        results["MemoryManager Import"] = "✅ Successful"
    except Exception as e:
        results["MemoryManager Import"] = f"❌ Failed: {str(e)}"

    # Test DecisionEngine
    try:
        d = FractiDecisionEngine()
        results["DecisionEngine Import"] = "✅ Successful"
    except Exception as e:
        results["DecisionEngine Import"] = f"❌ Failed: {str(e)}"

    # Test ProcessingUnit
    try:
        p = FractiProcessingUnit()
        results["ProcessingUnit Import"] = "✅ Successful"
    except Exception as e:
        results["ProcessingUnit Import"] = f"❌ Failed: {str(e)}"

    return results

def test_component_connectivity(engine):
    """Test connectivity to all FractiCognition components"""
    print("\n🔍 Testing FractiCognition Component Connectivity...")
    
    results = {
        "FractiCore": False,
        "MemoryManager": False,
        "DecisionEngine": False,
        "ProcessingUnit": False
    }
    
    try:
        # Test FractiCore
        if hasattr(engine, 'fractal_cognition'):
            if engine.fractal_cognition and hasattr(engine.fractal_cognition, 'activate'):
                results["FractiCore"] = True
                print("✅ FractiCore: Initialized successfully")
            else:
                print("❌ FractiCore: Missing required method 'activate'")
        else:
            print("❌ FractiCore: Component not found")
            
        # Test MemoryManager
        if hasattr(engine, 'memory_manager'):
            if engine.memory_manager and hasattr(engine.memory_manager, 'load_memory'):
                results["MemoryManager"] = True
                print("✅ Memory Manager: Initialized successfully")
            else:
                print("❌ Memory Manager: Missing required method 'load_memory'")
        else:
            print("❌ Memory Manager: Component not found")
            
        # Test DecisionEngine
        if hasattr(engine, 'decision_engine'):
            if engine.decision_engine and hasattr(engine.decision_engine, 'process_decision'):
                results["DecisionEngine"] = True
                print("✅ Decision Engine: Initialized successfully")
            else:
                print("❌ Decision Engine: Missing required method 'process_decision'")
        else:
            print("❌ Decision Engine: Component not found")
            
        # Test ProcessingUnit
        if hasattr(engine, 'processing_unit'):
            if engine.processing_unit and hasattr(engine.processing_unit, 'process'):
                results["ProcessingUnit"] = True
                print("✅ Processing Unit: Initialized successfully")
            else:
                print("❌ Processing Unit: Missing required method 'process'")
        else:
            print("❌ Processing Unit: Component not found")
            
    except Exception as e:
        print(f"❌ Component Test Error: {str(e)}")
        import traceback
        print(traceback.format_exc())

    return results

def generate_test_report(import_results, connectivity_results):
    """Generate HTML report of test results"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FractiCody Components Test Report</title>
        <style>
            body {{ font-family: 'Arial', sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .header {{ color: #2c3e50; text-align: center; }}
            .status-box {{ margin: 20px 0; padding: 15px; border-radius: 5px; }}
            .success {{ background: #e8f5e9; color: #2e7d32; }}
            .failure {{ background: #ffebee; color: #c62828; }}
            .summary {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="header">🤖 FractiCody Components Test Report</h1>
            
            <h2>Import Tests</h2>
            {''.join([f'''
            <div class="status-box {'success' if "✅" in result else 'failure'}">
                <h3>{component}</h3>
                <p>{result}</p>
            </div>
            ''' for component, result in import_results.items()])}

            <h2>Component Connectivity Tests</h2>
            {''.join([f'''
            <div class="status-box {'success' if connectivity_results[component] else 'failure'}">
                <h3>{'✅' if connectivity_results[component] else '❌'} {component}</h3>
                <p>Status: {'Initialized' if connectivity_results[component] else 'Failed'}</p>
            </div>
            ''' for component in connectivity_results])}

            <div class="summary">
                <h2>📊 Summary</h2>
                <p>Components Initialized: {sum(connectivity_results.values())}/{len(connectivity_results)}</p>
                <p>System Status: {'✅ Operational' if all(connectivity_results.values()) else '❌ Initialization Failed'}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    report_path = "test_results/component_test_report.html"
    os.makedirs("test_results", exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write(html_content)
    
    return report_path

def run_tests():
    """Run all component tests"""
    print("🚀 Starting FractiCody Components Test\n")
    
    # Test imports
    import_results = test_component_imports()
    
    # Initialize engine and test connectivity
    engine = FractiCodyEngine()
    engine.start()
    connectivity_results = test_component_connectivity(engine)
    
    # Generate and show report
    report_path = generate_test_report(import_results, connectivity_results)
    
    print(f"\n📊 Test Summary:")
    print(f"Total Components Initialized: {sum(connectivity_results.values())}/{len(connectivity_results)}")
    print(f"System Status: {'✅ Operational' if all(connectivity_results.values()) else '❌ Initialization Failed'}")
    print(f"\n📋 Detailed report generated: {report_path}")
    
    # Open report in browser
    webbrowser.open('file://' + os.path.realpath(report_path))
    
    return all(connectivity_results.values())

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 