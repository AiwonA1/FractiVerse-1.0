import torch
from fractal_lib import (
    UnipixelCore,
    PEFFSystem,
    AIVFIARForge,
    FractiNet,
    UnipixelInterface,
    FractalLayer,
    RealityLayer
)

def run_simulation(
    dimension: int = 512,
    num_nodes: int = 4,
    num_steps: int = 10
):
    """
    Run a full simulation of FractiVerse system.
    """
    print("🚀 Initializing FractiVerse 1.0 Components...")
    
    # Initialize core components
    unipixel = UnipixelCore(dimension=dimension)
    peff = PEFFSystem(dimension=dimension)
    forge = AIVFIARForge(reality_dim=dimension)
    network = FractiNet(network_dim=dimension, num_nodes=num_nodes)
    interface = UnipixelInterface(interface_dim=dimension)
    
    # Generate initial test pattern
    test_pattern = torch.randn(1, dimension)
    
    print("\n🔄 Starting Simulation Loop...")
    for step in range(num_steps):
        print(f"\nStep {step + 1}/{num_steps}")
        
        # 1. Process through Unipixel system
        print("📊 Processing through Unipixel...")
        unipixel_output, states = unipixel.forward(
            test_pattern,
            reality_channel="FractiVerse",
            return_states=True
        )
        print(f"  - Activation Level: {states['unipixel_state'].activation_level:.3f}")
        print(f"  - Entropy: {states['unipixel_state'].entropy:.3f}")
        
        # 2. Apply PEFF harmonization
        print("🌟 Applying PEFF harmonization...")
        harmonized = peff.forward(
            unipixel_output,
            target_layer=FractalLayer.QUANTUM
        )
        harmony = peff._measure_harmony(harmonized)
        print(f"  - Harmony Score: {harmony.item():.3f}")
        
        # 3. Create alternate reality
        print("🌌 Forging alternate reality...")
        reality = forge.forge_reality(
            seed_pattern=harmonized,
            target_layer=RealityLayer.FRACTAL
        )
        print(f"  - Reality Stability: {reality.stability_score:.3f}")
        
        # 4. Process through network
        print("🔄 Distributing through network...")
        for node_id in list(network.nodes.keys())[:2]:  # Process first 2 nodes
            node_output = network.process_node(
                node_id=node_id,
                input_state=harmonized,
                reality_channel="FractiVerse"
            )
            print(f"  - Node {node_id[:8]} bandwidth: {network.nodes[node_id].bandwidth:.3f}")
        
        # 5. Visualize results
        print("🎨 Generating visualization...")
        vis_data = interface.visualize_cluster(
            intelligence_cluster=harmonized,
            reality_mode=ViewMode.FRACTAL
        )
        
        # Update test pattern for next iteration
        test_pattern = harmonized
        
    print("\n✨ Simulation Complete!")
    return {
        'final_pattern': test_pattern,
        'reality': reality,
        'harmony': harmony.item(),
        'visualization': vis_data
    }

if __name__ == "__main__":
    print("🌟 FractiVerse 1.0 Simulation 🌟")
    print("=================================")
    
    try:
        results = run_simulation(
            dimension=256,  # Smaller dimension for testing
            num_nodes=4,
            num_steps=5
        )
        
        print("\n📊 Final Results:")
        print(f"Pattern Harmony: {results['harmony']:.3f}")
        print(f"Reality Stability: {results['reality'].stability_score:.3f}")
        
    except Exception as e:
        print(f"\n❌ Simulation Error: {str(e)}") 