"""
FractiVerse System Integration Tests
"""

import pytest
import torch
import asyncio
from core.fractiverse import FractiVerse
from core.integration.system_monitor import SystemMonitor

@pytest.fixture
async def monitored_system():
    """Create monitored FractiVerse system"""
    system = FractiVerse()
    monitor = SystemMonitor(system)
    await system.start()
    await monitor.start_monitoring()
    return system, monitor

@pytest.mark.asyncio
async def test_component_initialization(monitored_system):
    """Test proper initialization of all components"""
    system, monitor = monitored_system
    
    # Check component status
    assert all(monitor.component_status.values()), "Not all components initialized"
    
    # Verify specific components
    assert system.peff.is_initialized(), "PEFF not initialized"
    assert system.chain.is_running(), "Blockchain not running"
    assert system.network.is_connected(), "Network not connected"
    assert system.treasury.is_active(), "Treasury not active"
    assert system.token.total_supply() > 0, "Token not deployed"

@pytest.mark.asyncio
async def test_cognitive_learning(monitored_system):
    """Test cognitive learning and memory"""
    system, monitor = monitored_system
    
    # Create test patterns
    patterns = [torch.randn(256, 256).to(system.device) for _ in range(5)]
    patterns = [p / torch.norm(p) for p in patterns]
    
    # Process patterns
    for pattern in patterns:
        result = await system.process_pattern(pattern)
        assert result is not None
        
        # Wait for cognitive processing
        await asyncio.sleep(1)
        
    # Check cognitive events
    assert len(monitor.cognitive_events) > 0
    
    # Verify learning
    latest_event = monitor.cognitive_events[-1]
    assert latest_event.metrics['peff_coherence'] > 0.5
    assert latest_event.metrics['learning_rate'] > 0

@pytest.mark.asyncio
async def test_blockchain_integration(monitored_system):
    """Test blockchain integration"""
    system, monitor = monitored_system
    
    # Create and process pattern
    pattern = torch.randn(256, 256).to(system.device)
    pattern = pattern / torch.norm(pattern)
    
    result = await system.process_pattern(pattern)
    pattern_id = result['pattern_id']
    
    # Verify blockchain record
    block = await system.chain.get_pattern_block(pattern_id)
    assert block is not None
    assert block.pattern_hash is not None
    assert block.emergence_score > 0

@pytest.mark.asyncio
async def test_network_communication(monitored_system):
    """Test network communication"""
    system, monitor = monitored_system
    
    # Verify network status
    assert system.network.peer_count() > 0
    assert system.network.is_synced()
    
    # Test pattern broadcast
    pattern = torch.randn(256, 256).to(system.device)
    pattern = pattern / torch.norm(pattern)
    
    result = await system.process_pattern(pattern)
    pattern_id = result['pattern_id']
    
    # Verify pattern propagation
    propagation = await system.network.verify_pattern_propagation(pattern_id)
    assert propagation['success']
    assert propagation['peer_count'] > 0

@pytest.mark.asyncio
async def test_reward_distribution(monitored_system):
    """Test reward distribution"""
    system, monitor = monitored_system
    
    # Get initial balances
    initial_balances = system.treasury.get_balances()
    
    # Process patterns to trigger rewards
    patterns = [torch.randn(256, 256).to(system.device) for _ in range(3)]
    patterns = [p / torch.norm(p) for p in patterns]
    
    for pattern in patterns:
        await system.process_pattern(pattern)
        await asyncio.sleep(1)  # Wait for reward distribution
        
    # Verify reward distribution
    final_balances = system.treasury.get_balances()
    assert any(final_balances[k] > initial_balances[k] for k in final_balances)
    
    # Check token transfers
    transfers = system.token.get_recent_transfers()
    assert len(transfers) > 0
    assert all(t['amount'] > 0 for t in transfers)

@pytest.mark.asyncio
async def test_admin_ui(monitored_system):
    """Test admin UI functionality"""
    system, monitor = monitored_system
    
    # Verify UI launch
    assert monitor.admin_port > 0
    
    # Process pattern to generate UI update
    pattern = torch.randn(256, 256).to(system.device)
    pattern = pattern / torch.norm(pattern)
    
    result = await system.process_pattern(pattern)
    await asyncio.sleep(1)  # Wait for UI update
    
    # Verify UI state
    from core.integration.admin_ui import get_ui_state
    ui_state = get_ui_state()
    
    assert ui_state['connected']
    assert ui_state['last_pattern_id'] == result['pattern_id']
    assert ui_state['system_coherence'] > 0 