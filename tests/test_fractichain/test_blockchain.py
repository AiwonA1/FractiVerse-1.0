"""Tests for FractiChain blockchain operations."""
import pytest
import hashlib
import json
from datetime import datetime
from fractiverse.fractichain.blockchain import (
    Block,
    Chain,
    Transaction,
    SmartContract
)

@pytest.fixture
def test_chain():
    """Provide a test blockchain instance."""
    chain = Chain(test_mode=True)
    chain.initialize()
    return chain

@pytest.fixture
def test_transaction():
    """Create a test transaction."""
    return Transaction(
        sender="node_a",
        receiver="node_b",
        data={"type": "test", "value": 1.0},
        timestamp=datetime.now().isoformat()
    )

@pytest.fixture
def test_contract():
    """Create a test smart contract."""
    return SmartContract(
        code="""
        def execute(context):
            if context['value'] > 0:
                return {'status': 'success', 'result': context['value'] * 2}
            return {'status': 'error', 'message': 'Invalid value'}
        """,
        name="test_contract"
    )

def test_block_creation(test_chain):
    """Test block creation and validation."""
    # Create new block
    transactions = [
        Transaction(
            sender=f"node_{i}",
            receiver=f"node_{i+1}",
            data={"value": i}
        ) for i in range(3)
    ]
    
    block = test_chain.create_block(transactions)
    
    # Verify block properties
    assert block.index == len(test_chain.blocks)
    assert block.previous_hash == test_chain.get_last_block().hash
    assert len(block.transactions) == 3
    assert block.timestamp is not None
    assert block.validate()

def test_transaction_processing(test_chain, test_transaction):
    """Test transaction processing and inclusion in blocks."""
    # Add transaction
    tx_id = test_chain.add_transaction(test_transaction)
    assert tx_id is not None
    
    # Mine new block
    block = test_chain.mine_block()
    
    # Verify transaction inclusion
    assert any(tx.id == tx_id for tx in block.transactions)
    assert test_chain.get_transaction(tx_id) is not None
    
    # Verify transaction state
    tx_state = test_chain.get_transaction_state(tx_id)
    assert tx_state["status"] == "confirmed"
    assert tx_state["block_index"] == block.index

def test_chain_validation(test_chain):
    """Test blockchain validation mechanisms."""
    # Add some blocks
    for _ in range(3):
        tx = Transaction(
            sender="test",
            receiver="test",
            data={"value": 1.0}
        )
        test_chain.add_transaction(tx)
        test_chain.mine_block()
    
    # Validate entire chain
    assert test_chain.validate()
    
    # Try tampering with a block
    test_chain.blocks[1].transactions[0].data["value"] = 2.0
    assert not test_chain.validate()
    
    # Verify chain length
    assert len(test_chain.blocks) == 4  # Including genesis block

def test_smart_contract_execution(test_chain, test_contract):
    """Test smart contract deployment and execution."""
    # Deploy contract
    contract_id = test_chain.deploy_contract(test_contract)
    assert contract_id is not None
    
    # Create transaction to execute contract
    tx = Transaction(
        sender="test",
        receiver=contract_id,
        data={"value": 10.0}
    )
    
    # Execute contract
    result = test_chain.execute_contract(contract_id, tx)
    assert result["status"] == "success"
    assert result["result"] == 20.0
    
    # Test invalid input
    tx_invalid = Transaction(
        sender="test",
        receiver=contract_id,
        data={"value": -1.0}
    )
    result = test_chain.execute_contract(contract_id, tx_invalid)
    assert result["status"] == "error"

@pytest.mark.parametrize("block_size,difficulty", [
    (1, "easy"),
    (10, "medium"),
    (100, "hard")
])
def test_mining_performance(test_chain, block_size, difficulty):
    """Test mining performance with different parameters."""
    # Set difficulty
    test_chain.set_difficulty(difficulty)
    
    # Create transactions
    transactions = [
        Transaction(
            sender=f"node_{i}",
            receiver=f"node_{i+1}",
            data={"value": i}
        ) for i in range(block_size)
    ]
    
    # Add transactions
    for tx in transactions:
        test_chain.add_transaction(tx)
    
    # Time block mining
    start_time = datetime.now()
    block = test_chain.mine_block()
    mining_time = (datetime.now() - start_time).total_seconds()
    
    # Verify mining results
    assert len(block.transactions) == block_size
    assert block.validate()
    assert block.difficulty == difficulty
    
    # Check mining time constraints
    if difficulty == "easy":
        assert mining_time < 1.0
    elif difficulty == "medium":
        assert mining_time < 5.0
    else:  # hard
        assert mining_time < 30.0 