"""Blockchain implementation module."""
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class Transaction:
    """Blockchain transaction."""
    sender: str
    receiver: str
    data: Dict[str, Any]
    timestamp: str = None
    id: str = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.id is None:
            self.id = self._compute_hash()
            
    def _compute_hash(self):
        """Compute transaction hash."""
        content = {
            "sender": self.sender,
            "receiver": self.receiver,
            "data": self.data,
            "timestamp": self.timestamp
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()

@dataclass
class Block:
    """Blockchain block."""
    index: int
    transactions: List[Transaction]
    timestamp: str
    previous_hash: str
    nonce: int = 0
    difficulty: str = "medium"
    hash: str = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.hash is None:
            self.hash = self._compute_hash()
            
    def _compute_hash(self):
        """Compute block hash."""
        content = {
            "index": self.index,
            "transactions": [asdict(tx) for tx in self.transactions],
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "difficulty": self.difficulty
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()
        
    def validate(self):
        """Validate block integrity.
        
        Returns:
            bool: Whether block is valid
        """
        return self.hash == self._compute_hash()

class SmartContract:
    """Smart contract implementation."""
    
    def __init__(self, code: str, name: str):
        """Initialize smart contract.
        
        Args:
            code (str): Contract code
            name (str): Contract name
        """
        self.code = code
        self.name = name
        self.id = hashlib.sha256(code.encode()).hexdigest()
        
    def execute(self, context: Dict[str, Any]):
        """Execute contract code.
        
        Args:
            context (dict): Execution context
            
        Returns:
            dict: Execution result
        """
        try:
            # Create safe execution environment
            globals_dict = {"__builtins__": {}}
            locals_dict = {"context": context}
            
            # Execute code
            exec(self.code, globals_dict, locals_dict)
            result = locals_dict.get("execute")(context)
            
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

class Chain:
    """Blockchain implementation."""
    
    def __init__(self, test_mode=False):
        """Initialize blockchain.
        
        Args:
            test_mode (bool): Whether to run in test mode
        """
        self.test_mode = test_mode
        self.blocks: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.contracts: Dict[str, SmartContract] = {}
        self.difficulty = "medium"
        
    def initialize(self):
        """Initialize blockchain with genesis block."""
        genesis_block = Block(
            index=0,
            transactions=[],
            timestamp=datetime.now().isoformat(),
            previous_hash="0" * 64,
            difficulty=self.difficulty
        )
        self.blocks.append(genesis_block)
        
    def add_transaction(self, transaction: Transaction):
        """Add transaction to pending pool.
        
        Args:
            transaction (Transaction): Transaction to add
            
        Returns:
            str: Transaction ID
        """
        self.pending_transactions.append(transaction)
        return transaction.id
        
    def mine_block(self):
        """Mine new block with pending transactions.
        
        Returns:
            Block: Mined block
        """
        if not self.pending_transactions:
            return None
            
        block = Block(
            index=len(self.blocks),
            transactions=self.pending_transactions.copy(),
            timestamp=datetime.now().isoformat(),
            previous_hash=self.get_last_block().hash,
            difficulty=self.difficulty
        )
        
        # Simple proof of work
        while not block.hash.startswith("0" * self._get_difficulty_target()):
            block.nonce += 1
            block.hash = block._compute_hash()
            
        self.blocks.append(block)
        self.pending_transactions = []
        
        return block
        
    def get_last_block(self):
        """Get last block in chain.
        
        Returns:
            Block: Last block
        """
        return self.blocks[-1]
        
    def validate(self):
        """Validate entire blockchain.
        
        Returns:
            bool: Whether chain is valid
        """
        for i in range(1, len(self.blocks)):
            current = self.blocks[i]
            previous = self.blocks[i-1]
            
            if current.previous_hash != previous.hash:
                return False
            if not current.validate():
                return False
                
        return True
        
    def get_transaction(self, tx_id: str):
        """Get transaction by ID.
        
        Args:
            tx_id (str): Transaction ID
            
        Returns:
            Transaction: Found transaction or None
        """
        for block in self.blocks:
            for tx in block.transactions:
                if tx.id == tx_id:
                    return tx
        return None
        
    def get_transaction_state(self, tx_id: str):
        """Get transaction state.
        
        Args:
            tx_id (str): Transaction ID
            
        Returns:
            dict: Transaction state
        """
        for block in self.blocks:
            for tx in block.transactions:
                if tx.id == tx_id:
                    return {
                        "status": "confirmed",
                        "block_index": block.index
                    }
                    
        if any(tx.id == tx_id for tx in self.pending_transactions):
            return {"status": "pending"}
            
        return {"status": "unknown"}
        
    def deploy_contract(self, contract: SmartContract):
        """Deploy smart contract.
        
        Args:
            contract (SmartContract): Contract to deploy
            
        Returns:
            str: Contract ID
        """
        self.contracts[contract.id] = contract
        return contract.id
        
    def execute_contract(self, contract_id: str, transaction: Transaction):
        """Execute smart contract.
        
        Args:
            contract_id (str): Contract ID
            transaction (Transaction): Transaction context
            
        Returns:
            dict: Execution result
        """
        contract = self.contracts.get(contract_id)
        if not contract:
            return {
                "status": "error",
                "message": "Contract not found"
            }
            
        return contract.execute(asdict(transaction))
        
    def set_difficulty(self, difficulty: str):
        """Set mining difficulty.
        
        Args:
            difficulty (str): Difficulty level
        """
        self.difficulty = difficulty
        
    def _get_difficulty_target(self):
        """Get number of leading zeros required for current difficulty."""
        if self.difficulty == "easy":
            return 1
        elif self.difficulty == "medium":
            return 2
        else:  # hard
            return 3 