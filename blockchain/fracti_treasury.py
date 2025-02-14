"""
🏦 FractiTreasury - Governance & Rewards for AI Intelligence Economy
Handles treasury functions, rewards for AI contributions, and FractiMiner operations.
"""
class FractiTreasury:
    def __init__(self):
        self.treasury_balance = 10000  # Initial treasury balance
        self.rewards_pool = {}

    def distribute_rewards(self, user, amount):
        """Allocates AI rewards for valuable contributions."""
        if self.treasury_balance >= amount:
            self.treasury_balance -= amount
            self.rewards_pool[user] = self.rewards_pool.get(user, 0) + amount
            return f"🎉 {amount} FTN Rewarded to {user}"
        return "❌ Insufficient Treasury Balance"

    def check_treasury_balance(self):
        """Returns the current balance of the treasury."""
        return f"🏦 Treasury Balance: {self.treasury_balance} FTN"

    def check_rewards(self, user):
        """Returns the rewards earned by an AI node."""
        return f"🎖 {user} Rewards: {self.rewards_pool.get(user, 0)} FTN"

if __name__ == "__main__":
    treasury = FractiTreasury()
    print(treasury.distribute_rewards("AI-Node-3", 500))
    print(treasury.check_treasury_balance())
    print(treasury.check_rewards("AI-Node-3"))
