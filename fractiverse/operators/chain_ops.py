class FractiChain:
    def __init__(self):
        self.chain = []

    def __rrshift__(self, data):
        self.chain.append(data)
        return self

    def __repr__(self):
        return f"FractiChain({self.chain})"
