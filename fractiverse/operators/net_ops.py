class FractiNet:
    def __init__(self):
        self.network = {}
        self.spatial_grid = {}

    def __or__(self, unipixel):
        self.network[unipixel.id] = unipixel
        self.spatial_grid[unipixel.id] = unipixel.position
        return self

    def __repr__(self):
        return f"FractiNet(Pixels: {len(self.network)})"
