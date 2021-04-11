class Algorithm(object):
    """High Level interface for Algorithms"""

    def train(self, data):
        """Trains on the given data."""
        raise NotImplementedError

    def set_device(self, device):
        """Sets algorithm to use specified device."""
        pass
