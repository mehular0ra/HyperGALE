
class TotalMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float):
        self.sum += val
        self.count += 1

    def update_with_weight(self, val: float, count: int):
        self.sum += val*count
        self.count += count

    def reset(self):
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        if self.count == 0:
            return -1
        return self.sum / self.count
