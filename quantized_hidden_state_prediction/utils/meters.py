# from pytorch imagenet example
class AverageMeter(object):
    """Computes and stores the average, max and current value"""

    def __init__(self, keep_track_of_extrema=True):
        self.keep_track_of_extrema = keep_track_of_extrema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.min = float("inf")
        self.max = float("-inf")
        self.m = 0  # for running variance calculation (Knuth's algorithm)
        self.v = 0  # for running variance calculation
        self.var = 0
        self.count = 0
        self.just_reset = True

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.keep_track_of_extrema:
            if val < self.min:
                self.min = val
            if val > self.max:
                self.max = val
            if self.just_reset:
                self.just_reset = False
                self.m = val
            old_m = self.m
            self.m = old_m + (val - old_m) / self.count
            self.v = self.v + (val - old_m) * (val - self.m)
            if self.count - n > 0:
                self.var = self.v / (self.count - n)
        self.avg = self.sum / self.count


class MultiMeter(object):
    """Acts as average meter for multiple values at once"""

    def __init__(self, name_list=None, keep_track_of_extrema_list=None):
        self.meters = {}
        if name_list is not None:
            for i, name in enumerate(name_list):
                self.meters[name] = AverageMeter(
                    keep_track_of_extrema_list[i]
                    if keep_track_of_extrema_list is not None
                    else True
                )

    def add_meter(self, key, keep_track_of_extrema=True):
        if key in self.meters.keys():
            raise Exception(f"The meter '{key}' already exists!")
        self.meters[key] = AverageMeter(keep_track_of_extrema)

    def reset(self, name_list=None):
        if name_list is None:
            for m in self.meters.values():
                m.reset()
        else:
            for name in name_list:
                self.meters[name].reset()

    def update(self, val_dict, n=1):
        for k, v in val_dict.items():
            if k not in self.meters.keys():
                self.add_meter(k)
            self.meters[k].update(v, n=n)

    def __getitem__(self, key):
        return self.meters[key]
