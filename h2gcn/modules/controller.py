from collections import deque


class SlidingMeanEarlyStopping:
    def __init__(self, length):
        self.epoch_history = deque(maxlen=length)
        self.__mean_value = 0

    @property
    def length(self):
        return self.epoch_history.maxlen

    def reset(self):
        self.epoch_history.clear()
        self.__mean_value = 0

    def __call__(self, value):
        if self.length > 0:
            if (len(self.epoch_history) == self.length
                    and value > self.__mean_value):
                return True
            else:
                if len(self.epoch_history) == self.length:
                    self.__mean_value -= (self.epoch_history.popleft() /
                                          self.length)
                self.epoch_history.append(value)
                self.__mean_value += (value / self.length)
                return False
        else:
            return False
