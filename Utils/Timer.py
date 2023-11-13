from sys import platform
from resource import getrusage, RUSAGE_SELF
from time import time
from Utils.log import Log

L = Log(__name__).get_logger()


class Timer:
    def __init__(self):
        self.start_memory = 0
        self.total_memory = 0
        self.start_time = 0
        self.total_time = 0

    def start(self):
        # start the algorithm
        self.start_time = self.current_seconds()
        # self.start_memory = memory_usage_resource()

    def stop(self):
        # end the algorithm
        self.total_time += self.current_seconds() - self.start_time
        self.total_memory = memory_usage_resource()

    def print_timer(self):
        L.info('Duration Time: ' + str(self.total_time) + 's')
        L.info('Memory Usage: ' + str(self.total_memory) + 'MB')

    @staticmethod
    def current_seconds():
        return int(round(time() * 1000)) / 1000.0


class AccumulateTimer:
    def __init__(self):
        self.start_time = 0
        self.total_time = 0

    def start(self):
        # start the algorithm
        self.start_time = self.current_seconds()

    def stop(self):
        # end the algorithm
        self.total_time += self.current_seconds() - self.start_time

    def to_str(self):
        return str('Accumulate Time: ' + str(self.total_time) + 's')

    @staticmethod
    def current_seconds():
        return int(round(time() * 1000)) / 1000.0


def memory_usage_resource():
    # denominator for MB
    rusage_denominator = 1024
    # if the OS is MAC OSX
    if platform == 'darwin':
        # adjust the denominator
        rusage_denominator *= rusage_denominator

    # return the memory usage
    return getrusage(RUSAGE_SELF).ru_maxrss / rusage_denominator
