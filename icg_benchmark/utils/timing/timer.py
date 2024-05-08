import time

import numpy as np


class TimingModule:
    """A module to time the execution of a function.

    This is a context manager that can be used to time the execution of a function.
    It automatically saves the call count and the total execution time. And can be
    used to print the average execution time.
    """

    def __init__(self, name: str, transient_counts: int = 10):
        self.name = name
        self.call_count = 0.00001
        self.total_time = []
        self.start_time = 0
        self.transient_counts = transient_counts

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.call_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.call_count < self.transient_counts:  # dont time transient
            return
        self.total_time += [(time.perf_counter() - self.start_time) * 1000.0]

    def __str__(self):
        return f"{self.name}: {np.mean(self.total_time):.2f} $\pm$ {np.std(self.total_time):.2f} ms"

    def __repr__(self):
        return str(self)


class Timer:
    def __init__(self):
        self.timing_dict = {}

    def print_stats(self):
        for key, value in self.timing_dict.items():
            print(value)

    def get_stats(self):
        return {key: value.total_time for key, value in self.timing_dict.items()}

    def __getitem__(self, item):
        if item not in self.timing_dict:
            self.timing_dict[item] = TimingModule(item)
        return self.timing_dict[item]
