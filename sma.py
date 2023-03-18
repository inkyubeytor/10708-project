import numpy as np
import pandas as pd
import time

class SMA:

    def __init__(self, y, p):
        self.y = y
        self.p = p
    def fit(self):
        print("Fitting", end="")
        for _ in range(3):
            time.sleep(1)
            print(".", end="")
        print("\nModel fit.")
        return self

    def forecast(self, k):
        tmp_data = self.y[-self.p:].tolist()
        out = []
        for i in range(k):
            out.append(sum(tmp_data[-self.p:]) / self.p)
            tmp_data.append(out[-1])
        start_index = self.y.index.max() + 1
        return pd.Series(data=out, name="predicted_mean", index=range(start_index, start_index + k)).to_frame()
