#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class TrainTestSplit:
    #REWORK
    def __init__(self, size, split_fraction=0.1):
        self.all_indices = np.arange(size)
        train_idx, val_idx = train_test_split(
            self.all_indices,
            test_size=split_fraction,
        )
        self.splits = {
            "train": train_idx,
            "val": val_idx,
        }
        
class KFoldSplits:
    def __init__(self, size, folds=5):
        kf = KFold(n_splits=folds)
        all_indices = np.arange(size)
        self.splits = kf.split(all_indices)
