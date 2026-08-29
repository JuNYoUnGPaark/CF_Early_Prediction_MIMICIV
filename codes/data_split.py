# -*- coding: utf-8 -*-
# name: data_split.py
# author: JunYoung Park
# date: 2026-08-28


"""
    같은 subject_id의 모든 hadm_id / stay_id는 
    한 replicate 안에서는 반드시 같은 set에 들어가도록 
"""


import numpy as np
import pandas as pd 


# -------------------------------------------------------------------------------------------------------------------
# 1. Subject-level Train / Validation / Test split
# -------------------------------------------------------------------------------------------------------------------
def build_subject_splits(stays: pd.DataFrame, n_splits: int = 5, 
                         train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 42):
    """
        `splits.csv` -> subject_id | hadm_id | stay_id | split_id | set 
    """
    required = {"subject_id", "hadm_id", "stay_id"}
    missing = required - set(stays.columns)
    
    if missing:
        raise ValueError(f"Missing columns; {missing}")
    
    if stays["subject_id"].isna().any():
        raise ValueError("subejct_id에 missing value 존재")
    
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train ratio + val ratio < 1")
    
    subjects = stays["subject_id"].drop_duplicates().to_numpy()
    parts = []

    # Split replicate 생성 
    for split_id in range(1, n_splits + 1):
        # replicate마다 다른 random seed 사용 
        rng = np.random.default_rng(seed + split_id - 1)
        
        shuffled = subjects.copy()
        rng.shuffle(shuffled)
        
        n_subjects = len(shuffled)
        
        train_end = int(n_subjects * train_ratio)
        val_end = train_end + int(n_subjects * val_ratio)
        
        train_subjects = set(shuffled[:train_end])
        val_subjects = set(shuffled[train_end: val_end])
        test_subjects = set(shuffled[val_end:])
        
        # 동일 subject가 여러 set에 들어가지 않았는지 확인
        assert train_subjects.isdisjoint(val_subjects)
        assert train_subjects.isdisjoint(test_subjects)
        assert val_subjects.isdisjoint(test_subjects)
        
        split = stays[["subject_id", "hadm_id", "stay_id"]].copy()
        split["split_id"] = split_id
        split["set"] = np.select([split["subject_id"].isin(train_subjects),
                                  split["subject_id"].isin(val_subjects),
                                  split["subject_id"].isin(test_subjects)],
                                 ["train", "validation", "test"], default="unknown")
        
        if (split["set"] == "unknown").any():
            raise RuntimeError(f"Split {split_id}: 배정 안된 subject 존재")
        
        parts.append(split)
    
    splits = pd.concat(parts, ignore_index=True)
    return splits


# 특정 split의 stay_id 가져오기 
def get_split_stay_ids(splits: pd.DataFrame, split_id: int):
    split = splits.loc[splits["split_id"] == split_id]
    train_ids = split.loc[split["set"] == "train", "stay_id"].tolist()
    val_ids = split.loc[split["set"] == "validation", "stay_id"].tolist()
    test_ids = split.loc[split["set"] == "test", "stay_id"].tolist()
    
    return train_ids, val_ids, test_ids


# Split 결과 확인 필요할 때 
def summarize_splits(splits: pd.DataFrame):
    summary = (splits.groupby(["split_id", "set"]).agg(subjects=("subject_id", "nunique"),
                                                       admissions=("hadm_id", "nunique"),
                                                       stays=("stay_id", "nunique")).reset_index())
    return summary