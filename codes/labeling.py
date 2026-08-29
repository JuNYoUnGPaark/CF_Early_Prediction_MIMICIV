# -*- coding: utf-8 -*-
# name: labeling.py
# author: JunYoung Park
# date: 2026-08-27

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------------------------------------------------
# Future Circulatroy Failure Labeling 
# -------------------------------------------------------------------------------------------------------------------
def label_future_cf(state: pd.DataFrame, horizon_hours: float = 8.0):
    """
        positive (1): 현재 state == No_CF && 향후 8h 이내 CF가 하나라도 존재
        negative (0): 현재 state == No_CF && 향후 8h 이내 CF가 없음 
        excluded (NaN): 현재 state == CF 또는 AMMBIGUOUS
        
        HiRID source code의 future window는 오른쪽 경계가 exclusive임. [t, t + 8h)
    """
    labels = state[["stay_id", "gridtime", "state"]].copy()
    labels["gridtime"] = pd.to_datetime(labels["gridtime"], errors="coerce")
    labels = labels.dropna(subset=["stay_id", "gridtime"]).sort_values(["stay_id", "gridtime"]).reset_index(drop=True)
    labels["label"] = np.nan
    
    horizon = pd.Timedelta(hours=horizon_hours)
    
    # Stay별로 독립적으로 미래 CF 여부 확인 
    for sid, g in labels.groupby("stay_id", sort=False):
        idx = g.index.to_numpy()
        times = g["gridtime"].to_numpy(dtype="datetime64[ns]")
        states = g["state"].to_numpy(dtype="object")
        
        # 현재 No_CF가 아닌 시점은 Prediction에서 제외  
        for i in range(len(g)):
            if states[i] != "NO_CF":
                continue
            # [t, t + 8h) 범위 
            end_time = pd.Timestamp(times[i]) + horizon
            end = np.searchsorted(times, np.datetime64(end_time), side="left")
            future_states = states[i: end]
            # 미래 window 안에 CF가 하나라도 있으면 positive
            labels.loc[idx[i], "label"] = 1.0 if (future_states == "CF").any() else 0.0
            
    labels["label"] = labels["label"].astype("float32")
    return labels