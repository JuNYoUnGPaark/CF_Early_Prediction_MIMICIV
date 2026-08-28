# -*- coding: utf-8 -*-
# name: state_annotation.py
# author: JunYoung Park
# date: 2026-08-27


import pandas as pd
import numpy as np 


"""
    Lactate -> annotation 전용 interpolation
    MAP -> 60min grid의 실제 관측 MAP
    Drug -> 해당 시각 vasoactive/inotrope 투여 여부
    
    CF / No-CF / Ambiguous
"""


# -----------------------------------------------------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------------------------------------------------
# Vasoactive, Inotrope variables
VASOACTIVE_INOTROPE_VARIABLES = {
    "Norepinephrine",
    "Phenylephrin",
    "Vasopressin",
    "Epinephrine",
    "Dopamin",
    "Dobutamine",
    "Milrinone"
}


# -----------------------------------------------------------------------------------------------------------------------
# Annotation용 Lactate 생성 
# -----------------------------------------------------------------------------------------------------------------------
def prepare_annotation_lactate(time_grid: pd.DataFrame, nonpharma_merged: pd.DataFrame, 
                               threshold: float = 2.0, crossing_interp_hours: float = 6.0, 
                               edge_fill_hours: float = 3.0):
    """
        - Adaptive Imputation된 Lactate는 Training set용으로 fillna된거라 사용 X
        - Annotation 전용으로 Interpolation -> 미래 Lactate 값이 포함되도 상관 X

        실제 Lactate 측정값을 추출해서 
        1. Stay별로 정리해놓고
        2. 각 60분 grid point마다 가장 가까운 왼쪽 / 오른쪽 Lactate 측정값을 찾는다. 
        
        time_grid -> 최종 60min grid (stay_id | gridtime)
        nonpharma_merged -> 실제 non-pharma raw measurement (stay_id | charttime | variable | valuenum)
        threshold -> Lactate 정상 / 비정상 임계값 (threshold = 2)
        crossing_interp_hours -> Threshold 넘을시 전체 linear interpolation을 허용하는 최대 gap (6h)
        edge_fill_hours -> 비정상 Lactate를 앞 / 뒤로 유지할 최대 시간 
        
        Case 1: Grid point == Lactate charttime -> 그대로 적용 
        Case 2: Grid point != Lactate charttime
                [1] 현재 Grid point가 첫 Lactate보다 이전 
                    [1-1] Lactate <= 2 -> 계속 backward fill
                    [1-2] Lactate > 2 -> 3시간까지만 backward fill
                [2] 현재 Grid point가 첫 Lactate보다 이후 
                    [2-1] Lactate <= 2 -> 계속 forward fill
                    [2-2] Lactate > 2 -> 3시간까지만 forward fill
                [3] 현재 Grid point가 두 Lactate 측정 사이
                    [3-1] 두 Lactate 모두 <= 2 OR > 2 -> 전체 Linear Interpolation
                    [3-2] 두 Lactate가 2를 crossing 
                        [3-2-1] 두 Lactate 사이 Gap < 6h -> 전체 Linear Interpolation
                        [3-2-2] 두 Lactate 사이 Gap >= 6h -> 왼쪽 3h은 왼쪽값, 오른쪽 3h은 오른쪽값, 가운데는 missing
    """
    
    # 1. 실제 Lactate 측정값 가져오기 
    lactate = nonpharma_merged.loc[nonpharma_merged["variable"] == "Lactate", ["stay_id", "charttime", "valuenum"]].copy()
    lactate["charttime"] = pd.to_datetime(lactate["charttime"], errors="coerce")
    lactate["valuenum"] = pd.to_numeric(lactate["valuenum"], errors="coerce")
    lactate = lactate.dropna(subset=["stay_id", "charttime", "valuenum"])
    
    # 같은 시각에 Lactate 값이 여러개면 median 처리 
    lactate = (lactate.groupby(["stay_id", "charttime"], as_index=False)["valuenum"]
               .median().sort_values(["stay_id", "charttime"]))
    
    grid = time_grid[["stay_id", "gridtime"]].copy()
    grid["gridtime"] = pd.to_datetime(grid["gridtime"], errors="coerce")
    grid = grid.dropna(subset=["stay_id", "gridtime"])
    
    # Lactate를 stay별로 미리 나눠놓기 
    # {101: stay101의 Lactate df, 102: stay102의 Lactate df, ...}
    # 뒤에서 stay 하나를 처리할 때 lactate_by_stay[sid]로 바로 해당 환자의 Lactate를 꺼낼 수 있도록 만드는 것 
    lactate_by_stay = {sid: g for sid, g in lactate.groupby("stay_id", sort=False)}
    
    # 6시간짜리 객체 생성
    crossing_limit = pd.Timedelta(hours=crossing_interp_hours)
    # 3시간짜리 객체 생성
    edge_limit = pd.Timedelta(hours=edge_fill_hours)
    
    # stay별 결과 DataFrame을 차곡차곡 넣을 빈 List. 마지막에 전부 concat() 
    parts = []
    
    # 2. Stay별 annotation Lactate 계산 
    # sid: 현재 stay_id
    # stay_grid: 현재 stay의 모든 grid point 
    for sid, stay_grid in grid.groupby("stay_id", sort=False):
        stay_grid = stay_grid.copy()
        grid_times = stay_grid["gridtime"].to_numpy(dtype="datetime64[ns]")  # 현재 stay의 gridtime을 numpy array로
        values = np.full(len(stay_grid), np.nan, dtype=np.float32)  # 결과 Lactate를 담을 배열 
        
        # Lactate 측정이 한 번도 없는 stay -> 모두 missing 
        if sid not in lactate_by_stay:
            stay_grid["annotation_lactate"] = values  # 없으면 -> NaN일테니
            parts.append(stay_grid)  # 그 결과를 parts에 추가하고 다음 stay로... 
            continue

        measurements = lactate_by_stay[sid]  # 현재 stay의 Lactate measurement 가져오기 (charttime | valuenum)
        measure_times = measurements["charttime"].to_numpy(dtype="datetime64[ns]")  # charttime 배열
        measure_values = measurements["valuenum"].to_numpy(dtype=float)  # valuenum 배열 

        # 각 gridtime의 바로 오른쪽에 있는 첫 Lactate measurement가 몇 번째인지 찾는다.
        # 예를 들면, 
        # 10:20 -> 1.5  # index 0
        # 14:30 -> 3.0  # index 1
        # 18:10 -> 1.8  # index 2
        # 이고 우리가 궁금한 gridtime은 12:03이라고 하자.
        # 그러면 12:03은 10:20과 14:30 사이에 들어가야한다. 
        # 오른쪽을 보는거니까 결과는 1. 
        # 즉, index 1 위치에 12:03을 끼워 넣으면 정렬순서가 유지된다. 
        # 그래서 아래 반복문의 right = 1이 되는거고, left = right - 1을 하면 바로 찾을 수 있다. 
        right_indices = np.searchsorted(measure_times, grid_times, side="left")
        
        # grid point 하나씩 처리 
        for i, grid_time in enumerate(grid_times):
            right = int(right_indices[i])
            current = pd.Timestamp(grid_time)

            # 정확히 gridtime에 Lactate가 측정된 경우
            # 1. 배열 범위를 벗어나지 않으면서 
            # 2. 실제 같은 시각인지 확인되면 
            # -> 그대로 interpolation하지 않고 실제 측정값 그대로 사용 
            if right < len(measure_times) and measure_times[right] == grid_time:
                values[i] = measure_values[right]
                continue

            # 첫 Lactate 측정 이전에 있는 경우 
            # right가 0이라는건 현재 gridtime보다 오른쪽에 있는 측정값이 전체에서 첫번째 측정값이라는 뜻 
            if right == 0:
                first_time = pd.Timestamp(measure_times[0])
                first_value = float(measure_values[0])
                # 첫 값이 정상이라면 
                if first_value <= threshold: 
                    values[i] = first_value  # 첫 측정값 이전까지 backward fill
                # 첫 값이 비정상이라면 
                elif first_time - current <= edge_limit:
                    values[i] = first_value  # 최대 3시간까지만 backward fill 
                continue

            # 마지막 Lactate 측정 이후에 있는 경우 
            # 현재 grid가 마지막 Lactate보다 뒤에 있다는 뜻 
            if right >= len(measure_times):
                last_time = pd.Timestamp(measure_times[-1])
                last_value = float(measure_values[-1])
                # 마지막 값이 정상이라면 
                if last_value <= threshold:  # 계속 forward fill 
                    values[i] = last_value
                # 마지막 값이 비정상이라면 
                elif current - last_time <= edge_limit:
                    values[i] = last_value  # 최대 3시간까지만 forward fill 

                continue
            
            # 두 Lactate measurement 사이에 있는 경우 
            left = right - 1
            # 양쪽 측정 시각 
            left_time = pd.Timestamp(measure_times[left])
            right_time = pd.Timestamp(measure_times[right])
            # 양쪽 Lactate 값 
            left_value = float(measure_values[left])
            right_value = float(measure_values[right])

            # 두 측정 간격 계산 
            gap = right_time - left_time
            # 양쪽 값이 모두 정상인지 아닌지 (threshold 2를 넘는지 여부)
            crossed = (left_value > threshold) != (right_value > threshold)
            
            # Linear Interpolation 조건 
            # 1. threshold를 안 넘음 -> gap 길이에 상관없이 전체 interpolation
            # 2. threshold를 넘었고 gap < 6h -> 전체 interpolation 
            # 같은 threshold 영역이거나 threshold crossing이 6시간 미만
            # -> 전체 구간 linear interpolation
            if not crossed or gap < crossing_limit:
                # Interpolation 비율 계산하기 
                fraction = (current - left_time).total_seconds() / gap.total_seconds()
                values[i] = left_value + fraction * (right_value - left_value)
                continue

            # Threshold crossing + 6시간 이상
            # 1. 왼쪽에서 3시간 이내 
            if current - left_time <= edge_limit:
                values[i] = left_value
            # 2. 오른쪽에서 3시간 이내 
            elif right_time - current <= edge_limit:
                values[i] = right_value

        # 현재 stay 결과 저장 
        stay_grid["annotation_lactate"] = values
        parts.append(stay_grid)

    return pd.concat(parts, ignore_index=True)  # 모든 stay 결과 DataFrame으로 합쳐서 return 


# -----------------------------------------------------------------------------------------------------------------------
# Annotation용 MAP 생성
# -----------------------------------------------------------------------------------------------------------------------
def prepare_annotation_map(time_grid: pd.DataFrame, raw_grid_values: pd.DataFrame):
    map_values = raw_grid_values.loc[raw_grid_values["variable"] == "ABP mean",["stay_id", "gridtime", "valuenum"]].copy()
    map_values = map_values.rename(columns={"valuenum": "annotation_map"})

    annotation_map = time_grid[["stay_id", "gridtime"]].merge(map_values, on=["stay_id", "gridtime"], how="left")
    annotation_map["annotation_map"] = pd.to_numeric(annotation_map["annotation_map"], errors="coerce").astype("float32")

    return annotation_map


# -----------------------------------------------------------------------------------------------------------------------
# Vasoactive / Inotropic drug presence 생성
# -----------------------------------------------------------------------------------------------------------------------
def prepare_drug_presence(time_grid: pd.DataFrame, pharma_merged: pd.DataFrame):
    pharma = pharma_merged.loc[
        pharma_merged["pharma_variable"].isin(VASOACTIVE_INOTROPE_VARIABLES),
        ["stay_id", "pharma_variable", "starttime", "endtime"]
    ].copy()

    pharma["starttime"] = pd.to_datetime(pharma["starttime"], errors="coerce")
    pharma["endtime"] = pd.to_datetime(pharma["endtime"], errors="coerce")
    pharma = pharma.dropna(subset=["stay_id", "starttime", "endtime"])

    pharma_by_stay = {sid: g for sid, g in pharma.groupby("stay_id", sort=False)}
    parts = []

    # Stay별로 각 gridtime에 약물이 active한지 확인
    for sid, stay_grid in time_grid.groupby("stay_id", sort=False):
        stay_grid = stay_grid[["stay_id", "gridtime"]].copy()
        grid_times = pd.to_datetime(stay_grid["gridtime"]).to_numpy(dtype="datetime64[ns]")
        presence = np.zeros(len(stay_grid), dtype=np.int8)

        if sid in pharma_by_stay:
            p = pharma_by_stay[sid]
            starts = p["starttime"].to_numpy(dtype="datetime64[ns]")
            ends = p["endtime"].to_numpy(dtype="datetime64[ns]")

            for i, t in enumerate(grid_times):
                presence[i] = int(((starts <= t) & (t < ends)).any())

        stay_grid["vasoactive_inotrope"] = presence
        parts.append(stay_grid)

    return pd.concat(parts, ignore_index=True)


# -----------------------------------------------------------------------------------------------------------------------
# Circulatory State Annotation
# -----------------------------------------------------------------------------------------------------------------------
def annotate_circulatory_state(time_grid: pd.DataFrame, raw_grid_values: pd.DataFrame, nonpharma_merged: pd.DataFrame,
                               pharma_merged: pd.DataFrame, lactate_threshold: float = 2.0, map_threshold: float = 65.0,
                               grid_minutes: int = 60, window_minutes: int = 60):
    """
        MIMIC-III circEWS source code 기준

        Grid   : 60 min
        Window : 60 min

        따라서 window_points = 60 / 60 = 1
        → 각 현재 hourly grid point 하나를 판정

        1. MAP missing
           -> AMBIGUOUS (source: unknown)

        2. MAP > 65 AND vasoactive/inotrope 없음
           -> NO_CF (source: event 0)
           -> Lactate와 관계없이 stable

        3. MAP <= 65 OR vasoactive/inotrope 있음
           -> Lactate 확인
              - Lactate >= 2 -> CF
              - Lactate < 2  -> AMBIGUOUS (source: probably not)
              - Lactate NaN  -> AMBIGUOUS (source: maybe)
    """

    # MIMIC source의 60-min grid / 60-min window 확인
    if grid_minutes != 60 or window_minutes != 60:
        raise ValueError("MIMIC circEWS annotation uses 60-min grid and 60-min window.")

    window_points = window_minutes // grid_minutes
    assert window_points == 1

    # 1. Annotation에 필요한 값 생성
    lactate = prepare_annotation_lactate(time_grid=time_grid, nonpharma_merged=nonpharma_merged, threshold=lactate_threshold)
    map_values = prepare_annotation_map(time_grid=time_grid, raw_grid_values=raw_grid_values)
    drug = prepare_drug_presence(time_grid=time_grid, pharma_merged=pharma_merged) 

    # 2. 하나의 table로 합치기
    state = lactate.merge(map_values, on=["stay_id", "gridtime"], how="left")
    state = state.merge(drug, on=["stay_id", "gridtime"], how="left")
    state["vasoactive_inotrope"] = (state["vasoactive_inotrope"].fillna(0).astype("int8"))

    # 3. 기본 상태 = source의 unknown / maybe / probably not
    state["state"] = "AMBIGUOUS"
    has_map = state["annotation_map"].notna()

    # 4. MAP/drug criterion
    hemodynamic_failure = ((state["annotation_map"] <= map_threshold) | (state["vasoactive_inotrope"] == 1))

    # 5. 저자 코드의 event 0
    # MAP가 존재하면서 MAP>65이고 drug가 없으면 Lactate와 관계없이 stable
    is_no_cf = has_map & ~hemodynamic_failure
    state.loc[is_no_cf, "state"] = "NO_CF"

    # 6. MAP/drug criterion을 만족한 경우에만 Lactate 확인
    is_cf = (has_map & hemodynamic_failure & state["annotation_lactate"].notna() 
             & (state["annotation_lactate"] >= lactate_threshold))
    state.loc[is_cf, "state"] = "CF"

    return state[["stay_id", "gridtime", "annotation_lactate", "annotation_map", "vasoactive_inotrope", "state"]].copy()