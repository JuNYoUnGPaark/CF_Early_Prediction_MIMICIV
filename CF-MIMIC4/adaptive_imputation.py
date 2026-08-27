# -*- coding: utf-8 -*-
# name: adaptive_imputation.py
# author: JunYoung Park
# date: 2026-08-27


import numpy as np 
import pandas as pd


# -----------------------------------------------------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------------------------------------------------
# (1) Non-pharma default values 
DEFAULT_VALUES = {
    "Heart Rate": 70.0,
    "ABP systolic": 125.0,
    "ABP diastolic": 75.0,
    "ABP mean": 90.0,
    "SpO2": 98.0,
    "RASS": 0.0,
    "Ventilator peak pressure": 0.0,
    "Lactate": 1.0,
    "INR": 1.0,
    "Blood Glucose": 5.0,
    "C-reactive protein": 4.0,
}

# (2) Static Variable 
EMERGENCY_ADMISSION_TYPES = {
    "EW EMER.",
    "DIRECT EMER.",
}

SURGICAL_SERVICES = {
    "SURG", "CSURG", "NSURG", "PSURG", "TSURG", 
    "VSURG", "ORTHO", "ENT", "GU", "GYN", "DENT",
}


# -----------------------------------------------------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------------------------------------------------
# (1) 각 grid에서 가장 늦은 실제 charttime을 저장한다. 
# Imputation은 "마지막 측정 이후 몇 분 지났는가?"이고, 이것을 '올림'처리된 시간으로부터 계산하면 안된다. 
# 또한 `place_nonpharma_on_grid()`에서는 하나의 grid point에 중복이 생기면 Median 처리를 했었는데, 이것도 실제
# 마지막 측정의 실제 시간을 표현하고 있지는 않다. 
def _latest_measurement_time_by_grid_cell(events: pd.DataFrame, grid_stay_info: pd.DataFrame, grid_minutes: int = 60):
    events = events[["stay_id", "charttime", "variable", "valuenum"]].copy()
    events = events.merge(grid_stay_info[["stay_id", "grid_start", "grid_end"]], on="stay_id", how="inner")
    events["charttime"] = pd.to_datetime(events["charttime"], errors="coerce")
    events["grid_start"] = pd.to_datetime(events["grid_start"], errors="coerce")
    events["grid_end"] = pd.to_datetime(events["grid_end"], errors="coerce")
    events = events.dropna(subset=["charttime", "valuenum"]).copy()
    events = events.loc[(events["charttime"] >= events["grid_start"]) & (events["charttime"] <= events["grid_end"])].copy()

    interval_ns = (grid_minutes * 60 * 1_000_000_000)
    # Grid 전체 길이 계산 
    duration_ns = (events["grid_end"].astype("int64") - events["grid_start"].astype("int64"))
    # 마지막 grid의 index를 계산 
    last_idx = duration_ns // interval_ns
    # 마지막 실제 gridtime 계산 
    events["last_gridtime"] = (events["grid_start"] + pd.to_timedelta(last_idx * grid_minutes, unit="min"))
    # 측정값이 시작점에서 얼마나 떨어져 있는지 계산 
    delta_ns = (events["charttime"].astype("int64") - events["grid_start"].astype("int64"))
    # 어느 grid index에 들어갈지 '올림' 계산 
    grid_idx = (delta_ns + interval_ns - 1) // interval_ns
    # grid index를 실제 gridtime으로 변환 
    events["gridtime"] = (events["grid_start"] + pd.to_timedelta(grid_idx * grid_minutes, unit="min"))
    events = events.loc[events["gridtime"] <= events["last_gridtime"]].copy()
    # 실제 마지막 측정 시간(.max())정보로 수정됨.
    latest = (events.groupby(["stay_id", "gridtime", "variable"], as_index=False)["charttime"].max()
              .rename(columns={"charttime": "last_measurement_time"}))

    return latest  


# (2) Cardiac Output default 계산 (= 3.5 * 0.007184 * weight^0.425 * height^0.725)
def prepare_cardiac_output_defaults(stay_ids, height_weight_events: pd.DataFrame):
    hw = height_weight_events[height_weight_events["stay_id"].isin(set(stay_ids))].copy()
    hw["charttime"] = pd.to_datetime(hw["charttime"], errors="coerce")
    hw["value"] = pd.to_numeric(hw["value"], errors="coerce")
    hw = hw.dropna(subset=["stay_id", "charttime", "variable", "value"]).copy()
    hw = hw.loc[hw["variable"].isin(["Height", "Weight"])].copy()

    # Stay별 가장 이른 Height / Weight 사용
    hw = (hw.sort_values(["stay_id", "variable", "charttime"])
           .drop_duplicates(["stay_id", "variable"], keep="first"))
    hw = (hw.pivot(index="stay_id", columns="variable", values="value")
            .reindex(stay_ids))
    if "Height" not in hw.columns: hw["Height"] = np.nan
    if "Weight" not in hw.columns: hw["Weight"] = np.nan

    # Height/Weight가 없는 stay를 현재 전달된 stay들의 평균으로
    hw["Height"] = hw["Height"].fillna(hw["Height"].mean())
    hw["Weight"] = hw["Weight"].fillna(hw["Weight"].mean())
    hw["Cardiac Output"] = (3.5 * 0.007184 * np.power(hw["Weight"], 0.425) * np.power(hw["Height"], 0.725))
    
    return hw["Cardiac Output"].astype(float).to_dict()


# -----------------------------------------------------------------------------------------------------------------------
# Create 60-min Time Grid 
# -----------------------------------------------------------------------------------------------------------------------
def build_time_grid(stays: pd.DataFrame, nonpharma_merged: pd.DataFrame, grid_minutes: int = 60, max_days: int = 28):
    """
        stays -> ICU stay. (stay_id | intime | outtime)
        nonpharma_merged -> `data_preprocessing.py`에서 생성된 df. (stay_id | charttime | variable | valuenum)
        grid_minutes -> grid 시간 간격
        max_days -> 하나의 ICU stay 길이를 최대 28일까지로 제한 
    """
    
    # 1. ICU stay에 있는 Heart Rate값들만 가져오기 
    hr = nonpharma_merged.loc[nonpharma_merged["variable"] == "Heart Rate", ["stay_id", "charttime"]].copy()
    hr["charttime"] = pd.to_datetime(hr["charttime"], errors="coerce")
    hr = hr.dropna(subset=["charttime"])
    
    # 2. stays의 시간정보(intime, outtime) Datetime으로 변환 
    stay_windows = stays[["stay_id", "intime", "outtime"]].copy()
    stay_windows["intime"] = pd.to_datetime(stay_windows["intime"], errors="coerce")
    stay_windows["outtime"] = pd.to_datetime(stay_windows["outtime"], errors="coerce")
    
    # 3. ICU stay에 있었던 시간안에 있는 HR만 사용 
    hr = hr.merge(stay_windows, on="stay_id", how="inner")
    hr = hr.loc[(hr["charttime"] >= hr["intime"]) & (hr["charttime"] <= hr["outtime"])].copy()
    
    # 4. ICU stay별 first와 last HR 
    hr_window = (hr.groupby("stay_id").agg(first_hr=("charttime", "min"), 
                                           last_hr=("charttime", "max")).reset_index())
    
    # 5. Grid의 시작과 종료 시각 설정 
    stay_info = stay_windows.merge(hr_window, on="stay_id", how="left")
    stay_info["max_end"] = (stay_info["intime"] + pd.to_timedelta(max_days, unit="D"))
    stay_info["grid_start"] = stay_info["first_hr"]
    stay_info["grid_end"] = stay_info[["last_hr", "outtime", "max_end"]].min(axis=1)
    
    # 6. Vaild stay만 사용 
    valid = (stay_info["grid_start"].notna() & stay_info["grid_end"].notna() 
             & (stay_info["grid_end"] >= stay_info["grid_start"]))

    grid_stay_info = stay_info.loc[valid, ["stay_id", "grid_start", "grid_end"]].copy()
    
    # 7. Stay별 Time grid 생성 
    grids = []
    for row in grid_stay_info.itertuples(index=False):
        times = pd.date_range(start=row.grid_start, end=row.grid_end, freq=f"{grid_minutes}min")
        grids.append(pd.DataFrame({"stay_id": row.stay_id, "gridtime": times}))
    
    # 8. 모든 stay의 grid 합치기 
    if grids:
        time_grid = pd.concat(grids, ignore_index=True)
    else:
        time_grid = pd.DataFrame(columns=["stay_id", "gridtime"])
    
    return time_grid, grid_stay_info


# -----------------------------------------------------------------------------------------------------------------------
# Non-pharma variable들 time grid에 올리기 
# -----------------------------------------------------------------------------------------------------------------------
def place_nonpharma_on_grid(events: pd.DataFrame, grid_stay_info: pd.DataFrame, grid_minutes: int = 60):
    """
        events ->  `data_preprocessing.py`에서 생성된 df. (stay_id | charttime | variable | valuenum)
        grid_stay_info -> build_time_grid()에서 생성된 df. (stay_id | grid_start | grid_end)
        grid_minutes -> 현재 사용하는 grid 간격 
    """
    
    # 1. 각 measurement에 해당 stay의 grid 시작, 종료 시각 붙이기 
    events = events.merge(grid_stay_info[["stay_id", "grid_start", "grid_end"]], on="stay_id", how="inner").copy()
    
    # 2. 시간 variable Datetime으로 변환 
    events["charttime"] = pd.to_datetime(events["charttime"], errors="coerce")
    events["grid_start"] = pd.to_datetime(events["grid_start"], errors="coerce")
    events["grid_end"] = pd.to_datetime(events["grid_end"], errors="coerce")
    
    # 3. 분석 범위 안에 있는 측정값을 사용 
    events = events.loc[events["charttime"].notna() & (events["charttime"] >= events["grid_start"])
                        & (events["charttime"] <= events["grid_end"])].copy()
    
    # 4. Grid간격을 nano second 단위로 변환하기 
    # Pandas의 내부 단위가 nanosencond라서 시간 위치를 정수 계산으로 정확하게 구하기 위해서 변환 
    interval_ns = (grid_minutes * 60 * 1_000_000_000)
    
    # 5. 각 Stay의 실제 마지막 grid point를 계산 
    duration_ns = (events["grid_end"].astype("int64") - events["grid_start"].astype("int64"))  # 총 시간 차이 계산
    last_idx = duration_ns // interval_ns  # 60분 단위로 몇 칸 생기는지 계산 
    events["last_gridtime"] = (events["grid_start"] + pd.to_timedelta(last_idx * grid_minutes, unit="min"))
    
    # 6. 측정값을 측정 이후 첫 grid point에 배치 
    # 측정이 grid_start로부터 얼마나 지난 뒤에 수행되었는지 계산 
    delta_ns = (events["charttime"].astype("int64") - events["grid_start"].astype("int64"))  
    # 예를들어, 18분후가 나오면 18 / 60 = 0.3이고 이걸 올림 계산시켜서 1이됨.
    # 그러면 10:03 + 1시간 = 11:03의 값으로 지정됨. 
    # 이렇게 다음 grid point로 올림시켜서 보내는 이유는 미래의 측정값이 과거 시점에 들어오는 것을 방지하기 위해서 
    grid_idx = (delta_ns + interval_ns - 1) // interval_ns
    events["gridtime"] = (events["grid_start"] + pd.to_timedelta(grid_idx * grid_minutes, unit="min"))
    
    # 마지막 실제 grid point를 넘어가는 measurement는 제외
    events = events.loc[events["gridtime"] <= events["last_gridtime"]].copy()

    # 7. 같은 stay + gridtime + variable에 여러 측정값이 있으면 median
    binned = (events.groupby(["stay_id", "gridtime", "variable"], as_index=False)["valuenum"].median())
    binned["valuenum"] = (binned["valuenum"].astype("float32"))  # float32로 측정값 변환

    return binned


# -----------------------------------------------------------------------------------------------------------------------
# Pharma variable들 time grid에 올리기 
# -----------------------------------------------------------------------------------------------------------------------
def place_pharma_on_grid(events: pd.DataFrame, time_grid: pd.DataFrame, grid_stay_info: pd.DataFrame, grid_minutes: int = 60):
    """
        events -> data_preprocessing.py에서 생성된 pharma df. (stay_id|pharma_id|pharma_variable|
                                                              starttime|endtime|continuous_rate|continuous_rate_uom)
        time_grid -> build_time_grid()에서 생성된 전체 grid. (stay_id | gridtime)
        grid_stay_info -> build_time_grid()에서 생성된 df. (stay_id | grid_start | grid_end)
        grid_minutes -> 현재 사용하는 grid 간격
    """
    
    pharma = events.copy()
    # 1. 시간 / rate 자료형 분리하기 
    pharma["starttime"] = pd.to_datetime(pharma["starttime"], errors="coerce")
    pharma["endtime"] = pd.to_datetime(pharma["endtime"], errors="coerce")
    pharma["continuous_rate"] = pd.to_numeric(pharma["continuous_rate"], errors="coerce")
    pharma = pharma.dropna(subset=["stay_id", "pharma_id", "pharma_variable", "starttime", "endtime", "continuous_rate"]).copy()
    
    # 2. 오류 처리 
    # 음수 rate
    if (pharma["continuous_rate"] < 0).any():
        raise ValueError("Negative continuous_rate detected")
    
    # 같은 pharma variable인데, 단위가 여러개면 합칠 수 없으므로 오류처리 
    unit_count = (pharma.groupby("pharma_variable")["continuous_rate_uom"].nunique())
    if (unit_count > 1).any():
        raise ValueError("같은 pharma variable 안에 2개 이상의 unit이 존재함")
    
    # 3. 각 pharma interval에 해당 stay의 grid 범위 붙이기 
    pharma = pharma.merge(grid_stay_info[["stay_id", "grid_start", "grid_end"]], on="stay_id", how="inner")
    pharma["grid_start"] = pd.to_datetime(pharma["grid_start"], errors="coerce")
    pharma["grid_end"] = pd.to_datetime(pharma["grid_end"], errors="coerce") 
    
    # 4. 분석 범위 내 grid와 실제로 겹치는 약물 interval만 사용 
    pharma = pharma.loc[(pharma["endtime"] > pharma["grid_start"]) & (pharma["starttime"] <= pharma["grid_end"])].copy()
    
    # 5. Grid 간격을 nanosecond단위로 변환 
    interval_ns = (grid_minutes * 60 * 1_000_000_000)
    
    pieces = []

    # 7. 각 약물 interval을 활성화되어 있는 grid point들로 펼치기
    for row in pharma.itertuples(index=False):
        grid_start = pd.Timestamp(row.grid_start)
        grid_end = pd.Timestamp(row.grid_end)
        starttime = max(pd.Timestamp(row.starttime), grid_start)
        endtime = min(pd.Timestamp(row.endtime), grid_end + pd.Timedelta(minutes=grid_minutes))

        # 해당 stay의 마지막 grid index
        max_idx = int((grid_end.value - grid_start.value) // interval_ns)

        # starttime 이후 첫 grid point
        start_delta = (starttime.value - grid_start.value)
        first_idx = int((start_delta + interval_ns - 1) // interval_ns)

        # endtime보다 작은 마지막 grid point
        end_delta = (endtime.value - grid_start.value)
        last_idx = int((end_delta - 1) // interval_ns)

        first_idx = max(first_idx, 0)
        last_idx = min(last_idx, max_idx)

        if first_idx > last_idx: continue
        idx = np.arange(first_idx, last_idx + 1)
        gridtimes = (grid_start + pd.to_timedelta(idx * grid_minutes, unit="min"))
        pieces.append(pd.DataFrame({"stay_id": row.stay_id, "gridtime": gridtimes,
                                    "pharma_variable": row.pharma_variable,"rate": float(row.continuous_rate)}))

    # 8. 활성화된 약물 grid들을 하나로 합치기
    if pieces:
        pharma_grid_long = pd.concat(pieces, ignore_index=True)

        # 같은 시간에 같은 약물 infusion이 여러 개면 rate SUM
        pharma_grid_long = (pharma_grid_long.groupby(["stay_id", "gridtime", "pharma_variable"], as_index=False)["rate"].sum())

        # pharma variable들을 column으로 변환
        wide = (pharma_grid_long.pivot(index=["stay_id", "gridtime"], columns="pharma_variable", values="rate").reset_index())
        wide.columns.name = None
    else:
        wide = pd.DataFrame(columns=["stay_id", "gridtime"])

    # 9. 전체 time grid에 pharma 값 붙이기
    pharma_grid = time_grid.merge(wide, on=["stay_id", "gridtime"], how="left")

    # 10. 약물이 투여되지 않은 grid는 0
    pharma_variables = (events["pharma_variable"].dropna().unique())

    for variable in pharma_variables:
        if variable not in pharma_grid.columns:
            pharma_grid[variable] = 0.0
        pharma_grid[variable] = (
            pharma_grid[variable].fillna(0.0).astype("float32"))

    return pharma_grid
    

# -----------------------------------------------------------------------------------------------------------------------
# Adaptive Imputation
# 
# non-static variables
#   - non-pharma -> adpative imputation (여기서 완료 )
#   - pharma -> rate, 0 (place_nonpharma_on_grid로 완료)
# 
# static variables -> 별도 mean / mode impuation (다음 함수에서 완료)
# -----------------------------------------------------------------------------------------------------------------------
# 1. Adaptive Imputation parameter(sampling interval의 Median, IQR) 계산
def calculate_imputation_parameters(events: pd.DataFrame, grid_stay_info: pd.DataFrame, stay_ids=None):
    """
        events -> data_preprocessing.py에서 생성된 non-pharma df. (stay_id | charttime | variable | valuenum)
        grid_stay_info -> build_time_grid()에서 생성된 df. (stay_id | grid_start | grid_end)
        stay_ids -> parameter 계산에 사용할 stay들. !최종 학습에서는 training stay만 전달!
    """
    
    # 1. 필요한 column만 가져오기 
    events = events[["stay_id", "charttime", "variable", "valuenum"]].copy()
    windows = grid_stay_info[["stay_id", "grid_start", "grid_end"]].copy()
    
    # 2. 특정 stay만 -> 나중에 training stay만 사용할 때를 위한 조건
    if stay_ids is not None:
        stay_ids = set(stay_ids)
        events = events.loc[events["stay_id"].isin(stay_ids)].copy()
        windows = windows.loc[windows["stay_id"].isin(stay_ids)].copy()
    
    # 3. 시간 관련 variable Datetime으로 변환 
    events["charttime"] = pd.to_datetime(events["charttime"], errors="coerce")
    windows["grid_start"] = pd.to_datetime(windows["grid_start"], errors="coerce")
    windows["grid_end"] = pd.to_datetime(windows["grid_end"], errors="coerce")
    events = events.dropna(subset=["stay_id", "charttime", "variable", "valuenum"]).copy()  
    
    # 4. 각 측정값에 해당되는 stay의 grid 범위 붙이기 
    events = events.merge(windows, on="stay_id", how="inner")
    
    # 5. 실제 분석 window 안 측정값만 사용
    events = events.loc[(events["charttime"] >= events["grid_start"]) 
                        & (events["charttime"] <= events["grid_end"])].copy()   
    
    # 6. stay, variable별 시간순 정렬 
    events = events.sort_values(["stay_id", "variable", "charttime"])
    
    # 7. 이전 측정값과의 시간 차이 계산하기 
    events["sampling_interval_min"] = (events.groupby(["stay_id", "variable"])["charttime"].diff().dt.total_seconds() / 60)

    # 첫 측정값은 이전 측정값이 없으므로 제외 & 잘못된 시간 차이도 제외 
    intervals = events.loc[events["sampling_interval_min"].notna() & (events["sampling_interval_min"] > 0)].copy()
    
    # 8. variable별 sampling interval의 Median, Q1, Q3 계산 
    params = (intervals.groupby("variable")["sampling_interval_min"].agg(median_sampling_min="median",
                                                                         q1_sampling_min=lambda x: x.quantile(0.25),
                                                                         q3_sampling_min=lambda x: x.quantile(0.75))
                                                                        .reset_index())
    # IQR
    params["iqr_sampling_min"] = (params["q3_sampling_min"] - params["q1_sampling_min"])
    
    # 9. Adaptive imputation에 사용하는 시간 기준
    params["ffill_threshold_min"] = (params["median_sampling_min"] + params["iqr_sampling_min"])
    params["return_horizon_min"] = (2 * (params["median_sampling_min"] + 2 * params["iqr_sampling_min"]))

    return params


# 2. Adaptive Imputation 수행 
def adaptive_impute_nonpharma(time_grid: pd.DataFrame, raw_grid_values: pd.DataFrame,
                              nonpharma_merged: pd.DataFrame, grid_stay_info: pd.DataFrame,
                              imputation_params: pd.DataFrame, height_weight_events: pd.DataFrame,
                              grid_minutes: int = 60):
    """
        1. 실제 측정값 있음 -> 실제값 사용 -> 새로운 측정값 기준으로 상태 초기화 
        2. 아직 첫 측정 전 -> default values 사용 
        3-1. 마지막 실제 측정 후 시간이 m + IQR 미만 -> ffill
        3-2. 마지막 실제 측정 후 시간이 m + IQR 이상 -> 과거 측정값들의 median으로 linear하게 복귀 
            3-2-1. 복귀가 다 끝나면 median 값 그대로 유지 
            3-2-2. 새로운 측정값이 존재하면 다시 처음부터 
    """

    # 1. 전체 grid 정리
    grid = time_grid[["stay_id", "gridtime"]].copy()
    grid["gridtime"] = pd.to_datetime(grid["gridtime"], errors="coerce")
    grid = grid.dropna(subset=["stay_id", "gridtime"]).sort_values(["stay_id", "gridtime"]).reset_index(drop=True)
    grid["_grid_row"] = np.arange(len(grid))
    stay_ids = grid["stay_id"].drop_duplicates().tolist()

    # 2. Cardiac Output stay별 default
    cardiac_output_defaults = prepare_cardiac_output_defaults(
        stay_ids=stay_ids,
        height_weight_events=height_weight_events
    )

    # 3. 변수별 imputation parameter
    params = imputation_params.set_index("variable")
    variables = raw_grid_values["variable"].dropna().unique().tolist()

    # 4. 각 observed grid cell의 실제 마지막 측정시각
    latest_time = _latest_measurement_time_by_grid_cell(
        events=nonpharma_merged,
        grid_stay_info=grid_stay_info,
        grid_minutes=grid_minutes
    )

    observed = raw_grid_values[["stay_id", "gridtime", "variable", "valuenum"]].copy()
    observed = observed.merge(latest_time, on=["stay_id", "gridtime", "variable"], how="left")

    # 5. Historical median 계산용 raw measurement
    history = nonpharma_merged[["stay_id", "charttime", "variable", "valuenum"]].copy()
    history["charttime"] = pd.to_datetime(history["charttime"], errors="coerce")
    history["valuenum"] = pd.to_numeric(history["valuenum"], errors="coerce")
    history = history.dropna(subset=["stay_id", "charttime", "variable", "valuenum"]).copy()

    history = history.merge(grid_stay_info[["stay_id", "grid_start", "grid_end"]], on="stay_id", how="inner")
    history = history.loc[(history["charttime"] >= history["grid_start"]) &
                          (history["charttime"] <= history["grid_end"])].copy()

    # 6. Stay별 grid row 위치 저장
    stay_grids = {
        sid: (g["_grid_row"].to_numpy(), g["gridtime"].to_numpy(dtype="datetime64[ns]"))
        for sid, g in grid.groupby("stay_id", sort=False)
    }

    imputed_grid = grid[["stay_id", "gridtime"]].copy()

    # 7. Variable별 adaptive imputation
    for variable in variables:
        ffill_threshold = float(params.loc[variable, "ffill_threshold_min"])
        return_horizon = float(params.loc[variable, "return_horizon_min"])

        obs_v = observed.loc[observed["variable"] == variable]
        obs_by_stay = {sid: g.sort_values("gridtime") for sid, g in obs_v.groupby("stay_id")}

        hist_v = history.loc[history["variable"] == variable]
        hist_by_stay = {
            sid: (
                g["charttime"].to_numpy(dtype="datetime64[ns]"),
                g["valuenum"].to_numpy(dtype=float)
            )
            for sid, g in hist_v.groupby("stay_id")
        }

        values_all = np.empty(len(grid), dtype=np.float32)

        # 8. Stay 하나씩 처리
        for sid in stay_ids:
            row_idx, grid_times = stay_grids[sid]
            values = np.empty(len(grid_times), dtype=np.float32)

            # Cardiac Output만 stay별 default, 나머지는 고정 default
            if variable == "Cardiac Output":
                default_value = float(cardiac_output_defaults[sid])
            else:
                default_value = float(DEFAULT_VALUES[variable])

            # 해당 stay에서 이 variable이 한 번도 측정되지 않음
            if sid not in obs_by_stay:
                values[:] = default_value
                values_all[row_idx] = values
                continue

            obs = obs_by_stay[sid]

            # observed gridtime -> 현재 stay 내부 grid index
            grid_pos = pd.Series(np.arange(len(grid_times)), index=pd.DatetimeIndex(grid_times))
            obs_idx = grid_pos.reindex(pd.DatetimeIndex(obs["gridtime"])).to_numpy().astype(int)

            obs_lookup = {
                int(i): (float(v), t)
                for i, v, t in zip(
                    obs_idx,
                    obs["valuenum"].to_numpy(dtype=float),
                    obs["last_measurement_time"].to_numpy(dtype="datetime64[ns]")
                )
            }

            hist_times, hist_values = hist_by_stay.get(
                sid,
                (np.array([], dtype="datetime64[ns]"), np.array([], dtype=float))
            )

            have_measurement = False
            last_value = np.nan
            last_measurement_time = np.datetime64("NaT")
            target_median = np.nan
            return_entry_time = np.datetime64("NaT")

            # 9. Grid point를 시간순으로 진행
            for j, current_time in enumerate(grid_times):

                # 실제 measurement가 있는 grid
                if j in obs_lookup:
                    last_value, last_measurement_time = obs_lookup[j]
                    values[j] = last_value
                    have_measurement = True
                    target_median = np.nan
                    return_entry_time = np.datetime64("NaT")
                    continue

                # 첫 measurement 이전
                if not have_measurement:
                    values[j] = default_value
                    continue

                elapsed_min = float((current_time - last_measurement_time) / np.timedelta64(1, "m"))

                # Forward fill
                if elapsed_min < ffill_threshold:
                    values[j] = last_value
                    continue

                # Return mode에 처음 진입할 때 historical median 계산
                if np.isnan(target_median):
                    return_entry_time = last_measurement_time + np.timedelta64(int(round(ffill_threshold * 60)), "s")
                    history_start = return_entry_time - np.timedelta64(int(round(return_horizon * 60)), "s")

                    mask = (hist_times >= history_start) & (hist_times <= return_entry_time)
                    candidate_values = hist_values[mask]

                    if len(candidate_values):
                        target_median = float(np.median(candidate_values))
                    else:
                        target_median = float(last_value)

                # Historical median으로 선형 복귀
                since_entry = float((current_time - return_entry_time) / np.timedelta64(1, "m"))
                fraction = min(max(since_entry / return_horizon, 0.0), 1.0)

                values[j] = last_value + fraction * (target_median - last_value)

            values_all[row_idx] = values

        imputed_grid[variable] = values_all

    return imputed_grid

 
# -----------------------------------------------------------------------------------------------------------------------
# Static Variables mean / mode 처리하기 
# -----------------------------------------------------------------------------------------------------------------------
def static_variables(stays: pd.DataFrame, admissions: pd.DataFrame, patients: pd.DataFrame, 
                     services: pd.DataFrame, height_weight_events: pd.DataFrame, training_stay_ids = None):
    """
        - continuous missing -> training mean
        - categorical missing -> training mode
    """
    
    # 1. 기본 stay + admission + patient 정보 
    static = stays[["stay_id", "subject_id", "hadm_id", "intime"]].copy()
    static["intime"] = pd.to_datetime(static["intime"], errors="coerce")
    
    adm = admissions[["subject_id", "hadm_id", "admittime", "admission_type"]].copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    static = static.merge(adm, on=["subject_id", "hadm_id"], how="left")
    
    pat = patients[["subject_id", "anchor_age", "anchor_year", "gender"]].copy()
    static = static.merge(pat, on="subject_id", how="left")
    
    # 2. Age, Sex
    static["Age"] = (static["anchor_age"] + (static["admittime"].dt.year - static["anchor_year"])).astype(float)
    static["Sex"] = (static["gender"].astype("string").str.strip().str.upper())
    static.loc[~static["Sex"].isin(["M", "F"]), "Sex"] = pd.NA
    
    # 3. Emergency admission
    admission_type = static["admission_type"].astype("string").str.strip().str.upper()
    static["Emergency admission"] = np.where(admission_type.isna(), np.nan, 
                                             admission_type.isin(EMERGENCY_ADMISSION_TYPES).astype(float))
    
    # 4. Height: stay에서 가장 최초의 측정값으로 
    height = height_weight_events.loc[height_weight_events["variable"] == "Height", 
                                      ["stay_id", "charttime", "value"]].copy()
    height["charttime"] = pd.to_datetime(height["charttime"], errors="coerce")
    height["value"] = pd.to_numeric(height["value"], errors="coerce")
    height = (height.dropna(subset=["stay_id", "charttime", "value"])
                    .sort_values(["stay_id", "charttime"])
                    .drop_duplicates("stay_id", keep="first")
                    .rename(columns={"value": "Height"}))
    static = static.merge(height[["stay_id", "Height"]], on="stay_id", how="left")
    
    # 5. ICU 입실 시점의 hospital service를 선택 
    service = services[["hadm_id", "transfertime", "curr_service"]].copy()
    service["transfertime"] = pd.to_datetime(service["transfertime"], errors="coerce")
    service = service.dropna(subset=["hadm_id", "transfertime", "curr_service"])

    merged = static[["stay_id", "hadm_id", "intime"]].merge(service, on="hadm_id", how="left")

    before = (merged.loc[merged["transfertime"] <= merged["intime"]]
                    .sort_values(["stay_id", "transfertime"], ascending=[True, False])
                    .drop_duplicates("stay_id")
                    [["stay_id", "curr_service"]])

    found = set(before["stay_id"])

    after = (merged.loc[(~merged["stay_id"].isin(found)) & (merged["transfertime"] > merged["intime"])]
                   .sort_values(["stay_id", "transfertime"])
                   .drop_duplicates("stay_id")
                   [["stay_id", "curr_service"]])

    service_at_icu = pd.concat([before, after], ignore_index=True)
    static = static.merge(service_at_icu, on="stay_id", how="left")

    service_clean = static["curr_service"].astype("string").str.strip().str.upper()
    static["Surgical admission"] = np.where(service_clean.isna(), np.nan,
                                            service_clean.isin(SURGICAL_SERVICES).astype(float))   
    
    # 6. Imputation에 사용할 ICU stay -> training set only 설정 관련 
    if training_stay_ids is None:
        train_mask = pd.Series(True, index=static.index)
    else:
        train_mask = static["stay_id"].isin(set(training_stay_ids))
        
    # continuous -> mean
    static["Age"] = static["Age"].fillna(static.loc[train_mask, "Age"].mean())
    static["Height"] = static["Height"].fillna(static.loc[train_mask, "Height"].mean())

    # categorical -> mode
    sex_mode = static.loc[train_mask, "Sex"].dropna().mode().iloc[0]
    static["Sex"] = static["Sex"].fillna(sex_mode)
    
    emergency_mode = static.loc[train_mask, "Emergency admission"].dropna().mode().iloc[0]
    static["Emergency admission"] = static["Emergency admission"].fillna(emergency_mode).astype("int8")
    
    surgical_mode = static.loc[train_mask, "Surgical admission"].dropna().mode().iloc[0]
    static["Surgical admission"] = static["Surgical admission"].fillna(surgical_mode).astype("int8")
    
    return static[["stay_id", "Age", "Sex", "Height", "Emergency admission", "Surgical admission"]].copy()