# -*- coding: utf-8 -*-
# name: feature_generation.py
# author: JunYoung Park
# date: 2026-08-28


from pathlib import Path
import pandas as pd 
import numpy as np


# -----------------------------------------------------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------------------------------------------------
# 첫 30분은 제외해주는 함수 
def prepare_feature_grid(dynamic_grid: pd.DataFrame, grid_minutes: int = 60, ignore_first_minutes: int = 30):
    grid = dynamic_grid.copy()
    grid["gridtime"] = pd.to_datetime(grid["gridtime"], errors="coerce")
    grid = grid.dropna(subset=["stay_id", "gridtime"])
    grid = grid.sort_values(["stay_id", "gridtime"]).reset_index(drop=True)
    # 각 stay의 첫 gridtime
    first_time = grid.groupby("stay_id")["gridtime"].transform("min")
    elapsed_minutes = ((grid["gridtime"] - first_time).dt.total_seconds() / 60)
    # 현재 60-min grid에서는 첫 point(0 min)가 제외되고
    # 다음 point(60 min)부터 사용됨
    grid = grid.loc[elapsed_minutes >= ignore_first_minutes].copy()
    
    feature_grid = grid.reset_index(drop=True)
    return feature_grid


# -----------------------------------------------------------------------------------------------------------------------
# 1. Static Features 
# -----------------------------------------------------------------------------------------------------------------------
def add_static_features(feature_grid: pd.DataFrame, static_features: pd.DataFrame):
    static_cols = ["stay_id", "Age", "Surgical admission",
                   "Emergency admission", "Height", "Sex"]

    features = feature_grid[["stay_id", "gridtime"]].merge(
        static_features[static_cols],
        on="stay_id",
        how="left",
        validate="many_to_one"
    )

    return features


# -----------------------------------------------------------------------------------------------------------------------
# 2. Multi-resolution summaries
# -----------------------------------------------------------------------------------------------------------------------
def extract_multiresolution_features(
    dynamic_grid: pd.DataFrame,
    feature_grid: pd.DataFrame,
    imputation_params: pd.DataFrame,
    table4_path: str,
    grid_minutes: int = 60
):
    # 1. 전체 imputed grid 정리
    grid = dynamic_grid.copy()
    grid["gridtime"] = pd.to_datetime(grid["gridtime"], errors="coerce")
    grid = grid.dropna(subset=["stay_id", "gridtime"]).sort_values(
        ["stay_id", "gridtime"]
    ).reset_index(drop=True)

    stay_ids = grid["stay_id"]

    # 2. Frequency별 horizon
    horizons = {
        "high": [60, 240, 720],  # 30 min 제외
        "medium": [720, 1440, 2160, 2880],
        "low": [960, 1920, 2880, 4320]
    }

    # 3. Training set sampling interval로 non-pharma frequency 결정
    nonpharma_frequency = {}

    for row in imputation_params.itertuples(index=False):
        interval = float(row.median_sampling_min)

        if interval <= 15:
            frequency = "high"
        elif interval <= 8 * 60:
            frequency = "medium"
        else:
            frequency = "low"

        nonpharma_frequency[row.variable] = frequency

    # 4. Supplementary Table IV의 acting period로 pharma frequency 결정
    drugs = pd.read_excel(table4_path, sheet_name="drugs")
    drugs["temp: bern name"] = drugs["temp: bern name"].ffill()

    acting = drugs[
        ["temp: bern name", "acting period (individual)"]
    ].dropna().copy()

    pharma_frequency = {}

    for row in acting.itertuples(index=False):
        variable = str(row[0]).strip()
        period = str(row[1]).strip().lower()

        # Acting period -> minute
        if period.endswith("m"):
            acting_period_min = float(period[:-1])
        elif period.endswith("h"):
            acting_period_min = float(period[:-1]) * 60
        elif period.endswith("d"):
            acting_period_min = float(period[:-1]) * 24 * 60
        else:
            continue

        # 현재 MIMIC-IV dynamic grid에 존재하는 pharma variable만 사용
        if variable not in grid.columns:
            continue

        if acting_period_min <= 15:
            frequency = "high"
        elif acting_period_min <= 8 * 60:
            frequency = "medium"
        else:
            frequency = "low"

        pharma_frequency[variable] = frequency

    # Feature column들을 여기에 모아두고 마지막에 한 번에 붙임
    feature_dict = {}

    # 5. Non-pharma continuous variables
    for variable, frequency in nonpharma_frequency.items():
        if variable not in grid.columns:
            continue

        # 현재 시점 자체는 history에서 제외
        values = grid[variable].groupby(
            stay_ids,
            sort=False
        ).shift(1)

        for horizon_min in horizons[frequency]:
            window_points = horizon_min // grid_minutes

            rolling = values.groupby(
                stay_ids,
                sort=False
            ).rolling(
                window=window_points,
                min_periods=1
            )

            horizon_name = f"{horizon_min // 60}h"

            median = rolling.median().reset_index(
                level=0,
                drop=True
            ).reindex(grid.index)

            q75 = rolling.quantile(0.75).reset_index(
                level=0,
                drop=True
            ).reindex(grid.index)

            q25 = rolling.quantile(0.25).reset_index(
                level=0,
                drop=True
            ).reindex(grid.index)

            minimum = rolling.min().reset_index(
                level=0,
                drop=True
            ).reindex(grid.index)

            maximum = rolling.max().reset_index(
                level=0,
                drop=True
            ).reindex(grid.index)

            # trend = window의 마지막 값 - 첫 값
            last_value = values

            first_value = values.groupby(
                stay_ids,
                sort=False
            ).shift(window_points - 1)

            # Stay 시작부처럼 window 길이보다 history가 짧으면
            # 현재까지 존재하는 첫 과거값 사용
            first_available = values.groupby(
                stay_ids,
                sort=False
            ).transform("first")

            first_value = first_value.fillna(first_available)

            trend = last_value - first_value

            feature_dict[
                f"{variable}__{horizon_name}__median"
            ] = median.astype("float32")

            feature_dict[
                f"{variable}__{horizon_name}__iqr"
            ] = (q75 - q25).astype("float32")

            feature_dict[
                f"{variable}__{horizon_name}__min"
            ] = minimum.astype("float32")

            feature_dict[
                f"{variable}__{horizon_name}__max"
            ] = maximum.astype("float32")

            feature_dict[
                f"{variable}__{horizon_name}__trend"
            ] = trend.astype("float32")

        # Entire stay up to current point
        entire = values.groupby(
            stay_ids,
            sort=False
        ).expanding(
            min_periods=1
        ).median().reset_index(
            level=0,
            drop=True
        ).reindex(grid.index)

        feature_dict[
            f"{variable}__entire__median"
        ] = entire.astype("float32")

    # 6. Pharmaceutical variables
    for variable, frequency in pharma_frequency.items():
        if variable not in grid.columns:
            continue

        # 현재 시점 자체는 history에서 제외
        values = grid[variable].groupby(
            stay_ids,
            sort=False
        ).shift(1)

        for horizon_min in horizons[frequency]:
            window_points = horizon_min // grid_minutes

            rolling = values.groupby(
                stay_ids,
                sort=False
            ).rolling(
                window=window_points,
                min_periods=1
            )

            horizon_name = f"{horizon_min // 60}h"

            mean = rolling.mean().reset_index(
                level=0,
                drop=True
            ).reindex(grid.index)

            std = rolling.std(ddof=0).reset_index(
                level=0,
                drop=True
            ).reindex(grid.index)

            minimum = rolling.min().reset_index(
                level=0,
                drop=True
            ).reindex(grid.index)

            maximum = rolling.max().reset_index(
                level=0,
                drop=True
            ).reindex(grid.index)

            # trend = window의 마지막 값 - 첫 값
            last_value = values

            first_value = values.groupby(
                stay_ids,
                sort=False
            ).shift(window_points - 1)

            first_available = values.groupby(
                stay_ids,
                sort=False
            ).transform("first")

            first_value = first_value.fillna(first_available)

            trend = last_value - first_value

            feature_dict[
                f"{variable}__{horizon_name}__mean"
            ] = mean.astype("float32")

            feature_dict[
                f"{variable}__{horizon_name}__std"
            ] = std.astype("float32")

            feature_dict[
                f"{variable}__{horizon_name}__min"
            ] = minimum.astype("float32")

            feature_dict[
                f"{variable}__{horizon_name}__max"
            ] = maximum.astype("float32")

            feature_dict[
                f"{variable}__{horizon_name}__trend"
            ] = trend.astype("float32")

        # Entire stay up to current point
        entire = values.groupby(
            stay_ids,
            sort=False
        ).expanding(
            min_periods=1
        ).mean().reset_index(
            level=0,
            drop=True
        ).reindex(grid.index)

        feature_dict[
            f"{variable}__entire__mean"
        ] = entire.astype("float32")

    # 7. Feature column들을 한 번에 DataFrame으로 생성
    feature_values = pd.DataFrame(
        feature_dict,
        index=grid.index
    )

    features = pd.concat(
        [
            grid[["stay_id", "gridtime"]].reset_index(drop=True),
            feature_values.reset_index(drop=True)
        ],
        axis=1
    )

    # 8. 실제 feature generation 대상 grid만 남기기
    features = feature_grid[
        ["stay_id", "gridtime"]
    ].merge(
        features,
        on=["stay_id", "gridtime"],
        how="left",
        validate="one_to_one"
    )

    return features
        
        
# -----------------------------------------------------------------------------------------------------------------------
# 3. Instability history features
# -----------------------------------------------------------------------------------------------------------------------
def extract_instability_history_features(dynamic_grid: pd.DataFrame, feature_grid: pd.DataFrame,
                                         static_features: pd.DataFrame, grid_minutes: int = 60):
    grid = dynamic_grid.copy()
    grid["gridtime"] = pd.to_datetime(grid["gridtime"], errors="coerce")
    grid = grid.dropna(subset=["stay_id", "gridtime"]).sort_values(["stay_id", "gridtime"]).reset_index(drop=True)

    # 1. Weight 붙이기
    weight = static_features[["stay_id", "Weight"]].drop_duplicates("stay_id")\
    if "Weight" in static_features.columns else None
    
    if weight is None: raise ValueError("static_features에 Weight가 필요함")
    
    grid = grid.merge(weight, on="stay_id", how="left", validate="many_to_one")

    # 2. 필요한 변수가 없으면 0으로 생성
    required_drugs = ["Dopamin", "Milrinone", "Norepinephrine", "Epinephrine", "Vasopressin"]
    for variable in required_drugs:
        if variable not in grid.columns: grid[variable] = 0.0

    # 3. Norepinephrine / Epinephrine 단위 변환
    grid["Norepinephrine_absolute"] = grid["Norepinephrine"] * grid["Weight"]
    grid["Epinephrine_absolute"] = grid["Epinephrine"] * grid["Weight"]

    # 4. 기본 pathological sub-events
    subevents = {}
    subevents["map_low"] = grid["ABP mean"] <= 65
    subevents["lactate_high"] = grid["Lactate"] >= 2
    subevents["dopamin"] = grid["Dopamin"] > 0
    subevents["milrinone"] = grid["Milrinone"] > 0
    subevents["norepinephrine_low"] = (grid["Norepinephrine_absolute"] > 0) & (grid["Norepinephrine_absolute"] < 0.1)
    subevents["norepinephrine_high"] = grid["Norepinephrine_absolute"] >= 0.1
    subevents["epinephrine_low"] = (grid["Epinephrine_absolute"] > 0) & (grid["Epinephrine_absolute"] < 0.1)
    subevents["epinephrine_high"] = grid["Epinephrine_absolute"] >= 0.1
    subevents["vasopressin"] = grid["Vasopressin"] > 0

    # 5. Event L1 / L2 / L3
    lactate_high = subevents["lactate_high"]
    subevents["event_l1"] = lactate_high & (subevents["map_low"] | subevents["dopamin"] | subevents["milrinone"])
    subevents["event_l2"] = lactate_high & (subevents["norepinephrine_low"] | subevents["epinephrine_low"] | subevents["vasopressin"])
    subevents["event_l3"] = lactate_high & (subevents["norepinephrine_high"] | subevents["epinephrine_high"] | subevents["vasopressin"])

    # 6. History features
    features = grid[["stay_id", "gridtime"]].copy()
    horizons = [12, 24, 36, 48]

    for name, state in subevents.items():
        state = state.fillna(False).astype("int8")
        grouped = state.groupby(grid["stay_id"], sort=False)

        # current state
        features[f"{name}__current"] = state

        # time to last occurrence
        time_to_last = []
        for sid, idx in grid.groupby("stay_id", sort=False).groups.items():
            idx = list(idx)
            last_time = None
            for i in idx:
                current_time = grid.loc[i, "gridtime"]
                if state.loc[i] == 1: last_time = current_time
                if last_time is None: time_to_last.append(30 * 24)
                else: time_to_last.append((current_time - last_time).total_seconds() / 3600)
        features[f"{name}__time_to_last_h"] = pd.Series(time_to_last, index=features.index, dtype="float32")

        # fraction in fixed horizons
        for horizon_h in horizons:
            window_points = horizon_h * 60 // grid_minutes
            fraction = grouped.rolling(window=window_points, min_periods=1).mean().reset_index(level=0, drop=True).reindex(grid.index)
            features[f"{name}__fraction_{horizon_h}h"] = fraction.astype("float32")

        # entire stay fraction
        entire = grouped.expanding(min_periods=1).mean().reset_index(level=0, drop=True).reindex(grid.index)
        features[f"{name}__fraction_entire"] = entire.astype("float32")

    # 7. 실제 feature generation 대상 grid만 남기기
    features = feature_grid[["stay_id", "gridtime"]].merge(features, on=["stay_id", "gridtime"], how="left", validate="one_to_one")
    return features


# -----------------------------------------------------------------------------------------------------------------------
# 4. Measurement-intensity based features 
# -----------------------------------------------------------------------------------------------------------------------
def extract_measurement_intensity_features(
    raw_grid_values: pd.DataFrame,
    dynamic_grid: pd.DataFrame,
    feature_grid: pd.DataFrame,
    grid_minutes: int = 60
):
    # 1. 실제 measurement가 존재했던 grid 정리
    raw = raw_grid_values[["stay_id", "gridtime", "variable"]].copy()
    raw["gridtime"] = pd.to_datetime(raw["gridtime"], errors="coerce")
    raw = raw.dropna(subset=["stay_id", "gridtime", "variable"]).drop_duplicates(
        ["stay_id", "gridtime", "variable"]
    )

    # 2. 전체 60-min grid 사용
    # 첫 0-min grid도 history 계산에는 포함되어야 함
    grid = dynamic_grid[["stay_id", "gridtime"]].copy()
    grid["gridtime"] = pd.to_datetime(grid["gridtime"], errors="coerce")
    grid = (
        grid.dropna(subset=["stay_id", "gridtime"])
        .sort_values(["stay_id", "gridtime"])
        .reset_index(drop=True)
    )

    variables = raw["variable"].dropna().unique().tolist()
    features = grid.copy()

    # 3. Variable별 measurement intensity 계산
    for variable in variables:
        measured = raw.loc[
            raw["variable"] == variable,
            ["stay_id", "gridtime"]
        ].copy()

        measured["measured"] = 1

        temp = grid.merge(
            measured,
            on=["stay_id", "gridtime"],
            how="left"
        )

        temp["measured"] = temp["measured"].fillna(0).astype("int8")

        # Time since last actual measurement
        temp["last_measurement_time"] = temp["gridtime"].where(
            temp["measured"] == 1
        )

        temp["last_measurement_time"] = (
            temp.groupby("stay_id")["last_measurement_time"].ffill()
        )

        time_since_last = (
            (temp["gridtime"] - temp["last_measurement_time"])
            .dt.total_seconds() / 3600
        )

        # 한 번도 측정되지 않은 경우 30 days = 720 hours
        time_since_last = time_since_last.fillna(30 * 24)

        features[
            f"{variable}__time_since_last_measurement_h"
        ] = time_since_last.astype("float32")

        # Measurement ratio up to current point
        measured_count = temp.groupby("stay_id")["measured"].cumsum()
        grid_count = temp.groupby("stay_id").cumcount() + 1

        features[
            f"{variable}__measurement_ratio"
        ] = (measured_count / grid_count).astype("float32")

    # 4. History 계산 후 실제 feature sample만 남기기
    features = feature_grid[["stay_id", "gridtime"]].merge(
        features,
        on=["stay_id", "gridtime"],
        how="left",
        validate="one_to_one"
    )

    return features


# -----------------------------------------------------------------------------------------------------------------------
# 5. Shaplet-based features 
# -----------------------------------------------------------------------------------------------------------------------
def extract_shapelet_features(
    nonpharma_merged: pd.DataFrame,
    state: pd.DataFrame,
    feature_grid: pd.DataFrame,
    train_ids,
    imputation_params: pd.DataFrame,
    grid_minutes: int = 5,
    n_cases: int = 300,
    n_controls: int = 300,
    n_shapelets: int = 20,
    seed: int = 42
):
    """
        MIMIC-IV adaptation

        1. Shapelet 전용 5-min grid 생성
        2. Training stay만 사용
        3. CF onset 전 trajectory -> case
        4. CF가 없는 stable trajectory -> control
        5. candidate subsequence 생성
        6. case/control separation이 좋은 candidate부터 선택
        7. min-max 방식으로 최대 20개 representative shapelet 선택
        8. 각 patient history와 shapelet의 L2 distance 계산
        9. 최근 4h distance history를 60-min feature grid에 align

        현재 구현은 circEWS의 S3M binary 자체를 재사용하지 않고
        같은 목적의 lightweight adaptation으로 구현.
    """
    rng = np.random.default_rng(seed)
    train_ids = set(train_ids)

    # 1. Training set에서 variable frequency 결정
    frequency = {}
    for row in imputation_params.itertuples(index=False):
        interval = float(row.median_sampling_min)
        if interval <= 15: frequency[row.variable] = "high"
        elif interval <= 8 * 60: frequency[row.variable] = "medium"
        else: frequency[row.variable] = "low"

    # Supplementary Table 3
    settings = {
        "high": {"search_h": 4, "length_h": [0.5, 1]},
        "medium": {"search_h": 36, "length_h": [12, 24]},
        "low": {"search_h": 48, "length_h": [16, 32]}
    }

    # 2. Raw non-pharma를 5-min grid로 interpolation
    raw = nonpharma_merged[["stay_id", "charttime", "variable", "valuenum"]].copy()
    raw = raw.loc[raw["stay_id"].isin(train_ids)]
    raw["charttime"] = pd.to_datetime(raw["charttime"], errors="coerce")
    raw["valuenum"] = pd.to_numeric(raw["valuenum"], errors="coerce")
    raw = raw.dropna(subset=["stay_id", "charttime", "variable", "valuenum"])

    # 3. Training CF onset 찾기
    st = state[["stay_id", "gridtime", "state"]].copy()
    st = st.loc[st["stay_id"].isin(train_ids)]
    st["gridtime"] = pd.to_datetime(st["gridtime"], errors="coerce")
    st = st.sort_values(["stay_id", "gridtime"])
    st["prev_state"] = st.groupby("stay_id")["state"].shift(1)
    onsets = st.loc[(st["state"] == "CF") & (st["prev_state"] != "CF"), ["stay_id", "gridtime"]]

    selected_shapelets = {}

    # 4. Variable별 shapelet discovery
    for variable, freq in frequency.items():
        v = raw.loc[raw["variable"] == variable].copy()
        if v.empty: continue

        search_h = settings[freq]["search_h"]
        candidate_cases = []
        candidate_controls = []

        # Case: CF onset 직전 trajectory
        for row in onsets.itertuples(index=False):
            g = v.loc[v["stay_id"] == row.stay_id, ["charttime", "valuenum"]].sort_values("charttime")
            if g.empty: continue

            end = row.gridtime - pd.Timedelta(minutes=5)
            start = end - pd.Timedelta(hours=search_h)
            g = g.loc[(g["charttime"] >= start) & (g["charttime"] <= end)]
            if len(g) < 2: continue

            s = g.set_index("charttime")["valuenum"].resample(f"{grid_minutes}min").median().interpolate(limit_direction="both")
            if len(s): candidate_cases.append(s)

        if len(candidate_cases) > n_cases:
            selected_idx = rng.choice(len(candidate_cases), size=n_cases, replace=False)
            candidate_cases = [candidate_cases[i] for i in selected_idx]

        # Control: CF가 한 번도 없는 training stays
        cf_stays = set(st.loc[st["state"] == "CF", "stay_id"])
        control_stays = list(train_ids - cf_stays)
        rng.shuffle(control_stays)

        for sid in control_stays:
            g = v.loc[v["stay_id"] == sid, ["charttime", "valuenum"]].sort_values("charttime")
            if g.empty: continue

            s = g.set_index("charttime")["valuenum"].resample(f"{grid_minutes}min").median().interpolate(limit_direction="both")
            need = int(search_h * 60 / grid_minutes)
            if len(s) < need: continue

            start_idx = rng.integers(0, len(s) - need + 1)
            candidate_controls.append(s.iloc[start_idx:start_idx + need])

            if len(candidate_controls) >= n_controls: break

        if not candidate_cases or not candidate_controls: continue

        # 5. Candidate shapelets 생성
        candidates = []

        for length_h in settings[freq]["length_h"]:
            length = int(length_h * 60 / grid_minutes)
            if length < 2: continue

            for series in candidate_cases:
                values = series.to_numpy(dtype=float)
                if len(values) < length: continue

                for start in range(0, len(values) - length + 1, max(1, length // 2)):
                    shapelet = values[start:start + length]
                    if np.isnan(shapelet).any(): continue

                    case_dist = []
                    control_dist = []

                    for x in candidate_cases:
                        arr = x.to_numpy(dtype=float)
                        if len(arr) < length: continue
                        d = min(np.linalg.norm(arr[i:i + length] - shapelet)
                                for i in range(len(arr) - length + 1))
                        case_dist.append(d)

                    for x in candidate_controls:
                        arr = x.to_numpy(dtype=float)
                        if len(arr) < length: continue
                        d = min(np.linalg.norm(arr[i:i + length] - shapelet)
                                for i in range(len(arr) - length + 1))
                        control_dist.append(d)

                    if not case_dist or not control_dist: continue

                    threshold = (np.median(case_dist) + np.median(control_dist)) / 2
                    acc1 = (np.mean(np.array(case_dist) <= threshold) + np.mean(np.array(control_dist) > threshold)) / 2
                    acc2 = (np.mean(np.array(case_dist) > threshold) + np.mean(np.array(control_dist) <= threshold)) / 2
                    candidates.append((max(acc1, acc2), shapelet))

        if not candidates: continue

        # 6. Accuracy top 100
        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:100]

        # 7. Min-max selection
        chosen = [candidates[0][1]]

        while len(chosen) < min(n_shapelets, len(candidates)):
            best_shapelet = None
            best_distance = -1

            for _, candidate in candidates:
                if any(np.array_equal(candidate, s) for s in chosen): continue
                same_length = [s for s in chosen if len(s) == len(candidate)]
                if not same_length: min_distance = np.inf
                else: min_distance = min(np.linalg.norm(candidate - s) for s in same_length)

                if min_distance > best_distance:
                    best_distance = min_distance
                    best_shapelet = candidate

            if best_shapelet is None: break
            chosen.append(best_shapelet)

        selected_shapelets[variable] = chosen

    # 8. 선택된 shapelet을 전체 patient에 적용
    full_raw = nonpharma_merged[["stay_id", "charttime", "variable", "valuenum"]].copy()
    full_raw["charttime"] = pd.to_datetime(full_raw["charttime"], errors="coerce")
    full_raw["valuenum"] = pd.to_numeric(full_raw["valuenum"], errors="coerce")
    full_raw = full_raw.dropna(subset=["stay_id", "charttime", "variable", "valuenum"])

    features = feature_grid[["stay_id", "gridtime"]].copy()
    features["gridtime"] = pd.to_datetime(features["gridtime"])

    for variable, shapelets in selected_shapelets.items():
        v = full_raw.loc[full_raw["variable"] == variable]

        for shapelet_id, shapelet in enumerate(shapelets):
            rows = []

            for sid, g in v.groupby("stay_id"):
                s = g.set_index("charttime")["valuenum"].sort_index().resample(f"{grid_minutes}min").median().interpolate(limit_direction="both")
                arr = s.to_numpy(dtype=float)
                times = s.index
                length = len(shapelet)

                if len(arr) < length: continue

                distances = np.full(len(arr), np.nan)
                for i in range(length - 1, len(arr)):
                    distances[i] = np.linalg.norm(arr[i - length + 1:i + 1] - shapelet)

                d = pd.DataFrame({"stay_id": sid, "gridtime": times, "distance": distances}).dropna()
                rows.append(d)

            if not rows: continue

            distance_df = pd.concat(rows, ignore_index=True)

            # 9. 최근 4h distance history를 60-min prediction grid에 align
            for lag_h in range(0, 5):
                shifted = distance_df.copy()
                shifted["gridtime"] = shifted["gridtime"] + pd.Timedelta(hours=lag_h)
                shifted = shifted.rename(columns={"distance": f"{variable}__shapelet_{shapelet_id}__dist_t-{lag_h}h"})

                features = pd.merge_asof(
                    features.sort_values("gridtime"),
                    shifted.sort_values("gridtime"),
                    on="gridtime",
                    by="stay_id",
                    direction="backward",
                    tolerance=pd.Timedelta(minutes=grid_minutes)
                )

    return features, selected_shapelets


# -----------------------------------------------------------------------------------------------------------------------
# 6. Time since admission
# -----------------------------------------------------------------------------------------------------------------------
def extract_time_since_admission(dynamic_grid: pd.DataFrame, feature_grid: pd.DataFrame):
    grid = dynamic_grid[["stay_id", "gridtime"]].copy()
    grid["gridtime"] = pd.to_datetime(grid["gridtime"], errors="coerce")
    grid = grid.dropna(subset=["stay_id", "gridtime"])
    first_time = grid.groupby("stay_id")["gridtime"].min().rename("grid_start")

    features = feature_grid[["stay_id", "gridtime"]].copy()
    features["gridtime"] = pd.to_datetime(features["gridtime"], errors="coerce")
    features = features.merge(first_time, on="stay_id", how="left", validate="many_to_one")
    features["time_since_admission_min"] = \
        ((features["gridtime"] - features["grid_start"]).dt.total_seconds() / 60).astype("float32")

    return features.drop(columns="grid_start")


# -----------------------------------------------------------------------------------------------------------------------
# 7. Make CSV with labels
# -----------------------------------------------------------------------------------------------------------------------
def make_feature_csv(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    train_ids,
    val_ids,
    test_ids,
    output_path
):
    result = features.merge(
        labels[["stay_id", "gridtime", "label"]],
        on=["stay_id", "gridtime"],
        how="left",
        validate="one_to_one"
    )

    # 실제 prediction sample만 사용
    result = result.loc[result["label"].notna()].copy()
    result["label"] = result["label"].astype("int8")

    # Train / Validation / Test 표시
    train_ids = set(train_ids)
    val_ids = set(val_ids)
    test_ids = set(test_ids)

    result["set"] = None
    result.loc[result["stay_id"].isin(train_ids), "set"] = "train"
    result.loc[result["stay_id"].isin(val_ids), "set"] = "validation"
    result.loc[result["stay_id"].isin(test_ids), "set"] = "test"

    result = result.loc[result["set"].notna()].copy()

    # 보기 편하게 기본 정보 먼저
    feature_cols = [
        c for c in result.columns
        if c not in ["stay_id", "gridtime", "set", "label"]
    ]

    result = result[
        ["stay_id", "gridtime", "set", "label"] + feature_cols
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    return result