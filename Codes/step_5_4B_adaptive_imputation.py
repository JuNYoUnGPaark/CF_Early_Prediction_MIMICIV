import numpy as np
import pandas as pd


# ============================================================
# 5-4B. Patient-centered Adaptive Imputation
# ============================================================
#
# Paper rule (non-pharmaceutical variables):
#
# 1) Before the first measurement / no measurement
#       -> variable-specific default value
#
# 2) After a measurement, if
#       time since last measurement < m_i + IQR_i
#       -> forward fill
#
# 3) Otherwise
#       -> linearly return toward the median of measurements
#          in the previous 2 * (m_i + 2*IQR_i) minutes,
#          where the historical window is measured from the
#          moment this "return" mode begins.
#
#       -> return duration is also
#          2 * (m_i + 2*IQR_i) minutes.
#
# 4) After reaching that historical median
#       -> hold that median until a new measurement arrives.
#
# IMPORTANT
# ------------------------------------------------------------
# - This is the MODEL-DATA imputation branch.
# - It NEVER uses future measurements.
# - Lactate for CF annotation must be prepared separately using
#   the paper's annotation-specific interpolation rules.
#
# MIMIC-IV adaptation
# ------------------------------------------------------------
# - Main grid is 60 min rather than the paper's 5 min.
# - m_i / IQR_i are the Step 5-4A parameters recomputed from
#   the current MIMIC-IV cohort.
# - Time comparisons remain in real minutes; they are NOT
#   rounded to 60 min.
# ============================================================


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
    # MIMIC glucose values were converted mg/dL -> mmol/L in 4-1A.
    # The author's numerical default is 5.
    "Blood Glucose": 5.0,
    "C-reactive protein": 4.0,
}


def _prepare_cardiac_output_defaults(
    stay_ids,
    height_weight_events: pd.DataFrame,
):
    """
    Supplementary Table 4:
      Cardiac Output default
      = 3.5 * 0.007184 * weight^0.425 * height^0.725

    For each stay:
      - use the earliest available Height and Weight
      - if one is unavailable, fill that static quantity with the
        cohort mean of the stay-level available values

    This fallback follows the paper's general rule that missing
    continuous static variables are imputed using a training-data
    mean. In a final split experiment, means should be estimated
    from training stays only.
    """

    required = {
        "stay_id",
        "charttime",
        "variable",
        "value",
    }

    missing = required - set(height_weight_events.columns)
    if missing:
        raise ValueError(
            "height_weight_events에 필요한 column이 없습니다: "
            f"{missing}"
        )

    hw = height_weight_events[
        height_weight_events["stay_id"].isin(set(stay_ids))
    ].copy()

    hw["charttime"] = pd.to_datetime(
        hw["charttime"],
        errors="coerce",
    )

    hw["value"] = pd.to_numeric(
        hw["value"],
        errors="coerce",
    )

    hw = hw.dropna(
        subset=["stay_id", "charttime", "variable", "value"]
    ).copy()

    hw = hw.loc[
        hw["variable"].isin(["Height", "Weight"])
    ].sort_values(
        ["stay_id", "variable", "charttime"]
    )

    first_hw = (
        hw.drop_duplicates(
            subset=["stay_id", "variable"],
            keep="first",
        )
        .pivot(
            index="stay_id",
            columns="variable",
            values="value",
        )
        .reindex(stay_ids)
    )

    if "Height" not in first_hw.columns:
        first_hw["Height"] = np.nan

    if "Weight" not in first_hw.columns:
        first_hw["Weight"] = np.nan

    # continuous static fallback -> cohort mean
    mean_height = float(first_hw["Height"].mean())
    mean_weight = float(first_hw["Weight"].mean())

    missing_height = int(first_hw["Height"].isna().sum())
    missing_weight = int(first_hw["Weight"].isna().sum())

    first_hw["Height"] = first_hw["Height"].fillna(mean_height)
    first_hw["Weight"] = first_hw["Weight"].fillna(mean_weight)

    first_hw["Cardiac Output"] = (
        3.5
        * 0.007184
        * np.power(first_hw["Weight"], 0.425)
        * np.power(first_hw["Height"], 0.725)
    )

    defaults = first_hw["Cardiac Output"].astype(float).to_dict()

    report = {
        "mean_height_used_for_fallback": mean_height,
        "mean_weight_used_for_fallback": mean_weight,
        "stays_missing_height_filled": missing_height,
        "stays_missing_weight_filled": missing_weight,
        "co_default_min": float(first_hw["Cardiac Output"].min()),
        "co_default_median": float(first_hw["Cardiac Output"].median()),
        "co_default_max": float(first_hw["Cardiac Output"].max()),
    }

    return defaults, report


def _latest_measurement_time_by_grid_cell(
    nonpharma_merged: pd.DataFrame,
    grid_stay_info: pd.DataFrame,
    grid_minutes: int,
):
    """
    Step 5-2와 동일한 causal binning rule을 사용해,
    각 observed cell에 들어온 raw measurement 중
    가장 늦은 실제 charttime을 저장한다.

    measurement -> first grid point at or after measurement
    """

    e = nonpharma_merged[
        [
            "stay_id",
            "charttime",
            "variable",
            "valuenum",
        ]
    ].copy()

    windows = grid_stay_info[
        [
            "stay_id",
            "grid_start",
            "grid_end",
        ]
    ].copy()

    e["charttime"] = pd.to_datetime(
        e["charttime"],
        errors="coerce",
    )

    windows["grid_start"] = pd.to_datetime(
        windows["grid_start"],
        errors="coerce",
    )

    windows["grid_end"] = pd.to_datetime(
        windows["grid_end"],
        errors="coerce",
    )

    e = e.dropna(
        subset=["stay_id", "charttime", "variable", "valuenum"]
    )

    e = e.merge(
        windows,
        on="stay_id",
        how="inner",
    )

    e = e.loc[
        (e["charttime"] >= e["grid_start"])
        & (e["charttime"] <= e["grid_end"])
    ].copy()

    interval_ns = (
        int(grid_minutes)
        * 60
        * 1_000_000_000
    )

    duration_ns = (
        e["grid_end"].astype("int64")
        - e["grid_start"].astype("int64")
    )

    last_idx = duration_ns // interval_ns

    e["last_gridtime"] = (
        e["grid_start"]
        + pd.to_timedelta(
            last_idx * grid_minutes,
            unit="min",
        )
    )

    delta_ns = (
        e["charttime"].astype("int64")
        - e["grid_start"].astype("int64")
    )

    grid_idx = (
        delta_ns
        + interval_ns
        - 1
    ) // interval_ns

    e["gridtime"] = (
        e["grid_start"]
        + pd.to_timedelta(
            grid_idx * grid_minutes,
            unit="min",
        )
    )

    e = e.loc[
        e["gridtime"] <= e["last_gridtime"]
    ].copy()

    latest = (
        e.groupby(
            ["stay_id", "gridtime", "variable"],
            as_index=False,
            sort=False,
        )["charttime"]
        .max()
        .rename(
            columns={
                "charttime": "last_measurement_time"
            }
        )
    )

    return latest


def adaptive_impute_nonpharma(
    time_grid: pd.DataFrame,
    grid_stay_info: pd.DataFrame,
    raw_grid_values: pd.DataFrame,
    nonpharma_merged: pd.DataFrame,
    imputation_params: pd.DataFrame,
    height_weight_events: pd.DataFrame,
    grid_minutes: int = 60,
):
    """
    Returns
    -------
    imputed_grid : pd.DataFrame
        stay_id, gridtime + 12 imputed non-pharma variable columns

    imputation_report : pd.DataFrame
        variable별 observed/default/ffill/return/hold 비율

    report : dict
        전체 요약
    """

    # ========================================================
    # 0. Required columns
    # ========================================================

    required_grid = {"stay_id", "gridtime"}

    required_raw = {
        "stay_id",
        "gridtime",
        "variable",
        "valuenum",
    }

    required_params = {
        "variable",
        "median_sampling_min",
        "iqr_sampling_min",
        "ffill_threshold_min",
        "return_horizon_min",
    }

    for name, df, required in [
        ("time_grid", time_grid, required_grid),
        ("raw_grid_values", raw_grid_values, required_raw),
        ("imputation_params", imputation_params, required_params),
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{name}에 필요한 column이 없습니다: {missing}"
            )


    # ========================================================
    # 1. Base grid
    # ========================================================

    grid = time_grid[
        ["stay_id", "gridtime"]
    ].copy()

    grid["gridtime"] = pd.to_datetime(
        grid["gridtime"],
        errors="coerce",
    )

    grid = (
        grid.dropna(subset=["stay_id", "gridtime"])
        .sort_values(["stay_id", "gridtime"])
        .reset_index(drop=True)
    )

    grid["_grid_row"] = np.arange(
        len(grid),
        dtype=np.int64,
    )

    stay_ids = (
        grid["stay_id"]
        .drop_duplicates()
        .tolist()
    )


    # ========================================================
    # 2. Parameter dictionary
    # ========================================================

    params = (
        imputation_params
        .set_index("variable")
    )

    variables = sorted(
        raw_grid_values["variable"]
        .dropna()
        .unique()
        .tolist()
    )

    missing_params = [
        v for v in variables
        if v not in params.index
    ]

    if missing_params:
        raise ValueError(
            "Imputation parameter가 없는 variable: "
            f"{missing_params}"
        )


    # ========================================================
    # 3. Default values
    # ========================================================

    co_defaults, co_default_report = (
        _prepare_cardiac_output_defaults(
            stay_ids=stay_ids,
            height_weight_events=height_weight_events,
        )
    )

    missing_defaults = [
        v for v in variables
        if (
            v != "Cardiac Output"
            and v not in DEFAULT_VALUES
        )
    ]

    if missing_defaults:
        raise ValueError(
            "Default value가 정의되지 않은 variable: "
            f"{missing_defaults}"
        )


    # ========================================================
    # 4. Actual latest measurement time for each observed bin
    # ========================================================

    latest_time = _latest_measurement_time_by_grid_cell(
        nonpharma_merged=nonpharma_merged,
        grid_stay_info=grid_stay_info,
        grid_minutes=grid_minutes,
    )

    observed = raw_grid_values[
        [
            "stay_id",
            "gridtime",
            "variable",
            "valuenum",
        ]
    ].copy()

    observed["gridtime"] = pd.to_datetime(
        observed["gridtime"],
        errors="coerce",
    )

    observed = observed.merge(
        latest_time,
        on=["stay_id", "gridtime", "variable"],
        how="left",
        validate="one_to_one",
    )

    if observed["last_measurement_time"].isna().any():
        n_missing = int(
            observed["last_measurement_time"]
            .isna()
            .sum()
        )

        raise ValueError(
            "Observed grid cell 중 실제 measurement time을 "
            f"복구하지 못한 row가 있습니다: {n_missing}"
        )


    # ========================================================
    # 5. Raw exact measurements for history-median target
    # ========================================================

    raw_history = nonpharma_merged[
        [
            "stay_id",
            "charttime",
            "variable",
            "valuenum",
        ]
    ].copy()

    raw_history["charttime"] = pd.to_datetime(
        raw_history["charttime"],
        errors="coerce",
    )

    raw_history["valuenum"] = pd.to_numeric(
        raw_history["valuenum"],
        errors="coerce",
    )

    raw_history = raw_history.dropna(
        subset=[
            "stay_id",
            "charttime",
            "variable",
            "valuenum",
        ]
    ).copy()

    # Grid가 first HR에서 시작하므로, historical median도
    # 분석 grid 시작 이전의 measurement를 사용하지 않는다.
    history_windows = grid_stay_info[
        ["stay_id", "grid_start", "grid_end"]
    ].copy()

    history_windows["grid_start"] = pd.to_datetime(
        history_windows["grid_start"],
        errors="coerce",
    )

    history_windows["grid_end"] = pd.to_datetime(
        history_windows["grid_end"],
        errors="coerce",
    )

    raw_history = raw_history.merge(
        history_windows,
        on="stay_id",
        how="inner",
    )

    raw_history = raw_history.loc[
        (raw_history["charttime"] >= raw_history["grid_start"])
        & (raw_history["charttime"] <= raw_history["grid_end"])
    ].copy()

    raw_history = raw_history.sort_values(
        ["stay_id", "variable", "charttime"]
    )


    # ========================================================
    # 6. Stay row slices
    # ========================================================

    stay_slices = {}

    for sid, g in grid.groupby(
        "stay_id",
        sort=False,
    ):
        stay_slices[sid] = (
            g["_grid_row"].to_numpy(),
            g["gridtime"].to_numpy(
                dtype="datetime64[ns]"
            ),
        )


    # ========================================================
    # 7. Output base
    # ========================================================

    imputed_grid = grid[
        ["stay_id", "gridtime"]
    ].copy()

    report_rows = []


    # ========================================================
    # 8. Variable-wise adaptive imputation
    # ========================================================

    for variable in variables:

        p = params.loc[variable]

        ffill_threshold = float(
            p["ffill_threshold_min"]
        )

        return_horizon = float(
            p["return_horizon_min"]
        )

        if (
            not np.isfinite(ffill_threshold)
            or ffill_threshold < 0
        ):
            raise ValueError(
                f"{variable}: invalid ffill threshold "
                f"{ffill_threshold}"
            )

        if (
            not np.isfinite(return_horizon)
            or return_horizon <= 0
        ):
            raise ValueError(
                f"{variable}: invalid return horizon "
                f"{return_horizon}"
            )


        # ----------------------------------------------------
        # observed rows for this variable
        # ----------------------------------------------------

        obs_v = observed.loc[
            observed["variable"] == variable
        ].copy()

        obs_by_stay = {
            sid: g.sort_values("gridtime")
            for sid, g in obs_v.groupby(
                "stay_id",
                sort=False,
            )
        }


        # ----------------------------------------------------
        # raw history measurements for this variable
        # ----------------------------------------------------

        hist_v = raw_history.loc[
            raw_history["variable"] == variable
        ].copy()

        hist_by_stay = {
            sid: (
                g["charttime"].to_numpy(
                    dtype="datetime64[ns]"
                ),
                g["valuenum"].to_numpy(
                    dtype=float
                ),
            )
            for sid, g in hist_v.groupby(
                "stay_id",
                sort=False,
            )
        }


        values_all = np.empty(
            len(grid),
            dtype=np.float32,
        )

        source_all = np.empty(
            len(grid),
            dtype=np.int8,
        )

        # source code
        # 0 observed
        # 1 default_before_first_or_none
        # 2 forward_fill
        # 3 linear_return
        # 4 patient_median_hold


        # ----------------------------------------------------
        # stay-by-stay
        # ----------------------------------------------------

        for sid in stay_ids:

            row_idx, grid_times = stay_slices[sid]
            n = len(row_idx)

            vals = np.empty(
                n,
                dtype=np.float32,
            )

            src = np.empty(
                n,
                dtype=np.int8,
            )


            # -----------------------------------------------
            # stay-specific default
            # -----------------------------------------------

            if variable == "Cardiac Output":
                default_value = float(
                    co_defaults[sid]
                )
            else:
                default_value = float(
                    DEFAULT_VALUES[variable]
                )


            # -----------------------------------------------
            # no measurement -> default for whole stay
            # -----------------------------------------------

            if sid not in obs_by_stay:

                vals[:] = default_value
                src[:] = 1

                values_all[row_idx] = vals
                source_all[row_idx] = src
                continue


            # -----------------------------------------------
            # observed grid index -> value/time
            # -----------------------------------------------

            og = obs_by_stay[sid]

            # Map timestamps to local grid index
            grid_pos = pd.Series(
                np.arange(n),
                index=pd.DatetimeIndex(grid_times),
            )

            obs_local_idx = (
                grid_pos.reindex(
                    pd.DatetimeIndex(
                        og["gridtime"]
                    )
                )
                .to_numpy()
            )

            if np.isnan(obs_local_idx).any():
                raise ValueError(
                    f"{variable}, stay {sid}: "
                    "observed gridtime이 time_grid와 맞지 않습니다."
                )

            obs_local_idx = (
                obs_local_idx.astype(int)
            )

            obs_value = og["valuenum"].to_numpy(
                dtype=float
            )

            obs_actual_time = (
                og["last_measurement_time"]
                .to_numpy(
                    dtype="datetime64[ns]"
                )
            )

            obs_lookup = {
                int(i): (float(v), t)
                for i, v, t in zip(
                    obs_local_idx,
                    obs_value,
                    obs_actual_time,
                )
            }


            # -----------------------------------------------
            # raw exact history for target median
            # -----------------------------------------------

            hist_times, hist_values = (
                hist_by_stay.get(
                    sid,
                    (
                        np.array(
                            [],
                            dtype="datetime64[ns]",
                        ),
                        np.array(
                            [],
                            dtype=float,
                        ),
                    )
                )
            )


            # -----------------------------------------------
            # state variables
            # -----------------------------------------------

            have_measurement = False

            last_value = np.nan
            last_measurement_time = np.datetime64(
                "NaT"
            )

            target_median = np.nan
            return_entry_time = np.datetime64(
                "NaT"
            )


            for j in range(n):

                t = grid_times[j]


                # -------------------------------------------
                # Real observed value at this grid point
                # -------------------------------------------

                if j in obs_lookup:

                    last_value, last_measurement_time = (
                        obs_lookup[j]
                    )

                    vals[j] = last_value
                    src[j] = 0

                    have_measurement = True

                    # new measurement -> return mode reset
                    target_median = np.nan
                    return_entry_time = np.datetime64(
                        "NaT"
                    )

                    continue


                # -------------------------------------------
                # Before first measurement
                # -------------------------------------------

                if not have_measurement:

                    vals[j] = default_value
                    src[j] = 1
                    continue


                # -------------------------------------------
                # Minutes since the latest REAL measurement
                # -------------------------------------------

                elapsed_min = float(
                    (
                        t
                        - last_measurement_time
                    )
                    / np.timedelta64(1, "m")
                )


                # -------------------------------------------
                # Forward filling
                # paper: strictly less than m + IQR
                # -------------------------------------------

                if elapsed_min < ffill_threshold:

                    vals[j] = last_value
                    src[j] = 2
                    continue


                # -------------------------------------------
                # Enter return-to-history-median mode
                # -------------------------------------------

                if np.isnan(target_median):

                    return_entry_time = (
                        last_measurement_time
                        + np.timedelta64(
                            int(round(
                                ffill_threshold
                                * 60
                            )),
                            "s",
                        )
                    )

                    hist_start = (
                        return_entry_time
                        - np.timedelta64(
                            int(round(
                                return_horizon
                                * 60
                            )),
                            "s",
                        )
                    )

                    # only past/raw measurements
                    mask = (
                        (hist_times >= hist_start)
                        & (hist_times <= return_entry_time)
                    )

                    candidate_values = (
                        hist_values[mask]
                    )

                    if len(candidate_values):
                        target_median = float(
                            np.median(
                                candidate_values
                            )
                        )
                    else:
                        # practically the latest measurement
                        # should normally be inside the window.
                        target_median = float(
                            last_value
                        )


                # -------------------------------------------
                # Linear return
                # -------------------------------------------

                since_entry_min = float(
                    (
                        t
                        - return_entry_time
                    )
                    / np.timedelta64(1, "m")
                )

                fraction = (
                    since_entry_min
                    / return_horizon
                )

                fraction = min(
                    max(fraction, 0.0),
                    1.0,
                )

                vals[j] = (
                    last_value
                    + fraction
                    * (
                        target_median
                        - last_value
                    )
                )

                if fraction < 1.0:
                    src[j] = 3
                else:
                    src[j] = 4


            values_all[row_idx] = vals
            source_all[row_idx] = src


        # ----------------------------------------------------
        # Save imputed variable
        # ----------------------------------------------------

        imputed_grid[variable] = values_all


        # ----------------------------------------------------
        # Source summary
        # ----------------------------------------------------

        n_total = int(len(source_all))

        counts = {
            "observed": int((source_all == 0).sum()),
            "default": int((source_all == 1).sum()),
            "forward_fill": int((source_all == 2).sum()),
            "linear_return": int((source_all == 3).sum()),
            "median_hold": int((source_all == 4).sum()),
        }

        report_rows.append({
            "variable": variable,
            "total_grid_points": n_total,
            **{
                f"{k}_points": v
                for k, v in counts.items()
            },
            **{
                f"{k}_pct":
                    100.0 * v / n_total
                for k, v in counts.items()
            },
            "final_missing_points":
                int(
                    np.isnan(
                        values_all
                    ).sum()
                ),
            "final_min":
                float(
                    np.nanmin(values_all)
                ),
            "final_median":
                float(
                    np.nanmedian(values_all)
                ),
            "final_max":
                float(
                    np.nanmax(values_all)
                ),
        })


    # ========================================================
    # 9. Final report
    # ========================================================

    imputation_report = (
        pd.DataFrame(report_rows)
        .sort_values(
            "variable"
        )
        .reset_index(drop=True)
    )

    total_missing = int(
        imputation_report[
            "final_missing_points"
        ].sum()
    )


    print("=" * 70)
    print("5-4B. Patient-centered Adaptive Imputation")
    print("=" * 70)

    print(
        f"Main grid: {grid_minutes} min"
    )

    print(
        f"Grid points: {len(grid)}"
    )

    print(
        f"Variables imputed: {len(variables)}"
    )

    print(
        "Future measurements used: NO"
    )

    print(
        f"Final missing cells: {total_missing}"
    )


    print("\n[Imputation source by variable]")

    display_cols = [
        "variable",
        "observed_pct",
        "default_pct",
        "forward_fill_pct",
        "linear_return_pct",
        "median_hold_pct",
        "final_missing_points",
        "final_min",
        "final_median",
        "final_max",
    ]

    print(
        imputation_report[
            display_cols
        ].to_string(
            index=False,
            formatters={
                "observed_pct":
                    lambda x: f"{x:.2f}",
                "default_pct":
                    lambda x: f"{x:.2f}",
                "forward_fill_pct":
                    lambda x: f"{x:.2f}",
                "linear_return_pct":
                    lambda x: f"{x:.2f}",
                "median_hold_pct":
                    lambda x: f"{x:.2f}",
                "final_min":
                    lambda x: f"{x:.4g}",
                "final_median":
                    lambda x: f"{x:.4g}",
                "final_max":
                    lambda x: f"{x:.4g}",
            },
        )
    )


    print("\n[Cardiac Output default preparation]")

    for k, v in co_default_report.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


    print("\n[Example imputed grid]")

    print(
        imputed_grid
        .head(20)
        .to_string(index=False)
    )


    print("\n[Important]")

    print(
        "Observed values are the 60-min binned values from Step 5-2."
    )

    print(
        "Measurement age is calculated from the latest REAL raw "
        "measurement timestamp contributing to that grid cell."
    )

    print(
        "Historical-median targets use only raw measurements available "
        "before the return mode begins."
    )

    print(
        "This imputation output is for MODEL DATA. "
        "Do not use its Lactate column for CF state annotation."
    )

    print(
        "For final train/validation/test experiments, sampling parameters "
        "and static fallback means should be estimated on TRAINING stays only."
    )

    print("=" * 70)


    report = {
        "grid_minutes": int(grid_minutes),
        "grid_points": int(len(grid)),
        "variables": int(len(variables)),
        "final_missing_cells": total_missing,
        **co_default_report,
    }

    return (
        imputed_grid,
        imputation_report,
        report,
    )