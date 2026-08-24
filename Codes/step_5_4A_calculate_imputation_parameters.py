import numpy as np
import pandas as pd


# ============================================================
# 5-4A. Calculate Adaptive-Imputation Sampling Parameters
# ============================================================
#
# 원 논문에서 non-medication variable마다 사용하는:
#
#   m_i   = median sampling interval
#   iqr_i = IQR of sampling interval
#
# 를 현재 MIMIC-IV cohort에서 직접 계산한다.
#
# 중요:
# - 60-min grid에 binning된 값이 아니라,
#   Step 4-2A의 실제 measurement timestamp를 사용한다.
# - 서로 다른 ICU stay 사이의 시간 차이는 계산하지 않는다.
# - 같은 stay + variable 안에서 연속된 measurement 사이의
#   실제 시간 차이(minutes)만 사용한다.
# - 현재 분석 grid 범위(grid_start ~ grid_end) 안의 measurement만 사용한다.
#
# 논문과의 차이:
# 원 논문 MIMIC-III external validation에서는 HiRID에서 계산한
# parameter를 그대로 사용했지만, 현재 구현에서는 MIMIC-IV에서
# 직접 재계산하는 adaptation이다.
# ============================================================


def calculate_imputation_parameters(
    nonpharma_merged: pd.DataFrame,
    grid_stay_info: pd.DataFrame,
    stay_ids=None,
):

    required_events = {
        "stay_id",
        "charttime",
        "variable",
        "valuenum",
    }

    required_stays = {
        "stay_id",
        "grid_start",
        "grid_end",
    }

    missing = required_events - set(nonpharma_merged.columns)
    if missing:
        raise ValueError(
            f"nonpharma_merged에 필요한 column이 없습니다: {missing}"
        )

    missing = required_stays - set(grid_stay_info.columns)
    if missing:
        raise ValueError(
            f"grid_stay_info에 필요한 column이 없습니다: {missing}"
        )


    # --------------------------------------------------------
    # 1. 사용할 stay 선택
    # --------------------------------------------------------

    events = nonpharma_merged[
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

    if stay_ids is not None:
        stay_ids = set(stay_ids)

        events = events.loc[
            events["stay_id"].isin(stay_ids)
        ].copy()

        windows = windows.loc[
            windows["stay_id"].isin(stay_ids)
        ].copy()


    # --------------------------------------------------------
    # 2. Datetime 정리 + grid window 연결
    # --------------------------------------------------------

    events["charttime"] = pd.to_datetime(
        events["charttime"],
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

    events = events.dropna(
        subset=[
            "stay_id",
            "charttime",
            "variable",
            "valuenum",
        ]
    ).copy()

    events = events.merge(
        windows,
        on="stay_id",
        how="inner",
    )

    events = events.loc[
        (events["charttime"] >= events["grid_start"])
        & (events["charttime"] <= events["grid_end"])
    ].copy()


    # --------------------------------------------------------
    # 3. 같은 stay + variable 안에서 시간순 정렬
    # --------------------------------------------------------

    events = events.sort_values(
        [
            "stay_id",
            "variable",
            "charttime",
        ]
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # 4. consecutive sampling interval 계산
    # --------------------------------------------------------

    events["sampling_interval_min"] = (
        events.groupby(
            [
                "stay_id",
                "variable",
            ],
            sort=False,
        )["charttime"]
        .diff()
        .dt.total_seconds()
        / 60.0
    )


    # 첫 measurement는 이전 값이 없으므로 interval 없음.
    # 0 이하 interval은 사용하지 않는다.
    intervals = events.loc[
        events["sampling_interval_min"].notna()
        & (events["sampling_interval_min"] > 0)
    ].copy()


    # --------------------------------------------------------
    # 5. variable별 median / IQR
    # --------------------------------------------------------

    def q1(x):
        return x.quantile(0.25)

    def q3(x):
        return x.quantile(0.75)

    params = (
        intervals.groupby(
            "variable",
            as_index=False,
        )
        .agg(
            n_intervals=(
                "sampling_interval_min",
                "size",
            ),
            median_sampling_min=(
                "sampling_interval_min",
                "median",
            ),
            q1_sampling_min=(
                "sampling_interval_min",
                q1,
            ),
            q3_sampling_min=(
                "sampling_interval_min",
                q3,
            ),
        )
    )

    params["iqr_sampling_min"] = (
        params["q3_sampling_min"]
        - params["q1_sampling_min"]
    )


    # --------------------------------------------------------
    # 6. 논문 adaptive-imputation에서 바로 쓰는 시간값
    # --------------------------------------------------------
    #
    # forward-fill threshold:
    #   m_i + iqr_i
    #
    # return-to-median 관련 horizon:
    #   2 * (m_i + 2 * iqr_i)
    # --------------------------------------------------------

    params["ffill_threshold_min"] = (
        params["median_sampling_min"]
        + params["iqr_sampling_min"]
    )

    params["return_horizon_min"] = (
        2.0
        * (
            params["median_sampling_min"]
            + 2.0 * params["iqr_sampling_min"]
        )
    )


    # --------------------------------------------------------
    # 7. measurement / stay coverage도 붙이기
    # --------------------------------------------------------

    coverage = (
        events.groupby(
            "variable",
            as_index=False,
        )
        .agg(
            measurements=("valuenum", "size"),
            stays_with_measurement=("stay_id", "nunique"),
        )
    )

    params = coverage.merge(
        params,
        on="variable",
        how="left",
    )

    params = params.sort_values(
        "median_sampling_min",
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # 8. 이상 여부 간단 audit
    # --------------------------------------------------------

    total_stays = int(
        windows["stay_id"].nunique()
    )

    params["stay_coverage_pct"] = (
        100.0
        * params["stays_with_measurement"]
        / total_stays
    )


    # --------------------------------------------------------
    # 9. 출력
    # --------------------------------------------------------

    print("=" * 70)
    print("5-4A. Adaptive-Imputation Sampling Parameters")
    print("=" * 70)

    print(
        "Parameter source: current MIMIC-IV cohort "
        "(raw measurement timestamps)"
    )

    print(
        f"Stays used: {total_stays}"
    )

    print(
        f"Variables: {len(params)}"
    )

    print(
        f"Sampling intervals used: {len(intervals)}"
    )

    print("\n[Variable-specific parameters]")

    print(
        params.to_string(
            index=False,
            formatters={
                "stay_coverage_pct":
                    lambda x: f"{x:.2f}",

                "median_sampling_min":
                    lambda x: f"{x:.2f}",

                "q1_sampling_min":
                    lambda x: f"{x:.2f}",

                "q3_sampling_min":
                    lambda x: f"{x:.2f}",

                "iqr_sampling_min":
                    lambda x: f"{x:.2f}",

                "ffill_threshold_min":
                    lambda x: f"{x:.2f}",

                "return_horizon_min":
                    lambda x: f"{x:.2f}",
            },
        )
    )


    print("\n[Important]")
    print(
        "Sampling intervals are calculated from ORIGINAL measurement times, "
        "not from the 60-min resampled grid."
    )

    print(
        "Intervals are calculated only within the same stay and variable."
    )

    print(
        "ffill_threshold_min = median sampling interval + IQR."
    )

    print(
        "return_horizon_min = 2 * (median + 2 * IQR)."
    )

    print(
        "This is a MIMIC-IV-specific adaptation: parameters are recomputed "
        "from the current cohort rather than reused from HiRID."
    )

    print(
        "For a final train/validation/test experiment, these parameters "
        "should be recomputed from TRAINING stays only."
    )

    print("=" * 70)


    report = {
        "stays_used":
            total_stays,

        "variables":
            int(len(params)),

        "sampling_intervals":
            int(len(intervals)),
    }


    return (
        params,
        intervals,
        report,
    )