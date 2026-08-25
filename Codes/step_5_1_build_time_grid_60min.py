import numpy as np
import pandas as pd

def build_time_grid(
    stays: pd.DataFrame,
    nonpharma_merged: pd.DataFrame,
    grid_minutes: int = 60,
    comparison_minutes: int = 5,
    max_days: int = 28,
):
    # ICU stay 내 Heart Rate만 사용
    hr = nonpharma_merged.loc[
        nonpharma_merged["variable"] == "Heart Rate",
        [
            "stay_id",
            "charttime",
        ]
    ].copy()

    hr["charttime"] = pd.to_datetime(
        hr["charttime"],
        errors="coerce"
    )

    hr = hr.dropna(subset=["charttime"])


    stay_windows = stays[
        [
            "stay_id",
            "intime",
            "outtime",
        ]
    ].copy()

    stay_windows["intime"] = pd.to_datetime(
        stay_windows["intime"],
        errors="coerce"
    )

    stay_windows["outtime"] = pd.to_datetime(
        stay_windows["outtime"],
        errors="coerce"
    )

    hr = hr.merge(
        stay_windows,
        on="stay_id",
        how="inner"
    )

    hr = hr.loc[
        (hr["charttime"] >= hr["intime"])
        & (hr["charttime"] <= hr["outtime"])
    ].copy()


    # Stay별 first / last HR
    hr_window = (
        hr.groupby("stay_id")
        .agg(
            first_hr=("charttime", "min"),
            last_hr=("charttime", "max"),
        )
        .reset_index()
    )

    # Grid start / end
    stay_info = stay_windows.merge(
        hr_window,
        on="stay_id",
        how="left"
    )

    stay_info["max_end"] = (
        stay_info["intime"]
        + pd.to_timedelta(max_days, unit="D")
    )

    stay_info["grid_start"] = stay_info["first_hr"]

    stay_info["grid_end"] = stay_info[
        [
            "last_hr",
            "outtime",
            "max_end",
        ]
    ].min(axis=1)

    stay_info["truncated_at_28d"] = (
        stay_info["last_hr"]
        > stay_info["max_end"]
    )

    # 4. Valid stay
    valid = (
        stay_info["grid_start"].notna()
        & stay_info["grid_end"].notna()
        & (
            stay_info["grid_end"]
            >= stay_info["grid_start"]
        )
    )

    invalid_stays = stay_info.loc[
        ~valid,
        [
            "stay_id",
            "intime",
            "outtime",
            "first_hr",
            "last_hr",
            "grid_start",
            "grid_end",
        ]
    ].copy()

    valid_stays = stay_info.loc[
        valid
    ].copy()

    # 실제 60-min grid 생성
    freq = f"{grid_minutes}min"

    grids = []

    for row in valid_stays.itertuples(index=False):

        times = pd.date_range(
            start=row.grid_start,
            end=row.grid_end,
            freq=freq,
        )

        grids.append(
            pd.DataFrame({
                "stay_id": row.stay_id,
                "gridtime": times,
            })
        )

    if grids:
        grid = pd.concat(
            grids,
            ignore_index=True
        )
    else:
        grid = pd.DataFrame(
            columns=[
                "stay_id",
                "gridtime",
            ]
        )

    # 실제 grid point 수
    actual_counts = (
        grid.groupby("stay_id")
        .size()
        .rename("grid_points")
        .reset_index()
    )

    valid_stays = valid_stays.merge(
        actual_counts,
        on="stay_id",
        how="left"
    )

    valid_stays["grid_points"] = (
        valid_stays["grid_points"]
        .fillna(0)
        .astype(int)
    )

    # 7. 같은 stay window를 5-min으로 만들었다면 몇 point인지 계산만 함
    duration_minutes = (
        (
            valid_stays["grid_end"]
            - valid_stays["grid_start"]
        )
        .dt.total_seconds()
        / 60.0
    )

    comparison_col = (
        f"equivalent_{comparison_minutes}min_points"
    )

    valid_stays[comparison_col] = (
        np.floor(
            duration_minutes
            / comparison_minutes
        )
        .astype(int)
        + 1
    )

    report = {
        "grid_minutes":
            int(grid_minutes),

        "comparison_minutes":
            int(comparison_minutes),

        "input_stays":
            int(len(stays)),

        "valid_grid_stays":
            int(len(valid_stays)),

        "invalid_grid_stays":
            int(len(invalid_stays)),

        "truncated_at_28d":
            int(
                valid_stays[
                    "truncated_at_28d"
                ].sum()
            ),

        "total_grid_points":
            int(len(grid)),

        "median_grid_points_per_stay":
            float(
                valid_stays[
                    "grid_points"
                ].median()
            ),

        "max_grid_points_per_stay":
            int(
                valid_stays[
                    "grid_points"
                ].max()
            ),

        f"equivalent_total_{comparison_minutes}min_points":
            int(
                valid_stays[
                    comparison_col
                ].sum()
            ),
    }


    print("=" * 70)
    print("5-1. Patient Time Grid")
    print("=" * 70)

    print(
        f"Main grid interval: "
        f"{grid_minutes} min"
    )

    print(
        f"Comparison interval: "
        f"{comparison_minutes} min"
    )

    print(
        f"Input stays: "
        f"{report['input_stays']}"
    )

    print(
        f"Stays with valid grid: "
        f"{report['valid_grid_stays']}"
    )

    print(
        f"Invalid grid stays: "
        f"{report['invalid_grid_stays']}"
    )

    print(
        f"Stays truncated at 28 days: "
        f"{report['truncated_at_28d']}"
    )

    print(
        f"Total {grid_minutes}-min grid points: "
        f"{report['total_grid_points']}"
    )

    print(
        f"Median grid points per stay: "
        f"{report['median_grid_points_per_stay']:.1f}"
    )

    print(
        f"Max grid points per stay: "
        f"{report['max_grid_points_per_stay']}"
    )

    print(
        f"Equivalent total {comparison_minutes}-min "
        f"grid points: "
        f"{report[f'equivalent_total_{comparison_minutes}min_points']}"
    )


    print("\n[Example stay windows]")

    print(
        valid_stays[
            [
                "stay_id",
                "intime",
                "first_hr",
                "last_hr",
                "grid_start",
                "grid_end",
                "truncated_at_28d",
                "grid_points",
                comparison_col,
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


    if len(invalid_stays):
        print("\n[Invalid grid stays]")

        print(
            invalid_stays
            .head(20)
            .to_string(index=False)
        )


    print("\n[Important]")

    print(
        f"The actual pipeline grid is now {grid_minutes} minutes."
    )

    print(
        f"The {comparison_minutes}-min grid is NOT materialized here; "
        "only its hypothetical point count is calculated for audit."
    )

    print(
        "No measurements were assigned and no imputation was performed."
    )

    print("=" * 70)


    return (
        grid,
        valid_stays,
        invalid_stays,
        report,
    )