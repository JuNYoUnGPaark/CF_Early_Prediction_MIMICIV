import numpy as np
import pandas as pd

def _assign_to_grid(
    events: pd.DataFrame,
    stay_windows: pd.DataFrame,
    grid_minutes: int,
):

    e = events.merge(
        stay_windows[
            [
                "stay_id",
                "grid_start",
                "grid_end",
            ]
        ],
        on="stay_id",
        how="inner",
    ).copy()

    e["charttime"] = pd.to_datetime(
        e["charttime"],
        errors="coerce",
    )

    e["grid_start"] = pd.to_datetime(
        e["grid_start"],
        errors="coerce",
    )

    e["grid_end"] = pd.to_datetime(
        e["grid_end"],
        errors="coerce",
    )

    # 분석 window 안 measurement만
    e = e.loc[
        e["charttime"].notna()
        & (
            e["charttime"]
            >= e["grid_start"]
        )
        & (
            e["charttime"]
            <= e["grid_end"]
        )
    ].copy()


    interval_ns = (
        int(grid_minutes)
        * 60
        * 1_000_000_000
    )

    # 실제 마지막 grid point
    duration_ns = (
        e["grid_end"].astype("int64")
        - e["grid_start"].astype("int64")
    )

    last_idx = (
        duration_ns
        // interval_ns
    )

    e["last_gridtime"] = (
        e["grid_start"]
        + pd.to_timedelta(
            last_idx * grid_minutes,
            unit="min",
        )
    )

    # measurement 이후 처음 도달하는 grid point에 배치
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
        e["gridtime"]
        <= e["last_gridtime"]
    ].copy()

    # 같은 stay + bin + variable
    # 여러 raw measurement → median
    agg_dict = {
        "valuenum": "median",
        "charttime": "size",
    }

    if "cf_map_valuenum" in e.columns:
        agg_dict[
            "cf_map_valuenum"
        ] = "median"


    binned = (
        e.groupby(
            [
                "stay_id",
                "gridtime",
                "variable",
            ],
            as_index=False,
            sort=False,
        )
        .agg(agg_dict)
        .rename(
            columns={
                "charttime":
                    "measurements_in_bin"
            }
        )
    )

    return binned


def _grid_points_per_stay(
    grid_stay_info: pd.DataFrame,
    minutes: int,
):

    x = grid_stay_info[
        [
            "stay_id",
            "grid_start",
            "grid_end",
        ]
    ].copy()

    duration_minutes = (
        (
            x["grid_end"]
            - x["grid_start"]
        )
        .dt.total_seconds()
        / 60.0
    )

    x["grid_points"] = (
        np.floor(
            duration_minutes
            / minutes
        )
        .astype(int)
        + 1
    )

    return x[
        [
            "stay_id",
            "grid_points",
        ]
    ]


def _missingness_report(
    binned: pd.DataFrame,
    variables,
    grid_points_by_stay: pd.DataFrame,
    total_stays: int,
    suffix: str,
):

    total_grid_points = int(
        grid_points_by_stay[
            "grid_points"
        ].sum()
    )
    observed = (
        binned.groupby("variable")
        .agg(
            observed_points=(
                "gridtime",
                "size"
            ),
            stays_with_measurement=(
                "stay_id",
                "nunique"
            ),
        )
        .reindex(variables)
        .fillna(0)
        .reset_index()
    )

    observed[
        "observed_points"
    ] = observed[
        "observed_points"
    ].astype(int)

    observed[
        "stays_with_measurement"
    ] = observed[
        "stays_with_measurement"
    ].astype(int)

    observed[
        "total_grid_points"
    ] = total_grid_points

    observed[
        "missing_points"
    ] = (
        total_grid_points
        - observed[
            "observed_points"
        ]
    )

    observed[
        "missing_pct"
    ] = (
        100.0
        * observed[
            "missing_points"
        ]
        / total_grid_points
    )

    observed[
        "stay_coverage_pct"
    ] = (
        100.0
        * observed[
            "stays_with_measurement"
        ]
        / total_stays
    )

    stay_point_series = (
        grid_points_by_stay
        .set_index("stay_id")[
            "grid_points"
        ]
    )

    within_totals = {}

    for variable in variables:

        measured_stays = (
            binned.loc[
                binned["variable"]
                == variable,
                "stay_id"
            ]
            .drop_duplicates()
        )

        within_totals[
            variable
        ] = int(
            stay_point_series
            .reindex(
                measured_stays
            )
            .fillna(0)
            .sum()
        )


    observed[
        "within_total_points"
    ] = (
        observed["variable"]
        .map(within_totals)
        .fillna(0)
        .astype(int)
    )


    observed[
        "within_missing_pct"
    ] = np.where(
        observed[
            "within_total_points"
        ] > 0,

        100.0
        * (
            observed[
                "within_total_points"
            ]
            - observed[
                "observed_points"
            ]
        )
        / observed[
            "within_total_points"
        ],

        np.nan,
    )


    return observed.rename(
        columns={
            "observed_points":
                f"observed_{suffix}_points",

            "stays_with_measurement":
                f"stays_with_measurement_{suffix}",

            "total_grid_points":
                f"total_{suffix}_points",

            "missing_points":
                f"missing_{suffix}_points",

            "missing_pct":
                f"missing_{suffix}_pct",

            "stay_coverage_pct":
                f"stay_coverage_{suffix}_pct",

            "within_total_points":
                f"within_total_{suffix}_points",

            "within_missing_pct":
                f"within_missing_{suffix}_pct",
        }
    )


def place_measurements_on_grid(
    time_grid: pd.DataFrame,
    grid_stay_info: pd.DataFrame,
    nonpharma_merged: pd.DataFrame,
    grid_minutes: int = 60,
    comparison_minutes: int = 5,
):
    # Variables / stays
    variables = sorted(
        nonpharma_merged[
            "variable"
        ]
        .dropna()
        .unique()
    )

    total_stays = int(
        grid_stay_info[
            "stay_id"
        ].nunique()
    )

    # 2. 실제 60-min grid에 raw measurement 배치
    raw_grid_values = _assign_to_grid(
        events=nonpharma_merged,
        stay_windows=grid_stay_info,
        grid_minutes=grid_minutes,
    )


    # memory 절약
    raw_grid_values[
        "valuenum"
    ] = raw_grid_values[
        "valuenum"
    ].astype("float32")

    if (
        "cf_map_valuenum"
        in raw_grid_values.columns
    ):
        raw_grid_values[
            "cf_map_valuenum"
        ] = raw_grid_values[
            "cf_map_valuenum"
        ].astype("float32")


    raw_grid_values[
        "measurements_in_bin"
    ] = raw_grid_values[
        "measurements_in_bin"
    ].astype("int16")


    actual_grid_points = (
        _grid_points_per_stay(
            grid_stay_info,
            minutes=grid_minutes,
        )
    )


    miss_actual = _missingness_report(
        binned=raw_grid_values,
        variables=variables,
        grid_points_by_stay=
            actual_grid_points,
        total_stays=total_stays,
        suffix=f"{grid_minutes}min",
    )

    # (비교용) 5-min grid였다면 raw coverage가 어땠는지
    comparison_binned = _assign_to_grid(
        events=nonpharma_merged,
        stay_windows=grid_stay_info,
        grid_minutes=comparison_minutes,
    )


    comparison_grid_points = (
        _grid_points_per_stay(
            grid_stay_info,
            minutes=comparison_minutes,
        )
    )


    miss_comparison = (
        _missingness_report(
            binned=comparison_binned,
            variables=variables,
            grid_points_by_stay=
                comparison_grid_points,
            total_stays=total_stays,
            suffix=
                f"{comparison_minutes}min",
        )
    )


    # 60-min vs hypothetical 5-min variable 기준으로만 merge
    missingness = (
        miss_actual.merge(
            miss_comparison,
            on="variable",
            how="outer",
        )
    )


    actual_col = (
        f"missing_{grid_minutes}min_pct"
    )

    comparison_col = (
        f"missing_{comparison_minutes}min_pct"
    )


    # 5-min → 60-min으로 바꿨을 때
    # missingness가 몇 percentage point 줄었는지
    missingness[
        "missing_reduction_pp"
    ] = (
        missingness[
            comparison_col
        ]
        - missingness[
            actual_col
        ]
    )


    missingness = (
        missingness
        .sort_values(
            comparison_col,
            ascending=False,
        )
        .reset_index(drop=True)
    )

    measurement_rows_in_grid = int(
        raw_grid_values[
            "measurements_in_bin"
        ].sum()
    )

    observed_cells = int(
        len(raw_grid_values)
    )

    multi_measurement_cells = int(
        (
            raw_grid_values[
                "measurements_in_bin"
            ] > 1
        ).sum()
    )

    total_actual_points = int(
        actual_grid_points[
            "grid_points"
        ].sum()
    )

    total_comparison_points = int(
        comparison_grid_points[
            "grid_points"
        ].sum()
    )


    print("=" * 70)
    print(
        "5-2. Raw Measurements on Main Grid "
        "+ 5-min Missingness Audit"
    )
    print("=" * 70)

    print(
        f"Main grid: "
        f"{grid_minutes} min"
    )

    print(
        f"Hypothetical comparison grid: "
        f"{comparison_minutes} min"
    )

    print(
        f"Variables: "
        f"{len(variables)}"
    )

    print(
        f"Stays: "
        f"{total_stays}"
    )

    print(
        f"Total {grid_minutes}-min "
        f"grid points: "
        f"{total_actual_points}"
    )

    print(
        f"Observed {grid_minutes}-min "
        f"variable cells: "
        f"{observed_cells}"
    )

    print(
        "Raw measurement rows represented "
        f"on main grid: "
        f"{measurement_rows_in_grid}"
    )

    print(
        "Main-grid cells containing >1 "
        "measurement before median aggregation: "
        f"{multi_measurement_cells}"
    )

    print(
        f"Hypothetical total "
        f"{comparison_minutes}-min grid points: "
        f"{total_comparison_points}"
    )


    print(
        f"\n[Variable missingness: "
        f"actual {grid_minutes}-min "
        f"vs hypothetical "
        f"{comparison_minutes}-min]"
    )


    display_cols = [
        "variable",

        f"stays_with_measurement_{grid_minutes}min",
        f"stay_coverage_{grid_minutes}min_pct",

        f"observed_{grid_minutes}min_points",
        f"missing_{grid_minutes}min_pct",
        f"within_missing_{grid_minutes}min_pct",

        f"observed_{comparison_minutes}min_points",
        f"missing_{comparison_minutes}min_pct",
        f"within_missing_{comparison_minutes}min_pct",

        "missing_reduction_pp",
    ]


    print(
        missingness[
            display_cols
        ]
        .to_string(
            index=False,
            formatters={
                f"stay_coverage_{grid_minutes}min_pct":
                    lambda x: f"{x:.2f}",

                f"missing_{grid_minutes}min_pct":
                    lambda x: f"{x:.2f}",

                f"within_missing_{grid_minutes}min_pct":
                    lambda x: f"{x:.2f}",

                f"missing_{comparison_minutes}min_pct":
                    lambda x: f"{x:.2f}",

                f"within_missing_{comparison_minutes}min_pct":
                    lambda x: f"{x:.2f}",

                "missing_reduction_pp":
                    lambda x: f"{x:.2f}",
            },
        )
    )


    print(
        f"\n[Example observed "
        f"{grid_minutes}-min grid values]"
    )

    print(
        raw_grid_values
        .sort_values(
            [
                "stay_id",
                "gridtime",
                "variable",
            ]
        )
        .head(30)
        .to_string(index=False)
    )


    print("\n[Important]")

    print(
        f"raw_grid_values contains the ACTUAL "
        f"{grid_minutes}-min preprocessing data."
    )

    print(
        f"The {comparison_minutes}-min result is used only "
        "to quantify how sparse the data would have been "
        "under the paper-style finer grid."
    )

    print(
        "All missingness values are BEFORE adaptive imputation."
    )

    print(
        "Multiple raw measurements falling in the same "
        "time bin are summarized by median."
    )

    print("=" * 70)


    report = {
        "variables":
            int(len(variables)),

        "stays":
            total_stays,

        f"total_{grid_minutes}min_grid_points":
            total_actual_points,

        f"total_{comparison_minutes}min_grid_points":
            total_comparison_points,

        f"observed_{grid_minutes}min_cells":
            observed_cells,

        "measurement_rows_in_main_grid":
            measurement_rows_in_grid,

        "multi_measurement_cells_main_grid":
            multi_measurement_cells,
    }


    return (
        raw_grid_values,
        missingness,
        report,
    )