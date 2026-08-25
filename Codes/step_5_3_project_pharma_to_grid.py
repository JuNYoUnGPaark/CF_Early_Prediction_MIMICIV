import numpy as np
import pandas as pd

def project_pharma_to_grid(
    time_grid: pd.DataFrame,
    grid_stay_info: pd.DataFrame,
    pharma_merged: pd.DataFrame,
    grid_minutes: int = 60,
):
    # Basic preparation
    pharma = pharma_merged.copy()

    pharma["starttime"] = pd.to_datetime(
        pharma["starttime"],
        errors="coerce",
    )

    pharma["endtime"] = pd.to_datetime(
        pharma["endtime"],
        errors="coerce",
    )

    pharma["continuous_rate"] = pd.to_numeric(
        pharma["continuous_rate"],
        errors="coerce",
    )

    pharma = pharma.dropna(
        subset=[
            "stay_id",
            "pharma_id",
            "pharma_variable",
            "starttime",
            "endtime",
            "continuous_rate",
        ]
    ).copy()


    # rate는 음수가 될 수 없으므로 발견 시 중단해서 확인
    negative_rate_rows = int(
        (pharma["continuous_rate"] < 0).sum()
    )

    if negative_rate_rows > 0:
        raise ValueError(
            f"Negative continuous_rate rows detected: "
            f"{negative_rate_rows}"
        )

    zero_rate_rows = int(
        (pharma["continuous_rate"] == 0).sum()
    )

    # Unit check
    unit_check = (
        pharma.groupby(
            [
                "pharma_id",
                "pharma_variable",
            ],
            dropna=False,
        )["continuous_rate_uom"]
        .agg(
            unit_count=lambda x: x.dropna().nunique(),
            units=lambda x: " | ".join(
                sorted(
                    set(
                        x.dropna().astype(str)
                    )
                )
            ),
        )
        .reset_index()
    )

    mixed_unit = unit_check.loc[
        unit_check["unit_count"] > 1
    ]

    if len(mixed_unit):
        raise ValueError(
            "같은 pharma variable 안에 rate unit이 여러 개 있습니다.\n"
            + mixed_unit.to_string(index=False)
        )


    # Stay grid window 연결
    stay_window = grid_stay_info[
        [
            "stay_id",
            "grid_start",
            "grid_end",
        ]
    ].copy()

    stay_window["grid_start"] = pd.to_datetime(
        stay_window["grid_start"]
    )

    stay_window["grid_end"] = pd.to_datetime(
        stay_window["grid_end"]
    )

    pharma = pharma.merge(
        stay_window,
        on="stay_id",
        how="inner",
    )


    # 분석 window와 겹치는 interval만 사용
    overlap_window = (
        (pharma["endtime"] > pharma["grid_start"])
        & (pharma["starttime"] <= pharma["grid_end"])
    )

    intervals_outside_grid = int(
        (~overlap_window).sum()
    )

    pharma = pharma.loc[
        overlap_window
    ].copy()

    intervals_in_grid = int(
        len(pharma)
    )

    # Interval -> grid timestamp
    # starttime <= gridtime < endtime

    interval_ns = (
        int(grid_minutes)
        * 60
        * 1_000_000_000
    )

    pieces = []

    for row in pharma.itertuples(index=False):

        grid_start = pd.Timestamp(row.grid_start)
        grid_end = pd.Timestamp(row.grid_end)

        starttime = max(
            pd.Timestamp(row.starttime),
            grid_start,
        )

        endtime = min(
            pd.Timestamp(row.endtime),
            grid_end + pd.Timedelta(
                minutes=grid_minutes
            ),
        )

        # 실제 마지막 grid index
        max_idx = int(
            (
                grid_end.value
                - grid_start.value
            )
            // interval_ns
        )

        start_delta = (
            starttime.value
            - grid_start.value
        )

        # ceil(start_delta / interval)
        first_idx = int(
            (
                start_delta
                + interval_ns
                - 1
            )
            // interval_ns
        )

        end_delta = (
            endtime.value
            - grid_start.value
        )

        # gridtime < endtime
        # 따라서 endtime이 grid point와 정확히 같으면 그 point는 제외
        last_idx = int(
            (
                end_delta - 1
            )
            // interval_ns
        )

        first_idx = max(
            first_idx,
            0,
        )

        last_idx = min(
            last_idx,
            max_idx,
        )

        if first_idx > last_idx:
            continue

        idx = np.arange(
            first_idx,
            last_idx + 1,
            dtype=np.int32,
        )

        gridtimes = (
            grid_start
            + pd.to_timedelta(
                idx * grid_minutes,
                unit="min",
            )
        )

        n = len(idx)

        pieces.append(
            pd.DataFrame({
                "stay_id":
                    np.repeat(row.stay_id, n),

                "gridtime":
                    gridtimes,

                "pharma_id":
                    np.repeat(row.pharma_id, n),

                "pharma_variable":
                    np.repeat(row.pharma_variable, n),

                "rate_uom":
                    np.repeat(
                        row.continuous_rate_uom,
                        n,
                    ),

                "rate":
                    np.repeat(
                        float(row.continuous_rate),
                        n,
                    ),
            })
        )


    if pieces:
        expanded = pd.concat(
            pieces,
            ignore_index=True,
        )
    else:
        expanded = pd.DataFrame(
            columns=[
                "stay_id",
                "gridtime",
                "pharma_id",
                "pharma_variable",
                "rate_uom",
                "rate",
            ]
        )

    if len(expanded):

        pre_sum = (
            expanded.groupby(
                [
                    "stay_id",
                    "gridtime",
                    "pharma_id",
                    "pharma_variable",
                    "rate_uom",
                ],
                sort=False,
            )
            .agg(
                rate=(
                    "rate",
                    "sum",
                ),
                active_infusions=(
                    "rate",
                    "size",
                ),
            )
            .reset_index()
        )

    else:

        pre_sum = pd.DataFrame(
            columns=[
                "stay_id",
                "gridtime",
                "pharma_id",
                "pharma_variable",
                "rate_uom",
                "rate",
                "active_infusions",
            ]
        )


    overlapping_grid_cells = int(
        (
            pre_sum[
                "active_infusions"
            ] > 1
        ).sum()
    )

    pharma_active_long = pre_sum.copy()

    if len(pharma_active_long):
        pharma_active_long["rate"] = (
            pharma_active_long["rate"]
            .astype("float32")
        )

        pharma_active_long[
            "active_infusions"
        ] = pharma_active_long[
            "active_infusions"
        ].astype("int16")


    pharma_variables = (
        pharma_merged[
            [
                "pharma_id",
                "pharma_variable",
            ]
        ]
        .drop_duplicates()
        .sort_values("pharma_id")
    )

    variable_names = (
        pharma_variables[
            "pharma_variable"
        ]
        .tolist()
    )


    if len(pharma_active_long):

        wide = (
            pharma_active_long
            .pivot_table(
                index=[
                    "stay_id",
                    "gridtime",
                ],
                columns="pharma_variable",
                values="rate",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
        )

        wide.columns.name = None

    else:

        wide = pd.DataFrame(
            columns=[
                "stay_id",
                "gridtime",
            ]
            + variable_names
        )


    pharma_grid = time_grid.merge(
        wide,
        on=[
            "stay_id",
            "gridtime",
        ],
        how="left",
    )


    for variable in variable_names:

        if variable not in pharma_grid.columns:
            pharma_grid[variable] = 0.0

        pharma_grid[variable] = (
            pharma_grid[variable]
            .fillna(0.0)
            .astype("float32")
        )

    # 7. Drug-level report
    total_grid_points = int(
        len(time_grid)
    )

    rows = []

    for _, drug in pharma_variables.iterrows():

        pharma_id = drug["pharma_id"]
        variable = drug["pharma_variable"]

        active = pharma_active_long.loc[
            pharma_active_long[
                "pharma_id"
            ] == pharma_id
        ]

        active_points = int(
            len(active)
        )

        stays_with_drug = int(
            active["stay_id"].nunique()
        )

        if active_points > 0:
            rate_min = float(
                active["rate"].min()
            )
            rate_median = float(
                active["rate"].median()
            )
            rate_max = float(
                active["rate"].max()
            )
        else:
            rate_min = np.nan
            rate_median = np.nan
            rate_max = np.nan

        unit_row = unit_check.loc[
            unit_check[
                "pharma_id"
            ] == pharma_id
        ]

        units = (
            unit_row["units"].iloc[0]
            if len(unit_row)
            else ""
        )

        rows.append({
            "pharma_id":
                pharma_id,

            "pharma_variable":
                variable,

            "rate_uom":
                units,

            "stays_with_drug":
                stays_with_drug,

            "active_grid_points":
                active_points,

            "active_pct_all_grid":
                (
                    100.0
                    * active_points
                    / total_grid_points
                ),

            "inactive_grid_points":
                (
                    total_grid_points
                    - active_points
                ),

            "final_missing_pct":
                0.0,

            "rate_min_active":
                rate_min,

            "rate_median_active":
                rate_median,

            "rate_max_active":
                rate_max,
        })


    pharma_grid_report = (
        pd.DataFrame(rows)
        .sort_values(
            "active_grid_points",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("=" * 70)
    print(
        "5-3. Pharmaceutical Rates on "
        f"{grid_minutes}-min Grid"
    )
    print("=" * 70)

    print(
        f"Input pharma intervals: "
        f"{len(pharma_merged)}"
    )

    print(
        f"Intervals intersecting analysis grid: "
        f"{intervals_in_grid}"
    )

    print(
        f"Intervals outside analysis grid: "
        f"{intervals_outside_grid}"
    )

    print(
        f"Zero-rate intervals: "
        f"{zero_rate_rows}"
    )

    print(
        f"Total patient grid points: "
        f"{total_grid_points}"
    )

    print(
        f"Active pharma grid cells "
        f"(drug-specific): "
        f"{len(pharma_active_long)}"
    )

    print(
        f"Grid cells with >1 simultaneous "
        f"infusion of same pharma: "
        f"{overlapping_grid_cells}"
    )


    print("\n[Pharmaceutical grid summary]")

    print(
        pharma_grid_report
        .to_string(
            index=False,
            formatters={
                "active_pct_all_grid":
                    lambda x: f"{x:.2f}",

                "final_missing_pct":
                    lambda x: f"{x:.2f}",

                "rate_min_active":
                    lambda x: f"{x:.6g}",

                "rate_median_active":
                    lambda x: f"{x:.6g}",

                "rate_max_active":
                    lambda x: f"{x:.6g}",
            },
        )
    )


    print(
        f"\n[Example active pharma values]"
    )

    print(
        pharma_active_long
        .sort_values(
            [
                "stay_id",
                "gridtime",
                "pharma_variable",
            ]
        )
        .head(30)
        .to_string(index=False)
    )


    print(
        f"\n[Example dense pharma grid]"
    )

    print(
        pharma_grid
        .head(20)
        .to_string(index=False)
    )


    print("\n[Important]")

    print(
        "Inactive pharmaceutical states are represented as 0, "
        "not as missing values."
    )

    print(
        "If multiple intervals of the same pharmaceutical are active "
        "at one grid point, their rates are summed."
    )

    print(
        "No adaptive imputation is applied to these continuous "
        "pharmaceutical variables."
    )

    print(
        "CF annotation drug-presence logic should be constructed "
        "separately from raw infusion intervals."
    )

    print("=" * 70)


    report = {
        "input_intervals":
            int(len(pharma_merged)),

        "intervals_in_grid":
            intervals_in_grid,

        "intervals_outside_grid":
            intervals_outside_grid,

        "zero_rate_intervals":
            zero_rate_rows,

        "negative_rate_intervals":
            negative_rate_rows,

        "total_grid_points":
            total_grid_points,

        "active_pharma_grid_cells":
            int(len(pharma_active_long)),

        "overlapping_same_pharma_grid_cells":
            overlapping_grid_cells,

        "pharma_variables":
            int(len(variable_names)),
    }


    return (
        pharma_grid,
        pharma_active_long,
        pharma_grid_report,
        report,
    )