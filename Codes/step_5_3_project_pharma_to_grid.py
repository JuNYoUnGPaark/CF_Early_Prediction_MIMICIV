import numpy as np
import pandas as pd


# ============================================================
# 5-3. Project Continuous Pharmaceutical Rates onto 1-h Grid
# ============================================================
#
# 입력
# ------------------------------------------------------------
# time_grid       : Step 5-1 실제 60-min grid
# grid_stay_info  : Step 5-1 stay별 grid_start / grid_end
# pharma_merged   : Step 4-2B continuous pharma intervals
#
# 처리 원칙
# ------------------------------------------------------------
# 1) continuous infusion은 point measurement가 아니라 interval이다.
# 2) 각 grid timestamp에서
#
#       starttime <= gridtime < endtime
#
#    이면 해당 infusion이 active하다고 본다.
# 3) 같은 stay + time + pharma에 infusion이 겹치면 rate를 합산한다.
# 4) active infusion이 없으면 해당 pharma rate = 0.
#
# 따라서 pharma는 non-pharma처럼 "결측값"을 impute하는 구조가 아니다.
# 최종 pharma grid는 0 또는 active rate를 갖는다.
#
# 주의
# ------------------------------------------------------------
# - 현재 최종 7개 pharma variable은 각 variable 내부에서 unit이 하나로
#   확인된 상태를 전제로 한다.
# - 서로 다른 pharma variable끼리는 unit이 달라도 상관없다.
# - CF annotation의 45-min vasoactive/inotrope presence 판정은
#   이후 annotation branch에서 raw interval을 이용해 별도로 처리한다.
# ============================================================


def project_pharma_to_grid(
    time_grid: pd.DataFrame,
    grid_stay_info: pd.DataFrame,
    pharma_merged: pd.DataFrame,
    grid_minutes: int = 60,
):

    # --------------------------------------------------------
    # 0. Required columns
    # --------------------------------------------------------

    required_grid = {
        "stay_id",
        "gridtime",
    }

    required_stays = {
        "stay_id",
        "grid_start",
        "grid_end",
    }

    required_pharma = {
        "stay_id",
        "pharma_id",
        "pharma_variable",
        "starttime",
        "endtime",
        "continuous_rate",
        "continuous_rate_uom",
    }

    missing = required_grid - set(time_grid.columns)
    if missing:
        raise ValueError(
            f"time_grid에 필요한 column이 없습니다: {missing}"
        )

    missing = required_stays - set(grid_stay_info.columns)
    if missing:
        raise ValueError(
            f"grid_stay_info에 필요한 column이 없습니다: {missing}"
        )

    missing = required_pharma - set(pharma_merged.columns)
    if missing:
        raise ValueError(
            f"pharma_merged에 필요한 column이 없습니다: {missing}"
        )


    # --------------------------------------------------------
    # 1. Basic preparation
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 2. Unit check
    #
    # 같은 canonical pharma 안에서 unit이 여러 개면
    # 그대로 sum할 수 없으므로 중단한다.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 3. Stay grid window 연결
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 4. Interval -> grid timestamp
    #
    # 각 infusion에 대해:
    #   first active grid:
    #       grid_start 이후, starttime 이상인 첫 grid point
    #
    #   last active grid:
    #       endtime 미만인 마지막 grid point
    #
    # starttime <= gridtime < endtime
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 5. Simultaneous infusion aggregation
    #
    # 같은 canonical drug이 같은 grid point에서 여러 infusion으로
    # active하면 rate를 sum한다.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 6. Dense pharma grid
    #
    # active하지 않은 시점은 NaN이 아니라 0.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 7. Drug-level report
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 8. Print
    # --------------------------------------------------------

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