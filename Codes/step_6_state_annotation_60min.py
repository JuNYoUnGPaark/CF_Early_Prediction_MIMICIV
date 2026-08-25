import numpy as np
import pandas as pd


# ============================================================
# 6. Circulatory State Annotation - 60 min adaptation
# ============================================================
#
# ORIGINAL PAPER
# ------------------------------------------------------------
# State variables:
#   - Lactate
#   - MAP
#   - vasoactive / inotropic drug presence
#
# Original temporal rule:
#   - annotate every 5 min
#   - 45-min centered window
#   - each required condition must hold for >=30 min
#
# CURRENT MIMIC-IV ADAPTATION
# ------------------------------------------------------------
# Final annotation timestamps are the actual 60-min time_grid.
#
# Because the MIMIC-IV representation has been intentionally
# changed to 60-min resolution, the original 45-min / 30-min
# persistence rule cannot be represented faithfully from the
# hourly values without inventing additional sub-hourly MAP
# interpolation assumptions.
#
# Therefore this implementation keeps the PAPER'S STATE LOGIC
# but evaluates it at each 60-min grid point:
#
#   CF:
#       Lactate > 2
#       AND
#       (MAP <= 65 OR vasoactive/inotrope active)
#
#   NO_CF:
#       Lactate <= 2
#       AND MAP > 65
#       AND no vasoactive/inotrope active
#
#   AMBIGUOUS:
#       everything else, including missing MAP/lactate and
#       discordant states.
#
# IMPORTANT THRESHOLD NOTE
# ------------------------------------------------------------
# The main text/figure describes lactate >=2, but the Methods
# state-annotation section explicitly defines:
#     CF      : lactate > 2
#     NO_CF   : lactate <= 2
# We follow the Methods implementation wording here.
#
# Lactate annotation interpolation is SPECIAL and may use future
# information, exactly because it is used only for annotation.
# It must NOT be replaced by Step 5-4B model-data Lactate.
# ============================================================


VASOACTIVE_INOTROPE_VARIABLES = {
    "Norepinephrine",
    "Phenylephrin",
    "Vasopressin",
    "Epinephrine",
    "Dopamin",
    "Dobutamine",
    "Milrinone",
}


def _prepare_annotation_lactate(
    time_grid: pd.DataFrame,
    nonpharma_merged: pd.DataFrame,
    threshold: float = 2.0,
    crossing_full_interp_hours: float = 6.0,
    abnormal_edge_fill_hours: float = 3.0,
):
    """
    Paper's annotation-only Lactate interpolation.

    Rules
    -----
    Between two measurements:
      1) same side of threshold:
           linear interpolation for the full interval

      2) crosses threshold:
           gap < 6 h  -> linear interpolation for full interval
           gap >= 6 h -> forward fill left value max 3 h
                         backward fill right value max 3 h
                         middle remains missing

    Before first / after last measurement:
      - normal (<=2): fill indefinitely
      - abnormal (>2): fill for max 3 h

    Future information is intentionally allowed here because this
    series is ONLY for state annotation.
    """

    lactate = nonpharma_merged.loc[
        nonpharma_merged["variable"] == "Lactate",
        ["stay_id", "charttime", "valuenum"],
    ].copy()

    lactate["charttime"] = pd.to_datetime(
        lactate["charttime"],
        errors="coerce",
    )

    lactate["valuenum"] = pd.to_numeric(
        lactate["valuenum"],
        errors="coerce",
    )

    lactate = lactate.dropna(
        subset=["stay_id", "charttime", "valuenum"]
    ).copy()

    # Same stay + same timestamp Lactate should already be merged,
    # but keep this safety aggregation explicit.
    lactate = (
        lactate.groupby(
            ["stay_id", "charttime"],
            as_index=False,
        )["valuenum"]
        .median()
        .sort_values(["stay_id", "charttime"])
    )

    grid = time_grid[
        ["stay_id", "gridtime"]
    ].copy()

    grid["gridtime"] = pd.to_datetime(
        grid["gridtime"],
        errors="coerce",
    )

    grid = grid.dropna(
        subset=["stay_id", "gridtime"]
    ).copy()

    output_parts = []

    six_hours = pd.Timedelta(
        hours=crossing_full_interp_hours
    )
    three_hours = pd.Timedelta(
        hours=abnormal_edge_fill_hours
    )

    lactate_by_stay = {
        sid: g.sort_values("charttime")
        for sid, g in lactate.groupby("stay_id", sort=False)
    }

    for sid, gg in grid.groupby("stay_id", sort=False):

        times = gg["gridtime"].to_numpy(
            dtype="datetime64[ns]"
        )

        values = np.full(
            len(gg),
            np.nan,
            dtype=float,
        )

        source = np.full(
            len(gg),
            "missing",
            dtype=object,
        )

        if sid not in lactate_by_stay:
            part = gg.copy()
            part["annotation_lactate"] = values
            part["lactate_source"] = source
            output_parts.append(part)
            continue

        lm = lactate_by_stay[sid]

        mt = lm["charttime"].to_numpy(
            dtype="datetime64[ns]"
        )

        mv = lm["valuenum"].to_numpy(
            dtype=float
        )

        # searchsorted gives first measurement >= grid time
        right_idx = np.searchsorted(
            mt,
            times,
            side="left",
        )

        for i, t in enumerate(times):

            r = int(right_idx[i])

            # Exact measurement at t
            if r < len(mt) and mt[r] == t:
                values[i] = mv[r]
                source[i] = "measured"
                continue

            # Before first measurement
            if r == 0:
                first_t = pd.Timestamp(mt[0])
                first_v = float(mv[0])
                current_t = pd.Timestamp(t)

                if first_v <= threshold:
                    values[i] = first_v
                    source[i] = "normal_backward_fill"
                elif (first_t - current_t) <= three_hours:
                    values[i] = first_v
                    source[i] = "abnormal_backward_fill_3h"

                continue

            # After last measurement
            if r >= len(mt):
                last_t = pd.Timestamp(mt[-1])
                last_v = float(mv[-1])
                current_t = pd.Timestamp(t)

                if last_v <= threshold:
                    values[i] = last_v
                    source[i] = "normal_forward_fill"
                elif (current_t - last_t) <= three_hours:
                    values[i] = last_v
                    source[i] = "abnormal_forward_fill_3h"

                continue

            # Between two measurements
            l = r - 1

            left_t = pd.Timestamp(mt[l])
            right_t = pd.Timestamp(mt[r])

            left_v = float(mv[l])
            right_v = float(mv[r])

            current_t = pd.Timestamp(t)
            gap = right_t - left_t

            left_high = left_v > threshold
            right_high = right_v > threshold

            crossed_threshold = (
                left_high != right_high
            )

            # Same side: full linear interpolation
            # Crossed threshold + gap < 6h: full linear interpolation
            if (
                not crossed_threshold
                or gap < six_hours
            ):
                frac = (
                    (current_t - left_t).total_seconds()
                    / gap.total_seconds()
                )

                values[i] = (
                    left_v
                    + frac * (right_v - left_v)
                )

                source[i] = (
                    "linear_same_side"
                    if not crossed_threshold
                    else "linear_crossing_lt6h"
                )

                continue

            # Crossed threshold + gap >= 6h
            if (current_t - left_t) <= three_hours:
                values[i] = left_v
                source[i] = "crossing_forward_fill_3h"

            elif (right_t - current_t) <= three_hours:
                values[i] = right_v
                source[i] = "crossing_backward_fill_3h"

            # else remains missing

        part = gg.copy()
        part["annotation_lactate"] = values.astype(
            "float32"
        )
        part["lactate_source"] = source
        output_parts.append(part)

    result = pd.concat(
        output_parts,
        ignore_index=True,
    )

    return result


def _prepare_hourly_map(
    time_grid: pd.DataFrame,
    raw_grid_values: pd.DataFrame,
):
    """
    60-min adaptation:
    Use the Step 5-2 raw ABP mean value assigned to each hourly grid cell.

    IMPORTANT:
    cf_map_valuenum is used, so IABP-derived MAP source 224322
    remains excluded from CF annotation.
    """

    required = {
        "stay_id",
        "gridtime",
        "variable",
        "cf_map_valuenum",
    }

    missing = required - set(raw_grid_values.columns)

    if missing:
        raise ValueError(
            "raw_grid_values에 CF MAP annotation에 필요한 "
            f"column이 없습니다: {missing}"
        )

    map_values = raw_grid_values.loc[
        raw_grid_values["variable"] == "ABP mean",
        [
            "stay_id",
            "gridtime",
            "cf_map_valuenum",
        ],
    ].copy()

    map_values = map_values.rename(
        columns={
            "cf_map_valuenum": "annotation_map"
        }
    )

    result = time_grid[
        ["stay_id", "gridtime"]
    ].merge(
        map_values,
        on=["stay_id", "gridtime"],
        how="left",
        validate="one_to_one",
    )

    result["annotation_map"] = pd.to_numeric(
        result["annotation_map"],
        errors="coerce",
    ).astype("float32")

    return result


def _prepare_drug_presence(
    time_grid: pd.DataFrame,
    pharma_merged: pd.DataFrame,
):
    """
    At each hourly grid timestamp:
        starttime <= gridtime < endtime
    for any confirmed vasoactive/inotropic drug.

    Drug absence is 0, not missing.
    """

    required = {
        "stay_id",
        "pharma_variable",
        "starttime",
        "endtime",
    }

    missing = required - set(pharma_merged.columns)

    if missing:
        raise ValueError(
            f"pharma_merged에 필요한 column이 없습니다: {missing}"
        )

    p = pharma_merged.loc[
        pharma_merged["pharma_variable"].isin(
            VASOACTIVE_INOTROPE_VARIABLES
        )
    ].copy()

    p["starttime"] = pd.to_datetime(
        p["starttime"],
        errors="coerce",
    )

    p["endtime"] = pd.to_datetime(
        p["endtime"],
        errors="coerce",
    )

    p = p.dropna(
        subset=[
            "stay_id",
            "pharma_variable",
            "starttime",
            "endtime",
        ]
    ).copy()

    active_rows = []

    pharma_by_stay = {
        sid: g
        for sid, g in p.groupby("stay_id", sort=False)
    }

    for sid, gg in time_grid.groupby(
        "stay_id",
        sort=False,
    ):

        times = pd.to_datetime(
            gg["gridtime"]
        )

        drug_present = np.zeros(
            len(gg),
            dtype=np.int8,
        )

        active_drugs = np.full(
            len(gg),
            "",
            dtype=object,
        )

        if sid in pharma_by_stay:

            pp = pharma_by_stay[sid]

            starts = pp["starttime"].to_numpy(
                dtype="datetime64[ns]"
            )

            ends = pp["endtime"].to_numpy(
                dtype="datetime64[ns]"
            )

            names = pp["pharma_variable"].to_numpy(
                dtype=object
            )

            gt = times.to_numpy(
                dtype="datetime64[ns]"
            )

            for i, t in enumerate(gt):

                mask = (
                    (starts <= t)
                    & (t < ends)
                )

                if mask.any():
                    drug_present[i] = 1
                    active_drugs[i] = "|".join(
                        sorted(
                            set(names[mask].tolist())
                        )
                    )

        part = gg[
            ["stay_id", "gridtime"]
        ].copy()

        part["vasoactive_inotrope"] = drug_present
        part["active_drugs"] = active_drugs

        active_rows.append(part)

    return pd.concat(
        active_rows,
        ignore_index=True,
    )


def annotate_circulatory_state(
    time_grid: pd.DataFrame,
    raw_grid_values: pd.DataFrame,
    nonpharma_merged: pd.DataFrame,
    pharma_merged: pd.DataFrame,
    lactate_threshold: float = 2.0,
    map_threshold: float = 65.0,
):
    """
    Final hourly state:
        CF / NO_CF / AMBIGUOUS
    """

    lactate = _prepare_annotation_lactate(
        time_grid=time_grid,
        nonpharma_merged=nonpharma_merged,
        threshold=lactate_threshold,
    )

    map_values = _prepare_hourly_map(
        time_grid=time_grid,
        raw_grid_values=raw_grid_values,
    )

    drug = _prepare_drug_presence(
        time_grid=time_grid,
        pharma_merged=pharma_merged,
    )

    state = lactate.merge(
        map_values,
        on=["stay_id", "gridtime"],
        how="left",
        validate="one_to_one",
    )

    state = state.merge(
        drug,
        on=["stay_id", "gridtime"],
        how="left",
        validate="one_to_one",
    )

    state["vasoactive_inotrope"] = (
        state["vasoactive_inotrope"]
        .fillna(0)
        .astype("int8")
    )

    state["active_drugs"] = (
        state["active_drugs"]
        .fillna("")
    )

    # --------------------------------------------------------
    # Primitive conditions
    # --------------------------------------------------------

    state["lactate_available"] = (
        state["annotation_lactate"].notna()
    )

    state["map_available"] = (
        state["annotation_map"].notna()
    )

    state["lactate_high"] = (
        state["annotation_lactate"]
        > lactate_threshold
    )

    state["map_low"] = (
        state["annotation_map"]
        <= map_threshold
    )

    state["hemodynamic_criterion"] = (
        state["map_low"].fillna(False)
        | state["vasoactive_inotrope"].eq(1)
    )


    # --------------------------------------------------------
    # State definition
    #
    # Methods implementation:
    #
    # NO_CF:
    #   MAP >65
    #   AND no drug
    #   AND lactate <=2
    #
    # CF:
    #   lactate >2
    #   AND (MAP <=65 OR drug present)
    #
    # Missing MAP or lactate -> ambiguous.
    # Everything not satisfying a complete CF/NO_CF definition
    # also remains ambiguous.
    # --------------------------------------------------------

    complete = (
        state["lactate_available"]
        & state["map_available"]
    )

    is_cf = (
        complete
        & state["lactate_high"]
        & state["hemodynamic_criterion"]
    )

    is_no_cf = (
        complete
        & (
            state["annotation_lactate"]
            <= lactate_threshold
        )
        & (
            state["annotation_map"]
            > map_threshold
        )
        & state["vasoactive_inotrope"].eq(0)
    )

    state["state"] = "AMBIGUOUS"

    state.loc[
        is_no_cf,
        "state",
    ] = "NO_CF"

    state.loc[
        is_cf,
        "state",
    ] = "CF"


    # --------------------------------------------------------
    # Ambiguous reason audit
    # --------------------------------------------------------

    state["ambiguous_reason"] = ""

    amb = state["state"].eq("AMBIGUOUS")

    state.loc[
        amb
        & ~state["lactate_available"]
        & ~state["map_available"],
        "ambiguous_reason",
    ] = "missing_lactate_and_map"

    state.loc[
        amb
        & ~state["lactate_available"]
        & state["map_available"],
        "ambiguous_reason",
    ] = "missing_lactate"

    state.loc[
        amb
        & state["lactate_available"]
        & ~state["map_available"],
        "ambiguous_reason",
    ] = "missing_map"

    # Complete but discordant
    state.loc[
        amb
        & complete
        & (
            state["annotation_lactate"]
            <= lactate_threshold
        )
        & state["hemodynamic_criterion"],
        "ambiguous_reason",
    ] = "hemodynamic_abnormal_but_lactate_not_high"

    state.loc[
        amb
        & complete
        & state["lactate_high"]
        & ~state["hemodynamic_criterion"],
        "ambiguous_reason",
    ] = "lactate_high_but_hemodynamics_normal"


    # --------------------------------------------------------
    # CF source audit
    # --------------------------------------------------------

    state["cf_driver"] = ""

    state.loc[
        is_cf
        & state["map_low"]
        & state["vasoactive_inotrope"].eq(0),
        "cf_driver",
    ] = "MAP"

    state.loc[
        is_cf
        & ~state["map_low"]
        & state["vasoactive_inotrope"].eq(1),
        "cf_driver",
    ] = "DRUG"

    state.loc[
        is_cf
        & state["map_low"]
        & state["vasoactive_inotrope"].eq(1),
        "cf_driver",
    ] = "MAP+DRUG"


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    total = len(state)

    state_counts = (
        state["state"]
        .value_counts()
        .reindex(
            ["NO_CF", "CF", "AMBIGUOUS"],
            fill_value=0,
        )
    )

    ambiguous_counts = (
        state.loc[
            state["state"] == "AMBIGUOUS",
            "ambiguous_reason",
        ]
        .value_counts()
    )

    cf_drivers = (
        state.loc[
            state["state"] == "CF",
            "cf_driver",
        ]
        .value_counts()
    )

    stays_with_cf = int(
        state.loc[
            state["state"] == "CF",
            "stay_id",
        ].nunique()
    )

    print("=" * 70)
    print("6. Circulatory State Annotation - 60 min adaptation")
    print("=" * 70)

    print("Final annotation grid: 60 min")
    print(
        "State threshold implementation: "
        "Lactate >2 AND (MAP <=65 OR vasoactive/inotrope)"
    )
    print(
        "NO_CF: Lactate <=2 AND MAP >65 AND no vasoactive/inotrope"
    )
    print(
        "Original paper 45-min/30-min persistence rule: "
        "NOT applied after hourly adaptation"
    )
    print(
        "Annotation Lactate uses future-aware special interpolation: YES"
    )
    print(
        "Model-imputed Lactate from Step 5-4B used here: NO"
    )

    print(f"\nTotal grid points: {total}")

    print("\n[State distribution]")
    for name in ["NO_CF", "CF", "AMBIGUOUS"]:
        n = int(state_counts[name])
        print(
            f"{name}: {n} "
            f"({100.0 * n / total:.2f}%)"
        )

    print(
        f"\nStays with at least one CF point: "
        f"{stays_with_cf}"
    )

    print("\n[Input availability]")
    print(
        "Lactate missing:",
        int((~state["lactate_available"]).sum()),
        f"({100.0 * (~state['lactate_available']).mean():.2f}%)",
    )

    print(
        "MAP missing:",
        int((~state["map_available"]).sum()),
        f"({100.0 * (~state['map_available']).mean():.2f}%)",
    )

    print(
        "Drug active:",
        int(state["vasoactive_inotrope"].sum()),
        f"({100.0 * state['vasoactive_inotrope'].mean():.2f}%)",
    )

    print("\n[Ambiguous reasons]")
    print(
        ambiguous_counts.to_string()
        if len(ambiguous_counts)
        else "None"
    )

    print("\n[CF drivers]")
    print(
        cf_drivers.to_string()
        if len(cf_drivers)
        else "None"
    )

    print("\n[Lactate interpolation source]")
    print(
        state["lactate_source"]
        .value_counts()
        .to_string()
    )

    print("\n[Example annotated rows]")
    print(
        state[
            [
                "stay_id",
                "gridtime",
                "annotation_lactate",
                "annotation_map",
                "vasoactive_inotrope",
                "active_drugs",
                "state",
                "ambiguous_reason",
                "cf_driver",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

    print("\n[Important]")
    print(
        "This is an hourly MIMIC-IV adaptation, not exact reproduction "
        "of the paper's 5-min + 45-min-window annotation."
    )
    print(
        "The Methods section uses >2 for CF and <=2 for NO_CF; "
        "the paper's abstract/figure elsewhere uses >=2."
    )
    print(
        "AMBIGUOUS points should be excluded from later model "
        "training/evaluation labels, matching the paper."
    )
    print("=" * 70)

    report = {
        "total_points": int(total),
        "no_cf_points": int(state_counts["NO_CF"]),
        "cf_points": int(state_counts["CF"]),
        "ambiguous_points": int(state_counts["AMBIGUOUS"]),
        "stays_with_cf": stays_with_cf,
        "lactate_missing_points": int(
            (~state["lactate_available"]).sum()
        ),
        "map_missing_points": int(
            (~state["map_available"]).sum()
        ),
        "drug_active_points": int(
            state["vasoactive_inotrope"].sum()
        ),
    }

    return (
        state,
        report,
    )