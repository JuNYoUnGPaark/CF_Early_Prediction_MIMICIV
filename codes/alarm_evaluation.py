# -*- coding: utf-8 -*-
# name: alarm_evaluation.py
# author: JunYoung Park
# date: 2026-08-29

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import auc


# -----------------------------------------------------------------------------------------------------------------------
# 1. Extract CF events
# -----------------------------------------------------------------------------------------------------------------------
def extract_cf_events(state, stay_ids, grid_minutes=60):
    df = state.loc[
        state["stay_id"].isin(stay_ids),
        ["stay_id", "gridtime", "state"]
    ].copy()

    df["gridtime"] = pd.to_datetime(df["gridtime"], errors="coerce")
    df = df.dropna(subset=["stay_id", "gridtime"]).sort_values(
        ["stay_id", "gridtime"]
    )

    events = []

    for stay_id, g in df.groupby("stay_id", sort=False):
        g = g.sort_values("gridtime").reset_index(drop=True)

        in_event = False
        onset = None
        end = None

        for row in g.itertuples(index=False):
            if row.state == "CF":
                if not in_event:
                    onset = row.gridtime
                    in_event = True

                end = row.gridtime

            elif in_event:
                events.append({
                    "stay_id": stay_id,
                    "onset": onset,
                    "end": end
                })

                in_event = False
                onset = None
                end = None

        if in_event:
            events.append({
                "stay_id": stay_id,
                "onset": onset,
                "end": end
            })

    return pd.DataFrame(
        events,
        columns=["stay_id", "onset", "end"]
    )


# -----------------------------------------------------------------------------------------------------------------------
# 2. Build score thresholds
# -----------------------------------------------------------------------------------------------------------------------
def build_thresholds(probabilities, n_thresholds=200):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities[np.isfinite(probabilities)]

    quantiles = np.linspace(0, 1, n_thresholds)

    thresholds = np.quantile(
        probabilities,
        quantiles
    )

    thresholds = np.unique(thresholds)

    return np.sort(thresholds)


# -----------------------------------------------------------------------------------------------------------------------
# 3. Generate alarms with 60-min adapted silencing / reset
# -----------------------------------------------------------------------------------------------------------------------
def generate_alarms(
    predictions,
    cf_events,
    threshold,
    grid_minutes=60,
    silence_steps=1,
    reset_steps=1
):
    predictions = predictions.copy()
    predictions["gridtime"] = pd.to_datetime(
        predictions["gridtime"],
        errors="coerce"
    )

    predictions = predictions.sort_values(
        ["stay_id", "gridtime"]
    )

    alarm_rows = []

    for stay_id, g in predictions.groupby("stay_id", sort=False):
        g = g.sort_values("gridtime")

        stay_events = cf_events.loc[
            cf_events["stay_id"] == stay_id
        ].sort_values("onset")

        last_alarm_time = None

        for row in g.itertuples(index=False):
            current_time = row.gridtime

            if row.probability < threshold:
                continue

            # ---------------------------------------------------------------------------------------
            # 이전 alarm 이후 1 grid-step silencing
            # ---------------------------------------------------------------------------------------
            if last_alarm_time is not None:
                silence_until = last_alarm_time + pd.Timedelta(
                    minutes=grid_minutes * silence_steps
                )

                if current_time < silence_until:
                    continue

            alarm_rows.append({
                "stay_id": stay_id,
                "gridtime": current_time,
                "probability": row.probability
            })

            last_alarm_time = current_time

    return pd.DataFrame(
        alarm_rows,
        columns=["stay_id", "gridtime", "probability"]
    )


# -----------------------------------------------------------------------------------------------------------------------
# 4. Evaluate one threshold
# -----------------------------------------------------------------------------------------------------------------------
def evaluate_threshold(
    predictions,
    state,
    cf_events,
    threshold,
    horizon_hours=8,
    grid_minutes=60
):
    alarms = generate_alarms(
        predictions=predictions,
        cf_events=cf_events,
        threshold=threshold,
        grid_minutes=grid_minutes,
        silence_steps=1,
        reset_steps=1
    )

    horizon = pd.Timedelta(hours=horizon_hours)

    # -------------------------------------------------------------------------------------------------------------------
    # Alarm precision
    # -------------------------------------------------------------------------------------------------------------------
    true_alarms = 0
    false_alarms = 0

    for alarm in alarms.itertuples(index=False):
        future_events = cf_events.loc[
            (cf_events["stay_id"] == alarm.stay_id)
            & (cf_events["onset"] > alarm.gridtime)
            & (cf_events["onset"] <= alarm.gridtime + horizon)
        ]

        if len(future_events) > 0:
            true_alarms += 1
        else:
            false_alarms += 1

    # -------------------------------------------------------------------------------------------------------------------
    # Event recall + warning time
    # -------------------------------------------------------------------------------------------------------------------
    captured_events = 0
    missed_events = 0
    warning_times = []

    for event in cf_events.itertuples(index=False):
        event_alarms = alarms.loc[
            (alarms["stay_id"] == event.stay_id)
            & (alarms["gridtime"] < event.onset)
            & (alarms["gridtime"] >= event.onset - horizon)
        ]

        if len(event_alarms) > 0:
            captured_events += 1

            # 해당 CF event를 처음 경고한 alarm
            first_alarm = event_alarms["gridtime"].min()

            warning_hours = (
                event.onset - first_alarm
            ).total_seconds() / 3600

            warning_times.append(warning_hours)

        else:
            missed_events += 1

    precision = (
        true_alarms / (true_alarms + false_alarms)
        if true_alarms + false_alarms > 0
        else np.nan
    )

    recall = (
        captured_events / (captured_events + missed_events)
        if captured_events + missed_events > 0
        else np.nan
    )

    # -------------------------------------------------------------------------------------------------------------------
    # Alarm rate
    # prediction 가능한 patient-time 기준
    # -------------------------------------------------------------------------------------------------------------------
    patient_hours = len(predictions) * grid_minutes / 60

    alarm_rate = (
        len(alarms) / patient_hours
        if patient_hours > 0
        else np.nan
    )

    median_warning_time = (
        float(np.median(warning_times))
        if warning_times
        else np.nan
    )

    mean_warning_time = (
        float(np.mean(warning_times))
        if warning_times
        else np.nan
    )

    detected_over_2h = (
        float(np.mean(np.asarray(warning_times) > 2))
        if warning_times
        else np.nan
    )

    return {
        "threshold": threshold,
        "alarms": len(alarms),
        "true_alarms": true_alarms,
        "false_alarms": false_alarms,
        "captured_events": captured_events,
        "missed_events": missed_events,
        "precision": precision,
        "recall": recall,
        "alarm_rate_per_patient_hour": alarm_rate,
        "mean_warning_time_h": mean_warning_time,
        "median_warning_time_h": median_warning_time,
        "detected_over_2h": detected_over_2h
    }


# -----------------------------------------------------------------------------------------------------------------------
# 5. Evaluate one split
# -----------------------------------------------------------------------------------------------------------------------
def evaluate_alarm_split(
    prediction_path,
    state,
    split_id,
    output_dir,
    horizon_hours=8,
    grid_minutes=60,
    n_thresholds=200
):
    predictions = pd.read_csv(prediction_path)

    predictions["gridtime"] = pd.to_datetime(
        predictions["gridtime"],
        errors="coerce"
    )

    test_stay_ids = predictions["stay_id"].unique()

    cf_events = extract_cf_events(
        state=state,
        stay_ids=test_stay_ids,
        grid_minutes=grid_minutes
    )

    # CF onset 이전 8시간 안에 실제 prediction point가 하나라도 있는 event만 평가
    horizon = pd.Timedelta(hours=horizon_hours)
    evaluable_events = []

    for event in cf_events.itertuples(index=False):
        has_prediction = (
            (predictions["stay_id"] == event.stay_id)
            & (predictions["gridtime"] < event.onset)
            & (predictions["gridtime"] >= event.onset - horizon)
        ).any()

        if has_prediction:
            evaluable_events.append({
                "stay_id": event.stay_id,
                "onset": event.onset,
                "end": event.end
            })

    cf_events = pd.DataFrame(
        evaluable_events,
        columns=["stay_id", "onset", "end"]
    )

    thresholds = build_thresholds(
        predictions["probability"],
        n_thresholds=n_thresholds
    )

    print(f"\n{'=' * 100}")
    print(f"Alarm Evaluation - Split {split_id}")
    print(f"{'=' * 100}")
    print(f"Test stays : {len(test_stay_ids):,}")
    print(f"CF events  : {len(cf_events):,}")
    print(f"Thresholds : {len(thresholds):,}")

    results = []

    for threshold in thresholds:
        result = evaluate_threshold(
            predictions=predictions,
            state=state,
            cf_events=cf_events,
            threshold=threshold,
            horizon_hours=horizon_hours,
            grid_minutes=grid_minutes
        )

        results.append(result)

    results = pd.DataFrame(results)

    # -------------------------------------------------------------------------------------------------------------------
    # Event-based PR AUC
    # -------------------------------------------------------------------------------------------------------------------
    pr = results.dropna(
        subset=["precision", "recall"]
    ).copy()

    pr = pr.sort_values("recall")

    if len(pr) >= 2:
        event_auprc = auc(
            pr["recall"],
            pr["precision"]
        )
    else:
        event_auprc = np.nan

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / f"alarm_curve_split_{split_id}.csv",
        index=False
    )

    cf_events.to_csv(
        output_dir / f"cf_events_split_{split_id}.csv",
        index=False
    )

    print(f"Event-based AUPRC: {event_auprc:.4f}")

    return results, event_auprc


# -----------------------------------------------------------------------------------------------------------------------
# 6. Select one clinically useful operating point
# -----------------------------------------------------------------------------------------------------------------------
def select_operating_point(
    results,
    target_recall=0.80
):
    valid = results.dropna(
        subset=["precision", "recall"]
    ).copy()

    eligible = valid.loc[
        valid["recall"] >= target_recall
    ]

    if len(eligible) == 0:
        selected = valid.loc[
            valid["recall"].idxmax()
        ]
    else:
        selected = eligible.loc[
            eligible["precision"].idxmax()
        ]

    return selected


# -----------------------------------------------------------------------------------------------------------------------
# 7. Run 5 splits
# -----------------------------------------------------------------------------------------------------------------------
def run_alarm_evaluation(
    prediction_root,
    state,
    output_dir,
    n_splits=5,
    horizon_hours=8,
    grid_minutes=60,
    target_recall=0.80
):
    prediction_root = Path(prediction_root)
    output_dir = Path(output_dir)

    summary_rows = []

    for split_id in range(1, n_splits + 1):
        prediction_path = (
            prediction_root /
            f"lightgbm_predictions_split_{split_id}.csv"
        )

        results, event_auprc = evaluate_alarm_split(
            prediction_path=prediction_path,
            state=state,
            split_id=split_id,
            output_dir=output_dir,
            horizon_hours=horizon_hours,
            grid_minutes=grid_minutes
        )

        operating_point = select_operating_point(
            results,
            target_recall=target_recall
        )

        summary_rows.append({
            "split_id": split_id,
            "event_auprc": event_auprc,
            "threshold": operating_point["threshold"],
            "precision": operating_point["precision"],
            "recall": operating_point["recall"],
            "alarm_rate_per_patient_hour":
                operating_point["alarm_rate_per_patient_hour"],
            "mean_warning_time_h":
                operating_point["mean_warning_time_h"],
            "median_warning_time_h":
                operating_point["median_warning_time_h"],
            "detected_over_2h":
                operating_point["detected_over_2h"]
        })

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        output_dir / "alarm_summary.csv",
        index=False
    )

    print(f"\n{'=' * 100}")
    print("[Final Early-warning Result]")
    print(f"{'=' * 100}")

    for row in summary.itertuples(index=False):
        print(
            f"Split {row.split_id} | "
            f"AUPRC: {row.event_auprc:.4f} | "
            f"Precision: {row.precision:.4f} | "
            f"Recall: {row.recall:.4f} | "
            f"Alarm/h: {row.alarm_rate_per_patient_hour:.4f} | "
            f"Warning: {row.median_warning_time_h:.2f} h"
        )

    print()
    print(
        f"Event AUPRC : "
        f"{summary['event_auprc'].mean():.4f} ± "
        f"{summary['event_auprc'].std(ddof=1):.4f}"
    )

    print(
        f"Precision   : "
        f"{summary['precision'].mean():.4f} ± "
        f"{summary['precision'].std(ddof=1):.4f}"
    )

    print(
        f"Recall      : "
        f"{summary['recall'].mean():.4f} ± "
        f"{summary['recall'].std(ddof=1):.4f}"
    )

    print(
        f"Alarm rate  : "
        f"{summary['alarm_rate_per_patient_hour'].mean():.4f} ± "
        f"{summary['alarm_rate_per_patient_hour'].std(ddof=1):.4f} "
        f"alarms/patient-hour"
    )

    print(
        f"Warning time: "
        f"{summary['median_warning_time_h'].mean():.2f} ± "
        f"{summary['median_warning_time_h'].std(ddof=1):.2f} h"
    )

    print(
        f">2h detected: "
        f"{100 * summary['detected_over_2h'].mean():.2f}%"
    )

    return summary