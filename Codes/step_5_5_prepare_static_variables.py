import numpy as np
import pandas as pd


# ============================================================
# 5-5. Prepare Static Variables
# ============================================================
#
# Original paper static features:
#   1) Age
#   2) Surgical admission indicator
#   3) Emergency admission indicator
#   4) APACHE diagnostic group
#   5) Height
#   6) Sex
#
# Paper missing-value rule:
#   continuous static -> mean of training data
#   categorical static -> mode of training data
#
# MIMIC-IV adaptation:
#   - Age                 : patients + admissions
#   - Sex                 : patients.gender
#   - Height              : earliest available Height in current ICU stay
#   - Emergency admission : admissions.admission_type
#   - Surgical admission  : services.curr_service around ICU admission
#   - APACHE group        : unavailable in current cohort -> omitted
#
# IMPORTANT:
#   In a final train/validation/test experiment, mean/mode values
#   must be estimated from TRAINING stays only.
# ============================================================


# Explicit emergency labels in MIMIC-IV.
# "URGENT" is not automatically treated as emergency here.
EMERGENCY_ADMISSION_TYPES = {
    "EW EMER.",
    "DIRECT EMER.",
}


# Conservative service-code mapping for surgical admission.
# These are explicit surgical/surgical-specialty services.
SURGICAL_SERVICES = {
    "SURG",
    "CSURG",
    "NSURG",
    "PSURG",
    "TSURG",
    "VSURG",
    "ORTHO",
    "ENT",
    "GU",
    "GYN",
    "DENT",
}


def _mode_or_nan(series: pd.Series):
    x = series.dropna()
    if len(x) == 0:
        return np.nan
    m = x.mode()
    if len(m) == 0:
        return np.nan
    return m.iloc[0]


def _prepare_service_at_icu_admission(
    stays: pd.DataFrame,
    services: pd.DataFrame,
):
    """
    각 ICU stay의 intime 시점에 가장 가까운 hospital service를 선택.

    우선순위:
      1) ICU intime 이전/동일 시각의 가장 최근 curr_service
      2) 그런 row가 없으면 ICU intime 이후의 가장 이른 curr_service

    한 hospital admission에 service transition이 여러 번 있을 수 있으므로
    stay_id별로 따로 결정한다.
    """

    required_services = {
        "hadm_id",
        "transfertime",
        "curr_service",
    }

    missing = required_services - set(services.columns)
    if missing:
        raise ValueError(
            f"services에 필요한 column이 없습니다: {missing}"
        )

    s = services[
        ["hadm_id", "transfertime", "curr_service"]
    ].copy()

    s["transfertime"] = pd.to_datetime(
        s["transfertime"],
        errors="coerce",
    )

    s = s.dropna(
        subset=["hadm_id", "transfertime", "curr_service"]
    ).copy()

    base = stays[
        ["stay_id", "hadm_id", "intime"]
    ].copy()

    base["intime"] = pd.to_datetime(
        base["intime"],
        errors="coerce",
    )

    merged = base.merge(
        s,
        on="hadm_id",
        how="left",
    )

    merged["before_or_at_icu"] = (
        merged["transfertime"].notna()
        & (merged["transfertime"] <= merged["intime"])
    )

    # ICU 전/동시 service 중 가장 최근
    before = (
        merged.loc[merged["before_or_at_icu"]]
        .sort_values(
            ["stay_id", "transfertime"],
            ascending=[True, False],
        )
        .drop_duplicates("stay_id")
        [
            ["stay_id", "curr_service", "transfertime"]
        ]
        .rename(
            columns={
                "curr_service": "service_at_icu",
                "transfertime": "service_time",
            }
        )
    )

    found_before = set(before["stay_id"])

    # 이전 service가 없었던 stay는 이후의 가장 이른 service
    after = (
        merged.loc[
            ~merged["stay_id"].isin(found_before)
            & merged["transfertime"].notna()
            & (merged["transfertime"] > merged["intime"])
        ]
        .sort_values(
            ["stay_id", "transfertime"],
            ascending=[True, True],
        )
        .drop_duplicates("stay_id")
        [
            ["stay_id", "curr_service", "transfertime"]
        ]
        .rename(
            columns={
                "curr_service": "service_at_icu",
                "transfertime": "service_time",
            }
        )
    )

    result = base[["stay_id"]].merge(
        pd.concat(
            [before, after],
            ignore_index=True,
        ),
        on="stay_id",
        how="left",
    )

    return result


def prepare_static_variables(
    stays: pd.DataFrame,
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    services: pd.DataFrame,
    height_weight_events: pd.DataFrame,
    training_stay_ids=None,
):
    """
    Returns
    -------
    static_features : pd.DataFrame
        stay-level static features after mean/mode imputation

    static_audit : pd.DataFrame
        raw + imputation flags + source metadata

    report : dict
        summary
    """

    # ========================================================
    # 0. Required columns
    # ========================================================

    required_stays = {
        "stay_id",
        "subject_id",
        "hadm_id",
        "intime",
    }

    required_adm = {
        "subject_id",
        "hadm_id",
        "admittime",
        "admission_type",
    }

    required_pat = {
        "subject_id",
        "anchor_age",
        "anchor_year",
        "gender",
    }

    required_hw = {
        "stay_id",
        "charttime",
        "variable",
        "value",
    }

    for name, df, required in [
        ("stays", stays, required_stays),
        ("admissions", admissions, required_adm),
        ("patients", patients, required_pat),
        ("height_weight_events", height_weight_events, required_hw),
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{name}에 필요한 column이 없습니다: {missing}"
            )


    # ========================================================
    # 1. Base stay table
    # ========================================================

    static = stays[
        [
            "stay_id",
            "subject_id",
            "hadm_id",
            "intime",
        ]
    ].copy()

    static["intime"] = pd.to_datetime(
        static["intime"],
        errors="coerce",
    )


    # ========================================================
    # 2. Age + Emergency admission
    # ========================================================

    adm = admissions[
        [
            "subject_id",
            "hadm_id",
            "admittime",
            "admission_type",
        ]
    ].copy()

    adm["admittime"] = pd.to_datetime(
        adm["admittime"],
        errors="coerce",
    )

    static = static.merge(
        adm,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="many_to_one",
    )

    pat = patients[
        [
            "subject_id",
            "anchor_age",
            "anchor_year",
            "gender",
        ]
    ].copy()

    static = static.merge(
        pat,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    static["Age_raw"] = (
        static["anchor_age"]
        + (
            static["admittime"].dt.year
            - static["anchor_year"]
        )
    ).astype(float)

    static["Sex_raw"] = (
        static["gender"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    static.loc[
        ~static["Sex_raw"].isin(["M", "F"]),
        "Sex_raw",
    ] = pd.NA

    static["admission_type_clean"] = (
        static["admission_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    static["Emergency_raw"] = np.where(
        static["admission_type_clean"].isna(),
        np.nan,
        static["admission_type_clean"]
        .isin(EMERGENCY_ADMISSION_TYPES)
        .astype(float),
    )


    # ========================================================
    # 3. Height
    # ========================================================

    height = height_weight_events.loc[
        height_weight_events["variable"] == "Height",
        [
            "stay_id",
            "charttime",
            "value",
        ],
    ].copy()

    height["charttime"] = pd.to_datetime(
        height["charttime"],
        errors="coerce",
    )

    height["value"] = pd.to_numeric(
        height["value"],
        errors="coerce",
    )

    height = (
        height.dropna(
            subset=["stay_id", "charttime", "value"]
        )
        .sort_values(["stay_id", "charttime"])
        .drop_duplicates("stay_id", keep="first")
        .rename(
            columns={
                "value": "Height_raw",
                "charttime": "height_time",
            }
        )
    )

    static = static.merge(
        height[
            ["stay_id", "Height_raw", "height_time"]
        ],
        on="stay_id",
        how="left",
    )


    # ========================================================
    # 4. Surgical admission
    # ========================================================

    service = _prepare_service_at_icu_admission(
        stays=stays,
        services=services,
    )

    static = static.merge(
        service,
        on="stay_id",
        how="left",
    )

    static["service_clean"] = (
        static["service_at_icu"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    static["Surgical_raw"] = np.where(
        static["service_clean"].isna(),
        np.nan,
        static["service_clean"]
        .isin(SURGICAL_SERVICES)
        .astype(float),
    )


    # ========================================================
    # 5. APACHE diagnostic group
    # ========================================================
    #
    # 현재 3000-stay MIMIC-IV subset에서는 usable APACHE
    # diagnostic group source가 확인되지 않았으므로 생성하지 않는다.
    # ========================================================


    # ========================================================
    # 6. Training subset for imputation statistics
    # ========================================================

    if training_stay_ids is None:
        train_mask = pd.Series(
            True,
            index=static.index,
        )
        stats_source = "current cohort"
    else:
        train_ids = set(training_stay_ids)
        train_mask = static["stay_id"].isin(train_ids)
        stats_source = "provided training stays"

        if int(train_mask.sum()) == 0:
            raise ValueError(
                "training_stay_ids와 현재 stays가 겹치지 않습니다."
            )


    # continuous -> mean
    age_mean = float(
        static.loc[train_mask, "Age_raw"].mean()
    )

    height_mean = float(
        static.loc[train_mask, "Height_raw"].mean()
    )

    # categorical/binary -> mode
    sex_mode = _mode_or_nan(
        static.loc[train_mask, "Sex_raw"]
    )

    emergency_mode = _mode_or_nan(
        static.loc[train_mask, "Emergency_raw"]
    )

    surgical_mode = _mode_or_nan(
        static.loc[train_mask, "Surgical_raw"]
    )


    # ========================================================
    # 7. Imputation
    # ========================================================

    static["Age_imputed"] = static["Age_raw"].isna()
    static["Height_imputed"] = static["Height_raw"].isna()
    static["Sex_imputed"] = static["Sex_raw"].isna()
    static["Emergency_imputed"] = static["Emergency_raw"].isna()
    static["Surgical_imputed"] = static["Surgical_raw"].isna()

    static["Age"] = static["Age_raw"].fillna(age_mean)
    static["Height"] = static["Height_raw"].fillna(height_mean)
    static["Sex"] = static["Sex_raw"].fillna(sex_mode)

    static["Emergency admission"] = (
        static["Emergency_raw"]
        .fillna(emergency_mode)
        .astype("int8")
    )

    static["Surgical admission"] = (
        static["Surgical_raw"]
        .fillna(surgical_mode)
        .astype("int8")
    )

    static_features = static[
        [
            "stay_id",
            "Age",
            "Sex",
            "Height",
            "Emergency admission",
            "Surgical admission",
        ]
    ].copy()


    # ========================================================
    # 8. Report
    # ========================================================

    report = {
        "stays": int(len(static)),
        "static_features_available": 5,
        "apache_available": False,
        "stats_source": stats_source,

        "age_missing_before": int(static["Age_raw"].isna().sum()),
        "height_missing_before": int(static["Height_raw"].isna().sum()),
        "sex_missing_before": int(static["Sex_raw"].isna().sum()),
        "emergency_missing_before": int(static["Emergency_raw"].isna().sum()),
        "surgical_missing_before": int(static["Surgical_raw"].isna().sum()),

        "age_mean": age_mean,
        "height_mean": height_mean,
        "sex_mode": sex_mode,
        "emergency_mode": emergency_mode,
        "surgical_mode": surgical_mode,

        "emergency_stays": int(
            static_features["Emergency admission"].sum()
        ),
        "surgical_stays": int(
            static_features["Surgical admission"].sum()
        ),
    }


    print("=" * 70)
    print("5-5. Static Variable Preparation")
    print("=" * 70)

    print(f"Stays: {report['stays']}")
    print("Original paper static variables: 6")
    print("Available MIMIC-IV static variables: 5")
    print("APACHE diagnostic group: unavailable -> omitted")
    print(f"Imputation statistics source: {stats_source}")

    print("\n[Missing before imputation]")
    print(f"Age: {report['age_missing_before']}")
    print(f"Height: {report['height_missing_before']}")
    print(f"Sex: {report['sex_missing_before']}")
    print(f"Emergency admission: {report['emergency_missing_before']}")
    print(f"Surgical admission: {report['surgical_missing_before']}")

    print("\n[Imputation values]")
    print(f"Age mean: {age_mean:.3f}")
    print(f"Height mean: {height_mean:.3f}")
    print(f"Sex mode: {sex_mode}")
    print(f"Emergency mode: {emergency_mode}")
    print(f"Surgical mode: {surgical_mode}")

    print("\n[Final binary static features]")
    print(
        "Emergency admissions:",
        report["emergency_stays"],
        f"({100 * report['emergency_stays'] / len(static):.2f}%)"
    )
    print(
        "Surgical admissions:",
        report["surgical_stays"],
        f"({100 * report['surgical_stays'] / len(static):.2f}%)"
    )

    print("\n[Admission type distribution]")
    print(
        static["admission_type_clean"]
        .fillna("<missing>")
        .value_counts()
        .to_string()
    )

    print("\n[Service at ICU admission distribution]")
    print(
        static["service_clean"]
        .fillna("<missing>")
        .value_counts()
        .to_string()
    )

    print("\n[Final static feature examples]")
    print(
        static_features
        .head(20)
        .to_string(index=False)
    )

    print("\n[Important]")
    print(
        "Continuous static missing values are imputed with the mean; "
        "categorical/binary missing values with the mode."
    )
    print(
        "Emergency=1 only for explicit MIMIC-IV emergency admission labels: "
        f"{sorted(EMERGENCY_ADMISSION_TYPES)}"
    )
    print(
        "Surgical=1 is based on the explicit service-code set defined "
        "in SURGICAL_SERVICES."
    )
    print(
        "APACHE diagnostic group is not fabricated or mode-filled because "
        "the variable itself is unavailable in the current cohort."
    )
    print(
        "For final experiments, pass training_stay_ids so mean/mode values "
        "are estimated from training stays only."
    )
    print("=" * 70)

    return (
        static_features,
        static,
        report,
    )