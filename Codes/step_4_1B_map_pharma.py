import re
import pandas as pd

def _norm(x):
    if pd.isna(x):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def map_pharma_concepts(
    inputevents_classified: pd.DataFrame,
    mimic_vars_path,
    table4_path,
):

    # 현재 MIMIC-IV input ITEMID 요약
    items = (
        inputevents_classified
        .groupby(["itemid", "label"], dropna=False)
        .agg(
            rows=("itemid", "size"),
            stays=("stay_id", "nunique"),
            continuous_rows=(
                "administration_type",
                lambda x: int((x == "continuous").sum())
            ),
            noncontinuous_rows=(
                "administration_type",
                lambda x: int((x == "non-continuous").sum())
            ),
        )
        .reset_index()
    )

    items["name_norm"] = items["label"].apply(_norm)

    # mimic_vars.csv direct로 pharma mapping
    mimic = pd.read_csv(mimic_vars_path)

    include = (
        mimic["include"].astype(str).str.lower().eq("true")
    )

    mid = mimic["mID"].fillna("").astype(str)

    pharma_mid = (
        mid.str.lower().str.startswith("pm")
        | mid.str.lower().str.startswith("m_pm")
    )

    input_table = (
        mimic["table"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains("inputevents")
    )

    direct = mimic.loc[
        include & pharma_mid & input_table
    ].copy()

    direct["itemid"] = pd.to_numeric(
        direct["ITEM_ID"],
        errors="coerce"
    )

    direct = direct.dropna(subset=["itemid"])
    direct["itemid"] = direct["itemid"].astype(int)

    direct["pharma_variable"] = direct["varname2"]
    direct["pharma_variable"] = direct["pharma_variable"].fillna(
        direct["varname (mimic?)"]
    )

    direct = direct[
        ["itemid", "mID", "pharma_variable"]
    ].rename(columns={"mID": "pharma_id"})

    direct["mapping_basis"] = "mimic_vars_direct"
    direct = direct.drop_duplicates("itemid")

    # Table4 master list
    master = pd.read_excel(
        table4_path,
        sheet_name="master list"
    )

    eligible_pm_ids = set()

    for _, r in master.iterrows():

        pharma_id = str(r["ID"]).strip()

        if not pharma_id.lower().startswith("pm"):
            continue

        mimic_varids = r["MIMIC varids"]

        if pd.isna(mimic_varids):
            continue

        if re.search(r"\d+", str(mimic_varids)):
            eligible_pm_ids.add(pharma_id)


    # Table4 drugs exact-name mapping
    drugs = pd.read_excel(
        table4_path,
        sheet_name="drugs"
    )

    drugs["temp: ID"] = drugs["temp: ID"].ffill()
    drugs["temp: bern name"] = drugs["temp: bern name"].ffill()

    refs = []

    for col in ["drug", "constituent drugs (if relevant)"]:

        tmp = drugs[
            ["temp: ID", "temp: bern name", col]
        ].dropna(subset=[col]).copy()

        tmp = tmp[
            tmp["temp: ID"]
            .astype(str)
            .isin(eligible_pm_ids)
        ].copy()

        tmp["name_norm"] = tmp[col].apply(_norm)

        tmp = tmp.rename(
            columns={
                "temp: ID": "pharma_id",
                "temp: bern name": "pharma_variable",
            }
        )

        refs.append(
            tmp[
                ["name_norm", "pharma_id", "pharma_variable"]
            ]
        )

    if refs:
        table4_ref = (
            pd.concat(refs, ignore_index=True)
            .drop_duplicates()
        )

        table4_match = items.merge(
            table4_ref,
            on="name_norm",
            how="inner"
        )

        # 한 ITEMID가 여러 target에 연결되면 제외
        counts = (
            table4_match
            .groupby("itemid")["pharma_id"]
            .nunique()
        )

        safe_ids = set(
            counts[counts == 1].index
        )

        table4_match = (
            table4_match[
                table4_match["itemid"].isin(safe_ids)
            ][
                ["itemid", "pharma_id", "pharma_variable"]
            ]
            .drop_duplicates("itemid")
        )

    else:
        table4_match = pd.DataFrame(
            columns=[
                "itemid",
                "pharma_id",
                "pharma_variable",
            ]
        )

    table4_match["mapping_basis"] = "table4_exact"

    # 합치기 (mimic_vars direct가 우선)
    direct_ids = set(direct["itemid"])

    table4_only = table4_match[
        ~table4_match["itemid"].isin(direct_ids)
    ].copy()

    pharma_map = pd.concat(
        [direct, table4_only],
        ignore_index=True
    ).drop_duplicates("itemid")


    # 현재 데이터에 실제 존재하는 ITEMID만
    pharma_map = pharma_map.merge(
        items[
            [
                "itemid",
                "label",
                "rows",
                "stays",
                "continuous_rows",
                "noncontinuous_rows",
            ]
        ],
        on="itemid",
        how="inner"
    )

    pharma_map = pharma_map.sort_values(
        ["pharma_variable", "itemid"]
    ).reset_index(drop=True)


    # 실제 event rows에 canonical pharma 이름 붙이기
    pharma_events = inputevents_classified.merge(
        pharma_map[
            [
                "itemid",
                "pharma_id",
                "pharma_variable",
                "mapping_basis",
            ]
        ],
        on="itemid",
        how="inner"
    )
    
    pharma_report = (
        pharma_events
        .groupby(
            ["pharma_id", "pharma_variable"],
            dropna=False
        )
        .agg(
            rows=("itemid", "size"),
            stays=("stay_id", "nunique"),
            source_itemids=("itemid", "nunique"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )


    print("=" * 70)
    print("4-1B. Pharmaceutical Concept Mapping")
    print("=" * 70)

    print(f"Current input ITEMIDs: {items['itemid'].nunique()}")
    print(f"Mapped pharma ITEMIDs: {pharma_map['itemid'].nunique()}")
    print(
        f"Canonical pharma variables: "
        f"{pharma_map['pharma_variable'].nunique()}"
    )

    print(
        "mimic_vars direct ITEMIDs:",
        int((pharma_map["mapping_basis"] == "mimic_vars_direct").sum())
    )

    print(
        "Table4-added ITEMIDs:",
        int((pharma_map["mapping_basis"] == "table4_exact").sum())
    )

    print(
        "Table4 pharma IDs allowed by numeric MIMIC varids:",
        sorted(eligible_pm_ids)
    )


    print("\n[Canonical pharmaceutical variables]")
    print(pharma_report.to_string(index=False))


    print("\n[ITEMID → canonical pharmaceutical variable]")
    print(
        pharma_map[
            [
                "itemid",
                "label",
                "pharma_id",
                "pharma_variable",
                "mapping_basis",
                "continuous_rows",
                "noncontinuous_rows",
            ]
        ].to_string(index=False)
    )


    print("\n[Important]")
    print(
        "mimic_vars direct pharma mappings are preserved even if "
        "Table4 master-list MIMIC varids is blank."
    )
    print(
        "The MIMIC-varids filter is applied ONLY to additional "
        "Table4 drugs-sheet mappings."
    )
    print("No rate/presence aggregation was performed.")

    print("=" * 70)


    return (
        pharma_events,
        pharma_map,
        pharma_report,
    )