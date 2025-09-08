# backend/main.py
import os
import io
import re
import math
import base64
from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv
from PIL import Image

# Try to import OpenAI client (ensure you have the right SDK installed)
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None  # we'll surface a clear error if OCR is attempted without client

load_dotenv()

ALLOWED_MAIN_EXTENSIONS = {".xlsx"}
ALLOWED_NEW_EXTENSIONS = {".xlsx", ".csv", ".png", ".jpg", ".jpeg"}

EXPECTED_COLUMNS = [
    "Weight",
    "IIB State",
    "FGI State",
    "FGI Zone",
    "Status",
    "OD Outflow",
    "TP Outflow",
]

# Canonical India States and UTs (uppercase, standardized ampersands)
_CANON_STATES = [
    "ANDHRA PRADESH","ARUNACHAL PRADESH","ASSAM","BIHAR","CHHATTISGARH","GOA","GUJARAT","HARYANA","HIMACHAL PRADESH","JHARKHAND","KARNATAKA","KERALA","MADHYA PRADESH","MAHARASHTRA","MANIPUR","MEGHALAYA","MIZORAM","NAGALAND","ODISHA","PUNJAB","RAJASTHAN","SIKKIM","TAMIL NADU","TELANGANA","TRIPURA","UTTAR PRADESH","UTTARAKHAND","WEST BENGAL",
    "ANDAMAN & NICOBAR ISLANDS","CHANDIGARH","DADRA & NAGAR HAVELI AND DAMAN & DIU","DELHI","JAMMU & KASHMIR","LADAKH","LAKSHADWEEP","PUDUCHERRY"
]

# Build a normalization map using simplified keys and common aliases/abbreviations
_DEF_STATE_MAP: Dict[str, str] = {}

def _state_key(s: str) -> str:
    k = (s or "").upper()
    k = k.replace("\xa0", " ").replace(".", " ")
    k = re.sub(r"\s*&\s*", "&", k)
    k = k.replace("AND", "&")
    k = re.sub(r"\s+", "", k)
    return k

for name in _CANON_STATES:
    _DEF_STATE_MAP[_state_key(name)] = name

# Common aliases
_aliases = {
    "ORISSA": "ODISHA",
    "NCTDELHI": "DELHI",
    "NEWDELHI": "DELHI",
    "DELHINCT": "DELHI",
    "PONDICHERRY": "PUDUCHERRY",
    "DADRA&NAGARHAVELI": "DADRA & NAGAR HAVELI AND DAMAN & DIU",
    "DAMAN&DIU": "DADRA & NAGAR HAVELI AND DAMAN & DIU",
    "DNHDD": "DADRA & NAGAR HAVELI AND DAMAN & DIU",
    "J&K": "JAMMU & KASHMIR",
    "J&KSTATE": "JAMMU & KASHMIR",
    "JAMMUANDKASHMIR": "JAMMU & KASHMIR",
    "JAMMU&KASHMIR": "JAMMU & KASHMIR",
    "UTTARANCHAL": "UTTARAKHAND",
}
for k, v in _aliases.items():
    _DEF_STATE_MAP[_state_key(k)] = v

# Two-letter abbreviations
_abbrev = {
    "AP": "ANDHRA PRADESH","AR": "ARUNACHAL PRADESH","AS": "ASSAM","BR": "BIHAR","CG": "CHHATTISGARH","GA": "GOA","GJ": "GUJARAT","HR": "HARYANA","HP": "HIMACHAL PRADESH","JH": "JHARKHAND","KA": "KARNATAKA","KL": "KERALA","MP": "MADHYA PRADESH","MH": "MAHARASHTRA","MN": "MANIPUR","ML": "MEGHALAYA","MZ": "MIZORAM","NL": "NAGALAND","OD": "ODISHA","OR": "ODISHA","PB": "PUNJAB","RJ": "RAJASTHAN","SK": "SIKKIM","TN": "TAMIL NADU","TS": "TELANGANA","TG": "TELANGANA","TR": "TRIPURA","UP": "UTTAR PRADESH","UK": "UTTARAKHAND","WB": "WEST BENGAL","AN": "ANDAMAN & NICOBAR ISLANDS","CH": "CHANDIGARH","DD": "DADRA & NAGAR HAVELI AND DAMAN & DIU","DN": "DADRA & NAGAR HAVELI AND DAMAN & DIU","DL": "DELHI","JK": "JAMMU & KASHMIR","LA": "LADAKH","LD": "LAKSHADWEEP","PY": "PUDUCHERRY"
}
for k, v in _abbrev.items():
    _DEF_STATE_MAP[_state_key(k)] = v

app = FastAPI(title="Report Comparator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())


def normalize_and_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    canonical = { _norm(c): c for c in EXPECTED_COLUMNS }
    aliases = {
        "ibstate": "IIB State",
        "iibstate": "IIB State",
        "fgistate": "FGI State",
        "fgizone": "FGI Zone",
        "zone": "FGI Zone",
        "status": "Status",
        "stat": "Status",
        "odoutflow": "OD Outflow",
        "odoutflo": "OD Outflow",
        "odout": "OD Outflow",
        "od": "OD Outflow",
        "tpoutflow": "TP Outflow",
        "tpoutflo": "TP Outflow",
        "tpout": "TP Outflow",
    }
    lookup = {**canonical, **aliases}

    renamed = {}
    for col in list(df.columns):
        n = _norm(col)
        if n in lookup:
            renamed[col] = lookup[n]
        else:
            renamed[col] = col
    return df.rename(columns=renamed)


def _normalize_yes_no_token(s: str) -> str:
    if s in {"Y", "YES", "TRUE"}:
        return "YES"
    if s in {"N", "NO", "FALSE"}:
        return "NO"
    return s


def _normalize_state_value(val: object) -> str:
    s = str(val or "").strip()
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    key = _state_key(s)
    return _DEF_STATE_MAP.get(key, s.upper())


def _normalize_zone_value(val: object) -> str:
    s = str(val or "").strip().upper()
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.replace("-", " ")
    m = re.match(r"^(NORTH|SOUTH|EAST|WEST|CENTRAL)\s*([0-9])$", s)
    if m:
        name, num = m.group(1), m.group(2)
        if name == "CENTRAL":
            return "CENTRAL"
        return f"{name} {num}"
    return s


def _normalize_weight_value(val: object) -> str:
    s = str(val or "").strip().upper()
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*\+\s*", "+", s)
    patterns = [
        (r"^12K-20K$", "12K-20K"),
        (r"^20K-40K$", "20K-40K"),
        (r"^3\.?5K-7\.?5K$", "3.5K-7.5K"),
        (r"^7\.?5K-12K$", "7.5K-12K"),
        (r"^40K\+?$", "40K+"),
        (r"^3W\s*GCV$", "3W GCV"),
        (r"^AUTO$", "AUTO"),
        (r"^BOLERO$", "BOLERO"),
        (r"^TAXI$", "TAXI"),
        (r"^TRACTOR$", "TRACTOR"),
        (r"^BELOW\s*2\.5\s*TONS?$", "BELOW 2.5 TONS"),
        (r"^BELOW\s*3\.5\s*TONS?$", "BELOW 3.5 TONS"),
    ]
    for pat, canon in patterns:
        if re.match(pat, s, flags=re.IGNORECASE):
            return canon
    return s


def _normalize_key_value(val: object, col_name: str) -> str:
    s = str(val or "").strip()
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.upper()
    if col_name == "Weight":
        return _normalize_weight_value(s)
    if col_name == "Status":
        return _normalize_yes_no_token(s)
    if col_name in ("IIB State", "FGI State"):
        return _normalize_state_value(s)
    if col_name == "FGI Zone":
        return _normalize_zone_value(s)
    return s


def _apply_key_normalization(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    for c in key_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda v: _normalize_key_value(v, c))
    return df


def read_main_report(file_bytes: bytes) -> pd.DataFrame:
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read main xlsx: {exc}")
    df = normalize_and_rename_columns(df)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Main report missing columns: {missing}")
    return df[EXPECTED_COLUMNS].copy()


def _guess_mime_from_image(image_bytes: bytes) -> str:
    try:
        im = Image.open(io.BytesIO(image_bytes))
        fmt = (im.format or '').lower()
        if fmt in ("jpeg", "jpg"):
            return "image/jpeg"
        if fmt == "png":
            return "image/png"
        return "application/octet-stream"
    except Exception:
        return "application/octet-stream"


def _extract_csv_from_text(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```(?:csv)?\n([\s\S]*?)\n```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: take lines that look CSV-like
    lines = [ln for ln in text.splitlines() if ',' in ln]
    if lines:
        return "\n".join(lines)
    return text.strip()


# --- OCR ---
def ocr_image_to_dataframe(image_bytes: bytes) -> pd.DataFrame:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="OpenAI Python client not installed or importable")

    client = OpenAI(api_key=api_key)

    # First try: direct image payload (supported in newer SDKs). If the SDK doesn't accept bytes here,
    # we'll catch the exception and fallback to base64 data-url approach for compatibility.
    raw_text = None
    mime = _guess_mime_from_image(image_bytes)

    # Try direct image send
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the table strictly as CSV. Prefer the following column names when present: "
                        + ",".join(EXPECTED_COLUMNS)
                        + ". Only output CSV, no commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Return only CSV including the header row(s)."},
                        # Direct image attempt (may raise if client doesn't support it)
                        {"type": "image", "image": image_bytes},
                    ],
                },
            ],
        )
        raw_text = completion.choices[0].message.content
    except Exception as primary_exc:
        # Fallback: use data URL base64 (compatibility). This works even on older clients.
        try:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:{mime};base64,{b64}"
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract the table strictly as CSV. Prefer the following column names when present: "
                            + ",".join(EXPECTED_COLUMNS)
                            + ". Only output CSV, no commentary."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Return only CSV including the header row(s)."},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
            raw_text = completion.choices[0].message.content
        except Exception as fallback_exc:
            # Report both for debugging
            raise HTTPException(
                status_code=500,
                detail=f"OCR failed (direct attempt: {primary_exc}; fallback: {fallback_exc})"
            )

    csv_text = _extract_csv_from_text(raw_text)
    if not csv_text:
        raise HTTPException(status_code=500, detail="OCR returned no CSV-like text")

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse OCR CSV: {exc}")

    df = normalize_and_rename_columns(df)
    available = [c for c in EXPECTED_COLUMNS if c in df.columns]
    if not available:
        raise HTTPException(status_code=400, detail="OCR result has no recognizable columns")
    # Apply key normalization to OCR content too
    key_cols = [c for c in ("Weight", "IIB State", "FGI State", "FGI Zone", "Status") if c in df.columns]
    df = _apply_key_normalization(df, key_cols)
    return df[available].copy()


# --- Read new report (csv/xlsx/img) ---
def read_new_report(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    name_lower = (file_name or "").lower()
    try:
        if name_lower.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(file_bytes))
        elif name_lower.endswith(".csv"):
            # allow for possible BOM or encoding issues
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif any(name_lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
            df = ocr_image_to_dataframe(file_bytes)
        else:
            raise HTTPException(status_code=400, detail="Unsupported new report format")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read new report: {exc}")

    df = normalize_and_rename_columns(df)
    available = [c for c in EXPECTED_COLUMNS if c in df.columns]
    if not available:
        raise HTTPException(status_code=400, detail="New report has no recognizable columns")
    # Normalize keys early for non-image sources as well
    key_cols = [c for c in ("Weight", "IIB State", "FGI State", "FGI Zone", "Status") if c in df.columns]
    df = _apply_key_normalization(df, key_cols)
    return df[available].copy()


def _parse_percent_like_to_float(val):
    try:
        if isinstance(val, str):
            clean = val.strip().replace('%', '').replace(',', '')
            if clean == '':
                return float('nan')
            f = float(clean)
        else:
            f = float(val)
        # If between 0 and 1, treat as fraction; convert to percent value scale
        if 0 <= f <= 1:
            return f * 100.0
        return f
    except Exception:
        return float('nan')


def _format_percent_two_decimals(val) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return ""
    try:
        f = float(val)
    except Exception:
        return str(val)
    return f"{f:.2f}%"


def compare_and_update(main_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    key_cols_all = ["Weight", "IIB State", "FGI State", "FGI Zone", "Status"]
    value_cols_all = ["OD Outflow", "TP Outflow"]

    # Keys we can actually use from new (to find matches). If new doesn't have any of the identifying
    # columns we cannot proceed.
    key_cols = [c for c in key_cols_all if c in new_df.columns]
    if not key_cols:
        raise HTTPException(status_code=400, detail="New report is missing all identifying key columns")

    # Value columns present in either
    value_cols = [c for c in value_cols_all if c in new_df.columns or c in main_df.columns]

    # Normalize keys aggressively for both frames (create copies)
    main_keys = main_df.copy()
    new_keys = new_df.copy()
    main_keys = _apply_key_normalization(main_keys, key_cols)
    new_keys = _apply_key_normalization(new_keys, key_cols)

    # Numeric copies for comparison
    main_num = main_keys.copy()
    new_num = new_keys.copy()
    for v in value_cols_all:
        if v in main_num.columns:
            main_num[v] = main_num[v].apply(_parse_percent_like_to_float)
        if v in new_num.columns:
            new_num[v] = new_num[v].apply(_parse_percent_like_to_float)

    # merge left on main to keep main rows as canonical order
    merged_num = main_num.merge(new_num, on=key_cols, how="left", suffixes=("_main", "_new"))

    # Build output frame from merged_num (ensures alignment)
    out = pd.DataFrame()
    for c in key_cols_all:
        # If the key column exists in merged frame, take it; else try to fill with empty string
        out[c] = merged_num[c] if c in merged_num.columns else ""

    # Original values (formatted)
    if "OD Outflow_main" in merged_num:
        out["OD Outflow"] = merged_num["OD Outflow_main"].apply(_format_percent_two_decimals)
    else:
        out["OD Outflow"] = ""

    if "TP Outflow_main" in merged_num:
        out["TP Outflow"] = merged_num["TP Outflow_main"].apply(_format_percent_two_decimals)
    else:
        out["TP Outflow"] = ""

    # Determine updated values (only populate when a meaningful new value exists and differs)
    def updated_or_blank(row: pd.Series, col: str):
        new_val = row.get(f"{col}_new")
        main_val = row.get(f"{col}_main")
        if pd.notna(new_val) and pd.notna(main_val) and new_val != main_val:
            return _format_percent_two_decimals(new_val)
        if pd.notna(new_val) and (pd.isna(main_val) or main_val != main_val):  # main missing
            return _format_percent_two_decimals(new_val)
        return ""

    out["Updated OD Outflow"] = merged_num.apply(lambda r: updated_or_blank(r, "OD Outflow"), axis=1)
    out["Updated TP Outflow"] = merged_num.apply(lambda r: updated_or_blank(r, "TP Outflow"), axis=1)

    # Ensure column order consistent and present
    desired = [*key_cols_all, "OD Outflow", "TP Outflow", "Updated OD Outflow", "Updated TP Outflow"]
    # Keep only those that exist (safe if some value columns were entirely absent)
    final_cols = [c for c in desired if c in out.columns]
    return out[final_cols].copy()


@app.post("/compare", response_class=StreamingResponse)
async def compare_reports(main_file: UploadFile = File(...), new_file: UploadFile = File(...)):
    main_name = main_file.filename or "main.xlsx"
    new_name = new_file.filename or "new"

    if not main_name.lower().endswith(tuple(ALLOWED_MAIN_EXTENSIONS)):
        raise HTTPException(status_code=400, detail="Main report must be .xlsx")

    main_bytes = await main_file.read()
    new_bytes = await new_file.read()

    main_df = read_main_report(main_bytes)
    new_df = read_new_report(new_name, new_bytes)

    updated_df = compare_and_update(main_df, new_df)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        updated_df.to_excel(writer, index=False, sheet_name="Updated")
    output.seek(0)

    headers = {"Content-Disposition": 'attachment; filename="updated_report.xlsx"'}
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})
