# AvailMatrBuilder.py
import ast
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

# ---- Shifts definition (unchanged) ----
# (sid, day_abbr, start_hour_float, end_hour_float, staff_required)
SHIFTS: List[Tuple[str, str, float, float, int]] = [
    ("S1",  "Mon", 7.25, 9,  3),
    ("S2",  "Mon", 9,    12, 4),
    ("S3",  "Mon", 12,   15, 4),
    ("S4",  "Mon", 15,   17, 4),
    ("S5",  "Mon", 17,   19, 3),
    ("S6",  "Tue", 7.25, 9,  3),
    ("S7",  "Tue", 9,    12, 4),
    ("S8",  "Tue", 12,   15, 4),
    ("S9",  "Tue", 15,   17, 4),
    ("S10", "Tue", 17,   19, 3),
    ("S11", "Wed", 7.25, 9,  3),
    ("S12", "Wed", 9,    12, 4),
    ("S13", "Wed", 12,   15, 4),
    ("S14", "Wed", 15,   17, 4),
    ("S15", "Wed", 17,   19, 3),
    ("S16", "Thu", 7.25, 9,  3),
    ("S17", "Thu", 9,    12, 4),
    ("S18", "Thu", 12,   15, 4),
    ("S19", "Thu", 15,   17, 4),
    ("S20", "Thu", 17,   19, 3),
    ("S21", "Fri", 7.25, 9,  3),
    ("S22", "Fri", 9,    12, 4),
    ("S23", "Fri", 12,   15, 4),
    ("S24", "Fri", 15,   17, 4),
    ("S25", "Sat", 10,   14, 2),
    ("S26", "Sun", 10,   14, 2),
]

DAY_ABBR_TO_COL = {
    "Mon": "MONDAY", "Tue": "TUESDAY", "Wed": "WEDNESDAY",
    "Thu": "THURSDAY", "Fri": "FRIDAY", "Sat": "SATURDAY", "Sun": "SUNDAY"
}

def _hhmm_to_float(s: str) -> float:
    # s like "07:15:00" or "17:00:00"
    s = s.strip()
    dt = datetime.strptime(s, "%H:%M:%S")
    return dt.hour + dt.minute / 60.0

def _parse_ranges(cell_str: str) -> List[Tuple[float, float]]:
    """
    Cell contains a Python-like list of ranges:
    "['07:15:00 - 09:00:00', '12:00:00 - 15:00:00']"
    """
    if not isinstance(cell_str, str):
        return []
    try:
        items = ast.literal_eval(cell_str)
    except Exception:
        return []
    out: List[Tuple[float, float]] = []
    for rng in items:
        if not isinstance(rng, str) or "-" not in rng:
            continue
        start_s, end_s = rng.split("-", 1)
        out.append((_hhmm_to_float(start_s), _hhmm_to_float(end_s)))
    return out

@dataclass(frozen=True)
class InputData:
    students: List[str]
    availability_matrix: Dict[str, Dict[Tuple[str, float, float], int]]
    shifts: List[Tuple[str, str, float, float, int]]

def load_input_from_excel(xlsx_path: str) -> InputData:
    df = pd.read_excel(xlsx_path)

    # student list
    name_col = "STUDENT NAME"
    if name_col not in df.columns:
        raise ValueError(f"Expected column '{name_col}' in {xlsx_path}")
    students = (
        df[name_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.title()
        .tolist()
    )

    # availability matrix keyed by (day_abbr, start, end)
    availability_matrix: Dict[str, Dict[Tuple[str, float, float], int]] = {
        s: {} for s in students
    }

    # build per-row availability
    for _, row in df.iterrows():
        student = str(row[name_col]).strip().title()
        for (_, d, start, end, _) in SHIFTS:
            col = DAY_ABBR_TO_COL[d]
            parsed = _parse_ranges(row.get(col, ""))
            is_available = any(a <= start and end <= b for (a, b) in parsed)
            availability_matrix[student][(d, start, end)] = int(is_available)

    return InputData(students=students, availability_matrix=availability_matrix, shifts=SHIFTS)
