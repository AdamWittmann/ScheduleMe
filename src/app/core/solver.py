# solver.py
from dataclasses import dataclass
from typing import Dict, Tuple, List

from ortools.sat.python import cp_model

from AvailMatrBuilder import load_input_from_excel

# ---- CONFIG ----
MAX_WEEKLY_HOURS = 20
# Optional daily policies (toggle as needed; defaults keep your original behavior)
ENFORCE_MAX_SHIFTS_PER_DAY = False
MAX_SHIFTS_PER_DAY = 2            # e.g., set to 1 if you want at most one shift/day/person
ENFORCE_MAX_DAILY_HOURS = False
MAX_DAILY_HOURS = 8

# ---- Load input (no hard-coded paths) ----
INPUT_XLSX = "src/data/TestStudentAvailability_V3.xlsx"   # put your path here
data = load_input_from_excel(INPUT_XLSX)
students = data.students
availability_matrix = data.availability_matrix
shifts = data.shifts

# Precompute maps
ShiftId = Tuple[str, float, float]
shift_lengths: Dict[ShiftId, int] = {}
SCALE = 4  # 15-minute granularity because 7.25 (= 7:15)
for (_, day, start, end, _) in shifts:
    shift_lengths[(day, start, end)] = int(round((end - start) * SCALE))

# Build model
model = cp_model.CpModel()

# Decision vars x[(day,start,end), student] in {0,1}
x: Dict[Tuple[ShiftId, str], cp_model.IntVar] = {}
for (_, day, start, end, _) in shifts:
    sid = (day, start, end)
    for s in students:
        x[(sid, s)] = model.NewBoolVar(f"x_{day}_{start}_{end}__{s}")

# 1) Coverage: exactly required staff per shift
for (sid_label, day, start, end, required) in shifts:
    sid = (day, start, end)
    model.Add(sum(x[(sid, s)] for s in students) == required)

# 2) Availability: if not available, force 0
for (_, day, start, end, _) in shifts:
    sid = (day, start, end)
    for s in students:
        if availability_matrix.get(s, {}).get(sid, 0) == 0:
            model.Add(x[(sid, s)] == 0)

# 3) Weekly hour limit
for s in students:
    total_hours_scaled = sum(x[((day, start, end), s)] * shift_lengths[(day, start, end)]
                             for (_, day, start, end, _) in shifts)
    model.Add(total_hours_scaled <= MAX_WEEKLY_HOURS * SCALE)

# 4) Optional: at-most-K shifts per day per student
if ENFORCE_MAX_SHIFTS_PER_DAY:
    by_day: Dict[Tuple[str, str], List[ShiftId]] = {}
    for (_, day, start, end, _) in shifts:
        by_day.setdefault((day, ""), []).append((day, start, end))
    for s in students:
        for (day, _), shift_ids in by_day.items():
            model.Add(sum(x[((day, st, en), s)] for (_, st, en) in shift_ids) <= MAX_SHIFTS_PER_DAY)

# 5) Optional: max daily hours per student
if ENFORCE_MAX_DAILY_HOURS:
    by_day: Dict[str, List[ShiftId]] = {}
    for (_, day, start, end, _) in shifts:
        by_day.setdefault(day, []).append((day, start, end))
    for s in students:
        for day, sid_list in by_day.items():
            model.Add(
                sum(x[(sid, s)] * shift_lengths[sid] for sid in sid_list)
                <= MAX_DAILY_HOURS * SCALE
            )

# 6) Fairness objective: minimize (max - min) total hours
total_hours: Dict[str, cp_model.IntVar] = {
    s: model.NewIntVar(0, MAX_WEEKLY_HOURS * SCALE, f"tot_{s}") for s in students
}
for s in students:
    model.Add(total_hours[s] == sum(
        x[((day, start, end), s)] * shift_lengths[(day, start, end)]
        for (_, day, start, end, _) in shifts
    ))

max_h = model.NewIntVar(0, MAX_WEEKLY_HOURS * SCALE, "max_h")
min_h = model.NewIntVar(0, MAX_WEEKLY_HOURS * SCALE, "min_h")
model.AddMaxEquality(max_h, list(total_hours.values()))
model.AddMinEquality(min_h, list(total_hours.values()))
model.Minimize(max_h - min_h)

# ---- Solve ----
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30.0  # adjust as needed
status = solver.Solve(model)

# ---- Report ----
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print("Final Shift Assignment:")
    for (sid_label, day, start, end, required) in shifts:
        sid = (day, start, end)
        assigned = [s for s in students if solver.Value(x[(sid, s)]) == 1]
        print(f"  {sid_label} ({day} {start}-{end}) {len(assigned)}/{required}: {', '.join(assigned)}")

    print("\nTotal Assigned Hours per Student:")
    for s in students:
        print(f"  {s}: {solver.Value(total_hours[s]) / SCALE:.2f} hours")

    print(f"\nFairness gap (max - min): {(solver.Value(max_h) - solver.Value(min_h)) / SCALE:.2f} hours")
else:
    print("No feasible schedule found.")
    print(f"Status = {solver.StatusName(status)}")
