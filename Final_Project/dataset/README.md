# Dataset — UK Road Safety Data 2018

Source: UK Department for Transport, *Road Safety Data* (Accidents, Vehicles, Casualties tables for 2018).

| File | Rows | One row is... |
|---|---|---|
| `Accidents_2018.csv` | 122,635 | one accident (32 columns) |
| `Vehicles_2018.csv` | 226,409 | one vehicle involved |
| `Casualties_2018.csv` | 160,597 | one injured person |

The tables are linked with `Accident_Index`. Most columns are codes (e.g. `Weather_Conditions = 1` = fine, no high winds) and `-1` means missing. `Accident_Severity`: 1 = Fatal, 2 = Serious, 3 = Slight.
