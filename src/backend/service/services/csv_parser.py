"""
CSV Parser for experimental reflectivity data.
File: src/backend/service/services/csv_parser.py

Converts CSV files with Q, Reflectivity (and optional Error) columns
into the canonical numpy format expected by the ML pipeline.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CSVParseResult:
    """Result of parsing a CSV file."""
    payload: np.ndarray  # Shape: (1, 2, L) or (1, 3, L) if errors included
    metadata: dict[str, Any]
    report: dict[str, Any]


# Common column name variations
Q_COLUMN_NAMES = {'q', 'q (å⁻¹)', 'q (a^-1)', 'q (1/a)', 'q_value', 'qz', 'q (å-1)'}
R_COLUMN_NAMES = {'r', 'reflectivity', 'refl', 'intensity', 'i', 'r_value', 'counts'}
E_COLUMN_NAMES = {'error', 'err', 'e', 'sigma', 'uncertainty', 'dr', 'r_error'}


def parse_reflectivity_csv(file_content: str | bytes, filename: str = "data.csv") -> CSVParseResult:
    """
    Parse a CSV file containing reflectivity data.
    
    Expected formats:
    - Q, Reflectivity
    - Q, Reflectivity, Error
    - With or without headers
    
    Returns canonical numpy array format (1, 2, L) for the ML pipeline.
    """
    if isinstance(file_content, bytes):
        file_content = file_content.decode('utf-8')
    
    actions: list[str] = []
    warnings: list[str] = []
    
    # Parse CSV
    reader = csv.reader(io.StringIO(file_content))
    rows = list(reader)
    
    if len(rows) < 2:
        raise ValueError("CSV must have at least 2 rows (header + data or 2 data rows)")
    
    # Detect if first row is header
    first_row = rows[0]
    has_header = False
    q_col, r_col, e_col = 0, 1, None
    
    # Check if first row looks like a header
    try:
        float(first_row[0])
    except ValueError:
        has_header = True
        actions.append("detected_header_row")
        
        # Find columns by name
        headers_lower = [h.lower().strip() for h in first_row]
        
        q_col = _find_column(headers_lower, Q_COLUMN_NAMES, "Q")
        r_col = _find_column(headers_lower, R_COLUMN_NAMES, "Reflectivity")
        e_col = _find_column_optional(headers_lower, E_COLUMN_NAMES)
        
        if e_col is not None:
            actions.append("found_error_column")
    
    # Extract data rows
    data_rows = rows[1:] if has_header else rows
    
    # Parse numeric values
    q_values = []
    r_values = []
    e_values = [] if e_col is not None else None
    
    for i, row in enumerate(data_rows):
        if len(row) < 2:
            warnings.append(f"Skipped row {i + 1}: insufficient columns")
            continue
        
        try:
            q = float(row[q_col])
            r = float(row[r_col])
            
            # Validate values
            if not np.isfinite(q) or not np.isfinite(r):
                warnings.append(f"Skipped row {i + 1}: non-finite values")
                continue
            
            if q < 0:
                warnings.append(f"Skipped row {i + 1}: negative Q value")
                continue
                
            if r < 0:
                # Reflectivity can be negative in some cases (contrast matching)
                # but warn about it
                if len(warnings) < 5:  # Don't spam warnings
                    warnings.append(f"Row {i + 1}: negative reflectivity (R={r})")
            
            q_values.append(q)
            r_values.append(r)
            
            if e_values is not None and e_col is not None:
                try:
                    e = float(row[e_col]) if len(row) > e_col else 0.0
                    e_values.append(e)
                except ValueError:
                    e_values.append(0.0)
                    
        except (ValueError, IndexError) as exc:
            warnings.append(f"Skipped row {i + 1}: {exc}")
            continue
    
    if len(q_values) < 3:
        raise ValueError(f"CSV must have at least 3 valid data points, found {len(q_values)}")
    
    # Convert to numpy arrays
    q_arr = np.array(q_values, dtype=np.float64)
    r_arr = np.array(r_values, dtype=np.float64)
    
    # Sort by Q if not already sorted
    if not np.all(np.diff(q_arr) > 0):
        sort_idx = np.argsort(q_arr)
        q_arr = q_arr[sort_idx]
        r_arr = r_arr[sort_idx]
        if e_values is not None:
            e_values = [e_values[i] for i in sort_idx]
        actions.append("sorted_by_q")
        warnings.append("Data was not sorted by Q - sorted automatically")
    
    # Remove duplicate Q values
    unique_q, unique_idx = np.unique(q_arr, return_index=True)
    if len(unique_q) < len(q_arr):
        q_arr = unique_q
        r_arr = r_arr[unique_idx]
        if e_values is not None:
            e_values = [e_values[i] for i in unique_idx]
        actions.append("removed_duplicate_q")
        warnings.append(f"Removed {len(q_values) - len(unique_q)} duplicate Q values")
    
    # Build output array
    num_points = len(q_arr)
    
    if e_values is not None:
        payload = np.zeros((1, 3, num_points), dtype=np.float64)
        payload[0, 0, :] = q_arr
        payload[0, 1, :] = r_arr
        payload[0, 2, :] = np.array(e_values, dtype=np.float64)
        channels = 3
    else:
        payload = np.zeros((1, 2, num_points), dtype=np.float64)
        payload[0, 0, :] = q_arr
        payload[0, 1, :] = r_arr
        channels = 2
    
    # Build metadata and report
    q_range = (float(np.min(q_arr)), float(np.max(q_arr)))
    r_range = (float(np.min(r_arr)), float(np.max(r_arr)))
    
    metadata = {
        "shape": list(payload.shape),
        "num_points": num_points,
        "channels": channels,
        "q_range": q_range,
        "r_range": r_range,
        "has_errors": e_values is not None,
        "source_format": "csv",
        "canonicalized": True,
    }
    
    report = {
        "role": "experimental_nr_csv",
        "filename": filename,
        "original_rows": len(rows),
        "parsed_points": num_points,
        "skipped_rows": len(rows) - (1 if has_header else 0) - num_points,
        "q_range": q_range,
        "r_range": r_range,
        "has_errors": e_values is not None,
        "actions": actions,
        "warnings": warnings[:10],  # Limit warnings
        "total_warnings": len(warnings),
    }
    
    return CSVParseResult(payload=payload, metadata=metadata, report=report)


def _find_column(headers: list[str], valid_names: set[str], col_name: str) -> int:
    """Find a required column by checking against valid names."""
    for i, h in enumerate(headers):
        if h in valid_names:
            return i
    # If not found by name, use default position
    raise ValueError(f"Could not find {col_name} column. Expected one of: {valid_names}")


def _find_column_optional(headers: list[str], valid_names: set[str]) -> int | None:
    """Find an optional column by checking against valid names."""
    for i, h in enumerate(headers):
        if h in valid_names:
            return i
    return None


def parse_csv_file(file_path: Path) -> CSVParseResult:
    """Parse a CSV file from disk."""
    content = file_path.read_text(encoding='utf-8')
    return parse_reflectivity_csv(content, filename=file_path.name)