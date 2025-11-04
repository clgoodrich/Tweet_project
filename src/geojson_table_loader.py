"""Utilities for loading GeoJSON properties into ArcGIS tables without geometry.

This module provides helper functions that treat a GeoJSON FeatureCollection as
an attribute lookup table.  It extracts the properties from each feature,
infers reasonable field types, and writes the values into a file geodatabase
table.  The workflow avoids creating feature geometry so the data can be used
as a pure lookup keyed by textual attributes (e.g., GPE strings).

The functions expect to be executed inside an ArcGIS Pro Python environment
because they rely on :mod:`arcpy`.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import arcpy  # type: ignore


def _try_parse_datetime(value: Any) -> Optional[datetime]:
    """Attempt to parse an ISO-8601 timestamp from *value*.

    The GeoJSON export stores timestamps with ``+00:00`` offsets.  ArcPy accepts
    naive or timezone-aware ``datetime`` objects, so we normalise strings by
    replacing the ``Z`` suffix (if present) and delegating to
    :func:`datetime.fromisoformat`.
    """
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _infer_numeric_type(values: Iterable[Any]) -> Optional[str]:
    """Infer an ArcGIS numeric field type from *values*.

    Returns ``"LONG"`` for integral values, ``"DOUBLE"`` for general numeric
    data, or ``None`` when the values are not numeric.
    """
    has_float = False
    for value in values:
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            return "SHORT"
        if isinstance(value, int):
            continue
        if isinstance(value, float):
            has_float = True
            continue
        # Strings that can be coerced to numbers
        if isinstance(value, str):
            try:
                float(value)
            except ValueError:
                return None
            else:
                if value.isdigit():
                    continue
                has_float = True
                continue
        return None
    if has_float:
        return "DOUBLE"
    return "LONG"


def _determine_field_spec(values: List[Any]) -> Tuple[str, Optional[int]]:
    """Determine the ArcGIS field type and length for *values*."""
    non_null = [v for v in values if v not in (None, "")]
    if not non_null:
        return "TEXT", 255

    dt_value = _try_parse_datetime(non_null[0])
    if dt_value is not None:
        if all(_try_parse_datetime(v) is not None for v in non_null):
            return "DATE", None

    numeric_type = _infer_numeric_type(non_null)
    if numeric_type:
        return numeric_type, None

    # Default to text; size the field based on observed maximum length but cap
    # at 2048 characters to avoid extremely wide fields from alternate names.
    max_len = max(len(str(v)) for v in non_null)
    max_len = min(max(32, max_len), 2048)
    return "TEXT", max_len


def _coerce_value(value: Any, field_type: str) -> Any:
    """Coerce *value* into a representation compatible with *field_type*."""
    if value in (None, ""):
        return None

    if field_type == "DATE":
        parsed = _try_parse_datetime(value)
        return parsed

    if field_type in {"LONG", "SHORT"}:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    if field_type == "DOUBLE":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return str(value)


def geojson_properties_to_table(
    geojson_path: str,
    gdb_path: str,
    table_name: str,
) -> str:
    """Load GeoJSON properties into a geodatabase table and return its path.

    Parameters
    ----------
    geojson_path:
        Path to a GeoJSON file containing a ``FeatureCollection``.
    gdb_path:
        Target file geodatabase where the table will be created.
    table_name:
        Name of the table that will store the properties.  The name is validated
        with :func:`arcpy.ValidateTableName`.
    """
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"GeoJSON path does not exist: {geojson_path}")

    with open(geojson_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if data.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON must be a FeatureCollection")

    features = data.get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError("GeoJSON does not contain any features")

    # Gather property names and values
    field_values: Dict[str, List[Any]] = {}
    for feature in features:
        props = feature.get("properties", {}) or {}
        if not isinstance(props, dict):
            continue
        for key, value in props.items():
            field_values.setdefault(key, []).append(value)

    if not field_values:
        raise ValueError("GeoJSON features do not contain attribute properties")

    validated_name = arcpy.ValidateTableName(table_name, gdb_path)
    table_path = os.path.join(gdb_path, validated_name)

    if arcpy.Exists(table_path):
        arcpy.management.Delete(table_path)

    arcpy.management.CreateTable(gdb_path, validated_name)

    # Define fields
    field_specs: Dict[str, Tuple[str, Optional[int]]] = {}
    field_name_map: Dict[str, str] = {}
    for field_name, values in field_values.items():
        field_type, length = _determine_field_spec(values)
        field_specs[field_name] = (field_type, length)

        add_kwargs = {}
        if field_type == "TEXT" and length:
            add_kwargs["field_length"] = length
        valid_name = arcpy.ValidateFieldName(field_name, table_path)
        field_name_map[field_name] = valid_name
        arcpy.management.AddField(
            table_path,
            valid_name,
            field_type,
            **add_kwargs,
        )

    # Insert records
    original_field_order = list(field_specs.keys())
    cursor_fields = [field_name_map[name] for name in original_field_order]
    with arcpy.da.InsertCursor(table_path, cursor_fields) as cursor:
        for feature in features:
            props = feature.get("properties", {}) or {}
            row = []
            for field_name in original_field_order:
                value = props.get(field_name)
                field_type, _length = field_specs[field_name]
                row.append(_coerce_value(value, field_type))
            cursor.insertRow(row)

    return table_path

__all__ = ["geojson_properties_to_table"]
