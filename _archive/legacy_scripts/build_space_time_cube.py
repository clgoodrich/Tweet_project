"""Build a weighted, fused space-time cube from multiple feature classes.

This script assembles signals from polygon and point feature classes at a
common 5 km (configurable) grid resolution and exports a space-time cube for
ArcGIS Pro analysis.

Example command line usage::

    python build_space_time_cube.py \
        --in-states path/to/states \
        --in-counties path/to/counties \
        --in-cities path/to/cities \
        --in-facilities path/to/facilities \
        --time-field EVENT_TIME \
        --workspace-gdb path/to/workspace.gdb \
        --out-folder path/to/outputs \
        --grid-sr 5070 \
        --cell-km 5 \
        --time-step "1 Days" \
        --n-jobs 4

"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import logging
import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import arcpy
    from arcpy import env
    from arcpy.sa import KernelDensity, ZonalStatisticsAsTable
except ImportError as exc:  # pragma: no cover - ArcPy only available inside ArcGIS Pro
    raise ImportError("This script requires ArcPy and must be executed within ArcGIS Pro.") from exc


CELL_ID_FIELD = "cell_id"
VAL_BIN_FIELD = "val_bin"
VAL_CUM_FIELD = "val_cum"
START_FIELD = "START_TIME"
END_FIELD = "END_TIME"
SUMMARY_CSV_NAME = "time_bin_summary.csv"
LOG_FILENAME = "build_space_time_cube.log"

POLY_WEIGHTS = {
    "states": 0.1,
    "counties": 0.5,
    "cities": 1.0,
}
FACILITY_WEIGHT = 5.0


# ---------------------------------------------------------------------------
# Logging and utilities
# ---------------------------------------------------------------------------


def make_logger(out_folder: str) -> logging.Logger:
    """Create a logger that writes to the output folder."""
    os.makedirs(out_folder, exist_ok=True)
    log_path = os.path.join(out_folder, LOG_FILENAME)

    logger = logging.getLogger("space_time_cube")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs when rerunning in notebook sessions.
    while logger.handlers:
        logger.handlers.pop()

    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler(sys.stdout)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging initialized: %s", log_path)
    return logger


# ---------------------------------------------------------------------------
# Spatial reference helpers
# ---------------------------------------------------------------------------


def ensure_projected(in_fc: Optional[str], grid_sr: arcpy.SpatialReference, workspace: str, logger: logging.Logger) -> Optional[str]:
    """Project the input feature class to the grid spatial reference if necessary."""
    if not in_fc or not arcpy.Exists(in_fc):
        return None

    desc = arcpy.Describe(in_fc)
    in_sr = desc.spatialReference
    if in_sr and not in_sr.name:
        in_sr = None

    if in_sr and in_sr.factoryCode == grid_sr.factoryCode:
        logger.info("Input %s already in desired spatial reference (WKID=%s)", in_fc, grid_sr.factoryCode)
        return in_fc

    projected_name = f"{os.path.basename(in_fc)}_proj"
    projected_fc = os.path.join(workspace, arcpy.ValidateTableName(projected_name, workspace))

    logger.info("Projecting %s to %s -> %s", in_fc, grid_sr.factoryCode, projected_fc)
    arcpy.management.Project(in_fc, projected_fc, grid_sr)
    return projected_fc


# ---------------------------------------------------------------------------
# Fishnet creation
# ---------------------------------------------------------------------------


def _union_extent(extents: Iterable[arcpy.Extent]) -> Optional[arcpy.Extent]:
    extents = list(extents)
    if not extents:
        return None
    xmin = min(ext.XMin for ext in extents)
    ymin = min(ext.YMin for ext in extents)
    xmax = max(ext.XMax for ext in extents)
    ymax = max(ext.YMax for ext in extents)
    return arcpy.Extent(xmin, ymin, xmax, ymax)


def make_fishnet(extent: arcpy.Extent, grid_sr: arcpy.SpatialReference, cell_km: float, workspace: str, logger: logging.Logger) -> str:
    """Create a national fishnet covering the provided extent."""
    if extent is None:
        raise ValueError("Cannot create fishnet without extent.")

    cell_m = float(cell_km) * 1000.0
    if cell_m <= 0:
        raise ValueError("cell_km must be positive")

    width = extent.XMax - extent.XMin
    height = extent.YMax - extent.YMin
    n_cols = max(1, int(math.ceil(width / cell_m)))
    n_rows = max(1, int(math.ceil(height / cell_m)))

    logger.info(
        "Creating fishnet with %s rows x %s cols (cell %.1f km) covering extent %s",
        n_rows,
        n_cols,
        cell_km,
        extent,
    )

    origin_coord = f"{extent.XMin} {extent.YMin}"
    y_axis_coord = f"{extent.XMin} {extent.YMin + 10.0}"
    opposite_corner = f"{extent.XMin + n_cols * cell_m} {extent.YMin + n_rows * cell_m}"

    fishnet_name = arcpy.ValidateTableName(f"fishnet_{int(cell_km)}km", workspace)
    fishnet_fc = os.path.join(workspace, fishnet_name)
    if arcpy.Exists(fishnet_fc):
        logger.info("Removing existing fishnet %s", fishnet_fc)
        arcpy.management.Delete(fishnet_fc)

    arcpy.analysis.CreateFishnet(
        out_feature_class=fishnet_fc,
        origin_coord=origin_coord,
        y_axis_coord=y_axis_coord,
        cell_width=cell_m,
        cell_height=cell_m,
        number_rows=n_rows,
        number_columns=n_cols,
        corner_coord=opposite_corner,
        geometry_type="POLYGON",
        labels="NO_LABELS",
        template=None,
    )
    arcpy.management.DefineProjection(fishnet_fc, grid_sr)

    if CELL_ID_FIELD not in [f.name for f in arcpy.ListFields(fishnet_fc)]:
        arcpy.management.AddField(fishnet_fc, CELL_ID_FIELD, "LONG")

    logger.info("Assigning sequential cell IDs to fishnet")
    with arcpy.da.UpdateCursor(fishnet_fc, ["OID@", CELL_ID_FIELD]) as cursor:
        for oid, _ in cursor:
            cursor.updateRow((oid, oid))

    return fishnet_fc


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _parse_time_step(step: str) -> dt.timedelta:
    value, unit = step.strip().split(maxsplit=1)
    amount = float(value)
    unit = unit.lower()
    if unit.startswith("hour"):
        return dt.timedelta(hours=amount)
    if unit.startswith("day"):
        return dt.timedelta(days=amount)
    if unit.startswith("week"):
        return dt.timedelta(weeks=amount)
    if unit.startswith("minute"):
        return dt.timedelta(minutes=amount)
    if unit.startswith("second"):
        return dt.timedelta(seconds=amount)
    raise ValueError(f"Unsupported time step unit: {unit}")


def enumerate_time_bins(start: dt.datetime, end: dt.datetime, time_step: str) -> List[Tuple[dt.datetime, dt.datetime]]:
    delta = _parse_time_step(time_step)
    if delta.total_seconds() <= 0:
        raise ValueError("time_step must be positive")

    bins: List[Tuple[dt.datetime, dt.datetime]] = []
    current = start
    while current < end:
        next_edge = min(end, current + delta)
        bins.append((current, next_edge))
        current = next_edge
    return bins


def _format_timestamp(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def _build_time_query(feature_class: str, time_field: str, start: dt.datetime, end: dt.datetime) -> str:
    field = arcpy.AddFieldDelimiters(feature_class, time_field)
    start_str = _format_timestamp(start)
    end_str = _format_timestamp(end)
    return f"{field} >= timestamp '{start_str}' AND {field} < timestamp '{end_str}'"


def poly_signal_for_bin(polys_fc: Optional[str], fishnet_fc: str, time_field: str, time_bin: Tuple[dt.datetime, dt.datetime], weight: float, logger: logging.Logger) -> Dict[int, float]:
    if not polys_fc or not arcpy.Exists(polys_fc):
        return {}

    start, end = time_bin
    layer_name = arcpy.CreateUniqueName("poly_bin_layer")
    arcpy.management.MakeFeatureLayer(polys_fc, layer_name)
    out_fc = None
    try:
        where_clause = _build_time_query(polys_fc, time_field, start, end)
        arcpy.management.SelectLayerByAttribute(layer_name, "NEW_SELECTION", where_clause)

        count = int(arcpy.management.GetCount(layer_name)[0])
        if count == 0:
            logger.info("No polygon features from %s within bin %s -> %s", polys_fc, start, end)
            return {}

        out_fc = os.path.join("in_memory", arcpy.CreateUniqueName("poly_join"))
        arcpy.analysis.SpatialJoin(
            target_features=fishnet_fc,
            join_features=layer_name,
            out_feature_class=out_fc,
            join_operation="JOIN_ONE_TO_ONE",
            join_type="KEEP_COMMON",
            match_option="INTERSECT",
        )

        results: Dict[int, float] = {}
        with arcpy.da.SearchCursor(out_fc, [CELL_ID_FIELD, "Join_Count"]) as cursor:
            for cell_id, join_count in cursor:
                if join_count and join_count > 0:
                    results[int(cell_id)] = weight

        logger.info("Polygon bin %s-%s (%s) produced %s active cells", start, end, polys_fc, len(results))
        return results
    finally:
        if out_fc and arcpy.Exists(out_fc):
            arcpy.management.Delete(out_fc)
        if arcpy.Exists(layer_name):
            arcpy.management.Delete(layer_name)


def facility_signal_for_bin(points_fc: Optional[str], fishnet_fc: str, time_field: str, time_bin: Tuple[dt.datetime, dt.datetime], weight: float, kde_bandwidth_m: float, cell_km: float, logger: logging.Logger) -> Dict[int, float]:
    if not points_fc or not arcpy.Exists(points_fc):
        return {}

    start, end = time_bin
    layer_name = arcpy.CreateUniqueName("facility_layer")
    arcpy.management.MakeFeatureLayer(points_fc, layer_name)
    kd_path = None
    zonal_table = None
    out_fc = None
    try:
        where_clause = _build_time_query(points_fc, time_field, start, end)
        arcpy.management.SelectLayerByAttribute(layer_name, "NEW_SELECTION", where_clause)
        count = int(arcpy.management.GetCount(layer_name)[0])
        if count == 0:
            logger.info("No facilities within bin %s -> %s", start, end)
            return {}

        results: Dict[int, float] = {}
        cell_m = float(cell_km) * 1000.0
        if kde_bandwidth_m and kde_bandwidth_m > 0:
            logger.info("Running Kernel Density for facilities (%s features)", count)
            kd_raster = KernelDensity(layer_name, population_field=None, cell_size=cell_m, search_radius=kde_bandwidth_m)
            kd_path = os.path.join("in_memory", arcpy.CreateUniqueName("kd"))
            kd_raster.save(kd_path)

            zonal_table = os.path.join("in_memory", arcpy.CreateUniqueName("zonal"))
            ZonalStatisticsAsTable(fishnet_fc, CELL_ID_FIELD, kd_path, zonal_table, statistics_type="MEAN")
            with arcpy.da.SearchCursor(zonal_table, [CELL_ID_FIELD, "MEAN"]) as cursor:
                for cell_id, mean_val in cursor:
                    if mean_val is not None:
                        results[int(cell_id)] = float(mean_val) * weight
        else:
            logger.info("Aggregating facilities to fishnet cells via spatial join (%s features)", count)
            out_fc = os.path.join("in_memory", arcpy.CreateUniqueName("fac_join"))
            arcpy.analysis.SpatialJoin(
                target_features=fishnet_fc,
                join_features=layer_name,
                out_feature_class=out_fc,
                join_operation="JOIN_ONE_TO_ONE",
                join_type="KEEP_COMMON",
                match_option="INTERSECT",
            )
            with arcpy.da.SearchCursor(out_fc, [CELL_ID_FIELD, "Join_Count"]) as cursor:
                for cell_id, join_count in cursor:
                    if join_count and join_count > 0:
                        results[int(cell_id)] = float(join_count) * weight

        logger.info("Facility bin %s-%s produced %s active cells", start, end, len(results))
        return results
    finally:
        if kd_path and arcpy.Exists(kd_path):
            arcpy.management.Delete(kd_path)
        if zonal_table and arcpy.Exists(zonal_table):
            arcpy.management.Delete(zonal_table)
        if out_fc and arcpy.Exists(out_fc):
            arcpy.management.Delete(out_fc)
        if arcpy.Exists(layer_name):
            arcpy.management.Delete(layer_name)


# ---------------------------------------------------------------------------
# Accumulation and table writing
# ---------------------------------------------------------------------------


def accumulate_bins(bin_values: Sequence[Dict[int, float]]) -> Tuple[List[Dict[int, float]], List[Dict[int, float]], Sequence[int]]:
    cumulative: Dict[int, float] = {}
    seen: set[int] = set()
    cumulative_by_bin: List[Dict[int, float]] = []
    per_bin_ordered: List[Dict[int, float]] = []

    for bin_dict in bin_values:
        per_bin_ordered.append(dict(bin_dict))

    for bin_dict in per_bin_ordered:
        seen.update(bin_dict.keys())
        bin_cumulative: Dict[int, float] = {}
        for cell_id in seen:
            incremental = bin_dict.get(cell_id, 0.0)
            cumulative[cell_id] = cumulative.get(cell_id, 0.0) + incremental
            bin_cumulative[cell_id] = cumulative[cell_id]
        cumulative_by_bin.append(bin_cumulative)

    sorted_cells = sorted(seen)
    return per_bin_ordered, cumulative_by_bin, sorted_cells


def _create_output_table(table_path: str, workspace: str) -> None:
    if arcpy.Exists(table_path):
        arcpy.management.Delete(table_path)
    arcpy.management.CreateTable(workspace, os.path.basename(table_path))
    arcpy.management.AddField(table_path, CELL_ID_FIELD, "LONG")
    arcpy.management.AddField(table_path, START_FIELD, "DATE")
    arcpy.management.AddField(table_path, END_FIELD, "DATE")
    arcpy.management.AddField(table_path, VAL_BIN_FIELD, "DOUBLE")
    arcpy.management.AddField(table_path, VAL_CUM_FIELD, "DOUBLE")


def write_long_table(
    bins: Sequence[Tuple[dt.datetime, dt.datetime]],
    val_bin_by_bin: Sequence[Dict[int, float]],
    val_cum_by_bin: Sequence[Dict[int, float]],
    active_cells: Sequence[int],
    out_table: str,
    workspace: str,
    logger: logging.Logger,
) -> None:
    logger.info("Writing long table to %s", out_table)
    _create_output_table(out_table, workspace)

    records = 0
    fields = [CELL_ID_FIELD, START_FIELD, END_FIELD, VAL_BIN_FIELD, VAL_CUM_FIELD]
    with arcpy.da.InsertCursor(out_table, fields) as cursor:
        for (start, end), bin_vals, cum_vals in zip(bins, val_bin_by_bin, val_cum_by_bin):
            for cell_id in active_cells:
                bin_value = float(bin_vals.get(cell_id, 0.0))
                cum_value = float(cum_vals.get(cell_id, 0.0))
                cursor.insertRow((cell_id, start, end, bin_value, cum_value))
                records += 1
    logger.info("Inserted %s records into %s", records, out_table)


# ---------------------------------------------------------------------------
# Space-time cube
# ---------------------------------------------------------------------------


def create_stc_from_defined_locations(
    fishnet_fc: str,
    table: str,
    out_nc: str,
    time_step: str,
    logger: logging.Logger,
) -> None:
    logger.info("Creating space-time cube at %s", out_nc)
    if os.path.exists(out_nc):
        os.remove(out_nc)
    arcpy.stats.CreateSpaceTimeCubeFromDefinedLocations(
        in_features=fishnet_fc,
        out_space_time_cube=out_nc,
        unique_id_field=CELL_ID_FIELD,
        start_time_field=START_FIELD,
        time_step_interval=time_step,
        end_time_field=END_FIELD,
        related_table=table,
        value_fields=[[VAL_BIN_FIELD, "VALUE"], [VAL_CUM_FIELD, "VALUE"]],
    )


# ---------------------------------------------------------------------------
# Quality control exports
# ---------------------------------------------------------------------------


def _compute_stats(values: Iterable[float]) -> Dict[str, float]:
    data = sorted(values)
    if not data:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    n = len(data)
    total = sum(data)
    median = data[n // 2] if n % 2 == 1 else (data[n // 2 - 1] + data[n // 2]) / 2.0
    p90_index = min(n - 1, int(math.ceil(0.9 * n)) - 1)
    return {
        "count": n,
        "min": data[0],
        "mean": total / n,
        "median": median,
        "p90": data[p90_index],
        "max": data[-1],
    }


def _write_summary_csv(
    bins: Sequence[Tuple[dt.datetime, dt.datetime]],
    val_bin_by_bin: Sequence[Dict[int, float]],
    val_cum_by_bin: Sequence[Dict[int, float]],
    out_folder: str,
    logger: logging.Logger,
) -> str:
    csv_path = os.path.join(out_folder, SUMMARY_CSV_NAME)
    logger.info("Writing time bin summary CSV to %s", csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "bin_index",
            "start_time",
            "end_time",
            "val_bin_count",
            "val_bin_min",
            "val_bin_mean",
            "val_bin_median",
            "val_bin_p90",
            "val_bin_max",
            "val_cum_count",
            "val_cum_min",
            "val_cum_mean",
            "val_cum_median",
            "val_cum_p90",
            "val_cum_max",
        ])
        for idx, ((start, end), bin_vals, cum_vals) in enumerate(zip(bins, val_bin_by_bin, val_cum_by_bin)):
            bin_stats = _compute_stats(bin_vals.values())
            cum_stats = _compute_stats(cum_vals.values())
            writer.writerow([
                idx,
                start.isoformat(),
                end.isoformat(),
                bin_stats["count"],
                bin_stats["min"],
                bin_stats["mean"],
                bin_stats["median"],
                bin_stats["p90"],
                bin_stats["max"],
                cum_stats["count"],
                cum_stats["min"],
                cum_stats["mean"],
                cum_stats["median"],
                cum_stats["p90"],
                cum_stats["max"],
            ])
    return csv_path


def _make_slice_table(
    bin_vals: Dict[int, float],
    cum_vals: Dict[int, float],
    time_bin: Tuple[dt.datetime, dt.datetime],
    workspace: str,
) -> str:
    table_name = arcpy.ValidateTableName(arcpy.CreateUniqueName("slice_tbl"), workspace)
    table_path = os.path.join(workspace, table_name)
    arcpy.management.CreateTable(workspace, table_name)
    arcpy.management.AddField(table_path, CELL_ID_FIELD, "LONG")
    arcpy.management.AddField(table_path, START_FIELD, "DATE")
    arcpy.management.AddField(table_path, END_FIELD, "DATE")
    arcpy.management.AddField(table_path, VAL_BIN_FIELD, "DOUBLE")
    arcpy.management.AddField(table_path, VAL_CUM_FIELD, "DOUBLE")
    with arcpy.da.InsertCursor(table_path, [CELL_ID_FIELD, START_FIELD, END_FIELD, VAL_BIN_FIELD, VAL_CUM_FIELD]) as cursor:
        for cell_id, val in bin_vals.items():
            cursor.insertRow((cell_id, time_bin[0], time_bin[1], float(val), float(cum_vals.get(cell_id, 0.0))))
    return table_path


def qc_exports(
    fishnet_fc: str,
    bins: Sequence[Tuple[dt.datetime, dt.datetime]],
    val_bin_by_bin: Sequence[Dict[int, float]],
    val_cum_by_bin: Sequence[Dict[int, float]],
    workspace: str,
    out_folder: str,
    logger: logging.Logger,
) -> Tuple[str, str, str]:
    summary_csv = _write_summary_csv(bins, val_bin_by_bin, val_cum_by_bin, out_folder, logger)

    if not bins:
        return summary_csv, "", ""

    first_table = _make_slice_table(val_bin_by_bin[0], val_cum_by_bin[0], bins[0], workspace)
    last_table = _make_slice_table(val_bin_by_bin[-1], val_cum_by_bin[-1], bins[-1], workspace)

    first_fc = os.path.join(workspace, "fishnet_first_bin")
    last_fc = os.path.join(workspace, "fishnet_last_bin")
    for path in (first_fc, last_fc):
        if arcpy.Exists(path):
            arcpy.management.Delete(path)

    arcpy.management.CopyFeatures(fishnet_fc, first_fc)
    arcpy.management.CopyFeatures(fishnet_fc, last_fc)

    arcpy.management.JoinField(first_fc, CELL_ID_FIELD, first_table, CELL_ID_FIELD, [VAL_BIN_FIELD, VAL_CUM_FIELD, START_FIELD, END_FIELD])
    arcpy.management.JoinField(last_fc, CELL_ID_FIELD, last_table, CELL_ID_FIELD, [VAL_BIN_FIELD, VAL_CUM_FIELD, START_FIELD, END_FIELD])

    arcpy.management.Delete(first_table)
    arcpy.management.Delete(last_table)

    logger.info("Created QC feature classes: %s, %s", first_fc, last_fc)
    return summary_csv, first_fc, last_fc


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def _feature_time_range(feature_class: str, time_field: str) -> Optional[Tuple[dt.datetime, dt.datetime]]:
    if not feature_class or not arcpy.Exists(feature_class):
        return None
    values: List[dt.datetime] = []
    with arcpy.da.SearchCursor(feature_class, [time_field]) as cursor:
        for (value,) in cursor:
            if value:
                values.append(value)
    if not values:
        return None
    return min(values), max(values)


def determine_time_extent(
    time_ranges: Sequence[Tuple[dt.datetime, dt.datetime]],
    start_override: Optional[str],
    end_override: Optional[str],
    logger: logging.Logger,
) -> Tuple[dt.datetime, dt.datetime]:
    valid_ranges = [rng for rng in time_ranges if rng]
    if start_override:
        start_time = dt.datetime.fromisoformat(start_override)
    else:
        if not valid_ranges:
            raise ValueError("Could not determine start time from empty inputs. Provide start_time parameter.")
        start_time = min(r[0] for r in valid_ranges)

    if end_override:
        end_time = dt.datetime.fromisoformat(end_override)
    else:
        if not valid_ranges:
            raise ValueError("Could not determine end time from empty inputs. Provide end_time parameter.")
        end_time = max(r[1] for r in valid_ranges)

    if start_time >= end_time:
        raise ValueError("start_time must be earlier than end_time")

    logger.info("Using temporal extent: %s -> %s", start_time, end_time)
    return start_time, end_time


def collect_extent(inputs: Sequence[Optional[str]]) -> Optional[arcpy.Extent]:
    extents = []
    for fc in inputs:
        if fc and arcpy.Exists(fc):
            extents.append(arcpy.Describe(fc).extent)
    return _union_extent(extents)


def process_bins(
    bins: Sequence[Tuple[dt.datetime, dt.datetime]],
    fishnet_fc: str,
    layers: Dict[str, Optional[str]],
    time_field: str,
    weights: Dict[str, float],
    facilities_fc: Optional[str],
    kde_bandwidth_m: float,
    cell_km: float,
    logger: logging.Logger,
) -> List[Dict[int, float]]:
    results: List[Dict[int, float]] = []
    for start, end in bins:
        bin_total: Dict[int, float] = {}
        bin_tuple = (start, end)
        for name, fc in layers.items():
            if fc:
                partial = poly_signal_for_bin(fc, fishnet_fc, time_field, bin_tuple, weights[name], logger)
                for cell_id, val in partial.items():
                    bin_total[cell_id] = bin_total.get(cell_id, 0.0) + val
        if facilities_fc:
            facility_vals = facility_signal_for_bin(facilities_fc, fishnet_fc, time_field, bin_tuple, FACILITY_WEIGHT, kde_bandwidth_m, cell_km, logger)
            for cell_id, val in facility_vals.items():
                bin_total[cell_id] = bin_total.get(cell_id, 0.0) + val
        results.append(bin_total)
        logger.info("Aggregated bin %s -> %s with %s active cells", start, end, len(bin_total))
    return results


# ---------------------------------------------------------------------------
# Argument parsing and main execution
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a weighted space-time cube from multiple sources.")
    parser.add_argument("--in-states", dest="in_states", help="Input states polygon feature class", default=None)
    parser.add_argument("--in-counties", dest="in_counties", help="Input counties polygon feature class", default=None)
    parser.add_argument("--in-cities", dest="in_cities", help="Input cities polygon feature class", default=None)
    parser.add_argument("--in-facilities", dest="in_facilities", help="Input facilities point feature class", default=None)
    parser.add_argument("--time-field", dest="time_field", required=True, help="Time field shared across inputs")
    parser.add_argument("--workspace-gdb", dest="workspace_gdb", required=True, help="Workspace file geodatabase")
    parser.add_argument("--out-folder", dest="out_folder", required=True, help="Output folder for netCDF and CSVs")
    parser.add_argument("--grid-sr", dest="grid_sr", required=True, type=int, help="Projected spatial reference WKID")
    parser.add_argument("--cell-km", dest="cell_km", type=float, default=5.0, help="Fishnet cell size in kilometers")
    parser.add_argument("--time-step", dest="time_step", required=True, help="Time step interval (e.g., '1 Days')")
    parser.add_argument("--start-time", dest="start_time", help="Start time ISO-8601 (optional)")
    parser.add_argument("--end-time", dest="end_time", help="End time ISO-8601 (optional)")
    parser.add_argument("--kde-bandwidth-m", dest="kde_bandwidth_m", type=float, default=0.0, help="Facility kernel density bandwidth in meters (0 to disable)")
    parser.add_argument("--n-jobs", dest="n_jobs", type=int, default=1, help="Parallel processing factor")
    return parser.parse_args(argv)


@contextlib.contextmanager
def _env_manager(parallel_factor: int):
    previous_overwrite = env.overwriteOutput
    previous_parallel = env.parallelProcessingFactor
    env.overwriteOutput = True
    if parallel_factor:
        env.parallelProcessingFactor = str(parallel_factor)
    try:
        yield
    finally:
        env.overwriteOutput = previous_overwrite
        env.parallelProcessingFactor = previous_parallel


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logger = make_logger(args.out_folder)

    grid_sr = arcpy.SpatialReference(args.grid_sr)
    logger.info("Target spatial reference: %s (%s)", grid_sr.name, grid_sr.factoryCode)

    with _env_manager(args.n_jobs):
        arcpy.CheckOutExtension("Spatial")
        try:
            projected_states = ensure_projected(args.in_states, grid_sr, args.workspace_gdb, logger)
            projected_counties = ensure_projected(args.in_counties, grid_sr, args.workspace_gdb, logger)
            projected_cities = ensure_projected(args.in_cities, grid_sr, args.workspace_gdb, logger)
            projected_facilities = ensure_projected(args.in_facilities, grid_sr, args.workspace_gdb, logger)

            projected_inputs = [projected_states, projected_counties, projected_cities, projected_facilities]
            extent = collect_extent(projected_inputs)
            if not extent:
                raise ValueError("No input features available to determine spatial extent for fishnet.")

            fishnet_fc = make_fishnet(extent, grid_sr, args.cell_km, args.workspace_gdb, logger)

            time_ranges = [
                _feature_time_range(fc, args.time_field)
                for fc in [projected_states, projected_counties, projected_cities, projected_facilities]
            ]
            start_time, end_time = determine_time_extent(
                time_ranges,
                args.start_time,
                args.end_time,
                logger,
            )
            bins = enumerate_time_bins(start_time, end_time, args.time_step)
            logger.info("Generated %s time bins", len(bins))

            polygon_layers = {
                "states": projected_states,
                "counties": projected_counties,
                "cities": projected_cities,
            }
            bin_values = process_bins(
                bins=bins,
                fishnet_fc=fishnet_fc,
                layers=polygon_layers,
                time_field=args.time_field,
                weights=POLY_WEIGHTS,
                facilities_fc=projected_facilities,
                kde_bandwidth_m=args.kde_bandwidth_m,
                cell_km=args.cell_km,
                logger=logger,
            )

            val_bin_by_bin, val_cum_by_bin, active_cells = accumulate_bins(bin_values)

            out_table = os.path.join(args.workspace_gdb, "cell_time_values")
            write_long_table(bins, val_bin_by_bin, val_cum_by_bin, active_cells, out_table, args.workspace_gdb, logger)

            out_cube = os.path.join(args.out_folder, "heatmap_cube.nc")
            create_stc_from_defined_locations(fishnet_fc, out_table, out_cube, args.time_step, logger)

            summary_csv, first_fc, last_fc = qc_exports(
                fishnet_fc,
                bins,
                val_bin_by_bin,
                val_cum_by_bin,
                args.workspace_gdb,
                args.out_folder,
                logger,
            )

            logger.info("Space-time cube: %s", out_cube)
            logger.info("Cell time values table: %s", out_table)
            logger.info("Summary CSV: %s", summary_csv)
            logger.info("QC Feature classes: %s, %s", first_fc, last_fc)

            print("Space-time cube:", out_cube)
            print("Cell time values table:", out_table)
            print("Summary CSV:", summary_csv)
            print("QC first bin feature class:", first_fc)
            print("QC last bin feature class:", last_fc)
        finally:
            arcpy.CheckInExtension("Spatial")


if __name__ == "__main__":
    main()
