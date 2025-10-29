"""Utility for exporting pre-matched hurricane GeoJSON datasets.

This module runs the geographic matching pipeline once and writes new GeoJSON
files that already contain the matched geometries. These files can be used in
place of the original Helene and Francine tweet datasets to avoid re-running the
fuzzy matching step in future analyses.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import geopandas as gpd

try:  # pragma: no cover - runtime import resolution
    from . import config
    from . import data_loader
    from . import geographic_matching
except ImportError:  # pragma: no cover - direct execution fallback
    import config  # type: ignore
    import data_loader  # type: ignore
    import geographic_matching  # type: ignore


def _ensure_output_dir(output_dir: str) -> str:
    """Create the output directory if needed and return its absolute path."""
    absolute_path = os.path.abspath(output_dir)
    os.makedirs(absolute_path, exist_ok=True)
    return absolute_path


def _prepare_matched_geodataframe(expanded_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert the expanded tweet GeoDataFrame into a GeoJSON-ready structure.

    The input GeoDataFrame returned by ``expand_tweets_by_matches`` stores the
    fuzzy-matched geometry in a separate ``matched_geom`` column. This helper
    promotes that column to be the active geometry, while preserving the original
    tweet geometry for reference.
    """
    if "matched_geom" not in expanded_gdf.columns:
        raise KeyError("Expected 'matched_geom' column to exist on expanded GeoDataFrame")

    output_gdf = expanded_gdf.copy()

    # Preserve the original tweet geometry for auditing.
    output_gdf = output_gdf.rename_geometry("tweet_geometry")

    # Backfill any missing matched geometries with the tweet geometry so that
    # every row retains a valid geometry before export.
    output_gdf["matched_geom"] = output_gdf.apply(
        lambda row: row["tweet_geometry"] if row["matched_geom"] is None else row["matched_geom"],
        axis=1,
    )

    # Promote the matched geometry to be the active geometry column.
    output_gdf = output_gdf.set_geometry("matched_geom")
    output_gdf = output_gdf.rename_geometry("geometry")

    # Ensure CRS metadata is preserved for GeoJSON export.
    if output_gdf.crs is None and hasattr(expanded_gdf, "crs"):
        output_gdf.set_crs(expanded_gdf.crs, inplace=True)

    return output_gdf


def export_matched_geojsons(output_dir: str | None = None) -> Dict[str, str]:
    """Run fuzzy matching once and write reusable GeoJSON exports.

    Parameters
    ----------
    output_dir : str | None, optional
        Destination directory for the matched GeoJSON files. Defaults to
        ``config.MATCHED_GEOJSON_DIR``.

    Returns
    -------
    Dict[str, str]
        Mapping of dataset label (``"francine"`` / ``"helene"``) to the full
        path of the exported GeoJSON file.
    """
    print("\n=== EXPORTING PREMATCHED GEOJSONS ===")

    # Resolve export folder.
    export_root = _ensure_output_dir(output_dir or config.MATCHED_GEOJSON_DIR)
    print(f"Output directory: {export_root}")

    # Load hurricane tweets and reference geographies.
    francine_gdf, helene_gdf = data_loader.load_hurricane_data()
    states_gdf, counties_gdf, cities_gdf = data_loader.load_reference_shapefiles()

    # Build lookup dictionaries for the fuzzy matching step.
    lookups = geographic_matching.create_hierarchical_lookups(
        states_gdf, counties_gdf, cities_gdf
    )

    # Expand tweets so each fuzzy match becomes its own row.
    francine_expanded = geographic_matching.expand_tweets_by_matches(
        francine_gdf, lookups, "FRANCINE"
    )
    helene_expanded = geographic_matching.expand_tweets_by_matches(
        helene_gdf, lookups, "HELENE"
    )

    # Prepare GeoDataFrames ready for GeoJSON export.
    prepared: Dict[str, Tuple[gpd.GeoDataFrame, str]] = {
        "francine": (francine_expanded, os.path.join(export_root, "francine_matched.geojson")),
        "helene": (helene_expanded, os.path.join(export_root, "helene_matched.geojson")),
    }

    export_paths: Dict[str, str] = {}

    for dataset, (expanded, path) in prepared.items():
        print(f"\nProcessing {dataset.title()} matches → {path}")
        output_gdf = _prepare_matched_geodataframe(expanded)
        output_gdf.to_file(path, driver="GeoJSON")
        export_paths[dataset] = path
        print(
            f"  Saved {len(output_gdf)} rows with matched geometries."
            f"\n  Geometry column: {output_gdf.geometry.name}"
        )

    print("\nMatched GeoJSON export complete.")
    return export_paths


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    export_matched_geojsons()
