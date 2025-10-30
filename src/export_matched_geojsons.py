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
    if "matched_geom" not in expanded_gdf.columns:
        raise KeyError("Expected 'matched_geom' column to exist on expanded GeoDataFrame")

    gdf = expanded_gdf.copy()

    # Preserve original geometry as a normal (non-active) column first.
    # If the GeoDataFrame already has an active geometry, rename it.
    if gdf._geometry_column_name is not None:
        gdf = gdf.rename_geometry("tweet_geometry")
    else:
        # If somehow there's no active geometry, create a placeholder column
        # to keep the original points if present in data.
        if "tweet_geometry" not in gdf:
            gdf["tweet_geometry"] = None

    # Backfill matched geometries with the original tweet geometry when missing
    gdf["geometry"] = gdf["matched_geom"].where(gdf["matched_geom"].notna(), gdf["tweet_geometry"])

    # Remove helper column now that we've promoted it
    gdf = gdf.drop(columns=["matched_geom"])

    # Activate the promoted geometry
    gdf = gdf.set_geometry("geometry")

    # IMPORTANT: GeoJSON expects WGS84; reproject if needed.
    # First, ensure we retain CRS from the source if present.
    if gdf.crs is None and getattr(expanded_gdf, "crs", None) is not None:
        gdf = gdf.set_crs(expanded_gdf.crs)

    # Reproject to EPSG:4326 for GeoJSON (RFC 7946)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    # Convert ANY remaining geometry-typed columns (e.g., 'tweet_geometry') to WKT
    # so there is only one geometry column in the frame.
    for col in list(gdf.columns):
        if col != gdf.geometry.name:
            # GeoPandas 0.12+ helper:
            try:
                is_geom = gpd.array.is_geometry_type(gdf[col])
            except Exception:
                is_geom = False
            if is_geom:
                gdf[f"{col}_wkt"] = gdf[col].to_wkt()
                gdf = gdf.drop(columns=[col])

    return gdf



def export_matched_geojsons(
    francine_expanded: gpd.GeoDataFrame | None = None,
    helene_expanded: gpd.GeoDataFrame | None = None,
    output_dir: str | None = None,
) -> Dict[str, str]:
    """Write reusable GeoJSON exports containing pre-matched geometries.

    Parameters
    ----------
    francine_expanded : GeoDataFrame, optional
        Result of ``expand_tweets_by_matches`` for Hurricane Francine. When
        omitted, the dataset is loaded and expanded inside this function.
    helene_expanded : GeoDataFrame, optional
        Result of ``expand_tweets_by_matches`` for Hurricane Helene. When
        omitted, the dataset is loaded and expanded inside this function.
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

    # If expanded GeoDataFrames are not provided, load the raw data and run the
    # fuzzy matching pipeline locally. This keeps the function usable as a
    # stand-alone utility, while allowing callers (like the main pipeline) to
    # provide pre-expanded data so the matching only happens once.
    if francine_expanded is None or helene_expanded is None:
        print("Loading hurricane data and computing matches (stand-alone mode)...")
        francine_gdf, helene_gdf = data_loader.load_hurricane_data()
        states_gdf, counties_gdf, cities_gdf = data_loader.load_reference_shapefiles()

        lookups = geographic_matching.create_hierarchical_lookups(
            states_gdf, counties_gdf, cities_gdf
        )

        if francine_expanded is None:
            francine_expanded = geographic_matching.expand_tweets_by_matches(
                francine_gdf, lookups, "FRANCINE"
            )
        if helene_expanded is None:
            helene_expanded = geographic_matching.expand_tweets_by_matches(
                helene_gdf, lookups, "HELENE"
            )
    else:
        print("Using provided expanded GeoDataFrames (pipeline-integrated mode)...")

    # Prepare GeoDataFrames ready for GeoJSON export.
    prepared: Dict[str, Tuple[gpd.GeoDataFrame, str]] = {
        "francine": (francine_expanded, os.path.join(export_root, "francine_matched.geojson")),
        "helene": (helene_expanded, os.path.join(export_root, "helene_matched.geojson")),
    }

    export_paths: Dict[str, str] = {}

    for dataset, (expanded, path) in prepared.items():
        print(f"\nProcessing {dataset.title()} matches → {path}")
        output_gdf = _prepare_matched_geodataframe(expanded)
        print(output_gdf)
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
