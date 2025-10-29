"""
Export Pre-Matched Hurricane Data

Run this ONCE after fuzzy matching to create pre-matched GeoJSONs.
Future runs can load these directly and skip the expensive fuzzy matching step.
"""

import os
import geopandas as gpd
import pandas as pd
import json

# Configuration
LOCAL_PATH = r"C:\users\colto\documents\github\tweet_project"
GEOJSON_DIR = os.path.join(LOCAL_PATH, "data", "geojson")

# Output paths for pre-matched data
FRANCINE_MATCHED_PATH = os.path.join(GEOJSON_DIR, "francine_prematched.geojson")
HELENE_MATCHED_PATH = os.path.join(GEOJSON_DIR, "helene_prematched.geojson")

def export_prematched_geojson(expanded_gdf: gpd.GeoDataFrame, output_path: str, hurricane_name: str):
    """
    Export expanded (pre-matched) GeoDataFrame to GeoJSON.

    This saves all the fuzzy matching results so they don't need to be recomputed.
    """
    print(f"\nExporting pre-matched {hurricane_name} data...")
    print(f"  Rows: {len(expanded_gdf)}")
    print(f"  Columns: {list(expanded_gdf.columns)}")

    # Select essential columns for export
    export_cols = [
        'geometry',
        'time', 'timestamp', 'time_bin', 'unix_timestamp', 'bin_label',
        'scale_level', 'matched_name', 'match_score',
        'GPE', 'FAC', 'original_index'
    ]

    # Keep only columns that exist
    available_cols = [col for col in export_cols if col in expanded_gdf.columns]

    export_gdf = expanded_gdf[available_cols].copy()

    # Convert timestamp columns to strings for JSON compatibility
    if 'timestamp' in export_gdf.columns:
        export_gdf['timestamp'] = export_gdf['timestamp'].astype(str)
    if 'time_bin' in export_gdf.columns:
        export_gdf['time_bin'] = export_gdf['time_bin'].astype(str)

    # Export to GeoJSON
    export_gdf.to_file(output_path, driver='GeoJSON')

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"  Exported to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")

    # Print summary statistics
    scale_counts = export_gdf['scale_level'].value_counts()
    print(f"\n  Scale distribution:")
    for scale, count in scale_counts.items():
        print(f"    {scale}: {count}")

    return output_path


def load_prematched_geojson(input_path: str, hurricane_name: str) -> gpd.GeoDataFrame:
    """
    Load pre-matched GeoJSON.

    This skips all fuzzy matching and loads the previously computed results.
    """
    print(f"\nLoading pre-matched {hurricane_name} data...")
    print(f"  From: {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Pre-matched file not found: {input_path}")

    gdf = gpd.read_file(input_path)

    # Convert timestamp strings back to datetime
    if 'timestamp' in gdf.columns:
        gdf['timestamp'] = pd.to_datetime(gdf['timestamp'], utc=True)
    if 'time_bin' in gdf.columns:
        gdf['time_bin'] = pd.to_datetime(gdf['time_bin'], utc=True)

    print(f"  Loaded {len(gdf)} rows")

    scale_counts = gdf['scale_level'].value_counts()
    print(f"\n  Scale distribution:")
    for scale, count in scale_counts.items():
        print(f"    {scale}: {count}")

    return gdf


def create_match_summary_report(
    francine_expanded: gpd.GeoDataFrame,
    helene_expanded: gpd.GeoDataFrame,
    output_path: str = None
):
    """
    Create a detailed summary report of the fuzzy matching results.
    """
    if output_path is None:
        output_path = os.path.join(LOCAL_PATH, "data", "match_summary_report.txt")

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FUZZY MATCHING SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Francine Summary
        f.write("FRANCINE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total expanded rows: {len(francine_expanded)}\n\n")

        f.write("Scale Level Distribution:\n")
        for scale, count in francine_expanded['scale_level'].value_counts().items():
            f.write(f"  {scale:12s}: {count:6d}\n")

        f.write("\nTop 20 Matched Names by Scale:\n")
        for scale in ['STATE', 'COUNTY', 'CITY', 'FACILITY']:
            scale_data = francine_expanded[francine_expanded['scale_level'] == scale]
            if len(scale_data) > 0:
                f.write(f"\n  {scale}:\n")
                top_matches = scale_data['matched_name'].value_counts().head(20)
                for name, count in top_matches.items():
                    f.write(f"    {str(name)[:40]:40s}: {count:4d}\n")

        # Helene Summary
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("HELENE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total expanded rows: {len(helene_expanded)}\n\n")

        f.write("Scale Level Distribution:\n")
        for scale, count in helene_expanded['scale_level'].value_counts().items():
            f.write(f"  {scale:12s}: {count:6d}\n")

        f.write("\nTop 20 Matched Names by Scale:\n")
        for scale in ['STATE', 'COUNTY', 'CITY', 'FACILITY']:
            scale_data = helene_expanded[helene_expanded['scale_level'] == scale]
            if len(scale_data) > 0:
                f.write(f"\n  {scale}:\n")
                top_matches = scale_data['matched_name'].value_counts().head(20)
                for name, count in top_matches.items():
                    f.write(f"    {str(name)[:40]:40s}: {count:4d}\n")

        # Match Score Statistics
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("MATCH SCORE STATISTICS\n")
        f.write("-" * 40 + "\n")

        for hurricane_name, gdf in [("FRANCINE", francine_expanded), ("HELENE", helene_expanded)]:
            f.write(f"\n{hurricane_name}:\n")

            for scale in ['STATE', 'COUNTY', 'CITY']:
                scale_data = gdf[gdf['scale_level'] == scale]
                if len(scale_data) > 0 and 'match_score' in scale_data.columns:
                    scores = scale_data['match_score']
                    f.write(f"  {scale}:\n")
                    f.write(f"    Mean score: {scores.mean():.2f}\n")
                    f.write(f"    Min score:  {scores.min():.2f}\n")
                    f.write(f"    Max score:  {scores.max():.2f}\n")
                    f.write(f"    Exact matches (score=100): {(scores == 100).sum()}\n")

    print(f"\nMatch summary report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("=" * 80)
    print("EXPORT PRE-MATCHED HURRICANE DATA")
    print("=" * 80)
    print("\nNOTE: This script expects 'francine_gdf' and 'helene_gdf' to be")
    print("already expanded with fuzzy matching results.")
    print("\nRun this from the notebook after Step 5 (Expand Tweets by Matches)")
    print("=" * 80)
