"""
Tweet Project — Geographic Matching (Step-by-Step)
=================================================

This module parses place-like strings from tweets (GPE/FAC), applies fuzzy name
matching, and assigns **multi-level** geographic matches (STATE / COUNTY / CITY / FACILITY).

OVERVIEW (End-to-End Steps)
---------------------------
STEP 1 — NAME PREPROCESSING
    • Normalize strings (uppercasing, trim, expand common abbreviations).
    • Remove punctuation and extra whitespace.

STEP 2 — GPE STRING PARSING
    • Split free-text GPE fields into candidate entities (commas, ; & | separators).
    • De-duplicate while preserving order.

STEP 3 — LOOKUP CONSTRUCTION
    • Build dictionaries of candidate names → geometries for states, counties, cities.
    • Provide within-state nested dicts (county_by_state, city_by_state).
    • Track state name ↔ abbreviation mappings for contextual matches.

STEP 4 — FUZZY MATCHING
    • Try exact match first (fast path).
    • Otherwise use fuzzywuzzy ratio with thresholds from config (or inline constants).

STEP 5 — FIND ALL MATCHES
    • Resolve states first (to unlock contextual matching).
    • Match counties/cities globally then refine within matched states using a slightly
      lower threshold (contextual boost), replacing weaker global matches if needed.
    • De-duplicate (scale, name) pairs.

STEP 6 — PER-TWEET MULTI-LEVEL ASSIGNMENT
    • For each tweet: parse entities → find matches → include FACILITY if present.
    • If nothing matches, return a single UNMATCHED record (with geometry).

STEP 7 — EXPANSION
    • Expand the tweet GeoDataFrame: one output row per match with scale/name/geom/score.
    • Preserve CRS.

STEP 8 — INTERVAL COUNTS
    • Group by (unix_timestamp, scale_level, matched_name), carry one geometry, compute
      per-bin counts and cumulative counts (for rasterization).

Notes & Pitfalls
----------------
• Fuzzy library: `fuzzywuzzy` is used for compatibility; `rapidfuzz` is faster and pure-Python.
• Thresholds: Global (75) vs contextual (70) mirror `config` defaults; tune per data quality.
• Geometry types: States/counties are polygons; cities may be points. Rasterization handles this.
• Performance: The nested state-refined matching helps precision but costs some CPU. Cache inputs
  if calling repeatedly or consider vectorized approaches for very large datasets.

"""

from __future__ import annotations

# STEP 0 — IMPORTS
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import geopandas as gpd
from fuzzywuzzy import fuzz, process

import config


# ------------------------------------------------------------------------------
# STEP 1 — NAME PREPROCESSING
# ------------------------------------------------------------------------------

def preprocess_place_name(name: Any) -> Optional[str]:
    """
    Standardize place names for better matching.

    Operations
    ----------
    1) Return None for NaN / 'NAN'
    2) Uppercase + trim
    3) Expand common abbrevs: ST./MT./FT./N./S./E./W.
    4) Strip punctuation; collapse spaces

    Parameters
    ----------
    name : Any
        Raw place-like string or NaN.

    Returns
    -------
    str | None
        Cleaned name or None if not valid.
    """
    if pd.isna(name) or name == "NAN":
        return None

    name = str(name).upper().strip()

    # Abbreviation expansions
    name = re.sub(r"\bST\.?\b", "SAINT", name)
    name = re.sub(r"\bMT\.?\b", "MOUNT", name)
    name = re.sub(r"\bFT\.?\b", "FORT", name)
    name = re.sub(r"\bN\.?\b", "NORTH", name)
    name = re.sub(r"\bS\.?\b", "SOUTH", name)
    name = re.sub(r"\bE\.?\b", "EAST", name)
    name = re.sub(r"\bW\.?\b", "WEST", name)

    # Remove punctuation; collapse whitespace
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)

    return name.strip()


# ------------------------------------------------------------------------------
# STEP 2 — GPE STRING PARSING
# ------------------------------------------------------------------------------

def parse_gpe_entities(gpe_string: Any) -> List[Optional[str]]:
    """
    Parse a free-text GPE field into a list of cleaned candidate entities.

    Splits by:
        • commas first
        • then by ; & | within segments

    De-duplicates while preserving order.

    Parameters
    ----------
    gpe_string : Any
        Raw GPE-like string or NaN.

    Returns
    -------
    List[str]
        Cleaned entity candidates (no None, no empty strings).
    """
    if not gpe_string or pd.isna(gpe_string) or str(gpe_string).strip() == "":
        return []

    gpe_string = str(gpe_string).strip()
    entities: List[Optional[str]] = []

    # Primary split by comma
    parts = [part.strip() for part in gpe_string.split(",")]

    for part in parts:
        if part:
            # Further split by other separators
            sub_parts = re.split(r"[;&|]", part)
            for sub_part in sub_parts:
                sub_part = sub_part.strip()
                if sub_part and len(sub_part) > 1:
                    entities.append(preprocess_place_name(sub_part))

    # Remove None values and duplicates while preserving order
    clean_entities: List[str] = []
    seen: set[str] = set()
    for entity in entities:
        if entity and entity not in seen:
            clean_entities.append(entity)
            seen.add(entity)

    return clean_entities


# ------------------------------------------------------------------------------
# STEP 3 — LOOKUP CONSTRUCTION
# ------------------------------------------------------------------------------

def create_hierarchical_lookups(
    states_gdf: gpd.GeoDataFrame,
    counties_gdf: gpd.GeoDataFrame,
    cities_gdf: gpd.GeoDataFrame,
) -> Dict[str, Any]:
    """
    Create hierarchical lookup dictionaries for fuzzy matching.

    Returns a dict with:
        - state_lookup            : {state_name_or_abbrev -> geometry}
        - county_lookup           : {county_name -> geometry}
        - city_lookup             : {city_name -> geometry}
        - county_by_state         : {state_full_name -> {county_name -> geometry}}
        - city_by_state           : {state_full_name -> {city_name -> geometry}}
        - state_abbrev_to_name    : {STUSPS -> full_name}
        - state_name_to_abbrev    : {full_name -> STUSPS}

    Notes
    -----
    • County to state association:
      If a 'STATE_NAME' field exists on counties, it is used. Otherwise, the code
      matches county STATEFP to a state's STATEFP in states_gdf.
    • City to state association:
      Uses a 'ST' (state abbrev) column on cities; requires states lookup to map
      abbrev → full name.

    Side Effects
    ------------
    Prints a short “creating” banner.
    """
    print("\nCreating hierarchical lookup dictionaries...")

    # 1) States — names + USPS abbreviations
    state_lookup: Dict[str, Any] = {}
    state_abbrev_to_name: Dict[str, str] = {}
    state_name_to_abbrev: Dict[str, str] = {}

    for _, row in states_gdf.iterrows():
        state_name = preprocess_place_name(row["NAME"])
        if state_name:
            state_lookup[state_name] = row.geometry
            if "STUSPS" in row:
                abbrev = str(row["STUSPS"]).upper()
                state_abbrev_to_name[abbrev] = state_name
                state_name_to_abbrev[state_name] = abbrev
                state_lookup[abbrev] = row.geometry

    # 2) Counties — global and grouped by state
    county_by_state: Dict[str, Dict[str, Any]] = {}
    county_lookup: Dict[str, Any] = {}

    for _, row in counties_gdf.iterrows():
        county_name = preprocess_place_name(row["NAME"])
        state_fips = row.get("STATEFP", "")

        if county_name:
            county_lookup[county_name] = row.geometry

            # Resolve state name for this county
            state_name = None
            if "STATE_NAME" in row:
                state_name = preprocess_place_name(row["STATE_NAME"])
            else:
                # Fallback: match via STATEFP to states_gdf
                for _, s_row in states_gdf.iterrows():
                    if s_row.get("STATEFP", "") == state_fips:
                        state_name = preprocess_place_name(s_row["NAME"])
                        break

            if state_name:
                county_by_state.setdefault(state_name, {})
                county_by_state[state_name][county_name] = row.geometry

    # 3) Cities — global and grouped by state
    city_by_state: Dict[str, Dict[str, Any]] = {}
    city_lookup: Dict[str, Any] = {}

    for _, row in cities_gdf.iterrows():
        city_name = preprocess_place_name(row["NAME"])
        state_abbrev = str(row.get("ST", "")).upper()

        if city_name:
            city_lookup[city_name] = row.geometry

            if state_abbrev in state_abbrev_to_name:
                state_full = state_abbrev_to_name[state_abbrev]
                city_by_state.setdefault(state_full, {})
                city_by_state[state_full][city_name] = row.geometry

    return {
        "state_lookup": state_lookup,
        "county_lookup": county_lookup,
        "city_lookup": city_lookup,
        "county_by_state": county_by_state,
        "city_by_state": city_by_state,
        "state_abbrev_to_name": state_abbrev_to_name,
        "state_name_to_abbrev": state_name_to_abbrev,
    }


# ------------------------------------------------------------------------------
# STEP 4 — FUZZY MATCHING
# ------------------------------------------------------------------------------

def fuzzy_match_entity(
    entity: Optional[str],
    candidates: Dict[str, Any],
    threshold: int = config.FUZZY_THRESHOLD,
) -> Tuple[Optional[str], int]:
    """
    Fuzzy match an entity against candidate names.

    Fast path:
        • If exact key exists, return it with score=100.

    Otherwise:
        • Use fuzzywuzzy.ratio to find the best candidate.
        • Accept only if score >= threshold.

    Parameters
    ----------
    entity : str | None
        Cleaned entity text to match.
    candidates : Dict[str, Any]
        Mapping of candidate name -> payload (geometry, etc.).
    threshold : int
        Minimum score to accept (default: config.FUZZY_THRESHOLD).

    Returns
    -------
    (match_key, score) : Tuple[str | None, int]
        match_key is a key from `candidates` (or None), score is 0–100.
    """
    if not entity or not candidates:
        return None, 0

    # Exact match first
    if entity in candidates:
        return entity, 100

    # Fuzzy match
    match = process.extractOne(entity, candidates.keys(), scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return match[0], match[1]

    return None, 0


# ------------------------------------------------------------------------------
# STEP 5 — FIND ALL MATCHES
# ------------------------------------------------------------------------------

def find_all_geographic_matches(
    entities: List[str],
    lookups: Dict[str, Any],
) -> List[Tuple[str, Optional[str], Any, int]]:
    """
    Find ALL geographic matches (STATE, COUNTY, CITY) for the given entities.

    Matching Strategy
    -----------------
    1) Resolve STATEs first; remember found states for context.
    2) Resolve COUNTYs:
        - global match (strict threshold),
        - contextual re-match within each found state (slightly lower threshold),
          replace weaker global match if contextual is stronger.
    3) Resolve CITYs with the same global + contextual refinement pattern.
    4) De-duplicate by (scale, name).

    Parameters
    ----------
    entities : List[str]
        Preprocessed entity tokens derived from a tweet's GPE field.
    lookups : dict
        Output of `create_hierarchical_lookups()`.

    Returns
    -------
    List[Tuple[scale, name, geometry, score]]
        scale ∈ {'STATE','COUNTY','CITY'}, name may be None when absent (not typical).
    """
    if not entities:
        return []

    state_lookup = lookups["state_lookup"]
    county_lookup = lookups["county_lookup"]
    city_lookup = lookups["city_lookup"]
    county_by_state = lookups["county_by_state"]
    city_by_state = lookups["city_by_state"]

    all_matches: List[Tuple[str, Optional[str], Any, int]] = []
    found_states: set[str] = set()

    # -- STATES
    for entity in entities:
        state_match, state_score = fuzzy_match_entity(entity, state_lookup, threshold=75)
        if state_match:
            all_matches.append(("STATE", state_match, state_lookup[state_match], state_score))
            found_states.add(state_match)

    # -- COUNTIES
    for entity in entities:
        county_match, county_score = fuzzy_match_entity(entity, county_lookup, threshold=75)
        if county_match:
            all_matches.append(("COUNTY", county_match, county_lookup[county_match], county_score))

        # Contextual (within matched states) with slightly lower threshold
        for state_name in found_states:
            if state_name in county_by_state:
                state_counties = county_by_state[state_name]
                state_county_match, state_county_score = fuzzy_match_entity(
                    entity, state_counties, threshold=70
                )
                if state_county_match and state_county_score > county_score:
                    # Remove weaker global county match (if any) for this name
                    all_matches = [
                        m for m in all_matches
                        if not (m[0] == "COUNTY" and m[1] == county_match)
                    ]
                    all_matches.append(
                        ("COUNTY", state_county_match, state_counties[state_county_match], state_county_score)
                    )

    # -- CITIES
    for entity in entities:
        city_match, city_score = fuzzy_match_entity(entity, city_lookup, threshold=75)
        if city_match:
            all_matches.append(("CITY", city_match, city_lookup[city_match], city_score))

        # Contextual (within matched states)
        for state_name in found_states:
            if state_name in city_by_state:
                state_cities = city_by_state[state_name]
                state_city_match, state_city_score = fuzzy_match_entity(
                    entity, state_cities, threshold=70
                )
                if state_city_match and state_city_score > city_score:
                    # Remove weaker global city match (if any) for this name
                    all_matches = [
                        m for m in all_matches
                        if not (m[0] == "CITY" and m[1] == city_match)
                    ]
                    all_matches.append(
                        ("CITY", state_city_match, state_cities[state_city_match], state_city_score)
                    )

    # De-duplicate by (scale, name)
    unique_matches: List[Tuple[str, Optional[str], Any, int]] = []
    seen_combinations: set[Tuple[str, Optional[str]]] = set()
    for match in all_matches:
        combo = (match[0], match[1])
        if combo not in seen_combinations:
            unique_matches.append(match)
            seen_combinations.add(combo)

    return unique_matches


# ------------------------------------------------------------------------------
# STEP 6 — PER-TWEET MULTI-LEVEL ASSIGNMENT
# ------------------------------------------------------------------------------

def multi_level_assign_scale_levels(
    row: pd.Series,
    lookups: Dict[str, Any],
) -> List[Tuple[str, Optional[str], Any, int]]:
    """
    Return ALL geographic scale levels that match this tweet.

    For a single tweet row:
        1) Parse GPE into entities.
        2) Resolve STATE / COUNTY / CITY matches (possibly multiple).
        3) If FAC present, add ('FACILITY', FAC, row.geometry, 100).
        4) If nothing found, add ('UNMATCHED', None, row.geometry, 0).

    Returns
    -------
    List[(scale, name, geom, score)]
    """
    gpe = str(row.get("GPE", "")).strip()
    fac = str(row.get("FAC", "")).strip()

    matches: List[Tuple[str, Optional[str], Any, int]] = []

    # Parse GPE into multiple entities
    entities = parse_gpe_entities(gpe)
    if entities:
        geo_matches = find_all_geographic_matches(entities, lookups)
        matches.extend(geo_matches)

    # Add facility as separate match if present
    if fac and fac not in ["nan", "NAN", ""]:
        matches.append(("FACILITY", fac, row.geometry, 100))

    # If no matches found, return UNMATCHED
    if not matches:
        matches.append(("UNMATCHED", None, row.geometry, 0))

    return matches


# ------------------------------------------------------------------------------
# STEP 7 — EXPANSION
# ------------------------------------------------------------------------------

def expand_tweets_by_matches(
    gdf: gpd.GeoDataFrame,
    lookups: Dict[str, Any],
    dataset_name: str,
) -> gpd.GeoDataFrame:
    """
    Expand the GeoDataFrame so each tweet creates multiple rows (one per match).

    Adds columns:
        - scale_level   : {'STATE','COUNTY','CITY','FACILITY','UNMATCHED'}
        - matched_name  : normalized name (None for UNMATCHED)
        - matched_geom  : geometry of the matched place (or tweet geometry)
        - match_score   : 0–100 (fuzzy score or 100 for FACILITY)
        - original_index: original row index for traceability

    Parameters
    ----------
    gdf : GeoDataFrame
        Input tweet GeoDataFrame with columns like 'GPE', 'FAC', 'geometry', etc.
    lookups : dict
        Output from create_hierarchical_lookups().
    dataset_name : str
        Label for progress messages (e.g., 'FRANCINE', 'HELENE').

    Returns
    -------
    GeoDataFrame
        Row-per-match expansion, CRS preserved.
    """
    print(f"\nExpanding {dataset_name} tweets by geographic matches...")

    expanded_rows: List[pd.Series] = []

    for idx, row in gdf.iterrows():
        if idx % 100 == 0:
            print(f"  Processing tweet {idx}...")

        matches = multi_level_assign_scale_levels(row, lookups)

        # Create one row per match
        for scale, name, geom, score in matches:
            new_row = row.copy()
            new_row["scale_level"] = scale
            new_row["matched_name"] = name
            new_row["matched_geom"] = geom
            new_row["match_score"] = score
            new_row["original_index"] = idx
            expanded_rows.append(new_row)

    # Create new GeoDataFrame and preserve the original CRS
    expanded_gdf = gpd.GeoDataFrame(expanded_rows, crs=gdf.crs)

    print(f"  Expanded from {len(gdf)} to {len(expanded_gdf)} rows")

    # Show summary statistics
    scale_counts = expanded_gdf["scale_level"].value_counts()
    print(f"\n  Scale level distribution:")
    for scale, count in scale_counts.items():
        print(f"    {scale}: {count}")

    return expanded_gdf


# ------------------------------------------------------------------------------
# STEP 8 — INTERVAL COUNTS
# ------------------------------------------------------------------------------

def create_interval_counts(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Group tweets by time intervals and scale level for aggregation.

    Group-by keys:
        ('unix_timestamp', 'scale_level', 'matched_name')

    Outputs:
        - 'matched_geom'     : first geometry for that key (representative)
        - 'count'            : per-bin count
        - 'cumulative_count' : cumulative per (scale_level, matched_name)
        Sorted by 'unix_timestamp'.

    Parameters
    ----------
    gdf : GeoDataFrame
        Expanded rows (one per match) with 'unix_timestamp', 'scale_level',
        'matched_name', and 'matched_geom'.

    Returns
    -------
    DataFrame
        Aggregated per-bin counts with cumulative totals.
    """
    # Group by bin + scale + name, carry one geometry
    grouped = gdf.groupby(["unix_timestamp", "scale_level", "matched_name"])
    interval_counts = grouped.agg({"matched_geom": "first"}).reset_index()

    # Per-bin counts
    count_series = grouped.size()
    interval_counts["count"] = count_series.values

    # Sort by time
    interval_counts = interval_counts.sort_values("unix_timestamp")

    # Cumulative counts per (scale_level, matched_name)
    interval_counts["cumulative_count"] = (
        interval_counts.groupby(["scale_level", "matched_name"])["count"].cumsum()
    )

    return interval_counts
