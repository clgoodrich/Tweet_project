# Tweet Project - Spatial-Temporal Analysis Context

## Project Overview
Hurricane disaster response analysis using social media data. Goal: transform tweet streams into time-enabled geographic visualizations for ArcGIS Pro.

## Input Data Specifications

### Hurricane Tweet Data (GeoJSON)
**Files:**
- `francine.geojson` - Hurricane Francine tweets
- `helene.geojson` - Hurricane Helene tweets

**Expected Schema:**
- `geometry`: Point geometries (lat/lon coordinates)
- `time`: Timestamp field (string or datetime)
- `GPE`: Geographic entities mentioned in tweet text (may contain: states, counties, cities)
- `FAC`: Facility mentions (airports, landmarks, infrastructure)
- Additional metadata fields may be present

**Data Characteristics:**
- Tweets span multiple days during hurricane events
- Geographic entities may be mentioned at different scales (state/county/city/facility)
- Entity fields may contain comma-separated lists or be empty/null
- Temporal distribution is uneven (spikes during peak activity)

### Reference Geographic Data

**State Boundaries (Shapefile):**
- File: `cb_2023_us_state_20m.shp`
- Standard Census Bureau TIGER/Line format
- Fields include: NAME, STATEFP, STUSPS (2-letter code)
- Polygons, likely EPSG:4269 or similar

**County Boundaries (Shapefile):**
- File: `cb_2023_us_county_20m.shp`
- Standard Census Bureau format
- Fields include: NAME, STATEFP, COUNTYFP
- Polygons, coordinate system matches states

**Cities Reference (CSV):**
- File: `cities1000.csv`
- Likely contains: city names, coordinates, population, state associations
- Format to be determined during data exploration

## Technical Environment

### Python Environment
- Jupyter Notebook execution environment
- Standard geospatial stack available (geopandas, shapely, rasterio, etc.)
- ArcPy may or may not be available in notebook context
- Assume ArcGIS Pro 3.x as target for outputs

### Output Requirements

**ArcGIS Pro Compatibility:**
- Outputs must be loadable in ArcGIS Pro Map view
- Must support time slider functionality (time-enabled layers)
- Common formats: GeoTIFF with time metadata, feature classes in geodatabase, mosaic datasets, NetCDF/multidimensional formats

**Temporal Visualization:**
- User needs to see how tweet activity evolves over time
- Should support both cumulative and interval-based views
- Time granularity is a design decision (hourly, 4-hour, daily bins, etc.)

**Geographic Coverage:**
- Focus on hurricane-affected regions (Gulf Coast, Southeast US primarily)
- Must handle multi-scale geography (state → county → city → facility)
- Coordinate systems should be appropriate for regional analysis

## Domain Knowledge

### Hurricane Social Media Analysis
- Tweets contain real-time damage reports, location mentions, infrastructure status
- Volume of tweets correlates with impact severity and population density
- Geographic entities are mentioned at various scales simultaneously
- Facility mentions (airports, hospitals, highways) indicate critical infrastructure impacts

### Challenges to Consider
- Ambiguous place names (multiple cities named "Springfield")
- Incomplete geographic information in tweets
- Varying levels of geographic specificity
- Temporal clustering (burst patterns during landfall/peak impact)
- Multi-scale aggregation (how to weight state vs. city mentions)
- Missing or malformed data in entity fields

## Constraints & Freedoms

### Fixed (Non-Negotiable)
- Input file paths and formats
- Column names in tweet data (time, GPE, FAC, geometry)
- Output must work in ArcGIS Pro
- Must handle both hurricanes separately

### Open Design Decisions
- Spatial representation strategy (raster vs. vector vs. hybrid)
- Temporal binning approach and interval size
- Geographic matching/resolution methodology
- Aggregation and weighting schemes
- Output format selection
- Code architecture and modularity
- Library and algorithm choices
- Handling of ambiguous or missing data
- Performance optimization strategies

## Success Criteria

**Functional:**
✓ Loads all specified input data without errors
✓ Produces outputs that open in ArcGIS Pro
✓ Temporal controls (time slider) function correctly
✓ Geographic patterns are visually interpretable
✓ Handles both hurricane datasets

**Quality:**
✓ Code is well-documented and modular
✓ Methodology is justified and explained
✓ Edge cases are handled gracefully
✓ Performance is reasonable for dataset size
✓ Outputs are scientifically defensible

## Development Approach

1. **Exploration Phase**: Examine actual data structures and contents
2. **Design Phase**: Articulate solution strategy with rationale
3. **Implementation Phase**: Build modular, testable code
4. **Verification Phase**: Confirm ArcGIS Pro compatibility
5. **Documentation Phase**: Provide clear usage instructions

## Notes
- This is a research project context - innovative approaches are encouraged
- Existing codebase exists but you are implementing a fresh solution
- Focus on correctness and ArcGIS Pro integration over pure performance
- Document assumptions and design decisions clearly
- Jupyter notebook should be self-contained and reproducible
