# Tweet Analysis Methodology - Explained

## Data Structure

**Input**: `helene.geojson` - 3,007 tweet features

Each tweet has:
```json
{
  "properties": {
    "GPE": "Florida",           // ← Text field with place mentions
    "time": "2024-09-26 22:59:53+00:00",
    "Latitude": 27.7567667,
    "Longitude": -81.4639835
  },
  "geometry": {
    "type": "Point",
    "coordinates": [-81.4639835, 27.7567667]  // ← Actual tweet location
  }
}
```

**Key insight**: Tweets have BOTH text mentions AND geographic coordinates!

---

## Counting Methodology (test.ipynb)

### Part 1: Text-Based Mention Counting
**Purpose**: Count what places are MENTIONED in tweets (not where tweets are FROM)

**Process**:
1. Parse `GPE` field: "Florida, Tallahassee, Georgia" → ["FLORIDA", "TALLAHASSEE", "GEORGIA"]
2. Fuzzy match each entity to state/county/city databases
3. Increment count for each matched entity

**Example**:
- Tweet GPE: "Florida, Georgia, Tallahassee"
- Matches:
  - "Florida" → Florida state (+1)
  - "Georgia" → Georgia state (+1)
  - "Tallahassee" → Tallahassee city (+1)
- **Total: 3 counts from 1 tweet**

### Part 2: Spatial Cascade (ADDITIONAL counts)
**Purpose**: ALSO count where tweets are physically located

**Process**:
1. Use tweet's `geometry` (lat/lon point)
2. Find containing county → +1 to county
3. Find containing state (via county's STATEFP) → +1 to state
4. Find nearest city (within 50km) → +1 to city

**Example**:
- Tweet point: [-81.46, 27.76]
- Spatial operations:
  - Point is in Highlands County → +1
  - Highlands County is in Florida → +1
  - Nearest city is Sebring → +1
- **Total: 3 more counts from same tweet**

### Combined Result
**Same tweet can contribute to BOTH**:
- Text mentions: "Florida" in GPE → Florida +1
- Spatial cascade: Point in Florida → Florida +1
- **Florida total: +2 from one tweet!**

This is why Florida has 2,156 total counts from only 3,007 tweets.

---

## Why This Methodology?

**Problem**: Tweets about hurricanes mention many places but may be sent from elsewhere.

**Example**:
```
Tweet: "Praying for everyone in Tallahassee and Tampa. Stay safe Florida!"
Location: User tweeted from Atlanta, GA
```

**Traditional spatial-only analysis**:
- Would only count this for Georgia (where sent from)
- Misses that tweet is ABOUT Florida cities

**This dual methodology**:
- Text mentions: Tallahassee +1, Tampa +1, Florida +1
- Spatial cascade: Georgia +1 (where sent from)
- **Captures both what's mentioned AND where it's from**

---

## Output Geometries

**CRITICAL**: The output shapefiles use the **reference geography polygons**, NOT the tweet points!

**States output**:
```
Input: cb_2023_us_state_20m.shp (state polygons)
Process: Add tweet_count field
Result: Florida polygon with tweet_count = 2,156
```

The tweet point coordinates are ONLY used for:
1. Spatial containment checks (which county/state contains this point?)
2. Nearest city distance calculations

The FINAL output geometry is the STATE/COUNTY/CITY polygon from the reference shapefiles, with counts attached.

---

## ArcPy Implementation Needs

For proper ArcPy implementation, we need:

### 1. NO spatial operations needed for text mentions
```python
# Just parse text and match
entities = parse_gpe_field(row['GPE'])
for entity in entities:
    state_code = fuzzy_match(entity, state_lookup)
    state_counts[state_code] += 1
```

### 2. Spatial operations ONLY for cascade
```python
# Use tweet geometry point
tweet_point = row['SHAPE@']
containing_county = find_containing_polygon(tweet_point, counties_fc)
county_counts[county_id] += 1
```

### 3. Join counts to reference geography
```python
# Merge counts back to state polygons
states_with_counts = states_fc.merge(state_counts)
# Output: State polygons with tweet_count field
```

---

## Correct Understanding

**The workflow is**:
1. Import tweets (points with GPE text)
2. Import reference geography (state/county/city polygons)
3. Count tweets:
   - Parse GPE text → match to geographies
   - Use point location → spatial join to geographies
4. Merge counts to reference geography polygons
5. Export reference geography WITH counts

**NOT**: Create new geometries from tweet points
**YES**: Add counts to existing geography polygons

This is why test.ipynb does:
```python
states_with_counts = us_states_gdf.merge(state_counts_df, on='STUSPS')
# us_states_gdf = State polygons (unchanged geometry)
# state_counts_df = Just the counts
# Result = State polygons + counts
```
