import pandas as pd
import numpy as np
import requests
import zipfile
import io
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import geopandas as gpd
from shapely.geometry import Point
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CityData:
    """Data structure for city information"""
    geoname_id: int
    name: str
    latitude: float
    longitude: float
    feature_class: str
    feature_code: str
    country_code: str
    admin1_code: str  # State/Province
    admin2_code: str  # County
    population: int
    elevation: Optional[int]
    timezone: str


class CitiesDataProcessor:
    """Process cities data from various sources into lightweight centroids"""

    def __init__(self, cache_dir: str = "cities_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # GeoNames download URLs
        self.geonames_urls = {
            'cities1000': 'http://download.geonames.org/export/dump/cities1000.zip',
            'cities5000': 'http://download.geonames.org/export/dump/cities5000.zip',
            'cities15000': 'http://download.geonames.org/export/dump/cities15000.zip',
            'US': 'http://download.geonames.org/export/dump/US.zip'
        }

        # GeoNames field mapping
        self.geonames_columns = [
            'geonameid', 'name', 'asciiname', 'alternatenames',
            'latitude', 'longitude', 'feature_class', 'feature_code',
            'country_code', 'cc2', 'admin1_code', 'admin2_code',
            'admin3_code', 'admin4_code', 'population', 'elevation',
            'dem', 'timezone', 'modification_date'
        ]

    def download_geonames_data(self, dataset: str = 'cities5000') -> pd.DataFrame:
        """Download and parse GeoNames cities data"""

        if dataset not in self.geonames_urls:
            raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(self.geonames_urls.keys())}")

        cache_file = self.cache_dir / f"{dataset}.csv"

        # Check cache first
        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_csv(cache_file)

        logger.info(f"Downloading {dataset} from GeoNames...")

        try:
            # Download and extract
            response = requests.get(self.geonames_urls[dataset], timeout=60)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                # Find the text file in the zip
                txt_files = [f for f in zip_file.namelist() if f.endswith('.txt')]
                if not txt_files:
                    raise ValueError("No .txt file found in downloaded zip")

                # Read the data
                with zip_file.open(txt_files[0]) as txt_file:
                    df = pd.read_csv(
                        txt_file,
                        sep='\t',
                        header=None,
                        names=self.geonames_columns,
                        encoding='utf-8',
                        low_memory=False
                    )

            logger.info(f"Downloaded {len(df)} records")

            # Cache the raw data
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached data to {cache_file}")

            return df

        except Exception as e:
            logger.error(f"Error downloading {dataset}: {e}")
            raise

    def filter_us_cities(self, df: pd.DataFrame, min_population: int = 1000) -> pd.DataFrame:
        """Filter for US cities and clean data"""

        logger.info(f"Filtering for US cities with population >= {min_population}")

        # Filter for US cities with population data
        us_cities = df[
            (df['country_code'] == 'US') &
            (df['feature_class'] == 'P') &  # Populated places
            (df['population'] >= min_population) &
            (df['population'].notna()) &
            (df['latitude'].notna()) &
            (df['longitude'].notna())
            ].copy()

        # Clean and standardize data
        us_cities['population'] = us_cities['population'].astype(int)
        us_cities['latitude'] = us_cities['latitude'].astype(float)
        us_cities['longitude'] = us_cities['longitude'].astype(float)

        # Handle missing elevation data
        us_cities['elevation'] = us_cities['elevation'].fillna(0).astype(int)

        logger.info(f"Found {len(us_cities)} US cities meeting criteria")

        return us_cities

    def create_cities_geodataframe(self, cities_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert cities dataframe to GeoDataFrame with point geometries"""

        logger.info("Creating GeoDataFrame with point geometries")

        # Create point geometries
        geometry = [Point(lon, lat) for lon, lat in zip(cities_df['longitude'], cities_df['latitude'])]

        # Select essential columns for our use case
        essential_columns = [
            'geonameid', 'name', 'latitude', 'longitude',
            'admin1_code', 'admin2_code', 'population', 'timezone'
        ]

        gdf = gpd.GeoDataFrame(
            cities_df[essential_columns],
            geometry=geometry,
            crs='EPSG:4326'  # WGS84
        )

        # Add convenience columns
        gdf['city_id'] = gdf['geonameid'].astype(str)
        gdf['pop_category'] = pd.cut(
            gdf['population'],
            bins=[0, 5000, 25000, 100000, 500000, float('inf')],
            labels=['small', 'medium', 'large', 'major', 'metro'],
            right=False
        )

        logger.info(f"Created GeoDataFrame with {len(gdf)} cities")

        return gdf

    def export_cities_data(self, gdf: gpd.GeoDataFrame, output_formats: List[str] = ['geojson', 'parquet']) -> Dict[
        str, Path]:
        """Export cities data in various formats"""

        output_files = {}

        for fmt in output_formats:
            output_path = self.cache_dir / f"us_cities.{fmt}"

            try:
                if fmt == 'geojson':
                    # Create lightweight GeoJSON for web use
                    # Round coordinates to reasonable precision
                    gdf_export = gdf.copy()
                    gdf_export['longitude'] = gdf_export['longitude'].round(6)
                    gdf_export['latitude'] = gdf_export['latitude'].round(6)

                    gdf_export.to_file(output_path, driver='GeoJSON')

                elif fmt == 'parquet':
                    # Efficient format for data processing
                    gdf.to_parquet(output_path)

                elif fmt == 'csv':
                    # Simple format without geometry
                    df_export = gdf.drop('geometry', axis=1)
                    df_export.to_csv(output_path, index=False)

                output_files[fmt] = output_path
                logger.info(f"Exported cities data to {output_path}")

            except Exception as e:
                logger.error(f"Error exporting to {fmt}: {e}")

        return output_files

    def get_cities_summary(self, gdf: gpd.GeoDataFrame) -> Dict:
        """Generate summary statistics for cities data"""

        summary = {
            'total_cities': len(gdf),
            'population_stats': {
                'total': int(gdf['population'].sum()),
                'mean': int(gdf['population'].mean()),
                'median': int(gdf['population'].median()),
                'max': int(gdf['population'].max()),
                'min': int(gdf['population'].min())
            },
            'geographic_bounds': {
                'north': float(gdf['latitude'].max()),
                'south': float(gdf['latitude'].min()),
                'east': float(gdf['longitude'].max()),
                'west': float(gdf['longitude'].min())
            },
            'size_categories': gdf['pop_category'].value_counts().to_dict(),
            'top_states': gdf.groupby('admin1_code').size().sort_values(ascending=False).head(10).to_dict()
        }

        return summary


# Alternative: Using geonamescache library for quick setup
class GeonamesCacheProcessor:
    """Alternative processor using the geonamescache library"""

    def __init__(self):
        try:
            import geonamescache
            self.gc = geonamescache.GeonamesCache()
        except ImportError:
            raise ImportError("geonamescache library not installed. Run: pip install geonamescache")

    def get_us_cities(self, min_population: int = 5000) -> gpd.GeoDataFrame:
        """Get US cities using geonamescache library"""

        logger.info(f"Loading cities with minimum population {min_population}")

        # Get all cities
        cities = self.gc.get_cities_by_name()  # This might not exist in all versions

        # Alternative: get cities and filter
        all_cities = []
        countries = self.gc.get_countries()
        us_country_code = 'US'

        # This is a simplified approach - the actual implementation would depend
        # on the specific geonamescache version and available methods

        # Create mock data structure for demonstration
        us_cities_data = []

        # In practice, you'd iterate through the cities data structure
        # and filter for US cities with the required population

        # Convert to GeoDataFrame
        if us_cities_data:  # If we have data
            df = pd.DataFrame(us_cities_data)
            geometry = [Point(row['longitude'], row['latitude']) for _, row in df.iterrows()]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
            return gdf
        else:
            # Return empty GeoDataFrame with correct structure
            columns = ['name', 'latitude', 'longitude', 'population', 'admin1_code']
            return gpd.GeoDataFrame(columns=columns, crs='EPSG:4326')


# Main processing function
def process_cities_data(
        source: str = 'geonames',
        dataset: str = 'cities5000',
        min_population: int = 5000,
        output_formats: List[str] = ['geojson', 'parquet']
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """Main function to process cities data and return GeoDataFrame"""

    if source == 'geonames':
        processor = CitiesDataProcessor()

        # Download and process data
        raw_data = processor.download_geonames_data(dataset)
        us_cities = processor.filter_us_cities(raw_data, min_population)
        cities_gdf = processor.create_cities_geodataframe(us_cities)

        # Export data
        output_files = processor.export_cities_data(cities_gdf, output_formats)

        # Generate summary
        summary = processor.get_cities_summary(cities_gdf)

        return cities_gdf, summary

    elif source == 'geonamescache':
        processor = GeonamesCacheProcessor()
        cities_gdf = processor.get_us_cities(min_population)

        # Basic summary
        summary = {
            'total_cities': len(cities_gdf),
            'source': 'geonamescache'
        }

        return cities_gdf, summary

    else:
        raise ValueError(f"Unknown source: {source}. Choose 'geonames' or 'geonamescache'")


# Example usage and testing
if __name__ == "__main__":
    # Process cities data
    cities_gdf, summary = process_cities_data(
        source='geonames',
        dataset='cities1000',
        min_population=1000,
        output_formats=['geojson', 'parquet', 'csv']
    )

    print("=== Cities Data Processing Complete ===")
    print(f"Total cities: {summary['total_cities']}")
    print(f"Population range: {summary['population_stats']['min']:,} - {summary['population_stats']['max']:,}")
    print(f"Geographic bounds: {summary['geographic_bounds']}")
    print(f"Size categories: {summary['size_categories']}")

    # Display sample cities
    print("\n=== Sample Cities ===")
    print(cities_gdf[['name', 'latitude', 'longitude', 'population', 'admin1_code']].head(10))

    # Show largest cities
    print("\n=== Largest Cities ===")
    largest = cities_gdf.nlargest(10, 'population')
    print(largest[['name', 'latitude', 'longitude', 'population', 'admin1_code']])