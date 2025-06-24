import geopandas as gpd
import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Optional, List, Dict, Any, Union

from smart_query.data_repo.data_repository import DataRepository
from smart_query.data_repo.dataframe_annotation import DataFrameAnnotation
from smart_query.data_retriever.base_retriever import DataFrameRetriever
from smart_query.utils.logger import get_logger 

logger = get_logger(__name__)


class ArcGISFeatureLoader:
    def __init__(self, url: str, batch_size: int = 100, max_workers: int = 4, max_retries: int = 3):
        """
        Initialize the ArcGIS Feature Service loader.
        
        Args:
            url: The base URL of the ArcGIS Feature Service
            batch_size: Number of records to fetch per request
            max_workers: Maximum number of concurrent workers
            max_retries: Maximum number of retry attempts per failed request
        """
        self.url = url
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries

    def get_total_record_count(self, where: str) -> int:
        """Fetch the total number of records available."""
        params = {
            "where": where,
            "returnCountOnly": "true",
            "f": "json"
        }
        response = requests.get(self.url + "/query", params=params)
        response.raise_for_status()
        return response.json().get("count", 0)

    def fetch_batch(self, where: str, offset: int, bbox: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Fetch a batch of features with retry logic.
        
        Args:
            where: SQL where clause
            offset: Starting record offset
            bbox: Optional bounding box [minx, miny, maxx, maxy]
        """
        params = {
            "where": where,
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
            "outSR": "4326",
            "resultOffset": offset,
            "resultRecordCount": self.batch_size
        }
        
        if bbox:
            params.update({
                "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects"
            })

        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.url + "/query", params=params)
                response.raise_for_status()
                data = response.json()
                return data.get('features', [])
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch batch at offset {offset} after {self.max_retries} attempts: {str(e)}")
                    raise
                time.sleep(1 * (attempt + 1))  # Exponential backoff

    def load_features(self, where: str = "1=1", bbox: Optional[List[float]] = None) -> gpd.GeoDataFrame:
        """
        Load all features from the service using concurrent requests.
        
        Args:
            where: SQL where clause
            bbox: Optional bounding box [minx, miny, maxx, maxy]
            
        Returns:
            GeoDataFrame containing all features
        """
        total_records = self.get_total_record_count(where)
        logger.info(f"Total records to fetch: {total_records}")
        
        if total_records == 0:
            return gpd.GeoDataFrame(columns=['geometry'])

        offsets = range(0, total_records, self.batch_size)
        all_features = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_offset = {
                executor.submit(self.fetch_batch, where, offset, bbox): offset 
                for offset in offsets
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_offset):
                offset = future_to_offset[future]
                try:
                    features = future.result()
                    all_features.extend(features)
                    completed += len(features)
                    logger.info(f"Progress: {completed}/{total_records} features ({(completed/total_records)*100:.1f}%)")
                except Exception as e:
                    logger.error(f"Failed to fetch batch at offset {offset}: {str(e)}")

        if not all_features:
            return gpd.GeoDataFrame(columns=['geometry'])

        gdf = gpd.GeoDataFrame.from_features(all_features)
        logger.info(f"Successfully loaded {len(gdf)} features")
        return gdf


def load_features(self_url, where, wkid):
    url_string = self_url + "/query?where={}&returnGeometry=true&outFields={}&f=geojson".format(where, '*')
    resp = requests.get(url_string, verify=True)
    data = resp.json()
    if data['features']:
        return gpd.GeoDataFrame.from_features(data['features'], crs=f'EPSG:{wkid}')
    else:
        return gpd.GeoDataFrame(columns=['geometry'], crs=f'EPSG:{wkid}')


def load_coal_mines(where, bbox=None):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/CoalMines_US_EIA/FeatureServer/247"
    wkid = "3857"
    return get_arcgis_features(self_url, where, wkid, bbox)  


def load_coal_power_plants(where, bbox=None):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Coal_Power_Plants/FeatureServer/0"
    wkid = "3857"    
    return get_arcgis_features(self_url, where, wkid, bbox)  


def load_wind_power_plants(where, bbox=None):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Wind_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return get_arcgis_features(self_url, where, wkid, bbox)  


def load_renewable_diesel_fuel_and_other_biofuel_plants(where, bbox=None):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Renewable_Diesel_and_Other_Biofuels/FeatureServer/245"
    wkid = "3857"
    return get_arcgis_features(self_url, where, wkid, bbox)  


def load_battery_storage_plants(where, bbox=None):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Battery_Storage_Plants/FeatureServer/0"
    wkid = "3857"
    return get_arcgis_features(self_url, where, wkid, bbox)  


def load_geothermal_power_plants(where, bbox=None):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Geothermal_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return get_arcgis_features(self_url, where, wkid, bbox)  


def load_hydro_pumped_storage_power_plants(where, bbox=None):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Hydro_Pumped_Storage_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return get_arcgis_features(self_url, where, wkid, bbox)  


def load_natural_gas_power_plants(where, bbox=None):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Natural_Gas_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return get_arcgis_features(self_url, where, wkid, bbox)  


def load_nuclear_power_plants(where, bbox=None):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Nuclear_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return get_arcgis_features(self_url, where, wkid, bbox)  


def load_petroleum_power_plants(where, bbox=None):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Petroleum_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return get_arcgis_features(self_url, where, wkid, bbox)  


def load_solar_power_plants(where, bbox=None):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Solar_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return get_arcgis_features(self_url, where, wkid, bbox)  


def load_biodiesel_plants(where, bbox=None):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Biodiesel_Plants_US_EIA/FeatureServer/113"
    wkid = "3857"
    return get_arcgis_features(self_url, where, wkid, bbox)  


def get_arcgis_features(self_url, where, wkid=4326, bbox=None):
    if bbox is None:
        bbox = [-125.0, 24.396308, -66.93457, 49.384358]
    minx, miny, maxx, maxy = bbox
    params = {
        "where": where,
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": "4326",  # Ensure output is in WGS84                                                                                                                
        "resultOffset": 0,
        "resultRecordCount": 1000  # Increase this if needed                                                                                                         
    }

    response = requests.get(self_url + "/query", params=params)
    data = response.json()
    # st.code(response.url)
    # st.code(data)
    if data['features']:
        return gpd.GeoDataFrame.from_features(data['features'], crs=f"EPSG:{wkid}")
    else:
        return gpd.GeoDataFrame(columns=['geometry'], crs=f"EPSG:{wkid}")


def load_watersheds(where, bbox):
    self_url = "https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/Watershed_Boundary_Dataset_HUC_10s/FeatureServer/0"
    wkid = "3857"
    return get_arcgis_features(self_url, where, wkid, bbox)  


def load_basins(where, bbox):
    self_url = "https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/Watershed_Boundary_Dataset_HUC_6s/FeatureServer/0"
    wkid = "3857"
    return get_arcgis_features(self_url, where, wkid, bbox)  


def load_basins_2(where: str = "1=1", bbox: Optional[List[float]] = None) -> gpd.GeoDataFrame:
    """Load watershed boundary dataset using concurrent fetching."""
    url = "https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/Watershed_Boundary_Dataset_HUC_6s/FeatureServer/0"
    
    loader = ArcGISFeatureLoader(
        url=url,
        batch_size=100,
        max_retries=3
    )   
    if bbox is None and where == "1 = 1":
        raise Exception("Your request returned a large number of basins. Please refine your search.")
    gdf = loader.load_features(where=where, bbox=bbox)
    return gdf


class EnergyAtlasRetriever(DataFrameRetriever):

    def get_description(self):
        return f"""Dataframe Retriever: "{self.name}"
This retriever can load a GeoDataFrame for one of the following entity types at a time:
    1. Battery Storage Plant
    2. Coal Mine
    3. Coal Power Plant
    4. Geothermal Power Plant
    5. Wind Power Plant
    6. Renewable Diesel Fuel and Other Biofuel Plant
    7. Wind Power Plant
    8. Hydro Pumped Storage Power Plant
    9. Natural Gas Power Plant
    10. Nuclear Power Plant
    11. Petroleum Power Plant
    12. Solar Power Plant
    13. Biodiesel Plant
    14. Watershed
    15. Basin

Please note that all power plants and coal mines and storage plants have the attribute "County" and "State". 
In other words, this retriever can process the following queries directly:
    Find all coal mines in the Ohio state.
    Find all solar power plants in California.

"""

    def get_examples(self):
        return """Find all coal mines along Ohio River.
Find all counties downstream of coal mine with the name "Century Mine" along Ohio River.
Find renewable energy sources in Ohio state.
Find all coal-fired power plants along Ohio river."""

    # -------------------------------------------
    # The method does the query
    # -------------------------------------------
    def get_dataframe_annotation(self, data_repo: DataRepository, atomic_request: str):
        prompt = PromptTemplate(
            template="""
[ Data Repository ]
The Data Repository contains the following dataframes:
   {data_repo}
The meaning of each dataframe is defined by its title. Each dataframe can be accessed by the expression
given in "How to Access". For example, data_repo.dataframe_annotations[0].df can access the first dataframe. 
   
[ Definition 1 ] 
We have the following function to get coal mines from an ArcGIS Feature Service as a GeoDataFrame:
   load_coal_mines(where_condition)

The returned GeoDataFrame has the following columns:
   'geometry', 'OBJECTID', 'MSHA_ID', 'MINE_NAME', 'MINE_TYPE', 'MINE_STATE', 'STATE', 'FIPS_COUNTY', 
   'MINE_COUNTY', 'PRODUCTION', 'PHYSICAL_UNIT', 'REFUSE', 'Source', 'PERIOD', 'Longitude', 'Latitude'

Use the column 'STATE' rather than the column 'MINE_STATE' to find coal mines in a state. 
The values in the column 'STATE' are all in upper case like 'ALABAMA' or 'COLORADO' etc. 
The column 'COUNTY' contains values like 'Walker' or 'Jefferson'. 
To find all coal mines in Ohio, use the condition "STATE='OHIO'". 

[ Definition 2 ] 
We have the following functions to get coal power plants/wind power plants/battery storage plants/
geothermal power plants/hydro pumped storage power plants/natural gas power plants/nuclear power plants/
petroleum power plants/solar power plants from an ArcGIS Feature Service as a GeoDataFrame:
   load_coal_power_plants(where_condition)
   load_wind_power_plants(where_condition)
   load_battery_storage_plants(where_condition)
   load_geothermal_power_plants(where_condition)
   load_hydro_pumped_storage_power_plants(where_condition)
   load_natural_gas_power_plants(where_condition)
   load_nuclear_power_plants(where_condition)
   load_petroleum_power_plants(where_condition)
   load_solar_power_plants(where_condition)

The returned GeoDataFrame has the following columns:
   'geometry', 'OBJECTID', 'Plant_Code', 'Plant_Name', 'Utility_ID', 'Utility_Name', 'sector_name', 
   'Street_Address', 'City', 'County', 'State', 'Zip', 'PrimSource', 'source_desc', 'tech_desc', 
   'Install_MW', 'Total_MW', 'Bat_MW', 'Bio_MW', 'Coal_MW', 'Geo_MW', 'Hydro_MW', 'HydroPS_MW', 
   'NG_MW', 'Nuclear_MW', 'Crude_MW', 'Solar_MW', 'Wind_MW', 'Other_MW', 'Source', 'Period', 
   'Longitude', 'Latitude'

The values in the column 'State' are case sensitive like 'Nebraska' or 'Montana' or 'Kentucky' etc. 
The column 'County' contains values like 'Adams' or 'Yellowstone'. The column 'Total_MW' gives the 
Total Megawatts of the plants.

Note that use the case sensitive state names for the column 'State'.

[ Definition 3 ]
We have the following function to get renewable diesel fuel and other biofuel plants/biodiesel plants
from an ArcGIS Feature Service as a GeoDataFrame:
   load_renewable_diesel_fuel_and_other_biofuel_plants(where_condition)
   load_biodiesel_plants(where_condition)

The returned GeoDataFrame has the following columns:
   'geometry', 'OBJECTID', 'Company', 'Site', 'State', 'PADD', 'Cap_Mmgal',
  'Source', 'Period', 'Longitude', 'Latitude'

The values in the column 'State' are case sensitive like 'Nebraska' or 'Montana' etc.

To get all coal mines/coal power plants/wind power plants/renewable diesel fuel and 
other biofuel plants and etc, call the correspondent function with "1 = 1" as where_condition.

[ Definition 4 ]
We have the following function to get watersheds from an ArcGIS Feature Service as a GeoDataFrame:
   load_watersheds(where_condition, bbox)
where bbox is for a bounding box. Use None if bbox is unknown or not needed. 

The returned GeoDataFrame has the following columns:
   'geometry', 'OBJECTID', 'HUC10', 'NAME', 'HUTYPE', 'Shape__Area', 'Shape__Length'

[ Definition 5 ]
We have the following function to get basins from an ArcGIS Feature Service as a GeoDataFrame:
   load_basins(where_condition, bbox)
where bbox is for a bounding box. Use None if bbox is unknown or not needed. 

The returned GeoDataFrame has the following columns:
   'geometry', 'OBJECTID', 'HUC6', 'NAME', 'Shape__Area', 'Shape__Length'

Use the following condition when trying to get a watershed by a given watershed name (e.g., Headwaters Scioto River):
    UPPER(NAME) = UPPER('Headwaters Scioto River')
The reason for this is that there may be spaces in the name column of the ArcGIS Feature service.


[ Question ]
The following is the request from the user:
   {request}

Please return valid Python code in the following format to implement the user's request: 
   # other code which do not change Data Repository
   gdf = ......
   title = ......
   
Do not include any preamble, explanations, or Markdown formatting (e.g., no ```). Just return the plain Python code.  
Return only the Python code. Do not include any text like "Here is the Python code to implement the user's request:" or 
any other comments.

For example, if the request is "Find the Kanawha basin", you can return the following:
   gdf = load_basins("UPPER(NAME) = UPPER('Kanawha')", None)
   title = "the Kanawha basin"


[ Example 1]
Find all coal mines along Ohio River. 

Find out if the Data Repository has a GeoDataFrame containing Ohio River.

If the Data Repository doesn't have a GeoDataFrame containing Ohio River, then return the following code:
   raise Exception("The data for Ohio River is missing. Please load Ohio River first.")

If the Data Repository has a DataFrame with the title of Ohio River, then return the following valid Python code:
   gdf1 = <Use the expression specified in "How to Access" to load the dataframe for Ohio River, for example, data_repo.dataframe_annotations[0].df>
   gdf2 = load_coal_mines("1=1")
   # Keep the following line exactly as it is
   distance_threshold = 0.2
   gdf2['distance_to_river'] = gdf2.geometry.apply(lambda x: gdf1.distance(x).min())
   gdf = gdf2[gdf2['distance_to_river'] <= distance_threshold]
   gdf = gdf.drop(columns=['distance_to_river'])
   title = "All Coal Mines along Ohio River"

Do not include any preamble, explanations, or Markdown formatting (e.g., no ```). Just return the plain Python code.  
Return only the Python code. Do not include any text like "Here is the Python code to implement the user's request:" or 
any other comments.

[ Example 2 ]
Find all coal power plants along Ohio River.

Use the same way as Example 1 to implement it. Just replace load_coal_mines by load_coal_power_plants and change the title.
If none of the available variables are geodataframes containing Ohio River, then return the code raising the execption. 

[ Example 3 ]
If the request is for an attribute of a particular plant, first obtain the plant as gdf, and then store the answer 
to the user in gdf.answer. 

For example, find the capacity of the coal power plant Rockport.
   gdf = load_coal_power_plants("Plant_Name = 'Rockport'")
   title = "The Coal Power Plant Rockport"
   answer = f"The capacity of the coal power plant Rockport is {{gdf.iloc[0]['Total_MW']}} megawatt."
   
[ Note 1 ]
Use pandas.concat to concatenate two GeoDataFrame gdf1 and gdf2:
   gdf = pd.concat([gdf1, gdf2], ignore_index=True)
   gdf = gpd.GeoDataFrame(gdf, geometry='geometry')


[ Example 4 ]
Find all solar power plants in all counties the Scioto River flows through.

Find out if one of the available variables is a geodataframe containing all counties the Scioto River flows through.

If none of the available variables are geodataframes containing all counties the Scioto River flows through, then return the following code:
    raise Exception("The data for all counties the Scioto River flows through is missing. Please load it first.")
        
If you found a variable which is a geodataframe containing all counties the Scioto River flows through, then return the valid Python code in the following format:
    gdf1 = <replace by the variable of the geodataframe for all counties the Scioto River flows through if you found one>
    gdf2 = load_solar_power_plants("1 = 1")
    gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="intersects")
    gdf = gdf[~gdf.geometry.apply(lambda geom: geom.touches(gdf1.unary_union))]
    # Ensure all columns from gdf2 are retained
    for col in gdf2.columns:
        if col not in gdf.columns:
            gdf[col] = gdf2[col]
    gdf = gdf[gdf2.columns]
    title = "All solar power plants in all counties the Scioto River flows through"


[ Example 5 ]
Find all the watersheds that feed into the Scioto River.
        
Find out if one of the available variables is a geodataframe containing Scioto River.

If none of the available variables are geodataframes containing Scioto River, then return the following code:
    raise Exception("The data for the Scioto River is missing. Please load it first.")
        
If you found a variable which is a geodataframe containing Scioto River, then return the valid Python code in the following format:
    gdf1 = <replace by the variable of the geodataframe for the Scioto River if you found one>
    gdf2 = load_watersheds("1 = 1", gdf1.total_bounds)
    buffer_distance = 0.01
    gdf1_buffered = gdf1.copy()
    gdf1_buffered['geometry'] = gdf1_buffered['geometry'].buffer(buffer_distance)
    gdf = gpd.sjoin(gdf2, gdf1_buffered, how="inner", predicate="intersects")
    gdf = gdf[~gdf.geometry.apply(lambda geom: geom.touches(gdf1.unary_union))]
    # Ensure all columns from gdf2 are retained
    for col in gdf2.columns:
        if col not in gdf.columns:
            gdf[col] = gdf2[col]
    gdf = gdf[gdf2.columns]
    title = "All the watersheds that feed into the Scioto River"


[ Example 6 ]
Find all the watersheds in Ohio State.

Find out if one of the available variables is a geodataframe containing Ohio State.

If none of the available variables are geodataframes containing Ohio State, then return the following code:
    raise Exception("The data for the Ohio State is missing. Please load it first.")
        
If you found a variable which is a geodataframe containing Ohio State, then return the valid Python code in the following format:
    gdf1 = <replace by the variable of the geodataframe for the Ohio State if you found one>
    gdf2 = load_watersheds("1 = 1", gdf1.total_bounds)
    gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="intersects")
    gdf = gdf[~gdf.geometry.apply(lambda geom: geom.touches(gdf1.unary_union))]
    # Ensure all columns from gdf2 are retained
    for col in gdf2.columns:
        if col not in gdf.columns:
            gdf[col] = gdf2[col]
    gdf = gdf[gdf2.columns]
    title = "All the watersheds in Ohio State"


[ Example 7 ]
Find all the watersheds in Ross County in Ohio State.

Find out if one of the available variables is a geodataframe containing Ross County in Ohio State.

If none of the available variables are geodataframes containing Ross County in Ohio State, then return the following code:
    raise Exception("The data for Ross County in Ohio State is missing. Please load it first.")
        
If you found a variable which is a geodataframe containing Ohio State, then return the valid Python code in the following format:
    gdf1 = <replace by the variable of the geodataframe for Ross County in Ohio State if you found one>
    gdf2 = load_watersheds("1 = 1", gdf1.total_bounds)
    gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="intersects")
    gdf = gdf[~gdf.geometry.apply(lambda geom: geom.touches(gdf1.unary_union))]
    gdf = gdf[gdf2.columns]
    title = "All the watersheds in Ross County in Ohio State"


[ Example 7]
Find all basins through which the Scioto River flows. This request means "find all basins which are intersecting with the Scioto River".

Find out if one of the available variables is a geodataframe containing the Scioto River.

If none of the available variables are geodataframes containing the Scioto River, then return the following code:
    raise Exception("The data for the Scioto River is missing. Please load it first.")

If you found a variable which is a geodataframe containing the Scioto River, then return the valid Python code in the following format:
    gdf1 = <replace by the variable of the geodataframe for the Scioto River if you found one>
    gdf2 = load_basins("1 = 1", gdf1.total_bounds)
    gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="intersects")
    gdf = gdf[~gdf.geometry.apply(lambda geom: geom.touches(gdf1.unary_union))]
    # Ensure all columns from gdf2 are retained
    for col in gdf2.columns:
        if col not in gdf.columns:
            gdf[col] = gdf2[col]
    gdf = gdf[gdf2.columns]
    title = "All the basins through which the Scioto River flows"


[ Example 8 ]
Find all the basins in Ohio State.

Find out if one of the available variables is a geodataframe containing Ohio State.

If none of the available variables are geodataframes containing Ohio State, then return the following code:
    raise Exception("The data for the Ohio State is missing. Please load it first.")
        
If you found a variable which is a geodataframe containing Ohio State, then return the valid Python code in the following format:
    gdf1 = <replace by the variable of the geodataframe for the Ohio State if you found one>
    gdf2 = load_basins("1 = 1", gdf1.total_bounds)
    gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="intersects")
    gdf = gdf[~gdf.geometry.apply(lambda geom: geom.touches(gdf1.unary_union))]
    # Ensure all columns from gdf2 are retained
    for col in gdf2.columns:
        if col not in gdf.columns:
            gdf[col] = gdf2[col]
    gdf = gdf[gdf2.columns]
    title = "All the basins in Ohio State"

Please make sure the indentation is correct by removing the leading space. 


 """,
            input_variables=["request", "data_repo"],
        )
        code_chain = prompt | self.llm | StrOutputParser()

        data_repo_description = ""
        for index, adf in enumerate(data_repo.dataframe_annotations):
            data_repo_description = f"{data_repo_description}\n{str(adf)}\n" \
                                    f"How to Access: data_repo.dataframe_annotations[{index}].df\n"

        # prompt_instance = prompt.format(data_repo=str(data_repo_description), request=atomic_request)
        # print('-' * 70)
        # print(prompt_instance)
        # print('-' * 70)

        code = code_chain.invoke({"request": atomic_request, "data_repo": str(data_repo)})
        if code.startswith("```python"):
            start_index = code.find("```python") + len("```python")
            end_index = code.find("```", start_index)
            code = code[start_index:end_index].strip()
        elif code.startswith("```"):
            start_index = code.find("```") + len("```")
            end_index = code.find("```", start_index)
            code = code[start_index:end_index].strip()

        logger.debug(f"Translate the request into Python code: \n\n{code}\n")
        
        global_vars = {
            'requests': requests,
            'gpd': gpd,
            'data_repo': data_repo,
            'load_coal_mines': load_coal_mines,
            'load_coal_power_plants': load_coal_power_plants,
            'load_wind_power_plants': load_wind_power_plants,
            'load_renewable_diesel_fuel_and_other_biofuel_plants': load_renewable_diesel_fuel_and_other_biofuel_plants,
            'load_battery_storage_plants': load_battery_storage_plants,
            'load_geothermal_power_plants': load_geothermal_power_plants,
            'load_hydro_pumped_storage_power_plants': load_hydro_pumped_storage_power_plants,
            'load_natural_gas_power_plants': load_natural_gas_power_plants,
            'load_nuclear_power_plants': load_nuclear_power_plants,
            'load_petroleum_power_plants': load_petroleum_power_plants,
            'load_solar_power_plants': load_solar_power_plants,
            'load_biodiesel_plants': load_biodiesel_plants,
            'load_watersheds': load_watersheds,
            'load_basins': load_basins,
            'get_arcgis_features': get_arcgis_features
        }
        exec(code, global_vars)
        title = global_vars.get("title")
        gdf = global_vars.get("gdf")
        return DataFrameAnnotation(gdf, title)
