"""
Interactive map: Bluebikes stations sized by trip volume, with T stations on top.
Hover for details. Use the radio buttons to filter by rider type.

Reads (from the Project folder):
  clean/trips_2025_sample.csv     made by build_dataset.py
  MBTA_Gated_Station_Entries.csv  for the list of T stations and their lines
  MBTA_GTFS/stops.txt             for T station locations
  Boston and Cambridge neighborhood files, for the background map

Run:  python station_map.py
Output: station_map.html (open it in a browser)
"""

import glob

import altair as alt
import geopandas as gpd
import pandas as pd
import shapely


def find_file(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not find {patterns}")


BIKE_COLOR = "#1F6FB2"   # same colors as the hourly chart
T_COLOR = "#E07B00"

# ---------- background: Boston + Cambridge neighborhoods ----------
parts = []
for pattern in ["*oston*eighborhood*.json", "*ambridge*eighborhood*.geojson"]:
    g = gpd.read_file(find_file(pattern)).to_crs(4326)[["name", "geometry"]]
    parts.append(g)
nbhd = pd.concat(parts, ignore_index=True).rename(columns={"name": "neighborhood"})
nbhd = gpd.GeoDataFrame(nbhd, geometry="geometry", crs=4326)
nbhd["geometry"] = nbhd.geometry.simplify(0.0001)   # smaller file, same look
# Vega-Lite needs polygon outlines drawn clockwise, or the shapes fill the whole map
nbhd["geometry"] = shapely.orient_polygons(nbhd.geometry.values, exterior_cw=True)
minx, miny, maxx, maxy = nbhd.total_bounds

# ---------- Bluebikes: trips per start station and rider type ----------
trips = pd.read_csv("clean/trips_2025_sample.csv")
trips["rider_type"] = trips["rider_type"].str.capitalize()   # member -> Member
station_cols = ["start_station", "start_lat", "start_lng", "start_neighborhood",
                "near_transit", "nearest_t_station", "dist_to_t_m"]
counts = (trips.groupby(station_cols + ["rider_type"]).size()
               .reset_index(name="trips"))

# ---------- T stations ----------
gated = pd.read_csv(find_file("*Gated_Station_Entries*.csv"),
                    usecols=["stop_id", "station_name", "route_or_line"])
t_lines = (gated.drop_duplicates(["stop_id", "route_or_line"])
                .groupby(["stop_id", "station_name"])["route_or_line"]
                .apply(lambda s: ", ".join(sorted(s))).reset_index(name="lines"))
stops = pd.read_csv("MBTA_GTFS/stops.txt", usecols=["stop_id", "stop_lat", "stop_lon"])
t = t_lines.merge(stops, on="stop_id")
# keep only T stations inside the Boston/Cambridge area so the map doesn't stretch
t = t[t["stop_lon"].between(minx, maxx) & t["stop_lat"].between(miny, maxy)]

# ---------- chart ----------
rider = alt.param(name="rider", value="All",
                  bind=alt.binding_radio(options=["All", "Member", "Casual"],
                                         name="Rider type: "))

background = alt.Chart(nbhd).mark_geoshape(
    fill="#EEEEEE", stroke="white", strokeWidth=1
).encode(tooltip=alt.Tooltip("neighborhood:N", title="Neighborhood"))

bikes = alt.Chart(counts).transform_filter(
    "rider == 'All' || datum.rider_type == rider"
).transform_aggregate(
    trips="sum(trips)", groupby=station_cols
).mark_circle(
    color=BIKE_COLOR, opacity=0.6, stroke="white", strokeWidth=0.5
).encode(
    longitude="start_lng:Q",
    latitude="start_lat:Q",
    size=alt.Size("trips:Q", title="Trips started",
                  scale=alt.Scale(range=[10, 500])),
    tooltip=[
        alt.Tooltip("start_station:N", title="Bluebikes station"),
        alt.Tooltip("start_neighborhood:N", title="Neighborhood"),
        alt.Tooltip("trips:Q", title="Trips started (sample)"),
        alt.Tooltip("near_transit:N", title="Near or away from T"),
        alt.Tooltip("nearest_t_station:N", title="Nearest T station"),
        alt.Tooltip("dist_to_t_m:Q", title="Distance to T (m)", format=",.0f"),
    ],
).add_params(rider)

t_stops = alt.Chart(t).mark_point(
    shape="diamond", filled=True, color=T_COLOR, size=80,
    stroke="black", strokeWidth=0.6, opacity=1
).encode(
    longitude="stop_lon:Q",
    latitude="stop_lat:Q",
    tooltip=[alt.Tooltip("station_name:N", title="T station"),
             alt.Tooltip("lines:N", title="Line(s)")],
)

chart = alt.layer(background, bikes, t_stops).project(type="mercator").properties(
    width=650, height=650,
    title=alt.Title("Where do Bluebikes trips start, and how close are they to the T? (2025)",
                    subtitle=["Blue circles: Bluebikes stations, sized by trips started. "
                              "Orange diamonds: T stations.",
                              "Hover for details. Use the buttons below to filter by rider type."],
                    anchor="start", fontSize=16),
).configure_view(strokeWidth=0)

chart.save("station_map.html")
print(f"Saved station_map.html ({counts['start_station'].nunique()} Bluebikes stations, {len(t)} T stations)")