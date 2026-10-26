"""
Interactive map: Bluebikes stations sized by trip volume, with T stations on top.
Hover for details. Use the radio buttons to filter by rider type.

Reads (from the Project folder):
  clean/trips_2025_sample.csv     made by build_dataset.py
  clean/t_stations.csv            made by build_dataset.py (all T stations)
  Boston and Cambridge neighborhood files, for the background map

Run:  python station_map.py
Output: station_map.html (open it in a browser)
"""

import glob

import altair as alt
import geopandas as gpd
import pandas as pd
import shapely

from chart_style import BIKE_COLOR, T_COLOR


def find_file(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not find {patterns}")


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

# ---------- Bluebikes: trips per start station and rider type ----------
trips = pd.read_csv("clean/trips_2025_sample.csv")
trips["rider_type"] = trips["rider_type"].str.capitalize()   # member -> Member
station_cols = ["start_station", "start_lat", "start_lng", "start_neighborhood",
                "near_transit", "nearest_t_station", "dist_to_t_m"]
counts = (trips.groupby(station_cols + ["rider_type"]).size()
               .reset_index(name="trips"))

# ---------- T stations ----------
# every subway and light rail station, including street-level Green Line stops
t = pd.read_csv("clean/t_stations.csv").rename(columns={"t_station": "station_name"})
t["fare_gates"] = t["gated"].map({True: "Yes", False: "No (street-level stop)"})
# Show T stations that are inside Boston/Cambridge, plus any station outside that is
# the nearest stop for a bike station on the map (e.g. a Brookline stop next to the border).
# A simple bounding box doesn't work: Boston includes the Harbor Islands, so the box
# reaches down to Quincy.
area = nbhd.to_crs(26986).union_all()
t_points = gpd.GeoSeries(gpd.points_from_xy(t["stop_lon"], t["stop_lat"]), crs=4326).to_crs(26986)
inside = t_points.within(area).values
is_nearest = t["station_name"].isin(set(trips["nearest_t_station"])).values
t = t[inside | is_nearest]

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
             alt.Tooltip("lines:N", title="Line(s)"),
             alt.Tooltip("fare_gates:N", title="Fare gates")],
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