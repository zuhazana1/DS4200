"""
DS4200 project: build the cleaned 2025 Bluebikes + MBTA dataset.

Outputs (written to OUT_DIR):
  trips_2025_sample.csv  one row per sampled bike trip (~9,600 rows)
  system_hourly.csv      Bluebikes vs. T activity by hour, as a share of each
                         system's own total (for the hourly usage chart)
  stations.csv           one row per Bluebikes station with neighborhood,
                         nearest T stop, distance, and near/away label (for the map)
  t_stations.csv         every subway/light rail station with its line(s) and
                         whether it has fare gates (for the map)

Setup:
  pip install pandas numpy geopandas

Put these in the same folder as this script, then run it from that folder:
  202501-bluebikes-tripdata.csv ... 202512-bluebikes-tripdata.csv
  MBTA_Gated_Station_Entries.csv
  MBTA_GTFS/ folder                   (unzip MBTA_GTFS.zip; uses stops, routes, trips, stop_times)
  boston_neighborhood_boundaries.json
  Cambridge_Neighborhood_Polygons.geojson
"""

import glob
import os

import numpy as np
import pandas as pd

# ---------------- settings ----------------
# All files sit in the same folder as this script. The script finds them by
# pattern, so small differences in the file names are fine.
def find_file(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not find a file matching {patterns} in {os.getcwd()}")


BIKE_DIR = "."
GATED_FILE = find_file("*Gated_Station_Entries*.csv", "*gated*.csv")
GTFS_STOPS = find_file("MBTA_GTFS/stops.txt", "stops.txt")
BOSTON_NBHD = find_file("*oston*eighborhood*.json", "*oston*eighborhood*.geojson")
BOSTON_NAME_COL = "name"        # the script will tell you if this is wrong
CAMBRIDGE_NBHD = find_file("*ambridge*eighborhood*.geojson", "*ambridge*eighborhood*.json")
CAMBRIDGE_NAME_COL = "name"     # the script will tell you if this is wrong
OUT_DIR = "clean"

YEAR = 2025
NEAR_METERS = 400
SAMPLE_PER_MONTH = 800
SEED = 4200
MIN_MINUTES, MAX_MINUTES = 1, 180   # drop implausible trip durations
SERVICE_DAY_START_HOUR = 4          # T periods before 4 AM belong to the previous service date


def haversine_m(lat1, lng1, lat2, lng2):
    """Distance in meters. Works on numbers or numpy arrays (broadcasts)."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


# ---------------- 1. MBTA gated entries ----------------
def load_t_entries():
    g = pd.read_csv(GATED_FILE)
    g["service_date"] = pd.to_datetime(g["service_date"], format="mixed").dt.tz_localize(None).dt.normalize()

    # time_period looks like "(04:30:00)": take the hour
    g["hour"] = g["time_period"].str.strip("()").str[:2].astype(int)

    # periods after midnight belong to the previous service date, so move them
    # to the real calendar date so they line up with bike trip start times
    g["date"] = g["service_date"] + pd.to_timedelta(
        (g["hour"] < SERVICE_DAY_START_HOUR).astype(int), unit="D")
    g = g[g["date"].dt.year == YEAR]

    # one line per station: the line with the most entries there
    line = (g.groupby(["stop_id", "route_or_line"])["gated_entries"].sum()
              .reset_index().sort_values("gated_entries", ascending=False)
              .drop_duplicates("stop_id").set_index("stop_id")["route_or_line"])

    # hourly entries per station (sums across lines at transfer stations)
    hourly = (g.groupby(["stop_id", "date", "hour"], as_index=False)["gated_entries"].sum()
                .rename(columns={"gated_entries": "t_entries_hour"}))

    names = g.drop_duplicates("stop_id").set_index("stop_id")["station_name"]
    return hourly, line, names


def gtfs_line_name(route_id):
    for key, name in [("Green", "Green Line"), ("Red", "Red Line"), ("Orange", "Orange Line"),
                      ("Blue", "Blue Line"), ("Mattapan", "Mattapan Line")]:
        if route_id.startswith(key):
            return name
    return route_id


def load_t_stations(line, names):
    """All subway and light rail stations from GTFS, plus any other gated stations
    (Silver Line Waterfront). Street-level Green Line stops have no fare gates, so they
    are not in the entries data, but they still count when finding the nearest T stop."""
    gtfs_dir = os.path.dirname(GTFS_STOPS)
    stops = pd.read_csv(GTFS_STOPS, dtype=str)
    routes = pd.read_csv(os.path.join(gtfs_dir, "routes.txt"), dtype=str)
    trips = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"), dtype=str, usecols=["route_id", "trip_id"])

    # route_type 0 = light rail (Green Line, Mattapan), 1 = subway
    rail_routes = set(routes.loc[routes["route_type"].isin(["0", "1"]), "route_id"])
    trips = trips[trips["route_id"].isin(rail_routes)]
    trip_route = trips.set_index("trip_id")["route_id"]

    # every stop those trips serve (stop_times is big, so read it in chunks)
    pairs = []
    for chunk in pd.read_csv(os.path.join(gtfs_dir, "stop_times.txt"), dtype=str,
                             usecols=["trip_id", "stop_id"], chunksize=1_000_000):
        chunk = chunk[chunk["trip_id"].isin(trip_route.index)]
        chunk["route_id"] = chunk["trip_id"].map(trip_route)
        pairs.append(chunk[["stop_id", "route_id"]].drop_duplicates())
    pairs = pd.concat(pairs).drop_duplicates()

    # platforms -> their parent station ("place-..."), the same ids the entries data uses
    parent = stops.set_index("stop_id")["parent_station"]
    pairs["station_id"] = pairs["stop_id"].map(parent).fillna(pairs["stop_id"])
    pairs["line"] = pairs["route_id"].map(gtfs_line_name)
    gtfs_lines = pairs.groupby("station_id")["line"].apply(lambda s: sorted(set(s)))

    ids = sorted(set(gtfs_lines.index) | set(line.index))
    t = stops[stops["stop_id"].isin(ids)][["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
    t[["stop_lat", "stop_lon"]] = t[["stop_lat", "stop_lon"]].astype(float)
    missing = set(ids) - set(t["stop_id"])
    if missing:
        print(f"Warning: {len(missing)} T stations not found in stops.txt: {sorted(missing)[:10]}")

    t["gated"] = t["stop_id"].isin(line.index)
    t["t_station"] = t["stop_id"].map(names).fillna(t["stop_name"])
    # main line: the busiest line for gated stations, otherwise the GTFS line
    t["t_line"] = t["stop_id"].map(line).fillna(t["stop_id"].map(gtfs_lines).str[0])
    all_lines = t["stop_id"].map(gtfs_lines).apply(lambda x: x if isinstance(x, list) else [])
    t["lines"] = [", ".join(sorted(set(a) | {b})) for a, b in zip(all_lines, t["t_line"])]
    return t.drop(columns="stop_name").reset_index(drop=True)


# ---------------- 2. Bluebikes trips ----------------
BIKE_COLS = ["rideable_type", "started_at", "ended_at",
             "start_station_name", "start_station_id", "end_station_name", "end_station_id",
             "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"]


def load_trips():
    files = sorted(glob.glob(os.path.join(BIKE_DIR, f"{YEAR}*-bluebikes-tripdata.csv")))
    if len(files) != 12:
        print(f"Warning: found {len(files)} monthly files, expected 12")
    df = pd.concat((pd.read_csv(f, usecols=BIKE_COLS,
                                dtype={"start_station_id": str, "end_station_id": str})
                    for f in files), ignore_index=True)

    df["started_at"] = pd.to_datetime(df["started_at"], format="mixed")
    df["ended_at"] = pd.to_datetime(df["ended_at"], format="mixed")
    df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60

    before = len(df)
    df = df.dropna(subset=["start_station_id", "end_station_id",
                           "start_lat", "start_lng", "end_lat", "end_lng"])
    df = df[df["duration_min"].between(MIN_MINUTES, MAX_MINUTES)]
    df = df[df["started_at"].dt.year == YEAR]
    print(f"Trips: {before:,} raw -> {len(df):,} after removing missing stations and bad durations")
    return df


def build_station_table(trips):
    """One row per station id with median coordinates (e-bike coords vary a little)."""
    start = trips[["start_station_id", "start_station_name", "start_lat", "start_lng"]]
    start.columns = ["station_id", "station_name", "lat", "lng"]
    end = trips[["end_station_id", "end_station_name", "end_lat", "end_lng"]]
    end.columns = start.columns
    both = pd.concat([start, end])
    return both.groupby("station_id").agg(
        station_name=("station_name", "first"),
        lat=("lat", "median"),
        lng=("lng", "median")).reset_index()


def assign_neighborhoods(stations):
    """Spatial join stations to Boston + Cambridge neighborhoods. Outside both -> NaN."""
    import geopandas as gpd

    def read_polys(path, name_col):
        g = gpd.read_file(path).to_crs(4326)
        if name_col not in g.columns:
            raise KeyError(f"'{name_col}' is not a column in {path}. Columns are: {list(g.columns)}. "
                           f"Change the *_NAME_COL setting at the top of the script.")
        return g[[name_col, "geometry"]].rename(columns={name_col: "neighborhood"})

    bos = read_polys(BOSTON_NBHD, BOSTON_NAME_COL)
    cam = read_polys(CAMBRIDGE_NBHD, CAMBRIDGE_NAME_COL)
    bos["city"], cam["city"] = "Boston", "Cambridge"
    polys = pd.concat([bos, cam], ignore_index=True)

    pts = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations["lng"], stations["lat"]), crs=4326)
    joined = gpd.sjoin(pts, polys, how="left", predicate="within")
    joined = joined.drop_duplicates("station_id")  # a point on a border can match twice
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))


def add_nearest_t(stations, t):
    d = haversine_m(stations["lat"].values[:, None], stations["lng"].values[:, None],
                    t["stop_lat"].values[None, :], t["stop_lon"].values[None, :])
    idx = d.argmin(axis=1)
    stations["nearest_t_stop_id"] = t["stop_id"].values[idx]
    stations["nearest_t_station"] = t["t_station"].values[idx]
    stations["nearest_t_line"] = t["t_line"].values[idx]
    stations["nearest_t_gated"] = t["gated"].values[idx]
    stations["dist_to_t_m"] = d.min(axis=1).round(0)
    stations["near_transit"] = np.where(stations["dist_to_t_m"] <= NEAR_METERS, "Near", "Away")
    return stations


def time_of_day(hour):
    return pd.cut(hour, bins=[-1, 6, 9, 15, 18, 21, 23],
                  labels=["Overnight", "AM peak", "Midday", "PM peak", "Evening", "Overnight"],
                  ordered=False)


def add_time_features(df):
    df["date"] = df["started_at"].dt.normalize()
    df["hour"] = df["started_at"].dt.hour
    df["month"] = df["started_at"].dt.month_name()
    df["day_of_week"] = df["started_at"].dt.day_name()
    df["is_weekend"] = df["started_at"].dt.dayofweek >= 5
    df["time_of_day"] = time_of_day(df["hour"])
    return df


# ---------------- 3. hourly chart data ----------------
def build_system_hourly(trips, t_hourly):
    bike = trips.assign(day_type=np.where(trips["is_weekend"], "Weekend", "Weekday"))
    bike = bike.groupby(["day_type", "hour"]).size().rename("count").reset_index()
    bike["system"] = "Bluebikes"

    t = t_hourly.assign(day_type=np.where(t_hourly["date"].dt.dayofweek >= 5, "Weekend", "Weekday"))
    t = t.groupby(["day_type", "hour"])["t_entries_hour"].sum().rename("count").reset_index()
    t["system"] = "T"

    out = pd.concat([bike, t], ignore_index=True)
    out["share"] = out["count"] / out.groupby(["system", "day_type"])["count"].transform("sum")
    return out[["system", "day_type", "hour", "count", "share"]]


# ---------------- main ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    t_hourly, t_line, t_names = load_t_entries()
    t_stations = load_t_stations(t_line, t_names)
    print(f"T stations: {len(t_stations)} total, {t_stations['gated'].sum()} with gated entry data")
    t_stations.to_csv(os.path.join(OUT_DIR, "t_stations.csv"), index=False)

    trips = load_trips()
    stations = build_station_table(trips)
    stations = assign_neighborhoods(stations)
    stations = add_nearest_t(stations, t_stations)

    # keep trips that start in Boston or Cambridge
    st = stations.set_index("station_id")
    trips = trips[trips["start_station_id"].map(st["neighborhood"]).notna()].copy()
    trips = add_time_features(trips)
    print(f"Trips starting in Boston/Cambridge: {len(trips):,}")

    # hourly chart uses all filtered trips, not just the sample
    build_system_hourly(trips, t_hourly).to_csv(os.path.join(OUT_DIR, "system_hourly.csv"), index=False)

    # sample the same number of trips from each month
    counts = trips["month"].value_counts()
    if (counts < SAMPLE_PER_MONTH).any():
        raise ValueError(f"Some months have fewer than {SAMPLE_PER_MONTH} trips:\n{counts}")
    sample = trips.groupby("month").sample(n=SAMPLE_PER_MONTH, random_state=SEED).copy()

    # attach station info
    for side in ["start", "end"]:
        ids = sample[f"{side}_station_id"]
        sample[f"{side}_neighborhood"] = ids.map(st["neighborhood"]).fillna("Outside Boston/Cambridge")
        sample[f"{side}_lat"] = ids.map(st["lat"])
        sample[f"{side}_lng"] = ids.map(st["lng"])
    for col in ["nearest_t_stop_id", "nearest_t_station", "nearest_t_line", "dist_to_t_m", "near_transit"]:
        sample[col] = sample["start_station_id"].map(st[col])

    sample["distance_km"] = (haversine_m(sample["start_lat"], sample["start_lng"],
                                         sample["end_lat"], sample["end_lng"]) / 1000).round(2)
    sample["duration_min"] = sample["duration_min"].round(1)

    # T entries at the nearest T station during the trip's start hour
    sample = sample.merge(t_hourly, how="left",
                          left_on=["nearest_t_stop_id", "date", "hour"],
                          right_on=["stop_id", "date", "hour"]).drop(columns="stop_id")
    ungated = ~sample["start_station_id"].map(st["nearest_t_gated"]).astype(bool)
    n_missing = sample["t_entries_hour"].isna().sum()
    print(f"Trips with no T entry count: {n_missing} "
          f"({ungated.sum()} because the nearest stop is a street-level stop with no fare gates, "
          f"{n_missing - ungated.sum()} because a gated station had no data that hour)")

    sample = sample.rename(columns={"member_casual": "rider_type", "rideable_type": "bike_type",
                                    "start_station_name": "start_station",
                                    "end_station_name": "end_station"})
    keep = ["rider_type", "bike_type", "start_station", "end_station",
            "start_neighborhood", "end_neighborhood", "nearest_t_station", "nearest_t_line",
            "near_transit", "month", "day_of_week", "is_weekend", "time_of_day", "hour",
            "duration_min", "distance_km", "dist_to_t_m", "t_entries_hour",
            "start_lat", "start_lng", "end_lat", "end_lng"]
    sample[keep].to_csv(os.path.join(OUT_DIR, "trips_2025_sample.csv"), index=False)

    st.reset_index().to_csv(os.path.join(OUT_DIR, "stations.csv"), index=False)

    print(f"\nDone. trips_2025_sample.csv: {len(sample):,} rows x {len(keep)} columns")
    print(sample["near_transit"].value_counts().to_string())


if __name__ == "__main__":
    main()