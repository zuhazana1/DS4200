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
"""
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd


def find_file(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not find {patterns}")


# Settings
GATED_FILE = find_file("*Gated_Station_Entries*.csv", "*gated*.csv")
GTFS_DIR = "MBTA_GTFS"
BOSTON_NBHD = find_file("*oston*eighborhood*.json", "*oston*eighborhood*.geojson")
CAMBRIDGE_NBHD = find_file("*ambridge*eighborhood*.geojson", "*ambridge*eighborhood*.json")
NAME_COL = "name"
OUT_DIR = "clean"

YEAR = 2025
NEAR_METERS = 400
SAMPLE_PER_MONTH = 800
SEED = 4200
MIN_MINUTES, MAX_MINUTES = 1, 180
SERVICE_DAY_START_HOUR = 4

LINE_NAMES = {"Green": "Green Line", "Red": "Red Line", "Orange": "Orange Line",
              "Blue": "Blue Line", "Mattapan": "Mattapan Line"}


def haversine_m(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def load_t_entries():
    g = pd.read_csv(GATED_FILE)
    g["service_date"] = pd.to_datetime(g["service_date"], format="mixed").dt.tz_localize(None).dt.normalize()
    g["hour"] = g["time_period"].str.strip("()").str[:2].astype(int)

    # Periods after midnight belong to the previous service date
    g["date"] = g["service_date"] + pd.to_timedelta((g["hour"] < SERVICE_DAY_START_HOUR).astype(int), unit="D")
    g = g[g["date"].dt.year == YEAR]

    # Busiest line at each station
    line = (g.groupby(["stop_id", "route_or_line"])["gated_entries"].sum()
              .reset_index().sort_values("gated_entries", ascending=False)
              .drop_duplicates("stop_id").set_index("stop_id")["route_or_line"])

    hourly = (g.groupby(["stop_id", "date", "hour"], as_index=False)["gated_entries"].sum()
                .rename(columns={"gated_entries": "t_entries_hour"}))
    names = g.drop_duplicates("stop_id").set_index("stop_id")["station_name"]
    return hourly, line, names


def load_t_stations(line, names):
    # All subway and light rail stations, including street level stops with no fare gates
    stops = pd.read_csv(f"{GTFS_DIR}/stops.txt", dtype=str)
    routes = pd.read_csv(f"{GTFS_DIR}/routes.txt", dtype=str)
    trips = pd.read_csv(f"{GTFS_DIR}/trips.txt", dtype=str, usecols=["route_id", "trip_id"])

    rail = set(routes.loc[routes["route_type"].isin(["0", "1"]), "route_id"])
    trip_route = trips[trips["route_id"].isin(rail)].set_index("trip_id")["route_id"]

    # stop_times is large, so read in chunks
    pairs = []
    for chunk in pd.read_csv(f"{GTFS_DIR}/stop_times.txt", dtype=str,
                             usecols=["trip_id", "stop_id"], chunksize=1_000_000):
        chunk = chunk[chunk["trip_id"].isin(trip_route.index)]
        chunk["route_id"] = chunk["trip_id"].map(trip_route)
        pairs.append(chunk[["stop_id", "route_id"]].drop_duplicates())
    pairs = pd.concat(pairs).drop_duplicates()

    # Platforms to parent stations
    parent = stops.set_index("stop_id")["parent_station"]
    pairs["station_id"] = pairs["stop_id"].map(parent).fillna(pairs["stop_id"])
    pairs["line"] = pairs["route_id"].map(
        lambda r: next((v for k, v in LINE_NAMES.items() if r.startswith(k)), r))
    gtfs_lines = pairs.groupby("station_id")["line"].apply(lambda s: sorted(set(s)))

    ids = set(gtfs_lines.index) | set(line.index)
    t = stops[stops["stop_id"].isin(ids)][["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
    t[["stop_lat", "stop_lon"]] = t[["stop_lat", "stop_lon"]].astype(float)

    t["gated"] = t["stop_id"].isin(line.index)
    t["t_station"] = t["stop_id"].map(names).fillna(t["stop_name"])
    t["t_line"] = t["stop_id"].map(line).fillna(t["stop_id"].map(gtfs_lines).str[0])
    all_lines = t["stop_id"].map(gtfs_lines).apply(lambda x: x if isinstance(x, list) else [])
    t["lines"] = [", ".join(sorted(set(a) | {b})) for a, b in zip(all_lines, t["t_line"])]
    return t.drop(columns="stop_name").reset_index(drop=True)


def load_trips():
    cols = ["rideable_type", "started_at", "ended_at",
            "start_station_name", "start_station_id", "end_station_name", "end_station_id",
            "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"]
    files = sorted(glob.glob(f"{YEAR}*-bluebikes-tripdata.csv"))
    if len(files) != 12:
        print(f"Warning: found {len(files)} monthly files, expected 12")
    df = pd.concat((pd.read_csv(f, usecols=cols, dtype={"start_station_id": str, "end_station_id": str})
                    for f in files), ignore_index=True)

    df["started_at"] = pd.to_datetime(df["started_at"], format="mixed")
    df["ended_at"] = pd.to_datetime(df["ended_at"], format="mixed")
    df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60

    before = len(df)
    df = df.dropna(subset=["start_station_id", "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng"])
    df = df[df["duration_min"].between(MIN_MINUTES, MAX_MINUTES) & (df["started_at"].dt.year == YEAR)]
    print(f"Trips: {before:,} raw, {len(df):,} after cleaning")
    return df


def build_stations(trips, t):
    # One row per bike station
    start = trips[["start_station_id", "start_station_name", "start_lat", "start_lng"]]
    end = trips[["end_station_id", "end_station_name", "end_lat", "end_lng"]]
    start.columns = end.columns = ["station_id", "station_name", "lat", "lng"]
    st = pd.concat([start, end]).groupby("station_id").agg(
        station_name=("station_name", "first"), lat=("lat", "median"), lng=("lng", "median")).reset_index()

    # Neighborhood
    bos = gpd.read_file(BOSTON_NBHD).to_crs(4326)[[NAME_COL, "geometry"]].assign(city="Boston")
    cam = gpd.read_file(CAMBRIDGE_NBHD).to_crs(4326)[[NAME_COL, "geometry"]].assign(city="Cambridge")
    polys = pd.concat([bos, cam], ignore_index=True).rename(columns={NAME_COL: "neighborhood"})
    pts = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["lng"], st["lat"]), crs=4326)
    st = pd.DataFrame(gpd.sjoin(pts, polys, how="left", predicate="within")
                      .drop_duplicates("station_id").drop(columns=["geometry", "index_right"]))

    # Nearest T stop
    d = haversine_m(st["lat"].values[:, None], st["lng"].values[:, None],
                    t["stop_lat"].values[None, :], t["stop_lon"].values[None, :])
    idx = d.argmin(axis=1)
    st["nearest_t_stop_id"] = t["stop_id"].values[idx]
    st["nearest_t_station"] = t["t_station"].values[idx]
    st["nearest_t_line"] = t["t_line"].values[idx]
    st["nearest_t_gated"] = t["gated"].values[idx]
    st["dist_to_t_m"] = d.min(axis=1).round(0)
    st["near_transit"] = np.where(st["dist_to_t_m"] <= NEAR_METERS, "Near", "Away")
    return st.set_index("station_id")


def add_time_features(df):
    df["date"] = df["started_at"].dt.normalize()
    df["hour"] = df["started_at"].dt.hour
    df["month"] = df["started_at"].dt.month_name()
    df["day_of_week"] = df["started_at"].dt.day_name()
    df["is_weekend"] = df["started_at"].dt.dayofweek >= 5
    df["time_of_day"] = pd.cut(df["hour"], bins=[-1, 6, 9, 15, 18, 21, 23], ordered=False,
                               labels=["Overnight", "AM peak", "Midday", "PM peak", "Evening", "Overnight"])
    return df


def build_system_hourly(trips, t_hourly):
    # Hourly share of each system's total, for the hourly chart
    bike = (trips.assign(day_type=np.where(trips["is_weekend"], "Weekend", "Weekday"))
                 .groupby(["day_type", "hour"]).size().rename("count").reset_index().assign(system="Bluebikes"))
    t = (t_hourly.assign(day_type=np.where(t_hourly["date"].dt.dayofweek >= 5, "Weekend", "Weekday"))
                 .groupby(["day_type", "hour"])["t_entries_hour"].sum().rename("count").reset_index().assign(system="T"))
    out = pd.concat([bike, t], ignore_index=True)
    out["share"] = out["count"] / out.groupby(["system", "day_type"])["count"].transform("sum")
    return out[["system", "day_type", "hour", "count", "share"]]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    t_hourly, t_line, t_names = load_t_entries()
    t_stations = load_t_stations(t_line, t_names)
    t_stations.to_csv(f"{OUT_DIR}/t_stations.csv", index=False)
    print(f"T stations: {len(t_stations)} total, {t_stations['gated'].sum()} gated")

    trips = load_trips()
    st = build_stations(trips, t_stations)

    # Keep trips starting in Boston or Cambridge
    trips = add_time_features(trips[trips["start_station_id"].map(st["neighborhood"]).notna()].copy())
    print(f"Trips starting in Boston/Cambridge: {len(trips):,}")
    build_system_hourly(trips, t_hourly).to_csv(f"{OUT_DIR}/system_hourly.csv", index=False)

    # Same number of trips from each month
    if (trips["month"].value_counts() < SAMPLE_PER_MONTH).any():
        raise ValueError(f"Some months have fewer than {SAMPLE_PER_MONTH} trips")
    sample = trips.groupby("month").sample(n=SAMPLE_PER_MONTH, random_state=SEED).copy()

    # Station info
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

    # T entries at the nearest station during the trip's start hour
    sample = sample.merge(t_hourly, how="left", left_on=["nearest_t_stop_id", "date", "hour"],
                          right_on=["stop_id", "date", "hour"]).drop(columns="stop_id")
    print(f"Trips with no T entry count: {sample['t_entries_hour'].isna().sum()}")

    sample = sample.rename(columns={"member_casual": "rider_type", "rideable_type": "bike_type",
                                    "start_station_name": "start_station", "end_station_name": "end_station"})
    keep = ["rider_type", "bike_type", "start_station", "end_station",
            "start_neighborhood", "end_neighborhood", "nearest_t_station", "nearest_t_line",
            "near_transit", "month", "day_of_week", "is_weekend", "time_of_day", "hour",
            "duration_min", "distance_km", "dist_to_t_m", "t_entries_hour",
            "start_lat", "start_lng", "end_lat", "end_lng"]
    sample[keep].to_csv(f"{OUT_DIR}/trips_2025_sample.csv", index=False)
    st.reset_index().to_csv(f"{OUT_DIR}/stations.csv", index=False)

    print(f"Done: {len(sample):,} rows, {len(keep)} columns")
    print(sample["near_transit"].value_counts().to_string())


if __name__ == "__main__":
    main()