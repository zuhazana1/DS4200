"""
Hourly usage profile: Bluebikes trips vs. T entries by hour, weekday vs. weekend.
Each line is a share of that system's own total, so both systems fit on one
axis even though the T carries far more people. Hover for the values.

Reads (from the Project folder):
  clean/system_hourly.csv     made by build_dataset.py
  chart_style.py              shared colors and fonts

Run:  python hourly_chart.py
Output: hourly_usage.html (open it in a browser)
"""

import altair as alt
import pandas as pd

from chart_style import BIKE_COLOR, T_COLOR

hourly = pd.read_csv("clean/system_hourly.csv")
hourly["hour_label"] = [f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}" for h in hourly["hour"]]

# same 12-hour labels on the axis
HOUR_AXIS_LABEL = ("datum.value == 0 ? '12 AM' : datum.value < 12 ? datum.value + ' AM' : "
                   "datum.value == 12 ? '12 PM' : (datum.value - 12) + ' PM'")

x = alt.X("hour:Q", title="Hour of day", scale=alt.Scale(domain=[0, 23], nice=False),
          axis=alt.Axis(values=list(range(0, 24, 3)), labelExpr=HOUR_AXIS_LABEL, grid=False))
y = alt.Y("share:Q", title="Share of the day's activity",
          axis=alt.Axis(format=".0%", tickCount=6))
color = alt.Color("system:N", title=None,
                  scale=alt.Scale(domain=["Bluebikes", "T"], range=[BIKE_COLOR, T_COLOR]),
                  legend=alt.Legend(orient="top", symbolType="stroke", symbolStrokeWidth=2))

# hover anywhere in a panel to pick the nearest hour
hover = alt.selection_point(fields=["hour"], nearest=True, on="pointerover",
                            clear="pointerout", empty=False)

base = alt.Chart().encode(x=x)

lines = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round").encode(y=y, color=color)

points = base.mark_circle(size=70, opacity=1, stroke="white", strokeWidth=2).encode(
    y=y, color=color
).transform_filter(hover)

# one invisible rule per hour carries the hover and a tooltip with both systems
rule = base.transform_pivot(
    "system", value="share", groupby=["day_type", "hour", "hour_label"]
).mark_rule(color="#999999", strokeWidth=1).encode(
    opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
    tooltip=[alt.Tooltip("hour_label:N", title="Hour starting"),
             alt.Tooltip("Bluebikes:Q", title="Bluebikes (share of trips)", format=".1%"),
             alt.Tooltip("T:Q", title="T (share of entries)", format=".1%")],
).add_params(hover)

# name each line once, at its weekday peak, where the lines are far apart
labels = base.transform_filter(
    "datum.day_type == 'Weekday' && ((datum.system == 'Bluebikes' && datum.hour == 17) || "
    "(datum.system == 'T' && datum.hour == 8))"
).mark_text(dy=-12, fontSize=12, fontWeight="bold", color="#333333").encode(y=y, text="system:N")

chart = alt.layer(rule, lines, points, labels, data=hourly).properties(
    width=380, height=300
).facet(
    column=alt.Column("day_type:N", title=None, sort=["Weekday", "Weekend"])
).properties(
    title=alt.Title("Bluebikes and the T follow nearly the same daily rhythm (2025)",
                    subtitle=["Share of each system's own activity by hour: Bluebikes trips started "
                              "and T fare-gate entries.",
                              "The T starts earlier in the morning; Bluebikes stay busier in the "
                              "evening and late on weekend nights. Hover for values."]),
)

chart.save("hourly_usage.html")
print(f"Saved hourly_usage.html ({len(hourly)} rows)")
