"""
Shared colors and fonts for every chart in the project.

Import this in a chart script before building the chart:
  from chart_style import BIKE_COLOR, T_COLOR

Importing it also turns on the project theme (fonts, title, axes, legend),
so every chart looks the same without setting those things one by one.
"""

import altair as alt

BIKE_COLOR = "#1F6FB2"   # Bluebikes
T_COLOR = "#E07B00"      # MBTA subway / light rail

FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
TEXT_COLOR = "#333333"
MUTED_TEXT = "#666666"
GRID_COLOR = "#E6E6E6"


@alt.theme.register("ds4200", enable=True)
def ds4200_theme():
    return {
        "config": {
            "font": FONT,
            "view": {"continuousWidth": 300, "continuousHeight": 300, "strokeWidth": 0},
            "title": {"fontSize": 16, "anchor": "start", "color": TEXT_COLOR,
                      "subtitleFontSize": 12, "subtitleColor": MUTED_TEXT},
            "axis": {"labelFontSize": 11, "titleFontSize": 12, "labelColor": MUTED_TEXT,
                     "titleColor": TEXT_COLOR, "titleFontWeight": "normal",
                     "gridColor": GRID_COLOR, "domainColor": "#BBBBBB", "tickColor": "#BBBBBB"},
            "legend": {"labelFontSize": 12, "titleFontSize": 12, "labelColor": TEXT_COLOR,
                       "titleColor": TEXT_COLOR, "titleFontWeight": "normal"},
            "header": {"labelFontSize": 13, "labelColor": TEXT_COLOR, "titleFontSize": 13},
        }
    }
