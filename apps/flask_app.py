from pathlib import Path
from flask import Flask, jsonify, send_from_directory
import os
from dotenv import load_dotenv
import pandas as pd

# Load .env file (for local development)
load_dotenv(Path(__file__).resolve().parent.parent / '.env')
import folium
import json
import requests as _requests
from folium.plugins import MiniMap, MousePosition
from folium import FeatureGroup
from branca.element import Element

# ------------------- Paths -------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

app = Flask(__name__)


@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(PROJECT_ROOT / 'assets', filename)


# ------------------- Load & prepare data -------------------
stations_scotland = pd.read_csv(DATA_DIR / "stations.csv")
stations_scotland = stations_scotland[stations_scotland['constituentCountry'] == 'scotland']

# --- Pre-processed railway lines (filtered, named, dissolved) ---
railway_lines_geojson = open(DATA_DIR / "railways_processed.geojson").read()

# Scotland boundary & bounding box
scotland_geojson = open(DATA_DIR / "scotland.geojson").read()
minx, miny, maxx, maxy = -8.65000723, 54.63323825, -0.72460948, 60.86076145

# --- ORR usage data (pre-filtered Scotland from Table 1410) ---
_usage = pd.read_csv(DATA_DIR / "usage_scotland.csv")

# --- Top routes (pre-computed from ODM) ---
top_routes = pd.read_csv(DATA_DIR / "top_routes.csv")

# --- Merge: stations + ORR usage + top-route ---
joined_data = stations_scotland.merge(
    _usage[['tlc', 'entries_exits_all']].rename(columns={'entries_exits_all': 'total_journeys'}), left_on='crsCode', right_on='tlc', how='left')
joined_data = joined_data.merge(
    top_routes[['origin_tlc', 'destination_station_name', 'journeys']],
    left_on='crsCode', right_on='origin_tlc', how='left')

joined_data = joined_data[['stationName', 'crsCode', 'lat', 'long',
                           'destination_station_name', 'journeys', 'total_journeys']]
joined_data['_sort'] = joined_data['total_journeys'].fillna(-1)
joined_data = joined_data.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)

# --- Station performance ---
perf_data = pd.read_csv(DATA_DIR / "station_performance_scotland.csv")
perf_scotrail = perf_data[perf_data['operator_name'] == 'ScotRail'].copy()

PERF_NAME_MAP = {
    'Armadale': 'Armadale (West Lothian)', 'Bishopton': 'Bishopton (Renfrewshire)',
    'Burnside': 'Burnside (South Lanarkshire)', 'Exhibition Centre': 'Exhibition Centre (Glasgow)',
    'Falls of Cruachan': 'Falls of Cruachan (Request Stop)', 'High Street': 'High Street (Glasgow)',
    'Howwood': 'Howwood (Renfrewshire)', 'Johnstone': 'Johnstone (Renfrewshire)',
    'Kyle of Lochalsh': 'Kyle Of Lochalsh',
    'Muir of Ord': 'Muir Of Ord', 'Newton': 'Newton (South Lanarkshire)',
    'Possilpark & Parkhouse': 'Possilpark and Parkhouse',
    'Priesthill & Darnley': 'Priesthill and Darnley',
    'Edinburgh Waverley': 'Edinburgh', 'Balloch Central': 'Balloch',
    'Arrochar & Tarbet': 'Arrochar and Tarbet', 'Dunkeld & Birnam': 'Dunkeld and Birnam',
    'Bridge of Allan': 'Bridge Of Allan', 'Bridge of Orchy': 'Bridge Of Orchy',
}

all_periods_sorted = sorted(perf_scotrail['time_period'].unique())


# The last 6 periods — if the most recent non-null value falls within
# these, use it; otherwise mark as outdated.
_RECENT_PERIODS = set(all_periods_sorted[-6:])


def get_perf_for_station(map_name):
    perf_name = PERF_NAME_MAP.get(map_name, map_name)
    station_data = perf_scotrail[perf_scotrail['station_name'] == perf_name].copy()
    operator = 'ScotRail'
    if station_data.empty:
        station_data = perf_data[perf_data['station_name'] == perf_name].copy()
        if not station_data.empty:
            operator = station_data['operator_name'].value_counts().index[0]
            station_data = station_data[station_data['operator_name'] == operator]
    if station_data.empty:
        return None
    station_data = station_data.sort_values('time_period')

    # Find last non-null punctuality & cancellations
    valid_p = station_data[station_data['time_to_3_pct'].notna()]
    valid_c = station_data[station_data['cancellations_pct'].notna()]

    lp, lp_status = None, 'no_data'
    if not valid_p.empty:
        last_p_row = valid_p.iloc[-1]
        if last_p_row['time_period'] in _RECENT_PERIODS:
            lp = round(last_p_row['time_to_3_pct'], 1)
            lp_status = 'current'
        else:
            lp_status = 'outdated'

    lc, lc_status = None, 'no_data'
    if not valid_c.empty:
        last_c_row = valid_c.iloc[-1]
        if last_c_row['time_period'] in _RECENT_PERIODS:
            lc = round(last_c_row['cancellations_pct'], 1)
            lc_status = 'current'
        else:
            lc_status = 'outdated'

    return {
        'operator': operator,
        'latest_punctuality': lp,
        'latest_cancellations': lc,
        'punct_status': lp_status,
        'cancel_status': lc_status,
        'trend_punctuality': [round(v, 1) if pd.notna(v) else None for v in station_data['time_to_3_pct'].values],
        'trend_cancellations': [round(v, 1) if pd.notna(v) else None for v in station_data['cancellations_pct'].values],
    }


station_perf_lookup = {n: r for n in joined_data['stationName'].unique() if (r := get_perf_for_station(n))}

period_labels = []
for p in all_periods_sorted:
    parts = p.split()
    period_labels.append(f"P{parts[-1]} {parts[1][2:]}/{parts[4][2:]}")
PERIOD_LABELS_PIPE = '|'.join(period_labels)


# ------------------- Regional grouping (explicit mapping) -------------------
# Every station mapped to its correct Scottish region based on council area / geography.
_STATION_REGION = {
    # --- Glasgow & Clyde Valley ---
    # Glasgow City, Renfrewshire, East Renfrewshire, North Lanarkshire,
    # South Lanarkshire (Hamilton/East Kilbride), East Dunbartonshire,
    # West Dunbartonshire, Inverclyde
    'Airbles': 'Glasgow & Clyde Valley',
    'Airdrie': 'Glasgow & Clyde Valley',
    'Alexandra Parade': 'Glasgow & Clyde Valley',
    'Anderston': 'Glasgow & Clyde Valley',
    'Anniesland': 'Glasgow & Clyde Valley',
    'Argyle Street': 'Glasgow & Clyde Valley',
    'Ashfield': 'Glasgow & Clyde Valley',
    'Baillieston': 'Glasgow & Clyde Valley',
    'Bargeddie': 'Glasgow & Clyde Valley',
    'Barnhill': 'Glasgow & Clyde Valley',
    'Barrhead': 'Glasgow & Clyde Valley',
    'Bearsden': 'Glasgow & Clyde Valley',
    'Bellgrove': 'Glasgow & Clyde Valley',
    'Bellshill': 'Glasgow & Clyde Valley',
    'Bishopbriggs': 'Glasgow & Clyde Valley',
    'Bishopton': 'Glasgow & Clyde Valley',
    'Blairhill': 'Glasgow & Clyde Valley',
    'Blantyre': 'Glasgow & Clyde Valley',
    'Bogston': 'Glasgow & Clyde Valley',
    'Bowling': 'Glasgow & Clyde Valley',
    'Branchton': 'Glasgow & Clyde Valley',
    'Bridgeton': 'Glasgow & Clyde Valley',
    'Burnside': 'Glasgow & Clyde Valley',
    'Busby': 'Glasgow & Clyde Valley',
    'Caldercruix': 'Glasgow & Clyde Valley',
    'Cambuslang': 'Glasgow & Clyde Valley',
    'Cardonald': 'Glasgow & Clyde Valley',
    'Carfin': 'Glasgow & Clyde Valley',
    'Carmyle': 'Glasgow & Clyde Valley',
    'Carntyne': 'Glasgow & Clyde Valley',
    'Cartsdyke': 'Glasgow & Clyde Valley',
    'Cathcart': 'Glasgow & Clyde Valley',
    'Charing Cross (Glasgow)': 'Glasgow & Clyde Valley',
    'Chatelherault': 'Glasgow & Clyde Valley',
    'Clarkston': 'Glasgow & Clyde Valley',
    'Cleland': 'Glasgow & Clyde Valley',
    'Clydebank': 'Glasgow & Clyde Valley',
    'Coatbridge Central': 'Glasgow & Clyde Valley',
    'Coatbridge Sunnyside': 'Glasgow & Clyde Valley',
    'Coatdyke': 'Glasgow & Clyde Valley',
    'Corkerhill': 'Glasgow & Clyde Valley',
    'Croftfoot': 'Glasgow & Clyde Valley',
    'Crookston': 'Glasgow & Clyde Valley',
    'Crosshill': 'Glasgow & Clyde Valley',
    'Crossmyloof': 'Glasgow & Clyde Valley',
    'Croy': 'Glasgow & Clyde Valley',
    'Cumbernauld': 'Glasgow & Clyde Valley',
    'Dalmarnock': 'Glasgow & Clyde Valley',
    'Dalmuir': 'Glasgow & Clyde Valley',
    'Dalreoch': 'Glasgow & Clyde Valley',
    'Drumchapel': 'Glasgow & Clyde Valley',
    'Drumfrochar': 'Glasgow & Clyde Valley',
    'Drumgelloch': 'Glasgow & Clyde Valley',
    'Drumry': 'Glasgow & Clyde Valley',
    'Duke Street': 'Glasgow & Clyde Valley',
    'Dumbarton Central': 'Glasgow & Clyde Valley',
    'Dumbarton East': 'Glasgow & Clyde Valley',
    'Dumbreck': 'Glasgow & Clyde Valley',
    'East Kilbride': 'Glasgow & Clyde Valley',
    'Easterhouse': 'Glasgow & Clyde Valley',
    'Exhibition Centre': 'Glasgow & Clyde Valley',
    'Fort Matilda': 'Glasgow & Clyde Valley',
    'Garrowhill': 'Glasgow & Clyde Valley',
    'Garscadden': 'Glasgow & Clyde Valley',
    'Gartcosh': 'Glasgow & Clyde Valley',
    'Giffnock': 'Glasgow & Clyde Valley',
    'Gilshochill': 'Glasgow & Clyde Valley',
    'Glasgow Central': 'Glasgow & Clyde Valley',
    'Glasgow Queen Street': 'Glasgow & Clyde Valley',
    'Gourock': 'Glasgow & Clyde Valley',
    'Greenfaulds': 'Glasgow & Clyde Valley',
    'Greenock Central': 'Glasgow & Clyde Valley',
    'Greenock West': 'Glasgow & Clyde Valley',
    'Hairmyres': 'Glasgow & Clyde Valley',
    'Hamilton Central': 'Glasgow & Clyde Valley',
    'Hamilton West': 'Glasgow & Clyde Valley',
    'Hartwood': 'Glasgow & Clyde Valley',
    'Hawkhead': 'Glasgow & Clyde Valley',
    'High Street': 'Glasgow & Clyde Valley',
    'Hillfoot': 'Glasgow & Clyde Valley',
    'Hillington East': 'Glasgow & Clyde Valley',
    'Hillington West': 'Glasgow & Clyde Valley',
    'Holytown': 'Glasgow & Clyde Valley',
    'Howwood': 'Glasgow & Clyde Valley',
    'Hyndland': 'Glasgow & Clyde Valley',
    'Inverkip': 'Glasgow & Clyde Valley',
    'Johnstone': 'Glasgow & Clyde Valley',
    'Jordanhill': 'Glasgow & Clyde Valley',
    'Kelvindale': 'Glasgow & Clyde Valley',
    'Kennishead': 'Glasgow & Clyde Valley',
    'Kilpatrick': 'Glasgow & Clyde Valley',
    'Kings Park': 'Glasgow & Clyde Valley',
    'Kirkhill': 'Glasgow & Clyde Valley',
    'Kirkwood': 'Glasgow & Clyde Valley',
    'Langbank': 'Glasgow & Clyde Valley',
    'Langside': 'Glasgow & Clyde Valley',
    'Lenzie': 'Glasgow & Clyde Valley',
    'Lochwinnoch': 'Glasgow & Clyde Valley',
    'Maryhill': 'Glasgow & Clyde Valley',
    'Maxwell Park': 'Glasgow & Clyde Valley',
    'Merryton': 'Glasgow & Clyde Valley',
    'Milliken Park': 'Glasgow & Clyde Valley',
    'Milngavie': 'Glasgow & Clyde Valley',
    'Mosspark': 'Glasgow & Clyde Valley',
    'Motherwell': 'Glasgow & Clyde Valley',
    'Mount Florida': 'Glasgow & Clyde Valley',
    'Mount Vernon': 'Glasgow & Clyde Valley',
    'Muirend': 'Glasgow & Clyde Valley',
    'Neilston': 'Glasgow & Clyde Valley',
    'Newton': 'Glasgow & Clyde Valley',
    'Nitshill': 'Glasgow & Clyde Valley',
    'Paisley Canal': 'Glasgow & Clyde Valley',
    'Paisley Gilmour Street': 'Glasgow & Clyde Valley',
    'Paisley St James': 'Glasgow & Clyde Valley',
    'Partick': 'Glasgow & Clyde Valley',
    'Patterton': 'Glasgow & Clyde Valley',
    'Pollokshaws East': 'Glasgow & Clyde Valley',
    'Pollokshaws West': 'Glasgow & Clyde Valley',
    'Pollokshields East': 'Glasgow & Clyde Valley',
    'Pollokshields West': 'Glasgow & Clyde Valley',
    'Port Glasgow': 'Glasgow & Clyde Valley',
    'Possilpark & Parkhouse': 'Glasgow & Clyde Valley',
    'Priesthill & Darnley': 'Glasgow & Clyde Valley',
    'Queens Park (Glasgow)': 'Glasgow & Clyde Valley',
    'Robroyston': 'Glasgow & Clyde Valley',
    'Rutherglen': 'Glasgow & Clyde Valley',
    'Scotstounhill': 'Glasgow & Clyde Valley',
    'Shawlands': 'Glasgow & Clyde Valley',
    'Shettleston': 'Glasgow & Clyde Valley',
    'Shieldmuir': 'Glasgow & Clyde Valley',
    'Singer': 'Glasgow & Clyde Valley',
    'Springburn': 'Glasgow & Clyde Valley',
    'Stepps': 'Glasgow & Clyde Valley',
    'Summerston': 'Glasgow & Clyde Valley',
    'Thornliebank': 'Glasgow & Clyde Valley',
    'Thorntonhall': 'Glasgow & Clyde Valley',
    'Uddingston': 'Glasgow & Clyde Valley',
    'Wemyss Bay': 'Glasgow & Clyde Valley',
    'Westerton': 'Glasgow & Clyde Valley',
    'Whifflet': 'Glasgow & Clyde Valley',
    'Whinhill': 'Glasgow & Clyde Valley',
    'Whitecraigs': 'Glasgow & Clyde Valley',
    'Williamwood': 'Glasgow & Clyde Valley',
    'Wishaw': 'Glasgow & Clyde Valley',
    'Woodhall': 'Glasgow & Clyde Valley',
    'Yoker': 'Glasgow & Clyde Valley',
    # --- Edinburgh & Lothians ---
    # City of Edinburgh, East Lothian, Midlothian, West Lothian
    'Brunstane': 'Edinburgh & Lothians',
    'Curriehill': 'Edinburgh & Lothians',
    'Dalmeny': 'Edinburgh & Lothians',
    'Drem': 'Edinburgh & Lothians',
    'Dunbar': 'Edinburgh & Lothians',
    'East Linton': 'Edinburgh & Lothians',
    'Edinburgh Gateway': 'Edinburgh & Lothians',
    'Edinburgh Park': 'Edinburgh & Lothians',
    'Edinburgh Waverley': 'Edinburgh & Lothians',
    'Eskbank': 'Edinburgh & Lothians',
    'Gorebridge': 'Edinburgh & Lothians',
    'Haymarket': 'Edinburgh & Lothians',
    'Kingsknowe': 'Edinburgh & Lothians',
    'Kirknewton': 'Edinburgh & Lothians',
    'Livingston North': 'Edinburgh & Lothians',
    'Livingston South': 'Edinburgh & Lothians',
    'Longniddry': 'Edinburgh & Lothians',
    'Musselburgh': 'Edinburgh & Lothians',
    'Newcraighall': 'Edinburgh & Lothians',
    'Newtongrange': 'Edinburgh & Lothians',
    'North Berwick': 'Edinburgh & Lothians',
    'Prestonpans': 'Edinburgh & Lothians',
    'Shawfair': 'Edinburgh & Lothians',
    'Slateford': 'Edinburgh & Lothians',
    'South Gyle': 'Edinburgh & Lothians',
    'Uphall': 'Edinburgh & Lothians',
    'Wallyford': 'Edinburgh & Lothians',
    'Wester Hailes': 'Edinburgh & Lothians',
    'Addiewell': 'Edinburgh & Lothians',
    'Armadale': 'Edinburgh & Lothians',
    'Bathgate': 'Edinburgh & Lothians',
    'Blackridge': 'Edinburgh & Lothians',
    'Breich': 'Edinburgh & Lothians',
    'Fauldhouse': 'Edinburgh & Lothians',
    'Linlithgow': 'Edinburgh & Lothians',
    'Polmont': 'Edinburgh & Lothians',
    'West Calder': 'Edinburgh & Lothians',
    'Reston': 'Edinburgh & Lothians',
    # --- Ayrshire & Arran ---
    # East Ayrshire, North Ayrshire, South Ayrshire
    'Ardrossan Harbour': 'Ayrshire & Arran',
    'Ardrossan South Beach': 'Ayrshire & Arran',
    'Ardrossan Town': 'Ayrshire & Arran',
    'Ayr': 'Ayrshire & Arran',
    'Barassie': 'Ayrshire & Arran',
    'Dalry': 'Ayrshire & Arran',
    'Dunlop': 'Ayrshire & Arran',
    'Fairlie': 'Ayrshire & Arran',
    'Girvan': 'Ayrshire & Arran',
    'Glengarnock': 'Ayrshire & Arran',
    'Irvine': 'Ayrshire & Arran',
    'Kilmarnock': 'Ayrshire & Arran',
    'Kilmaurs': 'Ayrshire & Arran',
    'Kilwinning': 'Ayrshire & Arran',
    'Largs': 'Ayrshire & Arran',
    'Maybole': 'Ayrshire & Arran',
    'Newton-On-Ayr': 'Ayrshire & Arran',
    'Prestwick International Airport': 'Ayrshire & Arran',
    'Prestwick Town': 'Ayrshire & Arran',
    'Saltcoats': 'Ayrshire & Arran',
    'Stevenston': 'Ayrshire & Arran',
    'Stewarton': 'Ayrshire & Arran',
    'Troon': 'Ayrshire & Arran',
    'West Kilbride': 'Ayrshire & Arran',
    'Auchinleck': 'Ayrshire & Arran',
    'New Cumnock': 'Ayrshire & Arran',
    # --- Lanarkshire & South ---
    # South Lanarkshire (Carluke/Lanark/Carstairs), Shotts line
    'Carluke': 'Lanarkshire & South',
    'Carstairs': 'Lanarkshire & South',
    'Lanark': 'Lanarkshire & South',
    'Larkhall': 'Lanarkshire & South',
    'Shotts': 'Lanarkshire & South',
    # --- Stirling & Forth Valley ---
    # Stirling, Falkirk, Clackmannanshire, Helensburgh area
    'Alloa': 'Stirling & Forth Valley',
    'Balloch Central': 'Stirling & Forth Valley',
    'Bridge of Allan': 'Stirling & Forth Valley',
    'Camelon': 'Stirling & Forth Valley',
    'Cardross': 'Stirling & Forth Valley',
    'Craigendoran': 'Stirling & Forth Valley',
    'Dunblane': 'Stirling & Forth Valley',
    'Falkirk Grahamston': 'Stirling & Forth Valley',
    'Falkirk High': 'Stirling & Forth Valley',
    'Garelochhead': 'Stirling & Forth Valley',
    'Helensburgh Central': 'Stirling & Forth Valley',
    'Helensburgh Upper': 'Stirling & Forth Valley',
    'Larbert': 'Stirling & Forth Valley',
    'Stirling': 'Stirling & Forth Valley',
    'Alexandria': 'Stirling & Forth Valley',
    'Renton': 'Stirling & Forth Valley',
    # --- Fife & Dundee ---
    # Fife council area
    'Aberdour': 'Fife & Dundee',
    'Burntisland': 'Fife & Dundee',
    'Cameron Bridge': 'Fife & Dundee',
    'Cardenden': 'Fife & Dundee',
    'Cowdenbeath': 'Fife & Dundee',
    'Cupar': 'Fife & Dundee',
    'Dalgety Bay': 'Fife & Dundee',
    'Dunfermline City': 'Fife & Dundee',
    'Dunfermline Queen Margaret': 'Fife & Dundee',
    'Glenrothes with Thornton': 'Fife & Dundee',
    'Inverkeithing': 'Fife & Dundee',
    'Kinghorn': 'Fife & Dundee',
    'Kirkcaldy': 'Fife & Dundee',
    'Ladybank': 'Fife & Dundee',
    'Leuchars': 'Fife & Dundee',
    'Leven': 'Fife & Dundee',
    'Lochgelly': 'Fife & Dundee',
    'Markinch': 'Fife & Dundee',
    'North Queensferry': 'Fife & Dundee',
    'Rosyth': 'Fife & Dundee',
    'Springfield': 'Fife & Dundee',
    # --- Tayside & Perthshire ---
    # Perth & Kinross, Angus, Dundee City
    'Arbroath': 'Tayside & Perthshire',
    'Balmossie': 'Tayside & Perthshire',
    'Barry Links': 'Tayside & Perthshire',
    'Blair Atholl': 'Tayside & Perthshire',
    'Broughty Ferry': 'Tayside & Perthshire',
    'Carnoustie': 'Tayside & Perthshire',
    'Dalwhinnie': 'Tayside & Perthshire',
    'Dundee': 'Tayside & Perthshire',
    'Dunkeld & Birnam': 'Tayside & Perthshire',
    'Gleneagles': 'Tayside & Perthshire',
    'Golf Street': 'Tayside & Perthshire',
    'Invergowrie': 'Tayside & Perthshire',
    'Monifieth': 'Tayside & Perthshire',
    'Montrose': 'Tayside & Perthshire',
    'Perth': 'Tayside & Perthshire',
    'Pitlochry': 'Tayside & Perthshire',
    # --- North East & Aberdeen ---
    # Aberdeenshire
    'Aberdeen': 'North East & Aberdeen',
    'Dyce': 'North East & Aberdeen',
    'Huntly': 'North East & Aberdeen',
    'Insch': 'North East & Aberdeen',
    'Inverurie': 'North East & Aberdeen',
    'Kintore': 'North East & Aberdeen',
    'Laurencekirk': 'North East & Aberdeen',
    'Portlethen': 'North East & Aberdeen',
    'Stonehaven': 'North East & Aberdeen',
    # --- Highlands & Moray ---
    # Highland (Inverness/Ross-shire/Sutherland south), Moray
    'Achanalt': 'Highlands & Moray',
    'Achnasheen': 'Highlands & Moray',
    'Achnashellach': 'Highlands & Moray',
    'Alness': 'Highlands & Moray',
    'Ardgay': 'Highlands & Moray',
    'Attadale': 'Highlands & Moray',
    'Aviemore': 'Highlands & Moray',
    'Beauly': 'Highlands & Moray',
    'Carrbridge': 'Highlands & Moray',
    'Conon Bridge': 'Highlands & Moray',
    'Culrain': 'Highlands & Moray',
    'Dingwall': 'Highlands & Moray',
    'Duirinish': 'Highlands & Moray',
    'Duncraig': 'Highlands & Moray',
    'Dunrobin Castle': 'Highlands & Moray',
    'Elgin': 'Highlands & Moray',
    'Fearn': 'Highlands & Moray',
    'Forres': 'Highlands & Moray',
    'Garve': 'Highlands & Moray',
    'Golspie': 'Highlands & Moray',
    'Invergordon': 'Highlands & Moray',
    'Inverness': 'Highlands & Moray',
    'Inverness Airport': 'Highlands & Moray',
    'Invershin': 'Highlands & Moray',
    'Keith': 'Highlands & Moray',
    'Kingussie': 'Highlands & Moray',
    'Kyle of Lochalsh': 'Highlands & Moray',
    'Lochluichart': 'Highlands & Moray',
    'Muir of Ord': 'Highlands & Moray',
    'Nairn': 'Highlands & Moray',
    'Newtonmore': 'Highlands & Moray',
    'Plockton': 'Highlands & Moray',
    'Rogart': 'Highlands & Moray',
    'Strathcarron': 'Highlands & Moray',
    'Stromeferry': 'Highlands & Moray',
    'Tain': 'Highlands & Moray',
    # --- West Highlands ---
    # Highland (Fort William/Lochaber), Argyll & Bute (Oban line)
    'Ardlui': 'West Highlands',
    'Arisaig': 'West Highlands',
    'Arrochar & Tarbet': 'West Highlands',
    'Banavie': 'West Highlands',
    'Beasdale': 'West Highlands',
    'Bridge of Orchy': 'West Highlands',
    'Connel Ferry': 'West Highlands',
    'Corpach': 'West Highlands',
    'Corrour': 'West Highlands',
    'Crianlarich': 'West Highlands',
    'Dalmally': 'West Highlands',
    'Falls of Cruachan': 'West Highlands',
    'Fort William': 'West Highlands',
    'Glenfinnan': 'West Highlands',
    'Loch Awe': 'West Highlands',
    'Loch Eil Outward Bound': 'West Highlands',
    'Lochailort': 'West Highlands',
    'Locheilside': 'West Highlands',
    'Mallaig': 'West Highlands',
    'Morar': 'West Highlands',
    'Oban': 'West Highlands',
    'Rannoch': 'West Highlands',
    'Roy Bridge': 'West Highlands',
    'Spean Bridge': 'West Highlands',
    'Taynuilt': 'West Highlands',
    'Tulloch': 'West Highlands',
    'Tyndrum Lower': 'West Highlands',
    'Upper Tyndrum': 'West Highlands',
    # --- Far North ---
    # Highland (Caithness, Sutherland north)
    'Brora': 'Far North',
    'Forsinard': 'Far North',
    'Georgemas Junction': 'Far North',
    'Helmsdale': 'Far North',
    'Kildonan': 'Far North',
    'Kinbrace': 'Far North',
    'Lairg': 'Far North',
    'Scotscalder': 'Far North',
    'Thurso': 'Far North',
    'Wick': 'Far North',
    # --- Scottish Borders ---
    # Scottish Borders council area, plus Borders Railway in Midlothian
    'Galashiels': 'Scottish Borders',
    'Stow': 'Scottish Borders',
    'Tweedbank': 'Scottish Borders',
    # --- Dumfries & Galloway ---
    # Dumfries & Galloway council area
    'Annan': 'Dumfries & Galloway',
    'Barrhill': 'Dumfries & Galloway',
    'Dumfries': 'Dumfries & Galloway',
    'Gretna Green': 'Dumfries & Galloway',
    'Kirkconnel': 'Dumfries & Galloway',
    'Lockerbie': 'Dumfries & Galloway',
    'Sanquhar': 'Dumfries & Galloway',
    'Stranraer': 'Dumfries & Galloway',
}

def get_region(name):
    """Return the region for a station by name. Falls back to 'Other' if unmapped."""
    return _STATION_REGION.get(name, 'Other')

joined_data['region'] = joined_data['stationName'].map(get_region)

REGION_ORDER = [
    "Glasgow & Clyde Valley", "Edinburgh & Lothians", "Ayrshire & Arran",
    "Lanarkshire & South", "Stirling & Forth Valley", "Fife & Dundee",
    "Tayside & Perthshire", "North East & Aberdeen", "West Highlands",
    "Highlands & Moray", "Far North", "Scottish Borders", "Dumfries & Galloway",
]


# ------------------- Colour helpers -------------------
def get_station_colour(tj):
    if pd.isna(tj):        return '#888888'
    if tj >= 2_000_000:    return '#1a9641'
    if tj >= 500_000:      return '#2670D7'
    if tj >= 100_000:      return '#f4a742'
    return '#d7191c'


def get_punctuality_colour(pct):
    if pct is None: return '#888888'
    if pct >= 92:   return '#1a9641'
    if pct >= 85:   return '#2670D7'
    if pct >= 75:   return '#f4a742'
    return '#d7191c'


def get_cancellation_colour(pct):
    if pct is None: return '#888888'
    if pct <= 2:    return '#1a9641'
    if pct <= 4:    return '#2670D7'
    if pct <= 7:    return '#f4a742'
    return '#d7191c'


def format_journeys(j):
    return 'New Station' if pd.isna(j) else f'{int(j):,}'


# ------------------- Live Train Data API -------------------
RAIL_API_URL = "https://api1.raildata.org.uk/1010-live-arrival-and-departure-boards-arr-and-dep1_1/LDBWS/api/20220120/GetArrivalDepartureBoard"
RAIL_API_KEY = os.environ.get("RAIL_API_KEY", "")


@app.route('/api/live/<crs>')
def live_trains(crs):
    crs = crs.strip().upper()
    try:
        resp = _requests.get(
            f"{RAIL_API_URL}/{crs}",
            headers={"User-Agent": "", "x-apikey": RAIL_API_KEY},
            params={"numRows": 10, "timeOffset": 0, "timeWindow": 120},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    services = data.get("trainServices") or []
    departures, arrivals = [], []
    for s in services:
        origin = s.get("origin", [{}])[0].get("locationName", "Unknown")
        destination = s.get("destination", [{}])[0].get("locationName", "Unknown")
        platform = s.get("platform", "TBD")
        operator = s.get("operator", "")
        if s.get("std"):
            departures.append({"time": s["std"], "expected": s.get("etd", ""),
                               "destination": destination, "platform": platform, "operator": operator})
        if s.get("sta"):
            arrivals.append({"time": s["sta"], "expected": s.get("eta", ""),
                             "origin": origin, "platform": platform, "operator": operator})
    return jsonify({"station": crs, "departures": departures[:5], "arrivals": arrivals[:5]})


_cached_index_html = None
_cached_map_render = None

def _build_map():
    global _cached_index_html, _cached_map_render
    if _cached_index_html:
        return
    center_lat = (55.9533 + 55.8642) / 2
    center_lon = (-3.1883 + -4.2518) / 2

    station_map = folium.Map(
        location=[center_lat, center_lon],
        min_lat=miny, max_lat=maxy, min_lon=minx, max_lon=maxx,
        zoom_start=8, tiles='CartoDB Voyager',
        max_bounds=True, zoom_control=True,
    )

    # Scotland boundary
    folium.GeoJson(
        scotland_geojson, name='Scotland',
        interactive=False, control=False,
        style_function=lambda f: {
            'fillColor': '#2670D7', 'color': '#333', 'weight': 1.5,
            'opacity': 0.6, 'fillOpacity': 0.06, 'dashArray': '5 3',
        },
    ).add_to(station_map)

    # Railway lines
    folium.GeoJson(
        railway_lines_geojson, name='Railway Lines',
        tooltip=folium.GeoJsonTooltip(
            fields=['name', 'usage'], aliases=['Railway:', 'Type:'],
            style='background-color:white;color:#333;font-family:Arial;font-size:12px;padding:6px;border-radius:4px;box-shadow:0 2px 6px rgba(0,0,0,0.2);',
        ),
        zoom_on_click=True,
        style_function=lambda f: {
            'color': '#4CAF50' if f['properties'].get('usage') == 'branch' else '#1565C0',
            'weight': 2 if f['properties'].get('usage') == 'branch' else 2.5,
            'opacity': 0.8,
            'dashArray': '6 4' if f['properties'].get('usage') == 'branch' else '0',
        },
        highlight_function=lambda f: {'weight': 6, 'color': '#FF6F00', 'opacity': 1},
    ).add_to(station_map)

    # Station layers per region
    region_groups = {}
    for rn in REGION_ORDER:
        fg = FeatureGroup(name=rn, show=True); fg.add_to(station_map); region_groups[rn] = fg
    for rn in joined_data['region'].unique():
        if rn not in region_groups:
            fg = FeatureGroup(name=rn, show=True); fg.add_to(station_map); region_groups[rn] = fg

    for _, row in joined_data.iterrows():
        usage_colour = get_station_colour(row.get('total_journeys'))
        total_fmt = format_journeys(row.get('total_journeys'))
        top_route_fmt = format_journeys(row.get('journeys'))
        dest = row['destination_station_name'] if pd.notna(row.get('destination_station_name')) else 'Data Unavailable'

        perf = station_perf_lookup.get(row['stationName'])
        op = perf['operator'] if perf else 'Unknown'

        # Punctuality colour & label — respect recency
        if perf and perf['punct_status'] == 'current':
            pc = get_punctuality_colour(perf['latest_punctuality'])
            pl = f"{perf['latest_punctuality']}%"
        elif perf and perf['punct_status'] == 'outdated':
            pc = '#888888'
            pl = 'Outdated Data'
        else:
            pc = '#888888'
            pl = 'N/A'

        # Cancellation colour & label — respect recency
        if perf and perf['cancel_status'] == 'current':
            cc = get_cancellation_colour(perf['latest_cancellations'])
            cl = f"{perf['latest_cancellations']}%"
        elif perf and perf['cancel_status'] == 'outdated':
            cc = '#888888'
            cl = 'Outdated Data'
        else:
            cc = '#888888'
            cl = 'N/A'

        tp = '|'.join(str(v) if v is not None else '' for v in perf['trend_punctuality']) if perf else ''
        tc = '|'.join(str(v) if v is not None else '' for v in perf['trend_cancellations']) if perf else ''
        tc = '|'.join(str(v) if v is not None else '' for v in perf['trend_cancellations']) if perf else ''

        popup_html = f"""
        <div class="station-popup" style="font-family:'Segoe UI',Arial,sans-serif;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.15);">
            <div class="popup-header" data-usage-colour="{usage_colour}" data-punct-colour="{pc}" data-cancel-colour="{cc}" style="background:{usage_colour};color:white;padding:10px 12px;">
                <div style="font-size:15px;font-weight:600;">{row['stationName']}</div>
                <div style="font-size:11px;opacity:0.9;margin-top:2px;">Station Code: {row['crsCode']} &bull; {row['region']}</div>
                <div style="font-size:11px;opacity:0.85;margin-top:1px;">Operator: {op}</div>
            </div>
            <div class="popup-body" style="display:flex;background:white;">
                <div class="popup-left" style="padding:10px 12px;min-width:340px;max-width:380px;flex-shrink:0;">
                    <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #eee;">
                        <span style="color:#666;font-size:12px;">Annual Entries & Exits</span>
                        <span style="font-weight:600;font-size:12px;color:{usage_colour};">{total_fmt}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #eee;">
                        <span style="color:#666;font-size:12px;">Top Route</span>
                        <span style="font-weight:600;font-size:12px;">{dest} ({top_route_fmt})</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #eee;">
                        <span style="color:#666;font-size:12px;">Punctuality (On Time)</span>
                        <span style="font-weight:600;font-size:12px;color:{pc};">{pl}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #eee;">
                        <span style="color:#666;font-size:12px;">Cancellations</span>
                        <span style="font-weight:600;font-size:12px;color:#d7191c;">{cl}</span>
                    </div>
                    <div style="margin-top:8px;padding:6px 0;border-bottom:1px solid #eee;position:relative;">
                        <div style="font-size:11px;color:#666;margin-bottom:4px;">Punctuality Trend</div>
                        <div class="spark-wrap" style="position:relative;">
                            <svg class="sparkline-punct" data-values="{tp}" data-labels="{PERIOD_LABELS_PIPE}" width="100%" height="48" style="display:block;"></svg>
                            <div class="spark-tooltip" style="display:none;position:absolute;top:-6px;pointer-events:none;background:rgba(0,0,0,0.8);color:#fff;font-size:10px;padding:3px 6px;border-radius:4px;white-space:nowrap;z-index:5;transform:translateX(-50%);"></div>
                            <div class="spark-crosshair" style="display:none;position:absolute;top:0;width:1px;height:100%;background:rgba(0,0,0,0.2);pointer-events:none;z-index:4;"></div>
                        </div>
                    </div>
                    <div style="margin-top:4px;padding:6px 0;border-bottom:1px solid #eee;position:relative;">
                        <div style="font-size:11px;color:#666;margin-bottom:4px;">Cancellation Trend</div>
                        <div class="spark-wrap" style="position:relative;">
                            <svg class="sparkline-cancel" data-values="{tc}" data-labels="{PERIOD_LABELS_PIPE}" width="100%" height="48" style="display:block;"></svg>
                            <div class="spark-tooltip" style="display:none;position:absolute;top:-6px;pointer-events:none;background:rgba(0,0,0,0.8);color:#fff;font-size:10px;padding:3px 6px;border-radius:4px;white-space:nowrap;z-index:5;transform:translateX(-50%);"></div>
                            <div class="spark-crosshair" style="display:none;position:absolute;top:0;width:1px;height:100%;background:rgba(0,0,0,0.2);pointer-events:none;z-index:4;"></div>
                        </div>
                    </div>
                    <button onclick="loadLiveTrains(this, '{row['crsCode']}')"
                        style="margin-top:8px;width:100%;padding:7px;border:none;border-radius:5px;background:linear-gradient(135deg,#1565C0,#0D47A1);color:white;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;transition:opacity 0.2s;"
                        onmouseover="this.style.opacity='0.85'" onmouseout="this.style.opacity='1'">
                        &#128646; View Live Trains
                    </button>
                </div>
                <div id="live-{row['crsCode']}" class="popup-right" style="display:none;border-left:1px solid #eee;padding:10px 12px;min-width:240px;max-width:280px;max-height:380px;overflow-y:auto;"></div>
            </div>
        </div>
        """

        marker_icon_html = f"""
        <div class="station-marker" data-usage-colour="{usage_colour}" data-punct-colour="{pc}" data-cancel-colour="{cc}" data-region="{row['region']}">
            <div class="station-icon" style="background-color:{usage_colour};">
                <i class="fa fa-train" style="color:white;"></i>
            </div>
        </div>
        """

        z_offset = 0
        tj = row.get('total_journeys')
        if pd.notna(tj):
            if tj >= 2_000_000:   z_offset = 3000
            elif tj >= 500_000:   z_offset = 2000
            elif tj >= 100_000:   z_offset = 1000

        marker = folium.Marker(
            location=[row['lat'], row['long']],
            popup=folium.Popup(popup_html, max_width=640, max_height=500),
            tooltip=folium.Tooltip(row['stationName'], style='font-family:Arial;font-size:12px;'),
            icon=folium.DivIcon(icon_size=(22, 22), icon_anchor=(11, 11), html=marker_icon_html, class_name=''),
        )
        marker.options['zIndexOffset'] = z_offset
        marker.add_to(region_groups[row['region']])

    # Mouse position
    MousePosition(position='bottomleft', separator=' | ', prefix='Coords:', num_digits=4).add_to(station_map)

    # Layer control & fullscreen
    folium.LayerControl(collapsed=True).add_to(station_map)
    folium.plugins.Fullscreen(position="topright", title="Fullscreen",
                              title_cancel="Exit Fullscreen", force_separate_button=True).add_to(station_map)

    # Inject JS/CSS into the Folium map iframe
    station_map.get_root().html.add_child(Element(_LIVE_TRAINS_JS))
    station_map.get_root().html.add_child(Element(_SPARKLINE_JS))
    station_map.get_root().html.add_child(Element(_COLOUR_MODE_JS))
    station_map.get_root().html.add_child(Element(_ZOOM_SWAP_JS))
    station_map.get_root().html.add_child(Element(_STATION_CSS))

    # Build station list for search
    station_list = [{'name': r['stationName'], 'lat': float(r['lat']), 'lng': float(r['long'])}
                    for _, r in joined_data.iterrows()]
    stations_json = json.dumps(station_list)

    # Cache both the map render and the index page
    _cached_map_render = station_map.get_root().render()

    map_html = '<iframe src="/map" style="width:100%;height:100%;border:none;" allowfullscreen></iframe>'
    _cached_index_html = _build_page_html(map_html, stations_json)


@app.route('/')
def index():
    if not _cached_index_html:
        _build_map()
    return _cached_index_html


# ===================== JavaScript / CSS constants =====================

_LIVE_TRAINS_JS = """
<script>
function loadLiveTrains(btn, crs) {
    var container = document.getElementById('live-' + crs);
    if (!container) return;
    container.style.display = 'block';
    container.innerHTML = '<div style="text-align:center;padding:16px 8px;color:#999;font-size:12px;">Loading...</div>';
    btn.disabled = true; btn.style.opacity = '0.5';
    try { var pw = btn.closest('.leaflet-popup'); if (pw) { var cw = pw.querySelector('.leaflet-popup-content'); if (cw) cw.style.width = 'auto'; } } catch(e) {}

    fetch('/api/live/' + crs)
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.error) { container.innerHTML = '<div style="color:#d7191c;font-size:11px;padding:4px;">Error: ' + data.error + '</div>'; btn.disabled = false; btn.style.opacity = '1'; return; }
            var html = '<div style="font-size:13px;font-weight:700;color:#0D47A1;margin-bottom:6px;">Live Trains</div>';
            if (data.departures && data.departures.length) {
                html += '<div style="font-size:11px;font-weight:700;color:#1565C0;padding:4px 0 2px;">Departures</div>';
                html += '<table style="width:100%;font-size:10px;border-collapse:collapse;"><tr style="color:#999;"><td style="padding:2px 3px;">Time</td><td style="padding:2px 3px;">To</td><td style="padding:2px 3px;">Plat</td><td style="padding:2px 3px;">Exp</td></tr>';
                data.departures.forEach(function(d) {
                    var sc = d.expected === 'On time' ? '#1a9641' : (d.expected === 'Cancelled' ? '#d7191c' : '#f4a742');
                    html += '<tr><td style="padding:2px 3px;font-weight:600;">' + d.time + '</td><td style="padding:2px 3px;max-width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + d.destination + '</td><td style="padding:2px 3px;text-align:center;">' + d.platform + '</td><td style="padding:2px 3px;color:' + sc + ';font-weight:600;font-size:9px;">' + d.expected + '</td></tr>';
                });
                html += '</table>';
            }
            if (data.arrivals && data.arrivals.length) {
                html += '<div style="font-size:11px;font-weight:700;color:#4CAF50;padding:4px 0 2px;border-top:1px solid #eee;margin-top:4px;">Arrivals</div>';
                html += '<table style="width:100%;font-size:10px;border-collapse:collapse;"><tr style="color:#999;"><td style="padding:2px 3px;">Time</td><td style="padding:2px 3px;">From</td><td style="padding:2px 3px;">Plat</td><td style="padding:2px 3px;">Exp</td></tr>';
                data.arrivals.forEach(function(a) {
                    var sc = a.expected === 'On time' ? '#1a9641' : (a.expected === 'Cancelled' ? '#d7191c' : '#f4a742');
                    html += '<tr><td style="padding:2px 3px;font-weight:600;">' + a.time + '</td><td style="padding:2px 3px;max-width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + a.origin + '</td><td style="padding:2px 3px;text-align:center;">' + a.platform + '</td><td style="padding:2px 3px;color:' + sc + ';font-weight:600;font-size:9px;">' + a.expected + '</td></tr>';
                });
                html += '</table>';
            }
            if (!data.departures.length && !data.arrivals.length) html += '<div style="color:#999;font-size:11px;padding:4px;">No services found.</div>';
            container.innerHTML = html;
            btn.textContent = '\u21bb Refresh'; btn.disabled = false; btn.style.opacity = '1';
            try { var mapObj = null; for (var k in window) { try { if (window[k] && typeof window[k].getZoom === 'function' && window[k]._container) { mapObj = window[k]; break; } } catch(e) {} } if (mapObj) setTimeout(function() { mapObj.invalidateSize(); var ll = mapObj._popup && mapObj._popup.getLatLng(); if (ll) mapObj.panTo(ll); }, 120); } catch(e) {}
        })
        .catch(function() { container.innerHTML = '<div style="color:#d7191c;font-size:11px;padding:4px;">Failed to load data.</div>'; btn.disabled = false; btn.style.opacity = '1'; });
}
</script>
"""

_SPARKLINE_JS = """
<script>
function parseSparkData(str) {
    if (!str) return [];
    var parts = str.split('|'), out = [];
    for (var i = 0; i < parts.length; i++) if (parts[i] !== '') out.push({i:i, v:parseFloat(parts[i]), total:parts.length});
    return out;
}
function renderSparkline(svg) {
    if (svg.getAttribute('data-rendered')) return;
    var raw = svg.getAttribute('data-values') || '', labelStr = svg.getAttribute('data-labels') || '';
    var allLabels = labelStr ? labelStr.split('|') : [], vals = parseSparkData(raw);
    var W = 280, H = 48, ns = 'http://www.w3.org/2000/svg';
    if (vals.length < 2) {
        svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
        var txt = document.createElementNS(ns, 'text'); txt.setAttribute('x','4'); txt.setAttribute('y','26');
        txt.setAttribute('font-size','11'); txt.setAttribute('fill','#999'); txt.textContent = 'No data';
        svg.appendChild(txt); svg.setAttribute('data-rendered','1'); return;
    }
    var padY = 4, totalPts = vals[0].total;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H); svg.setAttribute('preserveAspectRatio','xMidYMid meet');
    svg.style.width = '100%'; svg.style.height = H + 'px'; svg.style.cursor = 'crosshair';
    var isCancel = svg.classList.contains('sparkline-cancel');
    var minV = isCancel ? 0 : Math.min.apply(null, vals.map(function(d){return d.v;}));
    var maxV = Math.max.apply(null, vals.map(function(d){return d.v;}));
    if (maxV === minV) maxV = minV + 1;
    var xStep = W / Math.max(totalPts - 1, 1);
    var cX = [], cY = [], cV = [];
    for (var j = 0; j < vals.length; j++) { cX.push(vals[j].i * xStep); cY.push(padY + (H - 2*padY) * (1 - (vals[j].v - minV) / (maxV - minV))); cV.push(vals[j]); }
    var pts = []; for (var j = 0; j < cX.length; j++) pts.push(cX[j].toFixed(1) + ',' + cY[j].toFixed(1));
    var colour = isCancel ? '#d7191c' : '#1565C0', fillC = isCancel ? 'rgba(215,25,28,0.10)' : 'rgba(21,101,192,0.10)';
    var area = document.createElementNS(ns, 'path');
    area.setAttribute('d', 'M' + cX[0].toFixed(1) + ',' + H + ' L' + pts.join(' L') + ' L' + cX[cX.length-1].toFixed(1) + ',' + H + ' Z');
    area.setAttribute('fill', fillC); svg.appendChild(area);
    var line = document.createElementNS(ns, 'path'); line.setAttribute('d', 'M' + pts.join(' L'));
    line.setAttribute('fill','none'); line.setAttribute('stroke',colour); line.setAttribute('stroke-width','1.5'); svg.appendChild(line);
    var endDot = document.createElementNS(ns, 'circle');
    endDot.setAttribute('cx', cX[cX.length-1].toFixed(1)); endDot.setAttribute('cy', cY[cY.length-1].toFixed(1));
    endDot.setAttribute('r','3'); endDot.setAttribute('fill',colour); endDot.setAttribute('class','spark-end-dot'); svg.appendChild(endDot);
    var hoverDot = document.createElementNS(ns, 'circle'); hoverDot.setAttribute('r','4'); hoverDot.setAttribute('fill',colour);
    hoverDot.setAttribute('stroke','white'); hoverDot.setAttribute('stroke-width','2'); hoverDot.style.display = 'none';
    hoverDot.setAttribute('class','spark-hover-dot'); svg.appendChild(hoverDot);
    var overlay = document.createElementNS(ns, 'rect'); overlay.setAttribute('x','0'); overlay.setAttribute('y','0');
    overlay.setAttribute('width',W); overlay.setAttribute('height',H); overlay.setAttribute('fill','transparent');
    overlay.style.cursor = 'crosshair'; svg.appendChild(overlay);
    svg._sparkData = {coordsX:cX, coordsY:cY, vals:cV, labels:allLabels, W:W, colour:colour, isCancel:isCancel};
    overlay.addEventListener('mousemove', function(e) {
        var rect = svg.getBoundingClientRect(), mouseX = (e.clientX - rect.left) / rect.width * W;
        var sd = svg._sparkData, best = 0, bestDist = 99999;
        for (var k = 0; k < sd.coordsX.length; k++) { var d = Math.abs(sd.coordsX[k] - mouseX); if (d < bestDist) { bestDist = d; best = k; } }
        var pt = sd.vals[best];
        hoverDot.setAttribute('cx', sd.coordsX[best].toFixed(1)); hoverDot.setAttribute('cy', sd.coordsY[best].toFixed(1)); hoverDot.style.display = '';
        endDot.style.display = 'none';
        var wrap = svg.parentElement;
        if (wrap) {
            var tip = wrap.querySelector('.spark-tooltip'), cross = wrap.querySelector('.spark-crosshair');
            if (tip) { var label = sd.labels[pt.i] || ('Period ' + (pt.i+1)); var suffix = sd.isCancel ? '% cancelled' : '% on time'; tip.textContent = label + ': ' + pt.v.toFixed(1) + suffix; tip.style.display = 'block'; tip.style.left = ((e.clientX - rect.left) / rect.width * 100) + '%'; }
            if (cross) { cross.style.display = 'block'; cross.style.left = ((e.clientX - rect.left) / rect.width * 100) + '%'; }
        }
    });
    overlay.addEventListener('mouseleave', function() {
        hoverDot.style.display = 'none'; endDot.style.display = '';
        var wrap = svg.parentElement;
        if (wrap) { var tip = wrap.querySelector('.spark-tooltip'), cross = wrap.querySelector('.spark-crosshair'); if (tip) tip.style.display = 'none'; if (cross) cross.style.display = 'none'; }
    });
    svg.setAttribute('data-rendered', '1');
}
function renderAllSparklines(el) { if (!el) el = document; var svgs = el.querySelectorAll('svg[class*=sparkline-]'); for (var i = 0; i < svgs.length; i++) renderSparkline(svgs[i]); }
(function() {
    var checkMap = setInterval(function() {
        var mapObj = null; for (var k in window) { try { if (window[k] && typeof window[k].getZoom === 'function' && window[k]._container) { mapObj = window[k]; break; } } catch(e) {} }
        if (!mapObj) return; clearInterval(checkMap);
        mapObj.on('popupopen', function() { setTimeout(function(){ renderAllSparklines(document); }, 80); setTimeout(function(){ renderAllSparklines(document); }, 250); });
    }, 200);
})();
</script>
"""

_COLOUR_MODE_JS = """
<script>
(function() {
    var currentMode = 'usage';
    window.addEventListener('message', function(e) { if (e.data && e.data.type === 'colourMode') { currentMode = e.data.mode; applyColourMode(currentMode); } });
    function colourForMode(el, mode) {
        if (mode === 'punctuality') return el.getAttribute('data-punct-colour');
        if (mode === 'cancellations') return el.getAttribute('data-cancel-colour');
        return el.getAttribute('data-usage-colour');
    }
    function applyColourMode(mode) {
        document.querySelectorAll('.station-marker').forEach(function(m) {
            var c = colourForMode(m, mode);
            var icon = m.querySelector('.station-icon');
            if (icon) icon.style.backgroundColor = c;
        });
        document.querySelectorAll('.popup-header').forEach(function(h) { h.style.background = colourForMode(h, mode); });
    }
    window.applyColourMode = applyColourMode;
    window.setColourMode = function(mode) { currentMode = mode; applyColourMode(mode); };
    var checkMap = setInterval(function() {
        var mapObj = null; for (var k in window) { try { if (window[k] && typeof window[k].getZoom === 'function' && window[k]._container) { mapObj = window[k]; break; } } catch(e) {} }
        if (!mapObj) return; clearInterval(checkMap);
        mapObj.on('popupopen', function() { setTimeout(function() { document.querySelectorAll('.popup-header').forEach(function(h) { h.style.background = colourForMode(h, currentMode); }); }, 30); });
    }, 200);
})();
</script>
"""

_ZOOM_SWAP_JS = """
<script>
(function() {
    var checkMap = setInterval(function() {
        var mapObj = null;
        for (var k in window) { try { if (window[k] && typeof window[k].getZoom === 'function' && window[k]._container) { mapObj = window[k]; break; } } catch(e) {} }
        if (!mapObj) return; clearInterval(checkMap);
        function update() {
            var z = mapObj.getZoom();
            var s = Math.min(1.0, Math.max(0.22, 0.22 + (z - 7) * 0.13));
            var markers = document.querySelectorAll('.station-marker');
            for (var i = 0; i < markers.length; i++) {
                markers[i].style.setProperty('--zoom-scale', s.toFixed(3));
            }
        }
        mapObj.on('zoomend', update); update();
    }, 200);
})();
</script>
"""

_STATION_CSS = """
<style>
.station-marker { width:22px; height:22px; display:flex; align-items:center; justify-content:center; position:relative;
    transform:scale(var(--zoom-scale, 0.22)); transition:transform 0.3s ease; transform-origin:center center; }
.station-icon {
    width:22px; height:22px;
    display:flex; align-items:center; justify-content:center;
    border-radius:4px; border:2px solid rgba(255,255,255,0.92);
    box-shadow:0 1px 4px rgba(0,0,0,0.3);
    transition:box-shadow 0.25s ease, background-color 0.3s ease;
}
.station-icon .fa { font-size:10px; line-height:1; color:white; }
.station-marker:hover { z-index:999 !important; }
.station-marker:hover .station-icon {
    box-shadow:0 0 0 3px rgba(255,255,255,0.5), 0 2px 10px rgba(0,0,0,0.35);
    filter:brightness(1.15);
}
.leaflet-popup-content-wrapper { padding:0 !important; border-radius:8px !important; overflow:hidden; }
.leaflet-popup-content { margin:0 !important; width:auto !important; }
.leaflet-popup-close-button { width:26px !important; height:26px !important; font-size:18px !important; font-weight:700 !important;
    color:white !important; background:rgba(0,0,0,0.35) !important; border-radius:50% !important;
    display:flex !important; align-items:center !important; justify-content:center !important;
    top:6px !important; right:6px !important; padding:0 !important; margin:0 !important;
    line-height:1 !important; transition:background 0.2s ease !important; z-index:10 !important; }
.leaflet-popup-close-button:hover { background:rgba(0,0,0,0.6) !important; color:white !important; }
</style>
"""


@app.route('/map')
def map_page():
    """Serve the folium map as a standalone page for the iframe."""
    if not _cached_map_render:
        _build_map()
    from flask import request, make_response
    import gzip as _gzip
    if 'gzip' in request.headers.get('Accept-Encoding', ''):
        compressed = _gzip.compress(_cached_map_render.encode('utf-8'), compresslevel=6)
        resp = make_response(compressed)
        resp.headers['Content-Encoding'] = 'gzip'
        resp.headers['Content-Type'] = 'text/html; charset=utf-8'
        resp.headers['Content-Length'] = len(compressed)
        resp.headers['Cache-Control'] = 'public, max-age=300'
        return resp
    return _cached_map_render


def _build_page_html(map_html, stations_json):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scotland Railway Map</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        html, body {{ height:100%; overflow:hidden; font-family:'Inter','Segoe UI',Arial,sans-serif; background:#f0f2f5; }}
        body {{ display:flex; flex-direction:column; }}

        .header {{
            background:rgba(12,20,40,0.92);
            backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
            color:white; padding:0 22px; height:56px;
            display:flex; align-items:center; justify-content:space-between;
            box-shadow:0 1px 8px rgba(0,0,0,0.18);
            z-index:10000; flex-shrink:0;
            border-bottom:1px solid rgba(255,255,255,0.06);
        }}
        .header-left {{ display:flex; align-items:center; gap:6px; }}
        .header-left .logo {{ width:28px; height:28px; border-radius:7px; background:linear-gradient(135deg,#4facfe,#00f2fe); display:flex; align-items:center; justify-content:center; font-size:14px; font-weight:700; color:#0a1428; flex-shrink:0; }}
        .header-left h1 {{ font-size:15px; font-weight:600; letter-spacing:-0.2px; }}
        .header-left h1 a {{ color:white; text-decoration:none; }}
        .header-left h1 a:hover {{ color:#4facfe; }}

        .header-centre {{ display:flex; align-items:center; gap:14px; }}

        .search-wrap {{ position:relative; }}
        .search-wrap input {{
            width:200px; padding:7px 12px 7px 30px; border:1px solid rgba(255,255,255,0.1);
            border-radius:8px; font-size:12.5px; outline:none;
            background:rgba(255,255,255,0.06); color:white;
            transition:all 0.2s;
        }}
        .search-wrap input:focus {{ background:rgba(255,255,255,0.12); border-color:rgba(79,172,254,0.4); width:240px; }}
        .search-wrap input::placeholder {{ color:rgba(255,255,255,0.4); }}
        .search-icon {{ position:absolute; left:9px; top:50%; transform:translateY(-50%); color:rgba(255,255,255,0.35); font-size:12px; pointer-events:none; }}
        #search-results {{
            display:none; position:absolute; top:calc(100% + 6px); left:0;
            background:rgba(20,28,48,0.96); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
            border-radius:10px; box-shadow:0 8px 30px rgba(0,0,0,0.35);
            max-height:240px; overflow-y:auto; width:280px; z-index:9999;
            border:1px solid rgba(255,255,255,0.08);
        }}

        .divider {{ width:1px; height:22px; background:rgba(255,255,255,0.1); }}

        .colour-wrap {{ display:flex; align-items:center; gap:7px; }}
        .colour-wrap label {{ font-size:11px; color:rgba(255,255,255,0.5); font-weight:500; text-transform:uppercase; letter-spacing:0.5px; }}
        .colour-select {{
            padding:5px 24px 5px 10px; border:1px solid rgba(255,255,255,0.1);
            border-radius:7px; font-size:12px; background:rgba(255,255,255,0.06);
            color:white; outline:none; cursor:pointer;
            -webkit-appearance:none; appearance:none;
            background-image:url('data:image/svg+xml;utf8,<svg fill="rgba(255,255,255,0.5)" xmlns="http://www.w3.org/2000/svg" width="10" height="6"><path d="M0 0l5 6 5-6z"/></svg>');
            background-repeat:no-repeat; background-position:right 8px center;
            transition:all 0.2s;
        }}
        .colour-select:hover {{ border-color:rgba(255,255,255,0.2); }}
        .colour-select option {{ color:#333; background:white; }}

        .legend {{ display:none; align-items:center; gap:10px; font-size:11px; font-weight:500; }}
        .legend.active {{ display:flex; }}
        .legend-item {{ display:flex; align-items:center; gap:4px; opacity:0.8; }}
        .legend-dot {{ width:8px; height:8px; border-radius:50%; }}

        .header-right {{ display:flex; align-items:center; }}
        .nav-btn {{
            color:rgba(255,255,255,0.7); text-decoration:none; font-size:12px; font-weight:500;
            padding:6px 14px; border-radius:7px; transition:all 0.2s;
            display:flex; align-items:center; gap:5px;
        }}
        .nav-btn:hover {{ color:white; background:rgba(255,255,255,0.08); }}

        .map-container {{ flex:1; min-height:0; position:relative; overflow:hidden; }}
        .map-container iframe, .map-container .folium-map {{ width:100% !important; height:100% !important; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <div class="logo">&#128507;</div>
            <h1><a href="/">Scotland Railway Map</a></h1>
        </div>

        <div class="header-centre">
            <div class="search-wrap">
                <span class="search-icon">&#128269;</span>
                <input id="station-search" type="text" placeholder="Search stations..." autocomplete="off" />
                <div id="search-results"></div>
            </div>

            <div class="divider"></div>

            <div class="colour-wrap">
                <label for="colour-mode">Colour</label>
                <select id="colour-mode" class="colour-select">
                    <option value="usage">Usage (Journeys)</option>
                    <option value="punctuality">Punctuality</option>
                    <option value="cancellations">Cancellations</option>
                </select>
            </div>

            <div class="divider"></div>

            <div id="legend-usage" class="legend active">
                <div class="legend-item"><div class="legend-dot" style="background:#1a9641;"></div>2M+</div>
                <div class="legend-item"><div class="legend-dot" style="background:#2670D7;"></div>500k\u20132M</div>
                <div class="legend-item"><div class="legend-dot" style="background:#f4a742;"></div>100k\u2013500k</div>
                <div class="legend-item"><div class="legend-dot" style="background:#d7191c;"></div>&lt;100k</div>
                <div class="legend-item"><div class="legend-dot" style="background:#888;"></div>N/A</div>
            </div>
            <div id="legend-punctuality" class="legend">
                <div class="legend-item"><div class="legend-dot" style="background:#1a9641;"></div>\u226592%</div>
                <div class="legend-item"><div class="legend-dot" style="background:#2670D7;"></div>85\u201392%</div>
                <div class="legend-item"><div class="legend-dot" style="background:#f4a742;"></div>75\u201385%</div>
                <div class="legend-item"><div class="legend-dot" style="background:#d7191c;"></div>&lt;75%</div>
                <div class="legend-item"><div class="legend-dot" style="background:#888;"></div>N/A</div>
            </div>
            <div id="legend-cancellations" class="legend">
                <div class="legend-item"><div class="legend-dot" style="background:#1a9641;"></div>\u22642%</div>
                <div class="legend-item"><div class="legend-dot" style="background:#2670D7;"></div>2\u20134%</div>
                <div class="legend-item"><div class="legend-dot" style="background:#f4a742;"></div>4\u20137%</div>
                <div class="legend-item"><div class="legend-dot" style="background:#d7191c;"></div>&gt;7%</div>
                <div class="legend-item"><div class="legend-dot" style="background:#888;"></div>N/A</div>
            </div>
        </div>

        <div class="header-right">
            <a href="/about" class="nav-btn">&#128100; About Me</a>
        </div>
    </div>

    <div class="map-container">
        {map_html}
    </div>

    <script>
    (function() {{
        var sel = document.getElementById('colour-mode');
        var lu = document.getElementById('legend-usage'), lp = document.getElementById('legend-punctuality'), lc = document.getElementById('legend-cancellations');
        sel.addEventListener('change', function() {{
            var mode = this.value;
            lu.className = 'legend' + (mode === 'usage' ? ' active' : '');
            lp.className = 'legend' + (mode === 'punctuality' ? ' active' : '');
            lc.className = 'legend' + (mode === 'cancellations' ? ' active' : '');
            var iframe = document.querySelector('.map-container iframe');
            if (iframe) {{
                try {{ var iWin = iframe.contentWindow; if (iWin && typeof iWin.setColourMode === 'function') iWin.setColourMode(mode); }} catch(ex) {{}}
                try {{ if (iframe.contentWindow) iframe.contentWindow.postMessage({{type:'colourMode', mode:mode}}, '*'); }} catch(ex) {{}}
            }}
        }});
    }})();
    (function() {{
        var stations = {stations_json};
        var input = document.getElementById('station-search'), resultsDiv = document.getElementById('search-results');
        var leafletMap = null, searchMarker = null;
        function findMap() {{
            if (leafletMap) return leafletMap;
            var iframe = document.querySelector('.map-container iframe');
            if (iframe) {{ try {{ var iWin = iframe.contentWindow; for (var key in iWin) {{ try {{ if (iWin[key] && typeof iWin[key].setView === 'function' && iWin[key]._zoom !== undefined) {{ leafletMap = iWin[key]; return leafletMap; }} }} catch(e) {{}} }} }} catch(e) {{}} }}
            return null;
        }}
        input.addEventListener('input', function() {{
            var q = this.value.trim().toLowerCase(); resultsDiv.innerHTML = '';
            if (q.length < 1) {{ resultsDiv.style.display = 'none'; return; }}
            var matches = stations.filter(function(s) {{ return s.name.toLowerCase().indexOf(q) !== -1; }}).slice(0, 15);
            if (!matches.length) {{ resultsDiv.style.display = 'none'; return; }}
            matches.forEach(function(s) {{
                var div = document.createElement('div'); div.textContent = s.name;
                div.style.cssText = 'padding:9px 14px;cursor:pointer;font-size:13px;font-family:Inter,Arial,sans-serif;color:rgba(255,255,255,0.85);border-bottom:1px solid rgba(255,255,255,0.06);transition:background 0.15s;';
                div.addEventListener('mouseenter', function() {{ this.style.background='rgba(79,172,254,0.12)'; }});
                div.addEventListener('mouseleave', function() {{ this.style.background='transparent'; }});
                div.addEventListener('click', function() {{
                    input.value = s.name; resultsDiv.style.display = 'none';
                    var m = findMap();
                    if (m) {{
                        m.setView([s.lat, s.lng], 14, {{animate:true}});
                        if (searchMarker) m.removeLayer(searchMarker);
                        searchMarker = L.circleMarker([s.lat, s.lng], {{radius:18, color:'#4facfe', weight:3, fillOpacity:0.12, fillColor:'#4facfe'}}).addTo(m);
                        setTimeout(function() {{ if (searchMarker) m.removeLayer(searchMarker); }}, 4000);
                    }}
                }});
                resultsDiv.appendChild(div);
            }});
            resultsDiv.style.display = 'block';
        }});
        document.addEventListener('click', function(e) {{ if (!input.contains(e.target) && !resultsDiv.contains(e.target)) resultsDiv.style.display = 'none'; }});
    }})();
    </script>
</body>
</html>"""


@app.route('/about')
def about():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About Me | Scotland Railway Map</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:'Inter','Segoe UI',Arial,sans-serif; background:#f0f2f5; color:#333; min-height:100vh; }
        .header {
            background:rgba(12,20,40,0.92);
            backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
            color:white; padding:0 22px; height:56px;
            display:flex; align-items:center; justify-content:space-between;
            box-shadow:0 1px 8px rgba(0,0,0,0.18);
            position:sticky; top:0; z-index:100;
            border-bottom:1px solid rgba(255,255,255,0.06);
        }
        .header h1 { font-size:15px; font-weight:600; letter-spacing:-0.2px; }
        .header h1 a { color:white; text-decoration:none; }
        .header h1 a:hover { color:#4facfe; }
        .nav-links { display:flex; gap:12px; align-items:center; }
        .nav-links a { color:rgba(255,255,255,0.7); text-decoration:none; font-size:12px; font-weight:500; padding:6px 14px; border-radius:7px; transition:all 0.2s; display:flex; align-items:center; gap:5px; }
        .nav-links a:hover { color:white; background:rgba(255,255,255,0.08); }
        .nav-links a.active { color:white; background:rgba(255,255,255,0.1); }
        .about-container { max-width:780px; margin:0 auto; padding:48px 24px 80px; }
        .profile-card { text-align:center; margin-bottom:48px; }
        .profile-photo-fallback { width:160px; height:160px; border-radius:50%; background:linear-gradient(135deg,#1565C0,#0D47A1); color:white; font-size:48px; font-weight:700; display:inline-flex; align-items:center; justify-content:center; border:4px solid white; box-shadow:0 4px 20px rgba(0,0,0,0.12); margin-bottom:20px; }
        .profile-photo { width:160px; height:160px; border-radius:50%; object-fit:cover; border:4px solid white; box-shadow:0 4px 20px rgba(0,0,0,0.12); margin-bottom:20px; }
        .profile-name { font-size:32px; font-weight:700; color:#1a1a2e; margin-bottom:6px; }
        .profile-title { font-size:16px; color:#666; font-weight:400; margin-bottom:16px; }
        .social-links { display:flex; justify-content:center; gap:10px; flex-wrap:wrap; }
        .linkedin-btn { display:inline-flex; align-items:center; gap:8px; padding:10px 24px; background:#0A66C2; color:white; text-decoration:none; border-radius:8px; font-size:14px; font-weight:600; transition:background 0.2s, transform 0.1s; }
        .linkedin-btn:hover { background:#004182; transform:translateY(-1px); }
        .linkedin-btn svg { width:18px; height:18px; fill:white; }
        .email-btn { display:inline-flex; align-items:center; gap:8px; padding:10px 24px; background:#333; color:white; text-decoration:none; border-radius:8px; font-size:14px; font-weight:600; transition:background 0.2s, transform 0.1s; }
        .email-btn:hover { background:#555; transform:translateY(-1px); }
        .section { background:white; border-radius:12px; padding:28px 32px; margin-bottom:24px; box-shadow:0 1px 4px rgba(0,0,0,0.06); }
        .section-title { font-size:18px; font-weight:700; color:#0D47A1; margin-bottom:16px; padding-bottom:10px; border-bottom:2px solid #e3f2fd; display:flex; align-items:center; gap:10px; }
        .section-title .icon { font-size:20px; }
        .section p { font-size:14px; line-height:1.7; color:#444; }
        .timeline { list-style:none; position:relative; padding-left:24px; }
        .timeline::before { content:''; position:absolute; left:6px; top:4px; bottom:4px; width:2px; background:#e0e0e0; }
        .timeline li { position:relative; margin-bottom:28px; padding-left:20px; }
        .timeline li:last-child { margin-bottom:0; }
        .timeline li::before { content:''; position:absolute; left:-20px; top:6px; width:10px; height:10px; border-radius:50%; background:#1565C0; border:2px solid white; box-shadow:0 0 0 2px #1565C0; }
        .timeline .role { font-size:15px; font-weight:600; color:#1a1a2e; }
        .timeline .org { font-size:13px; font-weight:500; color:#1565C0; }
        .timeline .period { font-size:12px; color:#999; margin:2px 0 6px; }
        .timeline .desc { font-size:13px; color:#555; line-height:1.6; }
        .timeline .desc ul { margin:4px 0 0 16px; }
        .timeline .desc ul li { margin-bottom:4px; padding-left:0; }
        .timeline .desc ul li::before { display:none; }
        .skills-grid { display:flex; flex-wrap:wrap; gap:8px; }
        .skill-tag { padding:6px 14px; background:#e3f2fd; color:#1565C0; border-radius:20px; font-size:12px; font-weight:500; }
        .lang-grid { display:flex; flex-wrap:wrap; gap:12px; margin-top:8px; }
        .lang-item { display:flex; align-items:center; gap:8px; font-size:13px; color:#444; }
        .lang-level { font-size:11px; color:#999; font-weight:500; }
        .project-highlight { background:linear-gradient(135deg,#e3f2fd,#bbdefb); border-radius:10px; padding:20px 24px; margin-top:12px; }
        .project-highlight .title { font-size:14px; font-weight:600; color:#0D47A1; margin-bottom:6px; }
        .project-highlight .desc { font-size:13px; color:#444; line-height:1.6; }
        .interests-grid { display:flex; flex-wrap:wrap; gap:10px; margin-top:4px; }
        .interest-tag { padding:6px 14px; background:#fff3e0; color:#e65100; border-radius:20px; font-size:12px; font-weight:500; }
        .charity-note { margin-top:16px; padding:14px 18px; background:#e8f5e9; border-radius:8px; font-size:13px; color:#2e7d32; line-height:1.5; }
        .charity-note strong { font-weight:600; }
        .footer-note { text-align:center; padding:24px; font-size:12px; color:#999; }
    </style>
</head>
<body>
    <div class="header">
        <div style="display:flex;align-items:center;gap:6px;">
            <div style="width:28px;height:28px;border-radius:7px;background:linear-gradient(135deg,#4facfe,#00f2fe);display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:#0a1428;flex-shrink:0;">&#128507;</div>
            <h1><a href="/">Scotland Railway Map</a></h1>
        </div>
        <div class="nav-links">
            <a href="/">&#127758; Map</a>
            <a href="/about" class="active">&#128100; About Me</a>
        </div>
    </div>
    <div class="about-container">
        <div class="profile-card">
            <img src="/assets/images/image.jpg" alt="Aleksejs Belasovs" class="profile-photo"
                 onerror="this.style.display='none'; var d=document.createElement('div'); d.className='profile-photo-fallback'; d.textContent='AB'; this.parentNode.insertBefore(d,this.nextSibling);">
            <div class="profile-name">Aleksejs Belasovs</div>
            <div class="profile-title">Data &amp; AI Scientist</div>
            <div class="social-links">
                <a href="https://www.linkedin.com/in/aleksejsbelasovs/" target="_blank" rel="noopener" class="linkedin-btn">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
                    LinkedIn
                </a>
                <a href="mailto:aleksejs.belasovs@gmail.com" class="email-btn">
                    &#9993; Email
                </a>
            </div>
        </div>

        <!-- About Me -->
        <div class="section">
            <div class="section-title"><span class="icon">&#128075;</span> About Me</div>
            
            <p style="margin-bottom: 1.5em; line-height: 1.6;">
                I am a data scientist with a strong passion for explainable modelling, transport planning, and economic policy analysis. I have a particular interest in how mass transport systems connect communities and support economic growth.
            </p>
            
            <p style="margin-bottom: 1.5em; line-height: 1.6;">
                I believe that transport networks are fundamental to creating more accessible, productive, and inclusive cities. Growing up in Norwich, England, I often wandered along old railway lines that had been converted into pedestrian paths, constantly imagining what they would have been like in their heyday. It was from that early curiosity that my passion for transport truly began.
            </p>
            
            <p style="margin-bottom: 1.5em; line-height: 1.6;">
                I created this page, and the projects it showcases, to help shape the conversation around transport. I believe discussions should not be limited to simple revenue-and-cost calculations for new projects. Instead, we should focus on the broader impact: the benefits to the communities affected and the wider economic opportunities that well-planned transport networks can generate.
            </p>
            
            <p style="line-height: 1.6;">
                I hope my data-first approach can inspire others to think differently about transport. If my work resonates with you, or if you would like to discuss ideas, projects, or collaborations, please feel free to reach out—I'd love to connect.
            </p>
        </div>

        <!-- Education -->
        <div class="section">
            <div class="section-title"><span class="icon">&#127891;</span> Education</div>
            <ul class="timeline">
                <li>
                    <div class="role">BSc Economics, Econometrics and Finance (with Placement Year)</div>
                    <div class="org">University of York &mdash; York, England</div>
                    <div class="period">Sep 2020 &ndash; Aug 2024</div>
                    <div class="desc">
                        <ul>
                            <li><strong>Grade: 1st Class Honours</strong></li>
                        </ul>
                    </div>
                </li>
            </ul>
        </div>

        <!-- Work Experience -->
        <div class="section">
            <div class="section-title"><span class="icon">&#128188;</span> Work Experience</div>
            <ul class="timeline">
                <li>
                    <div class="role">AI and Data Science Graduate</div>
                    <div class="org">Lloyds Banking Group &mdash; Edinburgh, Scotland</div>
                    <div class="period">Sep 2025 &ndash; Present</div>
                    <div class="desc">
                        <ul>
                            <li>Building cutting-edge fraud prevention models and supporting scam prevention use cases</li>
                            <li>Accelerating statistical learning model development in CIB, aiming to reduce client onboarding timeline by &gt;60%</li>
                            <li>Utilising Generative AI for FX conversion models with a potential of &pound;10M in business value</li>
                        </ul>
                    </div>
                </li>
                <li>
                    <div class="role">Private Bank Securities Based Lending Analyst</div>
                    <div class="org">JP Morgan and Chase &mdash; Edinburgh, Scotland</div>
                    <div class="period">Jan 2025 &ndash; Sep 2025</div>
                    <div class="desc">
                        <ul>
                            <li>Underwrote over &pound;500M of securities-backed Lombard loans for individuals and corporates</li>
                            <li>Assessed and calculated derivative trading limits based on client requirements</li>
                            <li>Conducted in-depth analysis of complex, high-value portfolios comprising liquid and tradable assets</li>
                            <li>Automated manual processes using VBA (saving 5+ hours weekly) and Python (portfolio rebalancing solution)</li>
                        </ul>
                    </div>
                </li>
                <li>
                    <div class="role">Research Development Fund Administrator</div>
                    <div class="org">University of York &mdash; York, England</div>
                    <div class="period">Feb 2024 &ndash; Dec 2024</div>
                    <div class="desc">
                        <ul>
                            <li>Managed &pound;4M+ in strategic research funding</li>
                            <li>Built out back-end data infrastructure for accurate financial reporting</li>
                            <li>Automated application workflows using JavaScript (saving ~4 hours per week)</li>
                            <li>Produced quarterly funding reports and financial forecasts based on historical trends</li>
                        </ul>
                    </div>
                </li>
                <li>
                    <div class="role">Technical Risk Analyst &mdash; Placement Scheme</div>
                    <div class="org">Lloyds Banking Group &mdash; Edinburgh, Scotland</div>
                    <div class="period">Jul 2022 &ndash; Jul 2023</div>
                    <div class="desc">
                        <ul>
                            <li>Analysed corporate credit portfolios with &pound;100Bn+ exposure</li>
                            <li>Delivered weekly risk briefings to senior stakeholder groups (up to 25 attendees)</li>
                            <li>Optimised SQL codebase, reducing update and processing times by 25%</li>
                            <li>Assisted in BASEL III: CRD IV model development</li>
                            <li>Led the technology team for the firm&rsquo;s Quants Expo Day, launching an interactive Jobs Board used by 400+ participants</li>
                            <li>Headed the data visualisation team as part of a pioneering Python migration project</li>
                        </ul>
                    </div>
                </li>
            </ul>
        </div>

        <!-- Skills & Tools -->
        <div class="section">
            <div class="section-title"><span class="icon">&#128736;</span> Skills &amp; Tools</div>
            <div class="skills-grid">
                <span class="skill-tag">Python</span>
                <span class="skill-tag">SQL</span>
                <span class="skill-tag">R</span>
                <span class="skill-tag">SAS</span>
                <span class="skill-tag">STATA</span>
                <span class="skill-tag">C</span>
                <span class="skill-tag">VBA</span>
                <span class="skill-tag">JavaScript</span>
                <span class="skill-tag">Flask</span>
                <span class="skill-tag">Pandas</span>
                <span class="skill-tag">GeoPandas</span>
                <span class="skill-tag">Folium / Leaflet.js</span>
                <span class="skill-tag">Power BI</span>
                <span class="skill-tag">Power Automate</span>
                <span class="skill-tag">HTML / CSS</span>
                <span class="skill-tag">Git</span>
                <span class="skill-tag">REST APIs</span>
                <span class="skill-tag">Office Suite</span>
            </div>
        </div>

        <!-- Languages -->
        <div class="section">
            <div class="section-title"><span class="icon">&#127760;</span> Languages</div>
            <div class="lang-grid">
                <div class="lang-item">&#127468;&#127463; English <span class="lang-level">(Fluent)</span></div>
                <div class="lang-item">&#127479;&#127482; Russian <span class="lang-level">(Fluent)</span></div>
                <div class="lang-item">&#127465;&#127466; German <span class="lang-level">(Basic)</span></div>
            </div>
        </div>

        <!-- Interests & Charity -->
        <div class="section">
            <div class="section-title"><span class="icon">&#9968;</span> Interests &amp; Charity</div>
            <div class="interests-grid">
                <span class="interest-tag">&#129495; Climbing</span>
                <span class="interest-tag">&#9968; Mountaineering</span>
                <span class="interest-tag">&#128200; Finance</span>
                <span class="interest-tag">&#129302; Machine Learning</span>
                <span class="interest-tag">&#128202; Royal Statistical Society</span>
                <span class="interest-tag">&#128642; Transportation</span>
            </div>
        </div>

        <!-- Scottish Railway Project -->
        <div class="section">
            <div class="section-title"><span class="icon">🚆</span> Scottish Railway Map Project</div>
            <a href="/" style="text-decoration: none; color: inherit;">
                <div class="project-highlight">
                    <div class="title">Scotland Railway Map &mdash; Interactive Station &amp; Line Explorer</div>
                    <div class="desc">A full-stack data visualisation application built with <strong>Python</strong>, <strong>Flask</strong>, <strong>Folium</strong> and <strong>Leaflet.js</strong>. Features include:</div>
                </div>
            </a>
            <ul style="margin-top:14px;padding-left:20px;font-size:13px;color:#555;line-height:2;">
                <li>Interactive map with 350+ Scottish railway stations and live train departures/arrivals</li>
                <li>Colour-coded markers by usage volume, punctuality, or cancellation rates</li>
                <li>Interactive sparkline trend charts showing 37 periods of performance data</li>
                <li>Integrated Origin-Destination Matrix (ODM) data for journey analysis</li>
                <li>Station search with instant results and map navigation</li>
                <li>Per-region zoom thresholds for optimal information density</li>
                <li>Real-time data from the National Rail API</li>
            </ul>
        </div>

        <div class="footer-note">Built with &#10084;&#65039; using Python, Flask, Folium &amp; Leaflet.js &bull; &copy; 2026 Aleksejs Belasovs</div>
    </div>
</body>
</html>"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
