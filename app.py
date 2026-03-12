import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide")

st.title("Graph Based Safe Route Optimization Using Traffic and Accident Prediction")

# ---------------------------------------------------
# Sidebar Weights
# ---------------------------------------------------

st.sidebar.header("Route Preference Weights")

alpha = st.sidebar.slider("Distance Weight (α)",0.0,1.0,0.4)
beta = st.sidebar.slider("Traffic Weight (β)",0.0,1.0,0.3)
gamma = st.sidebar.slider("Accident Risk Weight (γ)",0.0,1.0,0.3)

st.sidebar.write(
f"Cost = {alpha:.2f} × Distance + {beta:.2f} × Traffic + {gamma:.2f} × Accident"
)

# ---------------------------------------------------
# Load Sensor Locations
# ---------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("traffic/graph_sensor_locations.csv")
    data["sensor_id"] = data["sensor_id"].astype(int)
    return data

sensor_data = load_data()

# ---------------------------------------------------
# Load Prediction CSVs
# ---------------------------------------------------

traffic_data = pd.read_csv("outputs/traffic_predictions.csv")
accident_data = pd.read_csv("outputs/accident_predictions.csv")

traffic_dict = dict(zip(traffic_data.sensor_id, traffic_data.traffic_score))
accident_dict = dict(zip(accident_data.sensor_id, accident_data.risk))

# ---------------------------------------------------
# Build Graph
# ---------------------------------------------------

@st.cache_resource
def build_graph(data):

    G = nx.Graph()

    for _,row in data.iterrows():
        G.add_node(
            row["sensor_id"],
            pos=(row["latitude"],row["longitude"])
        )

    def dist(a,b,c,d):
        return np.sqrt((a-c)**2 + (b-d)**2)

    threshold = 0.05

    for i,r1 in data.iterrows():
        for j,r2 in data.iterrows():

            if i >= j:
                continue

            d = dist(
                r1.latitude,
                r1.longitude,
                r2.latitude,
                r2.longitude
            )

            if d < threshold:

                traffic_score = traffic_dict.get(r1.sensor_id,0.5)
                accident_score = accident_dict.get(r1.sensor_id,0.5)

                G.add_edge(
                    r1.sensor_id,
                    r2.sensor_id,
                    distance=abs(d),
                    traffic=traffic_score,
                    accident=accident_score
                )

    largest_component = max(nx.connected_components(G), key=len)

    return G.subgraph(largest_component).copy()


G = build_graph(sensor_data)

sensor_ids = sorted(list(G.nodes))

# ---------------------------------------------------
# UI Selection
# ---------------------------------------------------

source = st.selectbox("Source", sensor_ids)
destination = st.selectbox("Destination", sensor_ids)

center = [
sensor_data["latitude"].mean(),
sensor_data["longitude"].mean()
]

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "map" not in st.session_state:

    st.session_state.map = folium.Map(
        location=center,
        zoom_start=11,
        tiles="OpenStreetMap"
    )

if "routes" not in st.session_state:
    st.session_state.routes = None

if "route_generated" not in st.session_state:
    st.session_state.route_generated = False

# ---------------------------------------------------
# Button
# ---------------------------------------------------

if st.button("Find Route"):
    st.session_state.route_generated = True

# ---------------------------------------------------
# Route Calculation
# ---------------------------------------------------

if st.session_state.route_generated:

    if source == destination:
        st.warning("Source and Destination cannot be the same")
        st.stop()

    m = folium.Map(
        location=center,
        zoom_start=11,
        tiles="OpenStreetMap"
    )

    max_distance = max(nx.get_edge_attributes(G,"distance").values())

    for u,v,data in G.edges(data=True):

        data["distance_norm"] = data["distance"] / max_distance

        traffic_cost = data["traffic"]
        accident_cost = data["accident"]

        data["safest_cost"] = accident_cost + data["distance_norm"]*0.3

        data["ai"] = (
            alpha * data["distance_norm"]
            + beta * traffic_cost
            + gamma * accident_cost
        ) + 0.0001

    # ---------------------------------------------------
    # Draw Traffic Roads
    # ---------------------------------------------------

    for u,v,data in G.edges(data=True):

        lat1,lon1 = G.nodes[u]["pos"]
        lat2,lon2 = G.nodes[v]["pos"]

        traffic = data["traffic"]
        accident = data["accident"]

        if accident > 0.8:
            color = "black"
        elif traffic > 0.7:
            color = "red"
        elif traffic > 0.4:
            color = "orange"
        else:
            color = "yellow"

        folium.PolyLine(
            [(lat1,lon1),(lat2,lon2)],
            color=color,
            weight=3,
            opacity=0.6
        ).add_to(m)

    # ---------------------------------------------------
    # Calculate Routes
    # ---------------------------------------------------

    shortest = nx.shortest_path(G,source,destination,weight="distance")
    safest = nx.shortest_path(G,source,destination,weight="safest_cost")
    ai_route = nx.shortest_path(G,source,destination,weight="ai")

    def draw(route,color,weight):

        pts=[]

        for node in route:
            lat,lon = G.nodes[node]["pos"]
            pts.append((lat,lon))

        folium.PolyLine(
            pts,
            color=color,
            weight=weight
        ).add_to(m)

    draw(shortest,"blue",4)
    draw(safest,"green",6)
    draw(ai_route,"purple",8)

    folium.Marker(
        G.nodes[source]["pos"],
        popup="Source",
        icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        G.nodes[destination]["pos"],
        popup="Destination",
        icon=folium.Icon(color="red")
    ).add_to(m)

    st.session_state.map = m
    st.session_state.routes = (shortest,safest,ai_route)

# ---------------------------------------------------
# Display Map
# ---------------------------------------------------

st_folium(
    st.session_state.map,
    width=900,
    height=600
)

# ---------------------------------------------------
# Show Routes
# ---------------------------------------------------

if st.session_state.routes:

    shortest,safest,ai_route = st.session_state.routes

    st.subheader("Route Sensor Scores")

    for node in ai_route:

        traffic_val = traffic_dict.get(node,0)
        accident_val = accident_dict.get(node,0)

        st.write(
            f"Sensor {node} → Traffic: {traffic_val:.2f} | Accident Risk: {accident_val:.2f}"
        )

    st.write("Shortest Route:",shortest)
    st.write("Safest Route:",safest)
    st.write("AI Route:",ai_route)

# ---------------------------------------------------
# Legend
# ---------------------------------------------------

st.markdown("""
### Map Legend

🔴 Heavy Traffic  
🟠 Medium Traffic  
🟡 Low Traffic  

⚫ Accident-Prone Zone  

🔵 Shortest Route  
🟢 Safest Route  
🟣 AI Optimized Route
""")
