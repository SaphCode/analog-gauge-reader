from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
import io
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from streamlit_autorefresh import st_autorefresh
import numpy as np

import firebase_admin
from firebase_admin import firestore
from google.cloud import storage



# Page configuration
st.set_page_config(page_title="Analog Gauge Monitor", page_icon="📊", layout="wide")

# Add custom CSS for pulsing animation on the latest reading marker
st.markdown("""
<style>
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Target the latest reading marker in Plotly */
.js-plotly-plot .scatterlayer .trace:last-child .point {
    animation: pulse 2s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

# Initialize Firestore and Storage clients
# Make sure GOOGLE_APPLICATION_CREDENTIALS environment variable is set
# or you're running on GCP with appropriate permissions
@st.cache_resource
def get_firestore_client():
    """Initialize and cache the Firestore client"""
    # Check if Firebase app is already initialized
    if not firebase_admin._apps:
        # Application Default credentials are automatically created.
        firebase_admin.initialize_app()

    db = firestore.client()
    return db

@st.cache_resource
def get_storage_client():
    """Initialize and cache the Cloud Storage client"""
    return storage.Client()

def download_image_from_gcs(bucket_name, image_path):
    """
    Download an image from Google Cloud Storage

    Args:
        bucket_name (str): Name of the GCS bucket
        image_path (str): Path to the image within the bucket

    Returns:
        PIL.Image: The downloaded image, or None if error
    """
    try:
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(image_path)

        # Download image data
        image_data = blob.download_as_bytes()

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        st.error(f"Error downloading image: {e}")
        return None

def get_unique_devices(collection_name='readings'):
    """
    Fetch unique device IDs from Firestore

    Args:
        collection_name (str): Name of the Firestore collection

    Returns:
        list: List of unique device IDs
    """
    try:
        db = get_firestore_client()
        collection_ref = db.collection(collection_name)

        # Get all documents and extract unique device_ids
        docs = collection_ref.stream()
        device_ids = set()

        for doc in docs:
            doc_data = doc.to_dict()
            if 'device_id' in doc_data:
                device_ids.add(doc_data['device_id'])

        return sorted(list(device_ids))

    except Exception as e:
        st.error(f"Error fetching device IDs: {e}")
        return []

def get_total_records(collection_name='readings', device_id=None):
    """
    Get total count of records for a device

    Args:
        collection_name (str): Name of the Firestore collection
        device_id (str): Optional device ID to filter by

    Returns:
        int: Total number of records
    """
    try:
        db = get_firestore_client()
        collection_ref = db.collection(collection_name)

        # Apply device filter if specified
        if device_id and device_id != "All Devices":
            collection_ref = collection_ref.where('device_id', '==', device_id)

        # Count documents
        docs = list(collection_ref.stream())
        return len(docs)

    except Exception as e:
        st.error(f"Error counting records: {e}")
        return 100  # Default fallback

def fetch_gauge_readings(collection_name='readings', limit=100, device_id=None):
    """
    Fetch gauge readings from Firestore

    Args:
        collection_name (str): Name of the Firestore collection
        limit (int): Maximum number of documents to fetch
        device_id (str): Optional device ID to filter by

    Returns:
        pd.DataFrame: DataFrame with gauge readings
    """
    try:
        db = get_firestore_client()
        collection_ref = db.collection(collection_name)

        # Apply device filter if specified
        if device_id and device_id != "All Devices":
            collection_ref = collection_ref.where('device_id', '==', device_id)

        # Fetch documents ordered by timestamp (most recent first)
        docs = collection_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit).stream()

        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id  # Add document ID
            data.append(doc_data)

        if data:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error fetching data from Firestore: {e}")
        return pd.DataFrame()

# Main app
st.title("📊 Analog Gauge Monitor")

# Auto-refresh every 30 seconds (30000 milliseconds)
refresh_count = st_autorefresh(interval=30000, key="datarefresh")

# Show last refresh time
last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Last refreshed: {last_refresh}")

# Get available devices
devices = get_unique_devices()
device_options = ["All Devices"] + devices

# Controls
col1, col2 = st.columns([1, 2])
with col1:
    selected_device = st.selectbox("Device", device_options)
with col2:
    # Get total records for selected device to set slider max
    total_records = get_total_records(collection_name='readings', device_id=selected_device)
    max_records = max(10, total_records)  # Ensure at least 10
    num_records = st.slider("Number of Records", min_value=5, max_value=max_records, value=min(100, max_records))

# Fetch data based on selections
df = fetch_gauge_readings(collection_name='readings', limit=num_records, device_id=selected_device)

if not df.empty:
    # Get unit from first reading (assuming all readings have same unit)
    unit = df['unit'].iloc[0] if 'unit' in df.columns and len(df) > 0 else ""

    # Ensure timestamp is datetime (Firestore should return it as datetime, but handle strings from old data)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create tabs
    tab1, tab2 = st.tabs(["📈 Chart", "📋 Data"])

    with tab1:
        st.subheader(f"Measurements Over Time ({unit})")

        # Prepare data for chart (reverse order so oldest is first for proper time series)
        chart_df = df.sort_values('timestamp').copy()

        # Display chart and image side by side
        if 'timestamp' in chart_df.columns and 'measurement' in chart_df.columns:
            col_chart, col_image = st.columns([4, 1])

            with col_chart:
                # Create Plotly chart
                fig = go.Figure()

                # Add main line chart
                fig.add_trace(go.Scatter(
                    x=chart_df['timestamp'],
                    y=chart_df['measurement'],
                    mode='lines',
                    name='Measurement',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='%{y:.2f}<extra></extra>'
                ))

                # Add marker for the latest point
                latest_timestamp = df['timestamp'].iloc[0]
                latest_value = df['measurement'].iloc[0]

                fig.add_trace(go.Scatter(
                    x=[latest_timestamp],
                    y=[latest_value],
                    mode='markers',
                    name='Latest Reading',
                    marker=dict(color='red', size=12, symbol='circle'),
                    hovertemplate='Latest: %{y:.2f}<extra></extra>'
                ))

                # Update layout
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title=f"Measurement ({unit})",
                    hovermode='x unified',
                    showlegend=True,
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            with col_image:
                # Display latest image
                if 'image_path' in df.columns and len(df) > 0:
                    latest_image_path = df['image_path'].iloc[0]
                    bucket_name = "analog-gauge-images"  # From cloud function

                    image = download_image_from_gcs(bucket_name, latest_image_path)

                    if image:
                        st.image(image, caption="Latest Image", width='stretch')
                        st.markdown(f"**Reading: {latest_value:.2f} {unit}**")
                    else:
                        st.warning("Could not load image")
                else:
                    st.info("No image available")

        else:
            st.warning("Data missing required 'timestamp' or 'measurement' columns")

    with tab2:
        st.subheader("Raw Data")
        # Round measurement column to 2 decimals for display
        display_df = df.copy()
        if 'measurement' in display_df.columns:
            display_df['measurement'] = display_df['measurement'].round(2)
        st.dataframe(display_df, width='stretch')

else:
    st.warning("No data found in Firestore.")
    st.info("💡 Tip: Check that your Firebase credentials are configured correctly and the 'readings' collection contains data.")
