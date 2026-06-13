# Assam Flood Relief Dashboard: Predictive Analytics & NGO Logistics

**An automated, micro-level disaster prediction and resource allocation command center.**

## Overview
This project transforms raw topographical and meteorological data into actionable, non-technical dispatch plans for NGO relief workers. Instead of guessing where floods will happen or how many supplies to send, this AI-powered dashboard calculates the exact hydrological threat for a specific region and automatically drafts a logistics budget and deployment strategy.

## Core Features
* **Location-First Architecture:** Users simply select a known District and City. The system automatically fetches the exact GPS coordinates, elevation, slope, historical water occurrence, and population density.
* **Physics-Based ML Engine:** A Random Forest model trained not just on raw data, but on a custom "Hydrological Risk Equation" that understands water physics (e.g., steep slopes drain water, saturated soils hold it).
* **API Telemetry Simulation:** A dual-mode data pipeline that simulates live connections to OpenWeatherMap (48-hr rainfall forecasts) and NASA SMAP satellites (soil moisture saturation), alongside a Manual Override mode for stress-testing.
* **Automated Logistics Dispatch:** Automatically calculates the impact zone (1.5km radius) and itemizes exact supply needs (Water Kits, Med Kits, Tents, Boats, Personnel) based on standardized NGO heuristics.

---

## System Architecture & Engineering

### 1. The Data Pipeline (`build_locations.py`)
Raw GPS coordinates are useless to an NGO worker on the ground. We built a reverse-geocoding script using the offline `reverse_geocoder` library. It scans the raw coordinate dataset, snaps every point to the nearest recognized city/town (population > 1,000), and creates the structured `Location_Dictionary.csv`.

### 2. The Machine Learning Brain (`train_pipeline.py`)
Because historical datasets often lack perfectly accurate binary "Flood/No Flood" labels, we engineered our own target variable using a **Weighted Hydrological Formula**:
* `+` Heavy Rain & High Soil Moisture increase risk.
* `-` Steep Slopes & High Elevations decrease risk (natural drainage).
* `+` Gaussian Noise simulates the unpredictable nature of real-world weather.

We designated the top 20% most dangerous days as historical flood events and trained a **Random Forest Classifier** (`flood_brain.pkl`). By forcing the model to learn actual fluid mechanics rather than just memorizing data points, it achieved a highly realistic **Accuracy of 92.83%** with a **Model Error Rate (MAPE) of just 7.17%**.

### 3. The Command Center Dashboard (`app.py`)
Built with Streamlit, the UI is split into two halves:
* **⚙️ Mission Control (Left Sidebar):** For parameter configuration and telemetry selection.
* **➡️ Action Dashboard (Main Screen):** Outputs a readable 3-step action plan: Threat Assessment, Impact Zone Demographics, and the Recommended NGO Dispatch Plan.

---

## The Logistics & Financial Math (Assumptions)
To generate the final dispatch plan, the system uses the following operational heuristics:
* **Impact Radius:** 1.5 km circle around the target coordinate.
* **Demographics:** Assumes 30% children and 15% elderly for specialized medical calculations.
* **Supply Ratios:** * 1 Water Filtration Kit per 4 people.
    * 1 Pediatric/Trauma Med Kit per 50 vulnerable individuals.
    * 1 Emergency Tent per 8 people.
    * 1 Evacuation Boat per 100 people (with 3 rescue operators per boat).
* **Cost Estimates (INR):** Water Kits (₹500), Med Kits (₹1,500), Tents (₹3,000), Boats (₹10,000).

---

## Installation, Setup & Structure

You can deploy the environment, launch the Command Center, and verify your repository structure by referencing this single script block:

```bash
# ==========================================
# 1. SETUP INSTRUCTIONS
# ==========================================
# Clone the repository and enter the folder
git clone <your-repo-link>
cd <repository-folder>

# Create the virtual environment
python -m venv assam_env

# Activate the virtual environment
# --> FOR WINDOWS:
.\assam_env\Scripts\activate
# --> FOR MAC/LINUX (Uncomment the line below instead):
# source assam_env/bin/activate

# Install the required dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app.py

# ==========================================
# 2. REPOSITORY STRUCTURE REFERENCE
# ==========================================
# Your final folder should look exactly like this:
#
# ├── README.md
# ├── requirements.txt
# ├── app.py                                   # Main Streamlit Dashboard
# ├── train_pipeline.py                        # ML Model Training Script
# ├── build_locations.py                       # Geocoding Data Engineering Script
# ├── flood_brain.pkl                          # Trained Random Forest Model
# ├── Location_Dictionary.csv                  # Offline City Database
# └── Final_Master_Flood_Dataset_Corrected.csv # Master Topographical Dataset
