
# Menstrual Phase Predictor

This project presents a Streamlit-based application for estimating menstrual cycle phases using physiological sensor data, including body temperature, heart rate (BPM), and blood oxygen saturation (SPO2). The model applies unsupervised machine learning (KMeans clustering) to group patterns in the input data and maps them to probable menstrual phases: Follicular, Ovulatory, and Luteal.

## Features

- Predicts menstrual phases based on real-time user input.
- Accepts the following input signals:
  - Body Temperature (in °C)
  - Heart Rate (BPM)
  - Blood Oxygen Saturation (SPO2 %)
- Computes:
  - Rolling mean for Body Temperature and BPM (last 5 values)
  - Temperature Amplitude (max - min in last 5 readings)
- Provides a session-state buffer to visualize recent data trends.
- Logs all predictions with timestamps to a CSV file.

## Application Overview

This application is designed for educational and experimental use. It may serve as a prototype for future work in women's health analytics, cycle awareness, and digital health solutions.

## Model Details

- Algorithm: KMeans clustering
- Preprocessing: StandardScaler (scikit-learn)
- Input Features:
  - BodyTemp, BPM, SPO2
  - Rolling Mean of BodyTemp and BPM
  - Temperature Amplitude
- Output: Cluster label mapped to a menstrual phase using a predefined mapping dictionary

## Installation Instructions

### Clone the repository

```bash
git clone https://github.com/kurioux/fertility_tracker.git
cd fertility_tracker
