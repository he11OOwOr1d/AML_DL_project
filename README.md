# Skin Cancer Detection Project

## Overview
This repository contains the Phase 1 implementation for the population-scale skin cancer screening machine learning project. The objective is to build binary classifiers prioritizing sensitivity, moving from baseline metadata models to advanced computer vision+ML pipelines.

## Project Structure
- `archive/` : Directory containing the HAM10000 dataset (raw images and metadata).
- `notebooks/` :
  - `01_EDA_and_Data_Quality.ipynb`: Data exploration, dataset quality checks, and visualization.
  - `02_Baseline_ML_Metadata.ipynb`: Baseline Risk prediction modeling using patient metadata.
  - `03_Advanced_ML_CV.ipynb`: Advanced Machine Learning pipeline extracting explicit CV features.
- `report/` : LaTeX files for the final project report.

## Setup Instructions
1. Ensure you have Python 3.9+ installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Server:
   ```bash
   jupyter notebook
   ```

## Model Pipeline
The execution should ideally be done in the numbered order within the `notebooks` folder.
