# Silica concentrate prediction: an honest look

Predicting % silica in iron ore flotation concentrate from plant data.

The first version scored R² 0.998, but that came from data leakage: `% Iron Concentrate` (measured in the same lab assay as silica) was used as an input, and rows from the same lab hour ended up in both training and test sets. After fixing this (hourly averaging, time-based split), sensors alone explain about 7% of the variation (R² 0.07), and adding last hour's assay gives R² 0.59, level with simply repeating the last result (0.61).

**Live app:** add your Streamlit link here

## Results

| Model | R² | RMSE (% silica) |
|---|---|---|
| Predict the average | -0.001 | 1.145 |
| Random Forest, sensors only | 0.071 | 1.103 |
| Repeat last hour's assay | 0.612 | 0.713 |
| Random Forest, sensors + last hour | 0.592 | 0.731 |

## Files
- `app.py`: Streamlit app exploring the corrected results
- `silica_results.csv`, `rf_importances.csv`: result files the app reads
- `minig-data.ipynb`: full analysis, original and corrected versions

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

Data: public iron ore flotation plant dataset (Kaggle, March–September 2017).
