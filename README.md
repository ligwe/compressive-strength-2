# Concrete Strength Prediction and Mix Optimization

A Streamlit app using the trained SVR, scikit-learn ANN (MLPRegressor), and XGBoost pipelines from the research notebook.

## Features

- Predict compressive strength from cement, water, sand, granite and curing age.
- Compare predictions from the three trained models.
- View held-out test metrics recorded by the notebook.
- Run the notebook's constrained optimization objective across training-supported curing ages.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload `app.py`, `requirements.txt`, `models.joblib`, and `deployment_config.json` to a GitHub repository. In Streamlit Community Cloud, choose the repository and set `app.py` as the main file.

## Important

Export the model artifacts from a successfully executed version of the notebook. Do not mix model files from different training runs or feature sets. This app is a research demonstration, not a substitute for laboratory validation or professional concrete mix design.
