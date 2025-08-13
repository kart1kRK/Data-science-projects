import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

fastf1.Cache.enable_cache('cache')

# ------------------ DATA FETCHING ------------------
def fetch_f1_data(year, round_number):
    """Fetch qualifying data for a given race."""
    try:
        quali = fastf1.get_session(year, round_number, 'Q')
        quali.load()

        results = quali.results[['DriverNumber', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3']]
        results = results.rename(columns={'FullName': 'Driver'})

        # Convert lap times to seconds
        for col in ['Q1', 'Q2', 'Q3']:
            results[col + '_sec'] = results[col].apply(
                lambda x: x.total_seconds() if pd.notnull(x) else None
            )

        return results
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# ------------------ CLEANING ------------------
def clean_data(df):
    """Ensure times are in seconds and remove rows with missing data."""
    return df.dropna(subset=['Q1_sec', 'Q2_sec', 'Q3_sec'])

# ------------------ PERFORMANCE FACTORS ------------------
def apply_performance_factors(predictions_df):
    """Apply 2025-specific team/driver performance factors."""
    np.random.seed(42)  # For reproducible predictions

    base_time = 89.5  # Base lap time in seconds

    team_factors = {
        'Red Bull Racing': 0.997,
        'Ferrari': 0.998,
        'McLaren': 0.999,
        'Mercedes': 0.999,
        'Aston Martin': 1.001,
        'RB': 1.002,
        'Williams': 1.003,
        'Haas F1 Team': 1.004,
        'Kick Sauber': 1.004,
        'Alpine': 1.005,
    }

    driver_factors = {
        'Max Verstappen': 0.998,
        'Charles Leclerc': 0.999,
        'Carlos Sainz': 0.999,
        'Lando Norris': 0.999,
        'Oscar Piastri': 1.000,
        'Sergio Perez': 1.000,
        'Lewis Hamilton': 1.000,
        'George Russell': 1.000,
        'Fernando Alonso': 1.000,
        'Lance Stroll': 1.001,
        'Alexander Albon': 1.001,  # FIXED name to match dataset
        'Daniel Ricciardo': 1.001,
        'Yuki Tsunoda': 1.002,
        'Valtteri Bottas': 1.002,
        'Zhou Guanyu': 1.003,
        'Kevin Magnussen': 1.003,
        'Nico Hulkenberg': 1.003,
        'Logan Sargeant': 1.004,
        'Pierre Gasly': 1.004,
        'Esteban Ocon': 1.004,
    }

    for idx, row in predictions_df.iterrows():
        team_factor = team_factors.get(row['Team'], 1.005)
        driver_factor = driver_factors.get(row['Driver'], 1.002)
        base_prediction = base_time * team_factor * driver_factor
        random_variation = np.random.uniform(-0.1, 0.1)
        predictions_df.loc[idx, 'Predicted_Q3'] = round(base_prediction + random_variation, 3)

    return predictions_df

# ------------------ JAPANESE GP PREDICTION ------------------
def predict_japanese_gp():
    """Predict 2025 Japanese GP qualifying results."""
    driver_teams = {
        'Max Verstappen': 'Red Bull Racing',
        'Sergio Perez': 'Red Bull Racing',
        'Charles Leclerc': 'Ferrari',
        'Carlos Sainz': 'Ferrari',
        'Lewis Hamilton': 'Mercedes',
        'George Russell': 'Mercedes',
        'Lando Norris': 'McLaren',
        'Oscar Piastri': 'McLaren',
        'Fernando Alonso': 'Aston Martin',
        'Lance Stroll': 'Aston Martin',
        'Daniel Ricciardo': 'RB',
        'Yuki Tsunoda': 'RB',
        'Alexander Albon': 'Williams',
        'Logan Sargeant': 'Williams',
        'Valtteri Bottas': 'Kick Sauber',
        'Zhou Guanyu': 'Kick Sauber',
        'Kevin Magnussen': 'Haas F1 Team',
        'Nico Hulkenberg': 'Haas F1 Team',
        'Pierre Gasly': 'Alpine',
        'Esteban Ocon': 'Alpine'
    }

    results_df = pd.DataFrame(list(driver_teams.items()), columns=['Driver', 'Team'])
    results_df = apply_performance_factors(results_df)
    results_df = results_df.sort_values('Predicted_Q3')

    print("\nJapanese GP 2025 Qualifying Predictions:")
    print("=" * 100)
    print(f"{'Pos':<5}{'Driver':<20}{'Team':<25}{'Predicted Q3':<15}")
    print("-" * 100)

    for i, row in enumerate(results_df.itertuples(), start=1):
        print(f"{i:<5}{row.Driver:<20}{row.Team:<25}{row.Predicted_Q3:.3f}s")

# ------------------ MAIN EXECUTION ------------------
if __name__ == "__main__":
    print("Fetching recent race data...")

    all_data = []
    current_year = 2025

    # Fetch first 4 races of 2025
    for round_num in range(1, 5):
        df = fetch_f1_data(current_year, round_num)
        if df is not None:
            df = clean_data(df)
            df['Year'] = current_year
            df['Round'] = round_num
            all_data.append(df)

    # Fetch Japanese GP 2024
    df_2024 = fetch_f1_data(2024, 4)
    if df_2024 is not None:
        df_2024 = clean_data(df_2024)
        df_2024['Year'] = 2024
        df_2024['Round'] = 4
        all_data.append(df_2024)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Prepare model training data
        imputer = SimpleImputer(strategy='median')
        X = combined_df[['Q1_sec', 'Q2_sec']]
        y = combined_df['Q3_sec']

        X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        y_clean = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).ravel())

        # Train model
        model = LinearRegression()
        model.fit(X_clean, y_clean)

        # Predict Japanese GP
        predict_japanese_gp()

        # Evaluate model
        y_pred = model.predict(X_clean)
        mae = mean_absolute_error(y_clean, y_pred)
        r2 = r2_score(y_clean, y_pred)
        print("\nModel Performance Metrics:")
        print(f"Mean Absolute Error: {mae:.2f} seconds")
        print(f"R^2 Score: {r2:.2f}")
    else:
        print("No data available to train model.")
