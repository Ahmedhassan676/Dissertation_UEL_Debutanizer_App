
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Concatenate, TimeDistributed,
    AdditiveAttention, Conv1D, BatchNormalization, LeakyReLU
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_recall_fscore_support
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import ast
import seaborn as sns
import pydmd
import statsmodels.api as sm
import plotly.express as px
from sklearn.decomposition import PCA
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import nevergrad as ng
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="ML App",
    page_icon="🌊",
    layout="wide"
)

# --- Model & Data Functions (Cached for performance) ---

@st.cache_data
def load_data(uploaded_file):
    """Loads data from an uploaded CSV file."""
    df = pd.read_csv(uploaded_file)
    # Attempt to convert date/time columns to datetime objects
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass # Ignore columns that can't be converted
    return df

@st.cache_data
def create_sequences(input_data, target_data, n_in, n_out):
    """Converts time series data into supervised learning sequences."""
    X, y = [], []
    for i in range(len(input_data)):
        end_ix = i + n_in
        out_end_ix = end_ix + n_out
        if out_end_ix > len(input_data):
            break
        seq_x = input_data[i:end_ix, :]
        seq_y = target_data[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def build_simple_lstm_model(n_timesteps_in, n_input_features, n_timesteps_out, n_target_features, latent_dim, num_layers, learning_rate):
    """Builds a simple stacked LSTM model."""
    model_input = Input(shape=(n_timesteps_in, n_input_features))
    x = model_input
    # Add specified number of LSTM layers
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        x = LSTM(latent_dim, return_sequences=return_sequences, name=f'lstm_layer_{i+1}')(x)

    x = Dense(n_timesteps_out * n_target_features)(x)
    model_output = tf.keras.layers.Reshape((n_timesteps_out, n_target_features))(x)
    
    model = Model(model_input, model_output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

def build_enc_dec_attention_model(n_input_features, n_target_features, latent_dim, num_encoder_layers, num_decoder_layers, learning_rate):
    """Builds an LSTM Encoder-Decoder model with Attention."""
    # Encoder
    encoder_inputs = Input(shape=(None, n_input_features), name='encoder_inputs')
    x = encoder_inputs
    encoder_states = []
    for i in range(num_encoder_layers):
        return_sequences = True  # All layers must return sequences for stacking/attention
        return_state = (i == num_encoder_layers - 1)  # Only the last layer's state is needed
        encoder_lstm = LSTM(latent_dim, return_sequences=return_sequences, return_state=return_state, name=f'encoder_lstm_{i+1}')
        if return_state:
            x, state_h, state_c = encoder_lstm(x)
            encoder_states = [state_h, state_c]
        else:
            x = encoder_lstm(x)
    encoder_outputs = x

    # Decoder
    decoder_inputs = Input(shape=(None, n_target_features), name='decoder_inputs')
    x = decoder_inputs
    decoder_states = encoder_states
    for i in range(num_decoder_layers):
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name=f'decoder_lstm_{i+1}')
        # Use encoder states for the first decoder layer, then pass decoder states
        x, state_h, state_c = decoder_lstm(x, initial_state=decoder_states)
        decoder_states = [state_h, state_c]  # Update decoder states for the next iteration
    decoder_outputs = x

    # Attention
    attention_layer = AdditiveAttention(name='attention_layer')
    attention_result = attention_layer([decoder_outputs, encoder_outputs])
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])

    # Output
    output_layer = TimeDistributed(Dense(n_target_features, activation='linear'), name='output_layer')
    outputs = output_layer(decoder_concat_input)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

def build_enc_dec_attention_cnn_model(n_input_features, n_target_features, latent_dim, num_encoder_layers, num_decoder_layers, learning_rate, conv_filters, conv_kernel_size):
    """Builds the LSTM-Attention-CNN model."""
    # Encoder
    encoder_inputs = Input(shape=(None, n_input_features), name='encoder_inputs')
    x = encoder_inputs
    encoder_states = []
    for i in range(num_encoder_layers):
        return_sequences = True
        return_state = (i == num_encoder_layers - 1)
        encoder_lstm = LSTM(latent_dim, return_sequences=return_sequences, return_state=return_state, name=f'encoder_lstm_{i+1}')
        if return_state:
            x, state_h, state_c = encoder_lstm(x)
            encoder_states = [state_h, state_c]
        else:
            x = encoder_lstm(x)
    encoder_outputs = x

    # Decoder
    decoder_inputs = Input(shape=(None, n_target_features), name='decoder_inputs')
    x = decoder_inputs
    decoder_states = encoder_states
    for i in range(num_decoder_layers):
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name=f'decoder_lstm_{i+1}')
        x, state_h, state_c = decoder_lstm(x, initial_state=decoder_states)
        decoder_states = [state_h, state_c]
    decoder_outputs = x

    # Attention
    attention_layer = AdditiveAttention(name='attention_layer')
    attention_result = attention_layer([decoder_outputs, encoder_outputs])
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])

    # Intermediate & CNN
    intermediate_dense = TimeDistributed(Dense(latent_dim), name='intermediate_dense')(decoder_concat_input)
    conv1d_layer = Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, padding='causal', name='conv1d_layer')(intermediate_dense)
    conv1d_layer = BatchNormalization()(conv1d_layer)
    conv1d_layer = LeakyReLU()(conv1d_layer)

    # Output
    output_layer = TimeDistributed(Dense(n_target_features, activation='linear'), name='output_layer')
    outputs = output_layer(conv1d_layer)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

def make_lagged_df(df, lag, columns_to_drop, target_col, target_lag_n=0,show_lag_0 = False):
    """Creates a lagged dataframe for supervised learning."""
    # 1. Lag all features (predictors) by 'lag'
    df_copy = df.drop(columns=columns_to_drop, errors='ignore')
    lagged = df_copy.shift(lag).add_suffix(f"_lag{lag}")
    # We don't want the lagged version of the target as a predictor from this step
    lagged = lagged.drop(columns=[f"{target_col}_lag{lag}"], errors="ignore")

    # 2. Create a new DataFrame for the target and autoregressive features
    target_df = pd.DataFrame(index=df.index)
    target_df[target_col] = df[target_col] # The current target value (y)
    if target_lag_n > 0:
        for i in range(1, target_lag_n + 1):
            target_df[f'{target_col}_lag{i}'] = df[target_col].shift(i)
    # 3. Combine lagged predictors with the target and autoregressive feature
    if show_lag_0:
        st.dataframe(pd.concat([lagged, target_df], axis=1).dropna().head())
    return pd.concat([lagged, target_df], axis=1).dropna()

def make_multi_lagged_df(df, predictors, targets, input_lags=0, target_lags=0):
    """
    Creates a dataframe with multiple lags for both predictor and target variables.
    - input_lags: Number of past time steps for predictor variables (e.g., 2 means t-1, t-2).
    - target_lags: Number of past time steps for target variables to be used as features.
    """
    # Start with current targets and current predictors
    lagged_dfs = [df[targets], df[predictors]]
    # Create lagged predictors
    for i in range(1, input_lags + 1):
        lagged_dfs.append(df[predictors].shift(i).add_suffix(f'_lag{i}'))
    # Create lagged targets (autoregressive features)
    for i in range(1, target_lags + 1):
        lagged_dfs.append(df[targets].shift(i).add_suffix(f'_lag{i}'))

    combined = pd.concat(lagged_dfs, axis=1)
    return combined.dropna()

def make_multi_lagged_df_deprecated(df, lag, predictors, targets):
    """
    Creates a lagged dataframe for multi-target supervised learning.
    """
    if lag == 0:
        # Ensure only specified predictors and targets are returned
        return df[predictors + targets].copy()
    
    # Lag only the predictor columns
    lagged_predictors = df[predictors].shift(lag).add_suffix(f"_lag{lag}")
    # Combine lagged predictors with current targets
    combined = pd.concat([lagged_predictors, df[targets]], axis=1)
    return combined.dropna()


def evaluate_lagged_models(
    df, columns_to_drop, model_builder, max_lag=23, test_size=0.3, param_grid=None, target_col="C4 content", target_lag_n=0
):
    """
    Trains and evaluates models with lagged datasets for different lags.
    Supports both regression and classification depending on target dtype.
    """
    results = []
    models = {}
    for lag in range(max_lag + 1):
        df_lagged = make_lagged_df(df, lag, columns_to_drop, target_col, target_lag_n,show_lag_0= (max_lag==0))

        X = df_lagged.drop(columns=[target_col])
        y = df_lagged[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y if isinstance(y.dtype, CategoricalDtype) else None
        )
        if param_grid is not None:
            grid_search = GridSearchCV(
                model_builder(), param_grid,
                cv=3, scoring="accuracy" if y.dtype.name=="category" else "r2", n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model = model_builder()
            model.fit(X_train, y_train)
            best_params = None
        models[lag] = model
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Classification vs Regression
        if hasattr(model, "predict_proba"):  # classifier
            acc_train = accuracy_score(y_train, y_train_pred)
            acc_test  = accuracy_score(y_test, y_test_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="weighted", zero_division=0)

            results.append({
                "Lag": lag, 'Parameters':best_params, "Accuracy Train": acc_train,
                "Accuracy Test": acc_test, "Precision": precision, "Recall": recall, "F1": f1
            })
        else:  # regressor
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2  = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse  = np.sqrt(mean_squared_error(y_test, y_test_pred))

            results.append({
                "Lag": lag, 'Parameters':best_params, "Train R²": train_r2, "Test R²": test_r2,
                "Train RMSE": train_rmse, "Test RMSE": test_rmse
            })

    results_df = pd.DataFrame(results)
    is_classifier = (isinstance(df[target_col].dtype, CategoricalDtype) or
                     df[target_col].dtype == object or
                     pd.api.types.is_integer_dtype(df[target_col]))

    sort_col = "Accuracy Test" if is_classifier else "Test R²"
    results_df = results_df.sort_values(sort_col, ascending=False)
    
    best_lag = results_df.iloc[0]["Lag"]
    best_model = models[best_lag]
    return results_df, best_model

def evaluate_multi_lagged_models(
    df, predictors, targets, model_builder, columns_to_drop=None,
    max_lag=23, param_grid=None
):
    """
    Trains and evaluates multi-output regression models with lagged datasets for different lags.
    """

    if columns_to_drop is None:
        columns_to_drop = []

    results = []
    models = {}

    for lag in range(max_lag + 1):
        df_lagged = make_multi_lagged_df_deprecated(df, lag, predictors, targets)

        X = df_lagged.drop(columns=targets)
        y = df_lagged[targets]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        best_params_dict = {}
        # Multi-output wrapper if needed
        base_model = model_builder()
        if y.shape[1] > 1:
            if param_grid is not None:
                # For MultiOutputRegressor with GridSearch, we need to prefix the parameters
                prefixed_param_grid = {f'estimator__{key}': value for key, value in param_grid.items()}
                model_to_tune = MultiOutputRegressor(base_model)
                grid_search = GridSearchCV(
                    model_to_tune, prefixed_param_grid, cv=3, scoring="r2", n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                best_params_dict[lag] = grid_search.best_params_
            else:
                model = MultiOutputRegressor(base_model)
                model.fit(X_train, y_train)
                best_params_dict[lag] = "No grid search performed"
        else: # Single output (delegates to the other function's logic)
            if param_grid is not None:
                grid_search = GridSearchCV(
                    base_model, param_grid, cv=3, scoring="r2", n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                best_params_dict[lag] = grid_search.best_params_ 
            else:
                model = base_model
                model.fit(X_train, y_train)
                best_params_dict[lag] = "No grid search performed"

        models[lag] = model

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics per target
        train_r2 = [r2_score(y_train.iloc[:, i], y_train_pred[:, i]) for i in range(y.shape[1])]
        test_r2  = [r2_score(y_test.iloc[:, i], y_test_pred[:, i]) for i in range(y.shape[1])]
        train_rmse = [np.sqrt(mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i])) for i in range(y.shape[1])]
        test_rmse  = [np.sqrt(mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i])) for i in range(y.shape[1])]

        results.append({
            "Lag": lag,
            **{f"Train R² {col}": r2 for col, r2 in zip(targets, train_r2)},
            **{f"Test R² {col}": r2 for col, r2 in zip(targets, test_r2)},
            **{f"Train RMSE {col}": rmse for col, rmse in zip(targets, train_rmse)},
            **{f"Test RMSE {col}": rmse for col, rmse in zip(targets, test_rmse)},
            "Mean Test R²": np.mean(test_r2),
            "Mean Test RMSE": np.mean(test_rmse),
            "Best Parameters": best_params_dict.get(lag, "N/A")
        })

    results_df = pd.DataFrame(results).sort_values("Mean Test R²", ascending=False)
    

    best_lag = results_df.iloc[0]["Lag"]
    best_model = models[best_lag]

    return results_df, best_model

def plot_time_series(y_true, y_pred, target_feature, title):
    """Plots actual vs. predicted values as a time series."""
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(y_true, label='Actual Values', color='blue', alpha=0.7)
    ax.plot(y_pred, label='Predicted Values', color='red', linestyle='--', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel(target_feature)
    ax.legend()
    ax.grid(True)
    return fig

def plot_model_metrics_enhanced(df):
    """
    Enhanced version that creates two subplots showing R² and RMSE metrics 
    with markers for optimal performance points.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing model performance metrics
    """
    # Sort the dataframe by Lag to ensure proper ordering
    df = df.sort_values('Lag').reset_index(drop=True)
    
    # Find optimal points
    optimal_r2_lag = df.loc[df['Test R²'].idxmax(), 'Lag']
    optimal_rmse_lag = df.loc[df['Test RMSE'].idxmin(), 'Lag']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: R² values
    ax1.plot(df['Lag'], df['Train R²'], 'b-', marker='o', label='Train R²', linewidth=2, alpha=0.8)
    ax1.plot(df['Lag'], df['Test R²'], 'r-', marker='s', label='Test R²', linewidth=2, alpha=0.8)
    
    # Mark optimal R² point
    optimal_r2_idx = df['Test R²'].idxmax()
    ax1.scatter(df.loc[optimal_r2_idx, 'Lag'], df.loc[optimal_r2_idx, 'Test R²'], 
               s=150, c='red', edgecolors='black', zorder=5, 
               label=f'Best Test R² (Lag {optimal_r2_lag})')
    
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score vs Lag')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RMSE values
    ax2.plot(df['Lag'], df['Train RMSE'], 'b-', marker='o', label='Train RMSE', linewidth=2, alpha=0.8)
    ax2.plot(df['Lag'], df['Test RMSE'], 'r-', marker='s', label='Test RMSE', linewidth=2, alpha=0.8)
    
    # Mark optimal RMSE point
    optimal_rmse_idx = df['Test RMSE'].idxmin()
    ax2.scatter(df.loc[optimal_rmse_idx, 'Lag'], df.loc[optimal_rmse_idx, 'Test RMSE'], 
               s=150, c='red', edgecolors='black', zorder=5, 
               label=f'Best Test RMSE (Lag {optimal_rmse_lag})')
    
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE vs Lag')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def evaluate_regression(model, df, columns_to_drop, lag, target_name="Target", test_size=0.3,target_lag_n=0):
    """
    Evaluate a regression model with metrics and plots in a single figure.
    Returns the figure object for display in Streamlit.
    """
    df_lagged = make_lagged_df(df, lag, columns_to_drop, target_name, target_lag_n=target_lag_n,show_lag_0= (max_lag==0))
    X = df_lagged.drop(columns=[target_name])
    y = df_lagged[target_name]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # --- Predictions ---
    y_test_pred = model.predict(X_test)

    # --- Evaluation Metrics ---
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # --- Residuals (test set) ---
    residuals = y_test - y_test_pred

    # --- Combined Plot Layout ---
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.3, wspace=0.3)
    
    # Subplot 1: Predicted vs Actual (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, y_test_pred, alpha=0.6, edgecolor='k')
    ax1.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel(f"Actual {target_name}")
    ax1.set_ylabel(f"Predicted {target_name}")
    ax1.set_title("Predicted vs Actual (Test Set)")
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Residuals (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(residuals, bins=20, kde=True, color='skyblue', edgecolor='k', ax=ax2)
    ax2.axvline(0, color='red', linestyle='--')
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual Distribution (Test Set)")
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Time-series plot (bottom row, spans both columns)
    ax3 = fig.add_subplot(gs[1, :])
    test_indices = range(len(y_test))
    ax3.plot(test_indices, y_test.values, 'o-', label='Actual', alpha=0.7, markersize=4, linewidth=0.5, color='blue')
    ax3.plot(test_indices, y_test_pred, 'x-', label='Predicted', alpha=0.7, markersize=4, linewidth=0.5, color='red')
    ax3.set_ylabel(target_name)
    ax3.set_xlabel('Sample Index (from Test Set)')
    ax3.set_title(f"Actual vs Predicted {target_name} (Test Set Samples)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # --- Full Dataset Time Series Plot ---
    st.subheader(f"Full Dataset Time Series: Actual vs. Predicted")
    st.write("This plot shows the model's predictions across the entire dataset (training and test samples).")
    
    y_full_pred = model.predict(X)
    
    fig_full_ts, ax_full_ts = plt.subplots(figsize=(16, 6))
    full_indices = range(len(y))
    ax_full_ts.plot(full_indices, y.values, 'o-', label='Actual', alpha=0.7, markersize=4, linewidth=0.5, color='blue')
    ax_full_ts.plot(full_indices, y_full_pred, 'x-', label='Predicted', alpha=0.7, markersize=4, linewidth=0.5, color='red')
    ax_full_ts.set_ylabel(target_name)
    ax_full_ts.set_xlabel('Sample Index (Full Dataset)')
    ax_full_ts.set_title(f"Actual vs Predicted {target_name} (Full Dataset)")
    ax_full_ts.legend()
    ax_full_ts.grid(True, alpha=0.3)
    st.pyplot(fig_full_ts)

    # --- Feature Importance ---

    # Extract the final estimator from the pipeline if applicable
    final_estimator = model
    if isinstance(model, Pipeline):
        # Check if it's PCR specifically (PCA followed by a regressor)
        if 'pca' in model.named_steps and 'regressor' in model.named_steps and isinstance(model.named_steps['pca'], PCA):
            st.info("Direct feature importance for original features is not straightforward for Principal Component Regression. The model operates on transformed components.")
            return fig # Exit early for PCR, as direct importance is misleading
        else:
            st.subheader("Feature Importance (Principal Component Regression)")
            # For other pipelines (e.g., SVR with scaler), get the actual estimator
            final_estimator = model.steps[-1][1]

    if hasattr(final_estimator, 'feature_importances_'):
        # For tree-based models (RF, GB, XGB)
        importances = final_estimator.feature_importances_
        feature_names = X.columns
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        if not feature_importance_df.empty:
            fig_fi, ax_fi = plt.subplots(figsize=(10, max(6, len(feature_importance_df) * 0.4))) # Adjust height dynamically
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax_fi)
            ax_fi.set_title("Feature Importance")
            ax_fi.set_xlabel("Importance")
            ax_fi.set_ylabel("Feature")
            plt.tight_layout()
            st.pyplot(fig_fi)
            st.dataframe(feature_importance_df)
        else:
            st.info("No features to display importance for.")

    elif hasattr(final_estimator, 'coef_'):
        # For Linear Regression (and other linear models)
        feature_names = X.columns

        # Use statsmodels to get p-values for linear models
        if isinstance(final_estimator, (LinearRegression,)):
            st.info("Calculating p-values for coefficients using statsmodels OLS.")
            # statsmodels requires an explicit intercept to be added
            X_with_const = sm.add_constant(X)
            ols_model = sm.OLS(y, X_with_const).fit()
            
            # Create a dataframe from the OLS summary
            summary_df = ols_model.summary2().tables[1]
            summary_df = summary_df.rename(columns={'Coef.': 'Coefficient', 'P>|t|': 'p-value'})
            
            # Exclude the constant term from the feature importance plot
            feature_importance_df = summary_df.drop('const', errors='ignore')
            feature_importance_df = feature_importance_df.reset_index().rename(columns={'index': 'Feature'})
            feature_importance_df = feature_importance_df.sort_values(by='Coefficient', ascending=False, key=abs)
            
        else: # For other models with .coef_ but not LinearRegression (e.g., SVR with linear kernel)
            importances = final_estimator.coef_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': importances
            }).sort_values(by='Coefficient', ascending=False, key=abs) # Sort by absolute value for magnitude

        if not feature_importance_df.empty:
            fig_fi, ax_fi = plt.subplots(figsize=(10, max(6, len(feature_importance_df) * 0.4))) # Adjust height dynamically
            sns.barplot(x='Coefficient', y='Feature', data=feature_importance_df, ax=ax_fi)
            ax_fi.set_title("Feature Coefficients (Magnitude indicates importance)")
            ax_fi.set_xlabel("Coefficient Value")
            ax_fi.set_ylabel("Feature")
            plt.tight_layout()
            st.pyplot(fig_fi)
            # Display dataframe with p-values if available
            display_cols = ['Feature', 'Coefficient']
            if 'p-value' in feature_importance_df.columns:
                display_cols.append('p-value')
            st.dataframe(feature_importance_df[display_cols])
        else:
            st.info("No features to display coefficients for.")
    else:
        st.info("Direct feature importance or coefficients are not readily available for this model type. Consider using model-agnostic methods like Permutation Importance.")

    return fig

def evaluate_multi_regression(model, df, predictors, targets, input_lags, target_lags=0, test_size=0.3):
    """
    Evaluate a multi-target regression model with detailed plots for each target.
    """
    df_lagged = make_multi_lagged_df(df, predictors, targets, input_lags=input_lags, target_lags=target_lags)
    X = df_lagged.drop(columns=targets)
    y = df_lagged[targets]

    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    y_test_pred = model.predict(X_test)
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=targets, index=y_test.index)

    # --- Plotting in three columns ---
    n_targets = len(targets)
    n_cols = 3 # Predicted vs Actual, Residuals, Time Series
    fig, axes = plt.subplots(n_targets, n_cols, figsize=(20, 6 * n_targets), squeeze=False)

    for i, target in enumerate(targets):
        y_true_single = y_test[target]
        y_pred_single = y_test_pred_df[target]
        residuals = y_true_single - y_pred_single

        # Plot 1: Predicted vs Actual
        ax1 = axes[i, 0]
        ax1.scatter(y_true_single, y_pred_single, alpha=0.6, edgecolor='k')
        ax1.plot([y_true_single.min(), y_true_single.max()], [y_true_single.min(), y_true_single.max()], 'r--', lw=2)
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title(f"Predicted vs Actual for {target}")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residuals
        ax2 = axes[i, 1]
        sns.histplot(residuals, bins=20, kde=True, ax=ax2, color='skyblue')
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_title(f"Residual Distribution for {target}")
        ax2.set_xlabel("Residuals")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Time Series
        ax3 = axes[i, 2]
        test_indices = range(len(y_true_single))
        ax3.plot(test_indices, y_true_single.values, 'o-', label='Actual', alpha=0.7, markersize=4, linewidth=0.5, color='blue')
        ax3.plot(test_indices, y_pred_single.values, 'x-', label='Predicted', alpha=0.7, markersize=4, linewidth=0.5, color='red')
        ax3.set_ylabel(target)
        ax3.set_xlabel('Sample Index (from Test Set)')
        ax3.set_title(f"Actual vs Predicted Samples for {target}")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Add metrics text to the first plot of each row
        metrics_text = f"Test R²: {r2_score(y_true_single, y_pred_single):.3f}\nTest RMSE: {np.sqrt(mean_squared_error(y_true_single, y_pred_single)):.3f}"
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # --- Feature Importance / Coefficients ---
    # Check for MultiOutputRegressor wrapping a linear model
    if isinstance(model, MultiOutputRegressor):
        feature_names = X.columns

        for i, target in enumerate(targets):
            # Get the specific estimator for this target
            estimator = model.estimators_[i]
            
            # If the estimator is a pipeline, get the final step
            final_estimator = estimator
            if isinstance(estimator, Pipeline):
                final_estimator = estimator.steps[-1][1]

            # --- Case 1: Tree-based models with feature_importances_ ---
            if hasattr(final_estimator, 'feature_importances_'):
                if i == 0: st.subheader("Feature Importance per Target") # Show header only once
                st.write(f"---")
                st.write(f"#### Importance for Target: **{target}**")
                importances = final_estimator.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)

                fig_fi, ax_fi = plt.subplots(figsize=(10, max(6, len(feature_importance_df) * 0.4)))
                sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax_fi)
                ax_fi.set_title(f"Feature Importance for {target}")
                plt.tight_layout()
                st.pyplot(fig_fi)
                st.dataframe(feature_importance_df)

            # --- Case 2: Linear models with coef_ ---
            elif hasattr(final_estimator, 'coef_'):
                if i == 0: st.subheader("Feature Coefficients per Target") # Show header only once
                st.write(f"---")
                st.write(f"#### Coefficients for Target: **{target}**")

                # Use statsmodels to get p-values if it's a LinearRegression
                if isinstance(final_estimator, LinearRegression):
                    y_single = y[target]
                    X_with_const = sm.add_constant(X)
                    ols_model = sm.OLS(y_single, X_with_const).fit()
                    
                    summary_df = ols_model.summary2().tables[1]
                    summary_df = summary_df.rename(columns={'Coef.': 'Coefficient', 'P>|t|': 'p-value'})
                    
                    feature_importance_df = summary_df.drop('const', errors='ignore')
                    feature_importance_df = feature_importance_df.reset_index().rename(columns={'index': 'Feature'})
                    feature_importance_df = feature_importance_df.sort_values(by='Coefficient', ascending=False, key=abs)
                else:
                    # Fallback for other linear models without p-value calculation (e.g., MLP)
                    importances = final_estimator.coef_
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': importances
                    }).sort_values(by='Coefficient', ascending=False, key=abs)

                # Plotting
                fig_fi, ax_fi = plt.subplots(figsize=(10, max(6, len(feature_importance_df) * 0.4)))
                sns.barplot(x='Coefficient', y='Feature', data=feature_importance_df, ax=ax_fi)
                ax_fi.set_title(f"Feature Coefficients for {target}")
                plt.tight_layout()
                st.pyplot(fig_fi)

                # Display DataFrame
                display_cols = ['Feature', 'Coefficient']
                if 'p-value' in feature_importance_df.columns:
                    display_cols.append('p-value')
                st.dataframe(feature_importance_df[display_cols])

    return fig

def optimize_multi_target(
    model,
    df_train: pd.DataFrame,
    predictors: list,
    targets: list,
    target_to_optimize: str,
    optimization_goal: str,
    bounds_percent: float,
    constraints: dict | None = None,
    budget: int = 500,
    optimizer_name: str = "TwoPointsDE",
    fixed_values: dict | None = None,
):
    """
    A generalized multi-target optimizer using nevergrad.
    """
    
    
    fixed_values = fixed_values or {}
    # Correctly identify free predictors. These are the predictors
    # that are NOT in the fixed_values dictionary.
    # The `parametrization` object in nevergrad will only contain these free variables.
    free_predictors = [p for p in predictors if p not in fixed_values.keys()]
    # ---- Initial guess from the last row of the training data ----
    init_row_df = df_train.iloc[-1:]
    init_series = init_row_df[predictors].iloc[0]

    # ---- Build bounds based on a percentage of the initial values ----
    parametrization_dict = {}
    for col in free_predictors:
        initial_val = init_series[col]
        lower_bound = max(0.0, initial_val * (1 - bounds_percent))
        upper_bound = min(1.0, initial_val * (1 + bounds_percent))
        if lower_bound >= upper_bound: # Ensure bounds are valid
            lower_bound = max(0.0, initial_val - 0.05)
            upper_bound = min(1.0, initial_val + 0.05)
        parametrization_dict[col] = ng.p.Scalar(init=initial_val).set_bounds(lower_bound, upper_bound)

    if not parametrization_dict:
        st.warning("No variables are free to be optimized. Please un-fix at least one variable.")
        return None, None, None, None

    parametrization = ng.p.Dict(**parametrization_dict)
    parametrization.random_state = np.random.RandomState(42)

    # ---- Setup Optimizer ----
    optimizer_cls = getattr(ng.optimizers, optimizer_name)
    optimizer = optimizer_cls(parametrization=parametrization, budget=budget, num_workers=1)

    target_to_optimize_idx = targets.index(target_to_optimize)

    # ---- Initial Prediction ----
    initial_input_df = pd.DataFrame([init_series])
    initial_prediction = pd.Series(
        model.predict(initial_input_df[predictors]).ravel(),
        index=targets,
        name="initial_prediction"
    )

    # ---- Objective Function ----
    def objective(x_dict):
        # Reconstruct the full input vector for prediction
        input_vector = init_series.copy()
        
        # First, apply all fixed values
        for key, value in fixed_values.items():
            input_vector[key] = value
        
        # Then, overwrite with the current optimized values from nevergrad
        for key, value in x_dict.items():
            input_vector[key] = value
        
        # Predict
        y_pred = model.predict(pd.DataFrame([input_vector])[predictors]).ravel()
        
        # Objective value (what to minimize/maximize)
        objective_value = float(y_pred[target_to_optimize_idx])

        # Penalties for output constraints
        penalty = 0.0
        if constraints:
            for spec in constraints:
                if spec["name"] not in targets: continue
                j = targets.index(spec["name"])
                v = y_pred[j]
                lo, hi, w = spec.get("min", -np.inf), spec.get("max", np.inf), float(spec.get("weight", 1.0))
                penalty += (max(lo - v, 0) + max(v - hi, 0)) * w

        # If maximizing, we minimize the negative value
        if optimization_goal == "Maximize":
            return -objective_value + penalty
        else: # Minimize
            return objective_value + penalty

    # ---- Run Optimizer ----
    recommendation = optimizer.minimize(objective)
    best_input_dict = recommendation.value

    # ---- Prepare final results ----
    optimized_inputs_series = init_series.copy()
    for key, value in best_input_dict.items():
        optimized_inputs_series[key] = value

    optimized_input_df = pd.DataFrame([optimized_inputs_series])

    optimized_prediction = pd.Series(
        model.predict(optimized_input_df[predictors]).ravel(),
        index=targets,
        name="optimized_prediction"
    )

    return init_series, initial_prediction, optimized_inputs_series, optimized_prediction

def optimize_multiple_rows(
    model,
    df_full: pd.DataFrame,
    df_lagged: pd.DataFrame,
    row_indices: list,
    predictors: list,
    targets: list,
    target_to_optimize: str,
    optimization_goal: str,
    bounds_percent: float,
    free_vars: list,
    budget: int = 200,
):
    """
    Optimizes multiple selected rows from the original dataframe using the trained model.

    Args:
        model: The trained multi-target regression model.
        df_full (pd.DataFrame): The original, non-lagged dataframe.
        df_lagged (pd.DataFrame): The dataframe with lagged features.
        row_indices (list): A list of integer indices for the rows to be optimized.
        predictors (list): List of all predictor column names used by the model.
        targets (list): List of all target column names.
        target_to_optimize (str): The specific target variable to focus the optimization on.
        optimization_goal (str): "Maximize" or "Minimize".
        bounds_percent (float): The percentage to define the optimization bounds for free variables.
        free_vars (list): The list of input variables that the optimizer is allowed to change.
        budget (int): The computational budget for the optimizer for each row.

    Returns:
        list: A list of dictionaries, where each dictionary contains the detailed
              optimization results for a single row.
    """
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, row_idx in enumerate(row_indices):
        status_text.text(f"Optimizing row {i+1}/{len(row_indices)} (Index: {row_idx})...")
        
        # Find the corresponding row in the lagged dataframe
        try:
            # The index of df_lagged is the same as df_full, just shifted
            lagged_row_df = df_lagged.loc[[row_idx]]
        except KeyError:
            st.warning(f"Skipping row index {row_idx} as it's likely part of the initial lag window and has no corresponding lagged data.")
            continue

        # Determine fixed values based on the selected free variables
        # All lagged features are fixed because they are in the past.
        lagged_features = [p for p in predictors if '_lag' in p]
        
        # Current (non-lagged) predictors that were NOT selected to be optimized are also fixed.
        current_predictors = [p for p in predictors if '_lag' not in p]
        fixed_current_vars = [p for p in current_predictors if p not in free_vars]
        
        fixed_vars = {var: lagged_row_df[var].iloc[0] for var in lagged_features + fixed_current_vars}
        # Run single optimization
        initial_inputs, initial_pred, optimized_inputs, optimized_pred = optimize_multi_target(
            model=model,
            df_train=lagged_row_df, # Use the specific row for initial state
            predictors=predictors,
            targets=targets,
            target_to_optimize=target_to_optimize,
            optimization_goal=optimization_goal,
            bounds_percent=bounds_percent,
            constraints=None,
            fixed_values=fixed_vars,
            budget=budget
        )

        # Store comprehensive results for this row
        if initial_inputs is not None:
            actual_outputs = df_full.loc[row_idx, targets]
            result_row = {
                'row_index': row_idx,
                'actual_outputs': actual_outputs,
                'initial_inputs': initial_inputs,
                'initial_predictions': initial_pred,
                'optimized_inputs': optimized_inputs,
                'optimized_predictions': optimized_pred,
            }
            results.append(result_row)
        
        progress_bar.progress((i + 1) / len(row_indices))

    status_text.text("Batch optimization complete!")
    return results

def plot_optimization_results(results: list, target_to_optimize: str, goal: str):
    """
    Creates a comprehensive visualization of batch optimization results.

    This function generates a 3x2 grid of plots summarizing the outcome of the
    `optimize_multiple_rows` function, including target value comparisons,
    improvements, and success rates.

    Args:
        results (list): The list of result dictionaries from `optimize_multiple_rows`.
        target_to_optimize (str): The name of the target variable that was optimized.
        goal (str): The optimization goal ("Maximize" or "Minimize").
    """
    if not results:
        st.warning("No results to visualize!")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15)) # Increased figure size
    fig.suptitle('Multi-Row Optimization Results', fontsize=18, fontweight='bold')
    
    # --- Extract data for plotting ---
    row_indices = [r['row_index'] for r in results]
    actual_target_vals = [r['actual_outputs'][target_to_optimize] for r in results]
    initial_pred_vals = [r['initial_predictions'][target_to_optimize] for r in results]
    optimized_pred_vals = [r['optimized_predictions'][target_to_optimize] for r in results]
    
    # Calculate improvements based on the optimization goal
    if goal == "Maximize":
        improvements = [opt - init for init, opt in zip(initial_pred_vals, optimized_pred_vals)]
        improvements_pct = [((opt - init) / init * 100) if init != 0 else 0 for init, opt in zip(initial_pred_vals, optimized_pred_vals)]
    else: # Minimize
        improvements = [init - opt for init, opt in zip(initial_pred_vals, optimized_pred_vals)]
        improvements_pct = [((init - opt) / init * 100) if init != 0 else 0 for init, opt in zip(initial_pred_vals, optimized_pred_vals)]

    # 1. Target Value Comparison (Top-Left)
    n_results = len(results)
    x_pos = np.arange(n_results) * 2
    width = 0.5
    axes[0, 0].bar(x_pos - width, actual_target_vals, width=width, alpha=0.7, label=f'Actual {target_to_optimize}', color='skyblue')
    axes[0, 0].bar(x_pos, initial_pred_vals, width=width, alpha=0.7, label=f'Initial Predicted', color='salmon')
    axes[0, 0].bar(x_pos + width, optimized_pred_vals, width=width, alpha=0.7, label=f'Optimized Predicted', color='lightgreen')
    axes[0, 0].set_title(f'{target_to_optimize}: Actual vs. Predicted')
    axes[0, 0].set_xlabel('Row Index')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([str(idx) for idx in row_indices], rotation=45, ha="right")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Absolute Improvement Plot (Top-Right)
    colors = ['green' if x >= 0 else 'red' for x in improvements]
    axes[0, 1].bar(x_pos, improvements, color=colors, alpha=0.7)
    axes[0, 1].set_title(f'Absolute Improvement in {target_to_optimize}')
    axes[0, 1].set_xlabel('Row Index')
    axes[0, 1].set_ylabel('Improvement (Absolute)')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([str(idx) for idx in row_indices], rotation=45, ha="right")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Percentage Improvement (Middle-Left)
    colors_pct = ['green' if x >= 0 else 'red' for x in improvements_pct]
    axes[1, 0].bar(x_pos, improvements_pct, color=colors_pct, alpha=0.7)
    axes[1, 0].set_title(f'Percentage Improvement in {target_to_optimize}')
    axes[1, 0].set_xlabel('Row Index')
    axes[1, 0].set_ylabel('Improvement (%)')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([str(idx) for idx in row_indices], rotation=45, ha="right")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Input Variable Changes (Middle-Right)
    try:
        input_vars = results[0]['optimized_inputs'].index.tolist()
        input_changes_pct = []
        valid_vars = []
        
        for var in input_vars:
            changes = []
            for res in results:
                initial_val = res['initial_inputs'].get(var, 0)
                optimized_val = res['optimized_inputs'].get(var, 0)
                if initial_val != 0:
                    change_pct = ((optimized_val - initial_val) / initial_val) * 100
                    changes.append(change_pct)
            
            if changes:
                avg_change = np.mean(changes)
                input_changes_pct.append(avg_change)
                valid_vars.append(var)

        if valid_vars:
            axes[1, 1].barh(valid_vars, input_changes_pct, color='teal', alpha=0.7)
            axes[1, 1].set_title('Average Input Variable Changes (%)')
            axes[1, 1].set_xlabel('Average Change (%)')
            axes[1, 1].grid(True, alpha=0.3)
    except (IndexError, KeyError) as e:
        axes[1, 1].text(0.5, 0.5, f'Error displaying input changes: {e}', ha='center', va='center')

    # 5. Distribution of Improvements (Bottom-Left)
    if improvements:
        sns.histplot(improvements, bins=min(10, len(improvements)), ax=axes[2, 0], color='blue', kde=True)
        axes[2, 0].set_title(f'Distribution of {target_to_optimize} Improvements')
        axes[2, 0].set_xlabel('Improvement')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].grid(True, alpha=0.3)

    # 6. Success Rate (Bottom-Right)
    successful_optimizations = sum(1 for x in improvements if x > 0)
    total_runs = len(results)
    success_rate = (successful_optimizations / total_runs) * 100 if total_runs > 0 else 0
    
    labels = ['Successful', 'Unsuccessful']
    sizes = [success_rate, 100 - success_rate]
    colors = ['lightgreen', 'lightcoral']
    
    axes[2, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    axes[2, 1].set_title(f'Optimization Success Rate\n({successful_optimizations}/{total_runs} successful)')
    axes[2, 1].axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig

def plot_detailed_optimization_comparison(results: list, targets: list, free_vars: list):
    """
    Creates detailed bar plots comparing actual, initial, and optimized values.

    This function generates two sets of plots:
    1. For all target variables, comparing actual vs. initial predicted vs. optimized predicted values.
    2. For all manipulated (free) predictor variables, comparing their initial and optimized values.

    Args:
        results (list): The list of result dictionaries from `optimize_multiple_rows`.
        targets (list): A list of all target variable names.
        free_vars (list): A list of the predictor variables that were manipulated by the optimizer.
    """
    if not results:
        st.warning("No detailed results to visualize.")
        return

    row_indices = [r['row_index'] for r in results]
    n_results = len(results)
    x_pos = np.arange(n_results)

    # --- 1. Plot Target Variables ---
    st.subheader("Target Variables: Detailed Comparison")
    n_targets = len(targets)
    # Calculate rows needed for 2 columns
    n_rows_targets = (n_targets + 1) // 2
    fig_targets, axes_targets = plt.subplots(n_rows_targets, 2, figsize=(18, 5 * n_rows_targets), squeeze=False)
    fig_targets.suptitle('Target Variables: Actual vs. Initial Predicted vs. Optimized Predicted', fontsize=16, fontweight='bold')
    axes_targets = axes_targets.flatten()

    for i, target in enumerate(targets):
        actual_vals = [r['actual_outputs'].get(target, 0) for r in results]
        initial_pred_vals = [r['initial_predictions'].get(target, 0) for r in results]
        optimized_pred_vals = [r['optimized_predictions'].get(target, 0) for r in results]

        width = 0.25
        axes_targets[i].bar(x_pos - width, actual_vals, width=width, label='Actual', color='skyblue', alpha=0.8)
        axes_targets[i].bar(x_pos, initial_pred_vals, width=width, label='Initial Predicted', color='salmon', alpha=0.8)
        axes_targets[i].bar(x_pos + width, optimized_pred_vals, width=width, label='Optimized Predicted', color='lightgreen', alpha=0.8)

        axes_targets[i].set_title(f'{target}')
        axes_targets[i].set_ylabel('Value')
        axes_targets[i].set_xticks(x_pos)
        axes_targets[i].set_xticklabels([str(idx) for idx in row_indices], rotation=45, ha="right")
        axes_targets[i].legend()
        axes_targets[i].grid(True, axis='y', linestyle='--', alpha=0.6)

    # Hide any unused subplots for targets
    for i in range(n_targets, len(axes_targets)):
        axes_targets[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    st.pyplot(fig_targets)

    # --- 2. Plot Manipulated Predictor Variables ---
    st.subheader("Manipulated Variables: Detailed Comparison")
    if not free_vars:
        st.info("No variables were selected for manipulation by the optimizer.")
        return

    n_predictors = len(free_vars)
    # Calculate rows needed for 2 columns
    n_rows_predictors = (n_predictors + 1) // 2
    fig_predictors, axes_predictors = plt.subplots(n_rows_predictors, 2, figsize=(18, 5 * n_rows_predictors), squeeze=False)
    fig_predictors.suptitle('Manipulated Input Variables: Initial vs. Optimized', fontsize=16, fontweight='bold')
    axes_predictors = axes_predictors.flatten()

    for i, predictor in enumerate(free_vars):
        initial_vals = [r['initial_inputs'].get(predictor, 0) for r in results]
        optimized_vals = [r['optimized_inputs'].get(predictor, 0) for r in results]

        width = 0.35
        axes_predictors[i].bar(x_pos - width/2, initial_vals, width=width, label='Initial', color='lightblue', alpha=0.8)
        axes_predictors[i].bar(x_pos + width/2, optimized_vals, width=width, label='Optimized', color='darkorange', alpha=0.8)

        axes_predictors[i].set_title(f'{predictor}')
        axes_predictors[i].set_ylabel('Value')
        axes_predictors[i].set_xticks(x_pos)
        axes_predictors[i].set_xticklabels([str(idx) for idx in row_indices], rotation=45, ha="right")
        axes_predictors[i].legend()
        axes_predictors[i].grid(True, axis='y', linestyle='--', alpha=0.6)

    # Hide any unused subplots for predictors
    for i in range(n_predictors, len(axes_predictors)):
        axes_predictors[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    st.pyplot(fig_predictors)

def plot_dmdc_evaluation(y_true, y_pred, target_feature):
    """
    Creates a comprehensive evaluation plot for DMDc results, similar to evaluate_regression.
    """
    # --- Residuals ---
    residuals = y_true - y_pred

    # --- Combined Plot Layout ---
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.3, wspace=0.3)
    
    # Subplot 1: Predicted vs Actual (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
    ax1.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel(f"Actual {target_feature}")
    ax1.set_ylabel(f"Predicted {target_feature}")
    ax1.set_title("Predicted vs Actual (One-Step-Ahead)")
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Residuals (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(residuals, bins=20, kde=True, color='skyblue', edgecolor='k', ax=ax2)
    ax2.axvline(0, color='red', linestyle='--')
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual Distribution (One-Step-Ahead)")
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Time-series plot (bottom row, spans both columns)
    ax3 = fig.add_subplot(gs[1, :])
    test_indices = range(len(y_true))
    ax3.plot(test_indices, y_true, 'o-', label='Actual', alpha=0.7, markersize=4, linewidth=0.5, color='blue')
    ax3.plot(test_indices, y_pred, 'x-', label='Predicted', alpha=0.7, markersize=4, linewidth=0.5, color='red')
    ax3.set_ylabel(target_feature)
    ax3.set_xlabel('Time Step')
    ax3.set_title(f"One-Step-Ahead Prediction vs. Actual (Full Dataset)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_3d_plotly(df, x_col, y_col, z_col, color_col, 
                  color_scale='viridis', marker_size=5,
                  title='3D Plot with Color Map', opacity=0.8,
                  width=800, height=600):
    """
    Create an interactive 3D scatter plot with Plotly
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    x_col, y_col, z_col : str
        Column names for x, y, z coordinates
    color_col : str
        Column name for color mapping
    color_scale : str
        Plotly color scale name ('viridis', 'plasma', 'rainbow', etc.)
    marker_size : int or float
        Size of the markers
    title : str
        Plot title
    opacity : float
        Opacity of points (0-1)
    width, height : int
        Figure dimensions
    """
    
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                       color=color_col,
                       color_continuous_scale=color_scale,
                       opacity=opacity,
                       title=title)
    
    # Update marker size and layout
    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col, aspectmode='auto'), width=width, height=height)
    return fig

def perform_basic_eda(df):
    """
    Performs and displays basic EDA on the dataframe.
    """
    import io
    st.header("Basic Exploratory Data Analysis")

    # 1. Display df.info()
    st.subheader("Dataframe Info")
    st.write("Provides a concise summary of the dataframe, including data types and non-null values.")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # 2. Display df.describe()
    st.subheader("Descriptive Statistics")
    st.write("Generates descriptive statistics for the numerical columns.")
    st.dataframe(df.describe())

    # 3. Display Correlation Heatmap
    st.subheader("Correlation Heatmap")
    st.write("Shows the Pearson correlation between all numerical columns.")
    df_numeric = df.select_dtypes(include=np.number)
    if df_numeric.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(max(12, len(df_numeric.columns)), max(10, len(df_numeric.columns)*0.8)))
        sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns (at least 2 required) to generate a correlation heatmap.")

    # 4. Display Trends of each variable
    st.subheader("Variable Trends")
    st.write("Shows the trend of each numerical variable over its index.")
    numeric_cols_for_trends = df.select_dtypes(include=np.number).columns.tolist()
    
    if numeric_cols_for_trends:
        n_cols = 2
        n_rows = (len(numeric_cols_for_trends) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols_for_trends):
            ax = axes[i]
            ax.plot(df.index, df[col], label=col)
            ax.set_title(f"Trend of {col}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.5)

        # Hide any unused subplots
        for i in range(len(numeric_cols_for_trends), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found to plot trends.")

    # 5. Display Pair Plot
    st.subheader("Pairwise Relationships Plot")
    st.write("Visualizes pairwise relationships and distributions for numerical variables. This can be slow for datasets with many columns.")
    if df_numeric.shape[1] > 10:
        st.warning(f"Pair plot is disabled because the dataset has {df_numeric.shape[1]} numeric columns (more than 10). This is to prevent slow performance.")
    elif df_numeric.shape[1] > 1:
        st.pyplot(sns.pairplot(df_numeric))

def run_anomaly_detection(df, features, target_col, lag, methods_to_run, params):
    """
    Performs multiple anomaly detection methods on the data and displays results.
    """
    from sklearn.cross_decomposition import PLSRegression

    st.write(f"### Running Anomaly Detection for Target: `{target_col}` with Lag: `{lag}`")

    # Create a lagged dataframe for analysis
    df_lagged = make_lagged_df(df, lag, columns_to_drop=[], target_col=target_col, target_lag_n=0)
    
    # Prepare data for model-based methods
    X = df_lagged.drop(columns=[target_col]).fillna(0)
    y = df_lagged[target_col].fillna(0)

    anomaly_table = pd.DataFrame(index=df_lagged.index)

    # --- Method 1: Z-score (on target) ---
    if 'zscore' in methods_to_run:
        st.write("1. Calculating Z-score anomalies...")
        z_scores = zscore(y)
        anomaly_table["Zscore"] = (np.abs(z_scores) > params['z_threshold']).astype(int)

    # --- Method 2: Rolling Mean/Std (on target) ---
    if 'rolling' in methods_to_run:
        st.write("2. Calculating Rolling Statistics anomalies...")
        rolling_mean = y.rolling(params['rolling_window']).mean()
        rolling_std = y.rolling(params['rolling_window']).std()
        anomaly_table["RollingStats"] = (
            ((y > rolling_mean + params['z_threshold'] * rolling_std) |
             (y < rolling_mean - params['z_threshold'] * rolling_std))
        ).astype(int).fillna(0)

    # --- Method 3: Isolation Forest (on all features) ---
    if 'iso_forest' in methods_to_run:
        st.write("3. Running Isolation Forest...")
        iso = IsolationForest(contamination=params['iso_contamination'], random_state=42)
        iso_preds = iso.fit_predict(df_lagged.fillna(0))
        anomaly_table["IsolationForest"] = (iso_preds == -1).astype(int)

    # --- Method 4: Local Outlier Factor (on all features) ---
    if 'lof' in methods_to_run:
        st.write("4. Running Local Outlier Factor...")
        lof = LocalOutlierFactor(n_neighbors=params['lof_neighbors'], contamination=params['lof_contamination'])
        lof_preds = lof.fit_predict(df_lagged.fillna(0))
        anomaly_table["LOF"] = (lof_preds == -1).astype(int)

    # --- Method 5: PCA Reconstruction Error (on all features) ---
    if 'pca' in methods_to_run:
        st.write("5. Running PCA-based anomaly detection...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_lagged.fillna(0))
        pca = PCA(n_components=params['pca_components'], random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        X_reconstructed = pca.inverse_transform(X_pca)
        recon_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        threshold = np.percentile(recon_error, params['recon_error_percentile'])
        anomaly_table["PCA_Recon"] = (recon_error > threshold).astype(int)

    # --- Method 6: PLS Regression (prediction error) ---
    if 'pls' in methods_to_run:
        st.write("6. Running PLS Regression-based anomaly detection...")
        pls_components_safe = min(5, X.shape[1])
        pls = PLSRegression(n_components=pls_components_safe)
        pls.fit(X, y)
        y_pred_pls = pls.predict(X).ravel()
        pls_residuals = np.abs(y - y_pred_pls)
        pls_threshold = np.percentile(pls_residuals, params['recon_error_percentile'])
        anomaly_table["PLS_Residual"] = (pls_residuals > pls_threshold).astype(int)

    # --- Method 7: Linear Regression (residual error) ---
    if 'lin_reg' in methods_to_run:
        st.write("7. Running Linear Regression-based anomaly detection...")
        linreg = LinearRegression()
        linreg.fit(X, y)
        y_pred_lr = linreg.predict(X)
        lr_residuals = np.abs(y - y_pred_lr)
        lr_threshold = np.percentile(lr_residuals, params['recon_error_percentile'])
        anomaly_table["LinReg_Residual"] = (lr_residuals > lr_threshold).astype(int)

    # --- Combine Results ---
    if anomaly_table.empty:
        st.warning("No anomaly detection methods were selected. Please select at least one method to run.")
        return

    st.write("Combining results...")

    anomaly_table["Votes"] = anomaly_table.sum(axis=1)
    
    # --- Display Results ---
    st.subheader("Anomaly Detection Results")
    st.write(f"The table below shows data points flagged as anomalies by at least **{params['vote_threshold']}** method(s).")
    
    flagged_anomalies = anomaly_table[anomaly_table["Votes"] >= params['vote_threshold']].sort_values("Votes", ascending=False)
    
    if flagged_anomalies.empty:
        st.success("No significant anomalies found with the current settings.")
    else:
        st.dataframe(flagged_anomalies)

        # --- Create and offer the cleaned dataset for download ---
        st.subheader("Download Cleaned Data")
        indices_to_remove = flagged_anomalies.index
        df_cleaned = df.drop(indices_to_remove)

        st.write(f"A total of **{len(indices_to_remove)}** anomalous rows were identified based on the threshold. You can download the dataset with these rows removed.")
        st.write("Preview of the dataset with anomalies removed:")
        st.dataframe(df_cleaned.head())

        csv_cleaned = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Data.csv",
            data=csv_cleaned,
            file_name=f'cleaned_data_{target_col}.csv',
            mime='text/csv',
            key=f'download_cleaned_data_{target_col}' # Unique key per target
        )

        # Plot combined anomalies on the original time series
        st.subheader(f"Combined Anomalies in `{target_col}` Time Series")
        st.write(f"This plot highlights all points that were flagged by at least **{params['vote_threshold']}** method(s).")
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(df.index, df[target_col], label=f'Original {target_col}', color='blue', zorder=1)
        
        anomalous_points = df.loc[flagged_anomalies.index]
        ax.scatter(anomalous_points.index, anomalous_points[target_col], color='red', s=50, zorder=2, label=f'Detected Anomalies (Votes ≥ {params["vote_threshold"]})')
        
        ax.set_title(f"Combined Detected Anomalies for {target_col}")
        ax.set_xlabel("Index")
        ax.set_ylabel(target_col)
        ax.legend()
        ax.grid(True, alpha=0.5)
        st.pyplot(fig)

        # Plot anomalies for each method separately in a 2-column layout
        st.subheader("Individual Anomaly Detection Method Results")
        st.write("These plots show the specific anomalies detected by each individual method.")
        
        method_cols = [col for col in anomaly_table.columns if col != 'Votes']
        n_methods = len(method_cols)
        n_rows = (n_methods + 1) // 2  # Calculate rows needed for 2 columns

        fig_methods, axes = plt.subplots(n_rows, 2, figsize=(18, 5 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i, method in enumerate(method_cols):
            ax = axes[i]
            ax.plot(df.index, df[target_col], label=f'Original {target_col}', color='blue', alpha=0.6, zorder=1)
            
            method_anomalies_idx = anomaly_table[anomaly_table[method] == 1].index
            anomalous_points = df.loc[method_anomalies_idx]
            ax.scatter(anomalous_points.index, anomalous_points[target_col], color='red', s=40, zorder=2, label=f'Anomalies')
            
            ax.set_title(f"Anomalies Detected by: {method}")
            ax.set_xlabel("Index")
            ax.set_ylabel(target_col)
            ax.legend()
            ax.grid(True, alpha=0.4)

        # Hide any unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_methods)

def run_lag_correlation_analysis(df, target_col, max_lags):
    """
    Calculates and plots lag correlation heatmaps for Pearson, Kendall, and Spearman methods.
    """
    st.header(f"Lag Correlation Analysis for Target: `{target_col}`")
    st.write("This analysis helps identify the time delay (lag) at which each input variable has the strongest relationship with the target variable. The results can be used to create a feature-engineered dataset for modeling.")

    # Ensure we only work with numeric data
    df_numeric = df.select_dtypes(include=np.number)
    if target_col not in df_numeric.columns:
        st.error(f"Target column '{target_col}' is not numeric and cannot be used for correlation analysis.")
        return

    # Use tabs for each correlation method
    tab1, tab2, tab3 = st.tabs(["Pearson", "Kendall", "Spearman"])
    correlation_methods = [("pearson", tab1), ("kendall", tab2), ("spearman", tab3)]

    for method, tab in correlation_methods:
        with tab:
            st.subheader(f"{method.capitalize()}'s Correlation")
            with st.spinner(f"Calculating {method} correlations for lags 0-{max_lags-1}..."):
                # Prepare empty dataframe
                corr_df = pd.DataFrame(index=df_numeric.columns.drop(target_col))

                # Loop through lags
                for lag in range(max_lags):
                    if lag == 0:
                        # current correlation
                        corr_df[f'lag_{lag}'] = df_numeric.corr(method=method)[target_col].drop(target_col)
                    else:
                        # correlation with lagged features
                        df_lagged = df_numeric.shift(lag)
                        corr_df[f'lag_{lag}'] = df_lagged.corrwith(df_numeric[target_col], method=method).drop(target_col, errors='ignore')

                # --- Visualization ---
                fig, ax = plt.subplots(figsize=(18, max(8, len(corr_df) * 0.5)))
                sns.heatmap(
                    corr_df.fillna(0),
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    center=0,
                    ax=ax
                )
                ax.set_title(f"{method.capitalize()} Correlation of Variables with '{target_col}' (lags 0–{max_lags-1})", fontsize=16)
                ax.set_xlabel("Lag", fontsize=12)
                ax.set_ylabel("Variable", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)

                # --- 1. Find and display best lags ---
                st.subheader("Best Lag per Variable")
                st.write(f"The lag with the highest absolute {method.capitalize()} correlation for each variable.")
                
                # Find the column name with the max absolute value for each row
                best_lag_cols = corr_df.abs().idxmax(axis=1)
                # Extract the lag number from the column name
                best_lags = best_lag_cols.str.replace('lag_', '').astype(int)
                
                best_lags_df = pd.DataFrame(best_lags, columns=['Best Lag']).T
                st.dataframe(best_lags_df)

                # --- 2. Create and offer the modified dataset for download ---
                st.subheader("Create Lag-Optimized Dataset")
                st.write("A new dataset is created where each variable is shifted by its own best lag, as determined above. This can be a powerful feature engineering step.")

                # Create the new lagged dataframe
                new_lagged_features = []
                for var, lag_val in best_lags.items():
                    if var != target_col:
                        new_col = df_numeric[var].shift(lag_val).rename(f"{var}_lag{lag_val}")
                        new_lagged_features.append(new_col)
                
                # Combine the new lagged features with the original target column
                modified_df = pd.concat([df_numeric[target_col]] + new_lagged_features, axis=1).dropna()

                st.write("Preview of the lag-optimized dataset:")
                st.dataframe(modified_df.head())

                # --- Add heatmap for the modified_df ---
                st.subheader(f"Correlation Heatmap of Lag-Optimized Dataset ({method.capitalize()})")
                st.write("This heatmap shows the correlations between the optimally lagged features and the target variable. It also reveals correlations between the features themselves, which can be important for modeling.")
                
                fig_mod, ax_mod = plt.subplots(figsize=(max(10, len(modified_df.columns)*0.8), max(8, len(modified_df.columns)*0.6)))
                sns.heatmap(
                    modified_df.corr(method=method),
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    center=0,
                    ax=ax_mod
                )
                ax_mod.set_title(f"Correlations in Lag-Optimized Dataset ({method.capitalize()})")
                plt.tight_layout()
                st.pyplot(fig_mod)

                # --- 3. Make the new dataframe downloadable ---
                csv = modified_df.to_csv(index=False).encode('utf-8')
                each_lag_datset, selected_lag_dataset = st.columns(2)
                each_lag_datset.download_button(
                    label=f"Download Lag-Optimized Data ({method.capitalize()}).csv",
                    data=csv,
                    file_name=f'lag_optimized_data_{method}.csv',
                    mime='text/csv',
                    key=f'download_lag_{method}'
                )
                with selected_lag_dataset:
                    st.write("Or, select specific lags to download datasets for:")
                    selected_lags = st.multiselect(
                        "Select lags to generate datasets for",
                        options=range(max_lags),
                        key=f'multiselect_lags_{method}'
                    )

                    if selected_lags:
                        cols = st.columns(3)
                        for i, lag_val in enumerate(selected_lags):
                            with cols[i % 3]:
                                # Create a dataframe where all predictors are lagged by lag_val
                                predictors = df_numeric.columns.drop(target_col)
                                lagged_features = df_numeric[predictors].shift(lag_val).add_suffix(f'_lag{lag_val}')
                                single_lag_df = pd.concat([df_numeric[target_col], lagged_features], axis=1).dropna()
                                
                                csv_single_lag = single_lag_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label=f"Download Lag {lag_val} Data",
                                    data=csv_single_lag,
                                    file_name=f'data_lag_{lag_val}.csv',
                                    mime='text/csv',
                                    key=f'download_single_lag_{method}_{lag_val}'
                                )





def train_and_evaluate_dmdc(df, measured_features, control_features, target_feature,
                            hankel_window, svd_rank, prediction_horizon, split_fraction):
    """
    Manual DMDc implementation for complete control and prediction.
    """
    from scipy.linalg import svd
    
    # 1. Prepare data
    if not control_features:
        raise ValueError("DMD with Control requires at least one control feature.")
    if target_feature not in measured_features:
        raise ValueError("Target feature must be one of the measured features.")

    measured_data = df[measured_features].values
    control_data = df[control_features].values
    
    # Use separate scalers for measured and control data
    measured_scaler = StandardScaler()
    control_scaler = StandardScaler()
    measured_scaled = measured_scaler.fit_transform(measured_data)
    control_scaled = control_scaler.fit_transform(control_data)
    
    # 2. Create Hankel matrices manually
    n_samples = len(measured_scaled) - hankel_window
    if n_samples <= 0:
        raise ValueError("Hankel window is too large for the dataset length.")
        
    n_measured_features = len(measured_features)
    n_control_features = len(control_features)
    hankel_dim = hankel_window * n_measured_features

    X = np.zeros((hankel_dim, n_samples))
    X_prime = np.zeros((hankel_dim, n_samples))
    U = np.zeros((n_control_features, n_samples))
    
    for i in range(n_samples):
        X[:, i] = measured_scaled[i:i+hankel_window].flatten()
        X_prime[:, i] = measured_scaled[i+1:i+hankel_window+1].flatten()
        U[:, i] = control_scaled[i+hankel_window-1] # Control at the end of the window
    
    # 3. Split data
    split_idx = int(n_samples * split_fraction)
    X_train, X_prime_train, U_train = X[:, :split_idx], X_prime[:, :split_idx], U[:, :split_idx]
    X_test, U_test = X[:, split_idx:], U[:, split_idx:]
    
    # 4. Manual DMDc computation
    Omega = np.vstack([X_train, U_train])
    U_svd, s, Vh = svd(Omega, full_matrices=False)
    
    # Validate and use the SVD rank from user input
    if svd_rank > len(s):
        raise ValueError(
            f"SVD Rank ({svd_rank}) cannot be greater than the effective rank of the training data ({len(s)}). "
            "Please reduce the SVD Rank or increase the training data size."
        )
    rank = svd_rank
    
    # Truncate based on rank
    U_trunc = U_svd[:, :rank]
    s_trunc = s[:rank]
    Vh_trunc = Vh[:rank, :]
    
    # Check for numerical stability before inverting singular values
    if np.any(s_trunc < 1e-10):
        raise RuntimeError(
            f"Numerical instability detected. The chosen SVD rank ({rank}) includes singular values that are too small. "
            "Please try reducing the 'SVD Rank' parameter."
        )
    
    s_inv_trunc = np.diag(1/s_trunc)
    Omega_plus = Vh_trunc.T @ s_inv_trunc @ U_trunc.T
    
    # G = [A, B] = X'_train * pinv(Omega)
    G = X_prime_train @ Omega_plus
    
    A = G[:, :hankel_dim]
    B = G[:, hankel_dim:]
    
    # 5. Perform multi-step prediction (simulation) on the test set
    if X_test.shape[1] == 0:
        raise ValueError("Test set is empty. Reduce train data ratio or provide more data.")
        
    num_predictions = min(prediction_horizon, X_test.shape[1] - 1)
    if num_predictions <= 0:
        raise ValueError("Not enough test data to perform predictions for the given horizon.")

    predicted_hankel_snapshots = np.zeros((hankel_dim, num_predictions))
    current_hankel_snapshot = X_test[:, 0]

    for i in range(num_predictions):
        current_control = U_test[:, i]
        next_hankel_snapshot = (A @ current_hankel_snapshot) + (B @ current_control)
        predicted_hankel_snapshots[:, i] = next_hankel_snapshot.real # Use real part for stability
        current_hankel_snapshot = next_hankel_snapshot

    # 6. Extract target variable and evaluate
    target_idx_in_measured = measured_features.index(target_feature)
    
    # The value at the 'current' time is the first block of the Hankel matrix
    y_pred_scaled = predicted_hankel_snapshots[target_idx_in_measured, :]

    # The ground truth is the corresponding value from the future test snapshots
    y_true_hankel = X_test[:, 1:num_predictions+1]
    y_true_scaled = y_true_hankel[target_idx_in_measured, :]

    # 7. Inverse transform to original scale
    dummy_pred_array = np.zeros((len(y_pred_scaled), n_measured_features))
    dummy_pred_array[:, target_idx_in_measured] = y_pred_scaled
    y_pred_test = measured_scaler.inverse_transform(dummy_pred_array)[:, target_idx_in_measured]

    dummy_true_array = np.zeros((len(y_true_scaled), n_measured_features))
    dummy_true_array[:, target_idx_in_measured] = y_true_scaled
    y_true_test = measured_scaler.inverse_transform(dummy_true_array)[:, target_idx_in_measured]

    # 8. Calculate metrics for the multi-step test prediction
    rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    r2 = r2_score(y_true_test, y_pred_test)

    # 9. Perform one-step-ahead prediction on the ENTIRE dataset for visualization
    X_prime_pred_full = (A @ X) + (B @ U)
    
    y_pred_full_scaled = X_prime_pred_full[target_idx_in_measured, :]
    y_true_full_scaled = X_prime[target_idx_in_measured, :]

    dummy_pred_full_array = np.zeros((len(y_pred_full_scaled), n_measured_features))
    dummy_pred_full_array[:, target_idx_in_measured] = y_pred_full_scaled
    y_pred_full = measured_scaler.inverse_transform(dummy_pred_full_array)[:, target_idx_in_measured]

    dummy_true_full_array = np.zeros((len(y_true_full_scaled), n_measured_features))
    dummy_true_full_array[:, target_idx_in_measured] = y_true_full_scaled
    y_true_full = measured_scaler.inverse_transform(dummy_true_full_array)[:, target_idx_in_measured]

    return {
        "rmse": rmse, "r2": r2,
        "y_true_test": y_true_test, "y_pred_test": y_pred_test,
        "y_true_full": y_true_full, "y_pred_full": y_pred_full,
        "target_feature": target_feature
    }


# --- Model Definitions ---
# A dictionary of time-series models for easy identification
DEEP_LEARNING_MODELS = [
    "Simple LSTM",
    "LSTM Encoder-Decoder with Attention",
    "LSTM Encoder-Decoder with Attention + 1D CNN",
]

LAGGED_MODELS = [
    "Random Forest Regressor (Lagged)",
    "Gradient Boosting Regressor (Lagged)",
    "XGBoost Regressor (Lagged)",
    "Linear Regression (Lagged)",
    "Support Vector Regressor (Lagged)",
    "Principal Component Regression (Lagged)",
    "MLP Regressor (Lagged)",
]

MULTI_TARGET_MODELS = [
    "Multi-Target Linear Regression (Lagged)",
    "Multi-Target Random Forest (Lagged)",
    "Multi-Target XGBoost (Lagged)",
    "Multi-Target MLP (Lagged)",
]

DMD_MODELS = [
    "Hankel DMD with Control",
]

ALL_MODELS = DEEP_LEARNING_MODELS + LAGGED_MODELS + MULTI_TARGET_MODELS + DMD_MODELS

# --- Main Application UI ---

st.title("📈 ML Models Predictor")
st.write(
    "This application trains your Dataset to predict your chosen Target(s)."
)

# --- 1. Sidebar for Configuration ---
with st.sidebar:
    model_option = None
    train_button = False
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader")
    Required_calcs = st.selectbox("Choose your Phase", ['EDA','Anomaly Detection','Regression Models'], key="Phase_@_select")
    
    if Required_calcs == 'EDA':
        st.header("2. EDA Options")
        st.session_state.examine_lags = st.toggle("Examine Lags", key="examine_lags_toggle")
        basic_EDA = st.toggle("Basic EDA", key="basic_eda_toggle")
        examine_3D_feature = st.toggle('Examine features using 3D plots', key='examine_3D_feature')

    if Required_calcs == 'Anomaly Detection':
        
        st.header("2. Anomaly Detection Settings")
        st.subheader("Select Methods to Run")
        use_zscore = st.toggle("Z-Score", value=True, key="use_zscore")
        use_rolling = st.toggle("Rolling Stats", value=True, key="use_rolling")
        use_iso_forest = st.toggle("Isolation Forest", value=True, key="use_iso_forest")
        use_lof = st.toggle("Local Outlier Factor (LOF)", value=True, key="use_lof")
        use_pca = st.toggle("PCA Reconstruction", value=True, key="use_pca")
        use_pls = st.toggle("PLS Residuals", value=False, key="use_pls")
        use_lin_reg = st.toggle("Linear Regression Residuals", value=False, key="use_lin_reg")

        st.session_state.methods_selected = {
            'zscore': use_zscore, 'rolling': use_rolling, 'iso_forest': use_iso_forest,
            'lof': use_lof, 'pca': use_pca, 'pls': use_pls, 'lin_reg': use_lin_reg
        }
        num_methods_selected = sum(st.session_state.methods_selected.values())

        ad_lag = st.slider("Lag for Feature Engineering", 0, 24, 0, key="ad_lag")
        st.subheader("Method Parameters")
        ad_z_threshold = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.5, key="ad_z", help="For Z-Score and Rolling Stats methods.")
        ad_rolling_window = st.slider("Rolling Window Size", 5, 100, 20, key="ad_roll", help="For Rolling Stats method.")
        ad_iso_contamination = st.slider("Isolation Forest Contamination", 0.001, 0.1, 0.01, 0.001, format="%.3f", key="ad_iso", help="Expected proportion of outliers.")
        ad_lof_neighbors = st.slider("LOF Neighbors", 5, 50, 20, key="ad_lof_n", help="Number of neighbors for LOF.")
        ad_lof_contamination = st.slider("LOF Contamination", 0.001, 0.1, 0.01, 0.001, format="%.3f", key="ad_lof_c", help="Expected proportion of outliers.")
        ad_pca_components = st.slider("PCA Variance to Keep", 0.8, 0.99, 0.95, 0.01, key="ad_pca", help="For PCA-based method.")
        ad_recon_error_percentile = st.slider("Model Error Percentile", 90, 99, 99, key="ad_recon", help="For PCA, PLS, and Linear Regression residual methods.")
        
        st.subheader("Voting")
        # Make the max value of the slider dynamic based on the number of selected methods
        max_votes = max(1, num_methods_selected)
        default_vote = min(2, max_votes)
        if max_votes > 1:
            ad_vote_threshold = st.slider("Minimum Votes to Flag Anomaly", 1, max_votes, default_vote, key="ad_vote")
        else:
            st.session_state.ad_vote = 1
            ad_vote_threshold = 1
            
        run_ad_button = st.button("Run Anomaly Detection", type="primary", use_container_width=True, key="run_ad_button")


    if Required_calcs == 'Regression Models':
        st.header("2. Model Selection & Config")
        
        MODEL_CATEGORIES = ["Time Series Models", "Regression Models", "Multi-Target Models"]
        model_category = st.selectbox("Choose a Model Category", MODEL_CATEGORIES, key="model_category_select")

        if model_category == "Time Series Models":
            available_models = DEEP_LEARNING_MODELS + DMD_MODELS
        elif model_category == "Regression Models":
            available_models = LAGGED_MODELS
        elif model_category == "Multi-Target Models":
            available_models = MULTI_TARGET_MODELS

        model_option = st.selectbox("Choose a Model Architecture", available_models, key="model_option_select")

        is_deep_learning_model = model_option in DEEP_LEARNING_MODELS
        is_dmd_model = model_option in DMD_MODELS

        if is_deep_learning_model:
            st.subheader("Sequence Parameters")
            n_timesteps_in = st.number_input("Input Sequence Length (Past Steps)", min_value=1, max_value=100, value=24, key="n_timesteps_in")
            n_timesteps_out = st.number_input("Output Sequence Length (Future Steps to Predict)", min_value=1, max_value=100, value=12, key="n_timesteps_out")
            
            st.subheader("Architecture Parameters")
            learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=0.001, format="%.5f", key="learning_rate")
            
            if model_option == "Simple LSTM":
                num_lstm_layers = st.number_input("Number of LSTM Layers", min_value=1, max_value=5, value=2, key="num_lstm_layers")
                
            if "Encoder-Decoder" in model_option:
                num_encoder_layers = st.number_input("Number of Encoder LSTM Layers", min_value=1, max_value=5, value=1, key="num_encoder_layers")
                num_decoder_layers = st.number_input("Number of Decoder LSTM Layers", min_value=1, max_value=5, value=1, key="num_decoder_layers")
                
            if "CNN" in model_option:
                st.subheader("Convolutional Layer Parameters")
                conv_filters = st.number_input("CNN Filters", min_value=16, max_value=256, value=64, step=16, key="conv_filters")
                conv_kernel_size = st.number_input("CNN Kernel Size", min_value=2, max_value=10, value=3, key="conv_kernel_size")

        elif is_dmd_model:
            st.subheader("DMD Parameters")
            hankel_window = st.number_input("Hankel Window Size (State History)", min_value=2, max_value=200, value=50, key="hankel_window")
            svd_rank = st.number_input("SVD Rank (Model Order)", min_value=1, max_value=200, value=10, key="svd_rank")
            prediction_horizon = st.number_input("Prediction Horizon (Future Steps)", min_value=1, max_value=500, value=100, key="prediction_horizon")

        else: # Lagged models
            st.subheader("Lag Parameters")
            if model_option in MULTI_TARGET_MODELS:
                input_lags = st.number_input("Number of Lagged Inputs (e.g., t-1, t-2...)", min_value=0, max_value=24, value=1, key="input_lags")
                target_lags = st.number_input("Number of Lagged Targets (Autoregressive)", min_value=0, max_value=24, value=1, key="target_lags")
            else:
                target_lag_n = st.slider("Autoregressive Feature Lag (n)", min_value=0, max_value=5, value=1, help="Includes the target value from 'n' steps ago as a feature. 0 to disable.", key="target_lag_n")
            max_lag = st.number_input("Maximum Lag to Test", min_value=0, max_value=100, value=23, key="max_lag")
            st.subheader("Hyperparameter Tuning")
            use_grid_search = st.checkbox("Enable GridSearchCV (slower)", value=False, help="Applies to RF, GB, XGB, and PCR models.", key="use_grid_search")

            if use_grid_search:
                st.info("Define the parameter grid as a Python dictionary string.")
                
                if model_option == "Random Forest Regressor (Lagged)":
                    default_grid = """{
                            "n_estimators": [50, 100],
                            "max_depth": [10, 20, None],
                            "min_samples_split": [2, 5]
                        }"""
                    # {
                    #         "n_estimators": [100,200],
                    #         "max_depth": [3,10, 20],
                    #         "min_samples_split": [2, 5],
                    #     "min_samples_leaf": [1, 2, 4],          
                    #     "max_features": ["sqrt", "log2", 0.5],   
                    #     "min_impurity_decrease": [0.0, 0.01],   
                    #     "bootstrap": [True],                     
                    #     "max_samples": [0.8, 0.9, None],         
                    #     "ccp_alpha": [0.0, 0.01, 0.1] 
                    #     }
                    # param_grid = {
                    #     "n_estimators": [100, 200],
                    #     "max_depth": [3, 10, 20],
                    #     "min_samples_split": [2, 5],
                        
                    #     # Anti-overfitting parameters:
                    #     "min_samples_leaf": [1, 2, 4],           # Minimum samples per leaf
                    #     "max_features": ["sqrt", "log2", 0.5],   # Features to consider for splits
                    #     "min_impurity_decrease": [0.0, 0.01],    # Minimum impurity decrease for split
                    #     "bootstrap": [True],                     # Use bootstrap sampling
                    #     "max_samples": [0.8, 0.9, None],         # Samples per tree
                    #     "ccp_alpha": [0.0, 0.01, 0.1]           # Cost complexity pruning
                    # }
                elif model_option == "Support Vector Regressor (Lagged)":
                    default_grid = """{
                                    "svr__C": [0.1, 1, 10, 20],
                                    "svr__gamma": ["scale", "auto", 0.01, 0.1, 1],
                                    "svr__kernel": ["rbf"],
                                    "svr__epsilon": [0.01, 0.1, 0.2],
                                    "svr__tol": [1e-3, 1e-4]
                                }"""
                elif model_option == "Gradient Boosting Regressor (Lagged)":
                    default_grid = """{
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    }"""
                elif model_option == "XGBoost Regressor (Lagged)":
                    default_grid = """{
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8]
    }"""
                elif model_option == "Principal Component Regression (Lagged)":
                    default_grid = """{
        "pca__n_components": [0.95, 0.99, 5, 10]
    }"""
                elif model_option == "Multi-Target MLP (Lagged)":
                    default_grid = """{
        "mlp__hidden_layer_sizes": [(50,), (100,)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001]
    }"""
                elif model_option == "MLP Regressor (Lagged)":
                    default_grid = """{
                                "mlp__hidden_layer_sizes": [(50,), (100,), (50, 50)],
                                "mlp__activation": ["relu", "tanh"],
                                "mlp__alpha": [0.0001, 0.001]
                            }"""
                else:
                    default_grid = "{}"

                param_grid_str = st.text_area("Parameter Grid", value=default_grid, height=250, key="param_grid_str")

        st.subheader("Data Splitting")
        split_fraction = st.slider("Train Data Ratio", min_value=0.1, max_value=0.9, value=0.7, step=0.05, help="For DL models, this splits the data chronologically. For lagged models, this is the training set size for a random split.", key="split_fraction")

        st.header("3. Training Parameters")
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=20, disabled=not is_deep_learning_model, key="epochs", help="Only for Deep Learning models.")
        latent_dim = st.select_slider("LSTM Latent Dimension", options=[32, 64, 128, 256], value=128, disabled=not is_deep_learning_model, key="latent_dim", help="Only for Deep Learning models.")
        
        train_button = st.button("Train Model", type="primary", use_container_width=True, key="train_button")

# --- Main Panel ---
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())
    # --- Column Selection ---
    # Filter out columns from selection
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if Required_calcs == 'Regression Models':


        # --- Feature Selection UI ---
        if model_option in MULTI_TARGET_MODELS:
            st.subheader("Select Target Variables")
            target_features = st.multiselect("Select one or more target variables", options=numeric_columns, default=numeric_columns[-1:] if numeric_columns else [], key="target_features_multi")
            default_features = [col for col in numeric_columns if col not in target_features]
            target_feature = target_features[0] if target_features else None # For compatibility
            input_features = st.multiselect("Select Input Features", options=numeric_columns, default=default_features, key="input_features_multi")

        elif model_option in DMD_MODELS:
            st.subheader("Select Target, Measured, and Control Features")
            target_feature = st.selectbox("Select the Target Variable", options=numeric_columns, index=len(numeric_columns)-1 if len(numeric_columns) > 0 else 0, key="target_feature_dmd")
            
            # Ensure target is pre-selected in measured features
            default_measured = [target_feature] if target_feature in numeric_columns else []
            measured_features = st.multiselect("Select ALL Measured Features (including target)", options=numeric_columns, default=default_measured, key="measured_features_dmd")
            
            # Features not used as measured can be controls
            available_controls = [col for col in numeric_columns if col not in measured_features]
            control_features = st.multiselect("Select Control Features", options=available_controls, default=available_controls, key="control_features_dmd")

            # For compatibility with downstream code
            target_features = [target_feature]
            input_features = measured_features + control_features

        else:
            st.subheader("Select Target Variable")
            target_feature = st.selectbox("Select the Target Variable (e.g., 'C4 content')", options=numeric_columns, index=len(numeric_columns)-1 if len(numeric_columns) > 0 else 0, key="target_feature_single")
            default_features = [col for col in numeric_columns if col != target_feature]
            target_features = [target_feature] # Ensure it's a list for consistency
            input_features = st.multiselect("Select Input Features", options=numeric_columns, default=default_features, key="input_features_single")

        if not input_features: st.warning("Please select at least one input feature.")
        if not target_features: st.warning("Please select at least one target variable.")

        

        if train_button and input_features and target_features:
            is_deep_learning_model = model_option in DEEP_LEARNING_MODELS # Re-check inside button press

            if is_deep_learning_model:
                with st.spinner("Processing data and training DL model... This may take a few minutes."):
                    # --- Data Processing for DL Models ---
                    input_scaler = MinMaxScaler(feature_range=(0, 1))
                    target_scaler = MinMaxScaler(feature_range=(0, 1))

                    df_inputs_scaled = input_scaler.fit_transform(df[input_features])
                    df_target_scaled = target_scaler.fit_transform(df[[target_feature]])

                    split_idx = int(len(df_inputs_scaled) * split_fraction)
                    train_inputs_scaled, test_inputs_scaled = df_inputs_scaled[:split_idx], df_inputs_scaled[split_idx:]
                    train_target_scaled, test_target_scaled = df_target_scaled[:split_idx], df_target_scaled[split_idx:]

                    encoder_input_train, decoder_target_train = create_sequences(train_inputs_scaled, train_target_scaled, n_timesteps_in, n_timesteps_out)
                    decoder_input_train = np.zeros_like(decoder_target_train)
                    decoder_input_train[:, 1:, :] = decoder_target_train[:, :-1, :]

                    encoder_input_test, decoder_target_test = create_sequences(test_inputs_scaled, test_target_scaled, n_timesteps_in, n_timesteps_out)
                    decoder_input_test = np.zeros_like(decoder_target_test)
                    decoder_input_test[:, 1:, :] = decoder_target_test[:, :-1, :]

                    # --- DL Model Building, Training, and Evaluation ---
                    st.write(f"**Training Model: `{model_option}`**")
                    
                    if model_option == "Simple LSTM":
                        model = build_simple_lstm_model(
                            n_timesteps_in, len(input_features), n_timesteps_out, 1, latent_dim,
                            num_layers=num_lstm_layers, learning_rate=learning_rate
                        )
                        train_inputs, test_inputs = encoder_input_train, encoder_input_test
                        validation_data = (test_inputs, decoder_target_test)
                    else:
                        if model_option == "LSTM Encoder-Decoder with Attention":
                            model = build_enc_dec_attention_model(
                                len(input_features), 1, latent_dim,
                                num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, learning_rate=learning_rate
                            )
                        else: # LSTM Encoder-Decoder with Attention + 1D CNN
                            model = build_enc_dec_attention_cnn_model(
                                len(input_features), 1, latent_dim,
                                num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, learning_rate=learning_rate,
                                conv_filters=conv_filters, conv_kernel_size=conv_kernel_size
                            )
                        train_inputs = [encoder_input_train, decoder_input_train]
                        test_inputs = [encoder_input_test, decoder_input_test]
                        validation_data = (test_inputs, decoder_target_test)

                    model.fit(train_inputs, decoder_target_train, batch_size=64, epochs=epochs, validation_data=validation_data, verbose=0)
                    st.success("Model trained successfully!")

                    # --- Evaluation for DL Models ---
                    st.subheader("Model Performance")

                    # Create two columns for Train vs Test metrics
                    col1, col2 = st.columns(2)

                    # --- Train Set Evaluation ---
                    with col1:
                        st.write("#### Training Set")
                        train_predictions_scaled = model.predict(train_inputs)
                        train_predictions = target_scaler.inverse_transform(train_predictions_scaled.reshape(-1, 1))
                        decoder_target_train_orig = target_scaler.inverse_transform(decoder_target_train.reshape(-1, 1))
                        y_true_train, y_pred_train = decoder_target_train_orig.flatten(), train_predictions.flatten()
                        
                        st.metric("Root Mean Squared Error (RMSE)", f"{np.sqrt(mean_squared_error(y_true_train, y_pred_train)):.4f}")
                        st.metric("R-squared (R2)", f"{r2_score(y_true_train, y_pred_train):.4f}")

                    # --- Test Set Evaluation ---
                    with col2:
                        st.write("#### Test Set")
                        test_predictions_scaled = model.predict(test_inputs)
                        test_predictions = target_scaler.inverse_transform(test_predictions_scaled.reshape(-1, 1))
                        decoder_target_test_orig = target_scaler.inverse_transform(decoder_target_test.reshape(-1, 1))
                        y_true_test, y_pred_test = decoder_target_test_orig.flatten(), test_predictions.flatten()

                        st.metric("Root Mean Squared Error (RMSE)", f"{np.sqrt(mean_squared_error(y_true_test, y_pred_test)):.4f}")
                        st.metric("R-squared (R2)", f"{r2_score(y_true_test, y_pred_test):.4f}")

                    # --- Full Dataset Visualizations for DL Models ---
                    st.subheader("Full Dataset Visualizations")
                    st.write("The following plots show the model's performance across all available data points (training and test sets combined).")

                    # Create sequences for the entire dataset
                    encoder_input_full, decoder_target_full = create_sequences(
                        df_inputs_scaled,
                        df_target_scaled,
                        n_timesteps_in,
                        n_timesteps_out
                    )

                    # Prepare model inputs for the full dataset
                    if model_option == "Simple LSTM":
                        full_dataset_inputs = encoder_input_full
                    else:
                        decoder_input_full = np.zeros_like(decoder_target_full)
                        decoder_input_full[:, 1:, :] = decoder_target_full[:, :-1, :]
                        full_dataset_inputs = [encoder_input_full, decoder_input_full]

                    # Predict on the full dataset
                    full_predictions_scaled = model.predict(full_dataset_inputs)

                    # Inverse transform and flatten
                    full_predictions_orig = target_scaler.inverse_transform(full_predictions_scaled.reshape(-1, 1)).flatten()
                    full_actuals_orig = target_scaler.inverse_transform(decoder_target_full.reshape(-1, 1)).flatten()

                    # Scatter Plot
                    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
                    ax_scatter.scatter(full_actuals_orig, full_predictions_orig, alpha=0.5, edgecolor='k')
                    ax_scatter.plot([full_actuals_orig.min(), full_actuals_orig.max()], [full_actuals_orig.min(), full_actuals_orig.max()], 'r--', lw=2)
                    ax_scatter.set_xlabel("Actual Values")
                    ax_scatter.set_ylabel("Predicted Values")
                    ax_scatter.set_title(f"Full Dataset: Actual vs. Predicted for {target_feature}")
                    ax_scatter.grid(True)
                    
                    # Time Series Plot
                    fig_ts = plot_time_series(full_actuals_orig, full_predictions_orig, target_feature, f"Full Dataset: Actual vs. Predicted for {target_feature}")

                    col1_viz, col2_viz = st.columns(2)
                    with col1_viz:
                        st.pyplot(fig_scatter)
                    with col2_viz:
                        st.pyplot(fig_ts)

            #else:  # Lagged Models Workflow
            elif model_option in LAGGED_MODELS:
                with st.spinner(f"Evaluating {model_option} for lags 0 to {max_lag}..."):
                    df_filtered = df[input_features + [target_feature]].copy()

                    param_grid = None
                    if use_grid_search:
                        try:
                            # Safely parse the string from the text area into a dictionary
                            param_grid = ast.literal_eval(param_grid_str)
                            st.write("Using Custom Parameter Grid:")
                            st.json(param_grid)
                        except Exception as e:
                            st.error(f"Invalid Parameter Grid Dictionary: {e}")
                            st.stop() # Halt execution if the grid is invalid

                    if model_option == "Random Forest Regressor (Lagged)":
                        model_builder = lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    elif model_option == "Gradient Boosting Regressor (Lagged)":
                        model_builder = lambda: GradientBoostingRegressor(random_state=42)
                    elif model_option == "XGBoost Regressor (Lagged)":
                        model_builder = lambda: XGBRegressor(random_state=42, n_jobs=-1)
                    elif model_option == "Linear Regression (Lagged)":
                        model_builder = lambda: LinearRegression(n_jobs=-1)
                        if use_grid_search: st.warning("GridSearchCV is not applicable for Linear Regression."); param_grid = None
                    elif model_option == "Support Vector Regressor (Lagged)":
                        model_builder = lambda: Pipeline([
                            ('svr', SVR(kernel="rbf"))
                        ])
                    elif model_option == "Principal Component Regression (Lagged)":
                        model_builder = lambda: Pipeline([('pca', PCA()), ('regressor', LinearRegression())])
                    elif model_option == "MLP Regressor (Lagged)":
                        model_builder = lambda: Pipeline([
                            ('scaler', StandardScaler()),
                            ('mlp', MLPRegressor(random_state=42, max_iter=500))
                        ])

                    results_df, best_model = evaluate_lagged_models(
                        df=df_filtered, columns_to_drop=[], model_builder=model_builder,
                        max_lag=max_lag, test_size=1 - split_fraction, target_col=target_feature, param_grid=param_grid,target_lag_n=target_lag_n
                    )
                    st.success("Model evaluation complete!")

                    st.subheader("Lag Evaluation Results")
                    st.write("The model was trained and evaluated for each lag value. The best performing lag is selected based on Test R².")
                    
                    # Display new enhanced plot
                    fig_metrics = plot_model_metrics_enhanced(results_df)
                    st.pyplot(fig_metrics)
                    
                    st.dataframe(results_df.style.highlight_max(subset=['Test R²'], color='lightgreen'))

                    best_lag_row = results_df.iloc[0]
                    best_lag = int(best_lag_row["Lag"])
                    st.subheader(f"Best Model Performance (at Lag = {best_lag})")

                    col1, col2 = st.columns(2)
                    col1.metric("Test RMSE", f"{best_lag_row['Test RMSE']:.4f}")
                    col2.metric("Test R-squared (R2)", f"{best_lag_row['Test R²']:.4f}")

                    st.subheader(f"Detailed Evaluation of Best Model (Lag = {best_lag})")
                    st.write("The following plots show a detailed analysis of the best model's performance on the test set partition.")
                    fig_eval = evaluate_regression(
                        model=best_model, 
                        df=df_filtered, 
                        columns_to_drop=[], 
                        lag=best_lag, 
                        target_name=target_feature,
                        test_size=1 - split_fraction,target_lag_n=target_lag_n
                    )
                    st.pyplot(fig_eval)
            elif model_option in DMD_MODELS:
                        with st.spinner(f"Training {model_option} and predicting..."):
                            try:
                                results = train_and_evaluate_dmdc(
                                    df=df,
                                    measured_features=measured_features,
                                    control_features=control_features,
                                    target_feature=target_feature,
                                    hankel_window=hankel_window,
                                    svd_rank=int(svd_rank),
                                    prediction_horizon=prediction_horizon,
                                    split_fraction=split_fraction
                                )
                                st.success("DMDc model trained and evaluated successfully!")

                                st.subheader("Model Performance on Test Set (Multi-Step Simulation)")
                                col1, col2 = st.columns(2)
                                col1.metric("Root Mean Squared Error (RMSE)", f"{results['rmse']:.4f}")
                                col2.metric("R-squared (R2)", f"{results['r2']:.4f}")

                                st.subheader("Detailed Evaluation (One-Step-Ahead on Full Dataset)")
                                fig_dmdc_eval = plot_dmdc_evaluation(
                                    y_true=results['y_true_full'], y_pred=results['y_pred_full'], target_feature=results['target_feature']
                                )
                                st.pyplot(fig_dmdc_eval)

                            except Exception as e:
                                st.error(f"An error occurred during DMDc processing: {e}")
            elif model_option in MULTI_TARGET_MODELS:
                with st.spinner(f"Training {model_option}..."):
                    # --- Data Preparation ---
                    st.subheader("Lagged Data Preview")
                    st.write(f"Creating dataset with {input_lags} lag(s) for predictors and {target_lags} lag(s) for autoregressive features.")
                    df_lagged = make_multi_lagged_df(df, input_features, target_features, input_lags, target_lags)
                    st.dataframe(df_lagged.head())

                    X = df_lagged.drop(columns=target_features)
                    y = df_lagged[target_features]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=1 - split_fraction, random_state=42
                    )

                    param_grid = None
                    if use_grid_search:
                        try:
                            param_grid = ast.literal_eval(param_grid_str)
                            st.write("Using Custom Parameter Grid:"); #st.json(param_grid)
                        except Exception as e:
                            st.error(f"Invalid Parameter Grid Dictionary: {e}"); st.stop()

                    # --- Model Building ---
                    base_model = None
                    if "Random Forest" in model_option:
                        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
                    elif "XGBoost" in model_option:
                        base_model = XGBRegressor(random_state=42, n_jobs=-1)
                    elif "Gradient Boosting" in model_option:
                        base_model = GradientBoostingRegressor(random_state=42)
                    elif "MLP" in model_option:
                        base_model = Pipeline([
                            ('scaler', StandardScaler()),
                            ('mlp', MLPRegressor(random_state=42, max_iter=500))
                        ])
                        if use_grid_search and param_grid:
                            param_grid = {f'mlp__{k}': v for k, v in param_grid.items()}
                    else: # Linear Regression
                        base_model = LinearRegression(n_jobs=-1)
                        if use_grid_search: st.warning("GridSearchCV is not applicable for Linear Regression."); param_grid = None

                    # --- Training and Evaluation ---
                    model_to_train = MultiOutputRegressor(base_model)
                    if use_grid_search and param_grid:
                        st.write("Performing GridSearchCV...")
                        prefixed_param_grid = {f'estimator__{key}': value for key, value in param_grid.items()}
                        grid_search = GridSearchCV(model_to_train, prefixed_param_grid, cv=3, scoring="r2", n_jobs=-1)
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                        st.write("Best Parameters Found:")
                        st.json(grid_search.best_params_)
                    else:
                        model_to_train.fit(X_train, y_train)
                        best_model = model_to_train

                    st.success("Multi-Target model training complete!")

                    # --- Save model and related info to session state for later use ---
                    st.session_state['best_multi_target_model'] = best_model
                    st.session_state['multi_target_input_features'] = input_features
                    st.session_state['multi_target_target_features'] = target_features
                    st.session_state['multi_target_input_lags'] = input_lags
                    st.session_state['multi_target_target_lags'] = target_lags
                    st.session_state['multi_target_df'] = df
                    st.session_state['multi_target_split_fraction'] = split_fraction

                    # --- Add flag and data to show tabs after training ---
                    st.session_state['multi_target_model_trained'] = True
                    st.session_state['X_test_multi'] = X_test
                    st.session_state['y_test_multi'] = y_test
                    st.session_state['X_train_multi'] = X_train
                    st.session_state['y_train_multi'] = y_train

                    # --- Create tabs for organized output ---
                    tab1, tab2 = st.tabs(["Model Evaluation", "New Tab"])

                    with tab1:
                        y_test_pred = best_model.predict(X_test)
                        y_train_pred = best_model.predict(X_train)

                        # --- Metrics Display ---
                        st.subheader("Model Performance")
                        test_r2_scores = [r2_score(y_test.iloc[:, i], y_test_pred[:, i]) for i in range(y_test.shape[1])]
                        test_rmse_scores = [np.sqrt(mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i])) for i in range(y_test.shape[1])]
                        train_r2_scores = [r2_score(y_train.iloc[:, i], y_train_pred[:, i]) for i in range(y_train.shape[1])]
                        train_rmse_scores = [np.sqrt(mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i])) for i in range(y_train.shape[1])]

                        mean_test_r2 = np.mean(test_r2_scores)
                        mean_test_rmse = np.mean(test_rmse_scores)

                        col1, col2 = st.columns(2)
                        col1.metric("Mean Test RMSE", f"{mean_test_rmse:.4f}")
                        col2.metric("Mean Test R-squared (R2)", f"{mean_test_r2:.4f}")

                        # Create and display the table
                        metrics_data = {
                            "Train R²": train_r2_scores,
                            "Test R²": test_r2_scores,
                            "Train RMSE": train_rmse_scores,
                            "Test RMSE": test_rmse_scores,
                        }
                        metrics_df = pd.DataFrame(metrics_data, index=target_features)
                        st.write("### Detailed Metrics per Target")
                        st.dataframe(metrics_df.style.format("{:.4f}"))

                        st.subheader(f"Detailed Evaluation of Multi-Target Model")
                        fig_multi_eval = evaluate_multi_regression(
                            model=best_model, df=df, predictors=input_features, targets=target_features,
                            input_lags=input_lags, target_lags=target_lags, test_size=1 - split_fraction
                        )
                        st.pyplot(fig_multi_eval)

        # --- Display Multi-Target Results and Optimizer in Tabs (if model is trained) ---
        if st.session_state.get('multi_target_model_trained', False):
            tab1, tab2 = st.tabs(["Model Evaluation", "Process Optimizer"])

            with tab1:
                st.header("Model Evaluation")
                st.write("This tab shows the performance of the most recently trained multi-target model.")
                
                # Re-display metrics and plots using saved data
                best_model = st.session_state['best_multi_target_model']
                X_test = st.session_state['X_test_multi']
                y_test = st.session_state['y_test_multi']
                X_train = st.session_state['X_train_multi']
                y_train = st.session_state['y_train_multi']
                target_cols = st.session_state['multi_target_target_features']
                
                y_test_pred = best_model.predict(X_test)
                y_train_pred = best_model.predict(X_train)

                st.subheader("Model Performance")
                test_r2_scores = [r2_score(y_test.iloc[:, i], y_test_pred[:, i]) for i in range(y_test.shape[1])]
                test_rmse_scores = [np.sqrt(mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i])) for i in range(y_test.shape[1])]
                train_r2_scores = [r2_score(y_train.iloc[:, i], y_train_pred[:, i]) for i in range(y_train.shape[1])]
                train_rmse_scores = [np.sqrt(mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i])) for i in range(y_train.shape[1])]

                mean_test_r2 = np.mean(test_r2_scores)
                mean_test_rmse = np.mean(test_rmse_scores)

                col1, col2 = st.columns(2)
                col1.metric("Mean Test RMSE", f"{mean_test_rmse:.4f}")
                col2.metric("Mean Test R-squared (R2)", f"{mean_test_r2:.4f}")

                # Create and display the table
                metrics_data = {
                    "Train R²": train_r2_scores,
                    "Test R²": test_r2_scores,
                    "Train RMSE": train_rmse_scores,
                    "Test RMSE": test_rmse_scores,
                }
                metrics_df = pd.DataFrame(metrics_data, index=target_cols)
                st.write("### Detailed Metrics per Target")
                st.dataframe(metrics_df.style.format("{:.4f}"))

                st.subheader(f"Detailed Evaluation of Multi-Target Model")
                fig_multi_eval = evaluate_multi_regression(
                    model=st.session_state['best_multi_target_model'], df=st.session_state['multi_target_df'], 
                    predictors=st.session_state['multi_target_input_features'], targets=st.session_state['multi_target_target_features'], 
                    input_lags=st.session_state['multi_target_input_lags'], target_lags=st.session_state['multi_target_target_lags'],
                    test_size=1 - st.session_state['multi_target_split_fraction']
                )
                st.pyplot(fig_multi_eval)

            with tab2:
                st.header("Process Optimizer")
                # --- Optimizer UI ---
                st.subheader("Optimization Settings")
                
                # Retrieve saved model and data
                model = st.session_state['best_multi_target_model']
                df_full = st.session_state['multi_target_df']
                input_cols = st.session_state['multi_target_input_features']
                target_cols = st.session_state['multi_target_target_features']

                target_to_optimize = st.selectbox("Select Target to Optimize", options=target_cols, key="optimizer_target")
                optimization_goal = st.radio("Optimization Goal", ["Maximize", "Minimize"], horizontal=True, key="optimizer_goal")

                st.write("**Select Variables to Optimize (Manipulated Variables):**")
                # Only current (non-lagged) variables can be manipulated.
                all_manipulable_vars = [
                    col for col in input_cols if not any(lag_str in col for lag_str in ['_lag', '_mode'])
                ]
                
                free_vars = st.multiselect(
                    "Select which input variables the optimizer can change.",
                    options=all_manipulable_vars,
                    default=all_manipulable_vars,
                    key="optimizer_free_vars"
                )
                fixed_vars = {var: df_full[var].iloc[-1] for var in all_manipulable_vars if var not in free_vars}

                bounds_percent = st.slider("Optimization Bounds (%)", 1, 50, 10, help="Allow variables to change by ±X% of their last known value.", key="optimizer_bounds") / 100.0

                st.subheader("Select Rows to Optimize")
                st.info("Select one or more rows from the original dataset to run the optimizer on. The optimizer will use each row as a starting point.")
                row_indices_to_optimize = st.multiselect(
                    "Select Row Indices",
                    options=df_full.index.tolist(),
                    default=[df_full.index[-1]] # Default to the last row
                )

                if st.button("Run Batch Optimization", type="primary", key="optimizer_run_button"):
                    with st.spinner("Finding optimal settings..."):
                        start_time = time.time()
                        # Retrieve the necessary data from session state
                        df_lagged = make_multi_lagged_df(
                            st.session_state['multi_target_df'],
                            input_cols, # Use the full input_cols list
                            st.session_state['multi_target_target_features'],
                            st.session_state['multi_target_input_lags'],
                            st.session_state['multi_target_target_lags']
                        )
                        X_cols = st.session_state['X_test_multi'].columns.tolist()
                        st.write(X_cols)
                        batch_results = optimize_multiple_rows(
                            model=model,
                            df_full=df_full,
                            df_lagged=df_lagged,
                            row_indices=row_indices_to_optimize,
                            predictors=X_cols,
                            targets=target_cols,
                            target_to_optimize=target_to_optimize,
                            optimization_goal=optimization_goal,
                            bounds_percent=bounds_percent,
                            free_vars=free_vars
                        )
                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        if batch_results:
                            st.success(f"Optimization complete in {elapsed_time:.2f} seconds.")

                            st.subheader("Batch Optimization Results")
                            fig = plot_optimization_results(batch_results, target_to_optimize, optimization_goal)
                            st.pyplot(fig)
                            
                            # Call the new detailed plotting function
                            plot_detailed_optimization_comparison(batch_results, target_cols, free_vars)

                            st.subheader("Detailed Comparison Table")
                            st.write("This table shows the initial and optimized values for all inputs and targets for each selected row.")
                            
                            all_results_dfs = []
                            for res in batch_results:
                                row_idx = res['row_index']
                                
                                # Combine inputs and predictions for the initial state
                                initial_series = pd.concat([res['initial_inputs'], res['initial_predictions']])
                                initial_series['State'] = 'Initial'
                                initial_series['Row Index'] = row_idx
                                
                                # Create the "Actual" state with initial inputs and actual outputs
                                actual_series = pd.concat([res['initial_inputs'], res['actual_outputs']])
                                actual_series['State'] = 'Actual'
                                actual_series['Row Index'] = row_idx

                                # Combine inputs and predictions for the optimized state
                                optimized_series = pd.concat([res['optimized_inputs'], res['optimized_predictions']])
                                optimized_series['State'] = 'Optimized'
                                optimized_series['Row Index'] = row_idx
                                
                                # Add all three states to the list
                                all_results_dfs.extend([initial_series.to_frame().T, optimized_series.to_frame().T, actual_series.to_frame().T])

                            if all_results_dfs:
                                final_df = pd.concat(all_results_dfs, ignore_index=True)
                                # Reorder columns for better readability
                                final_df = final_df.set_index(['Row Index', 'State']).sort_index()
                                st.dataframe(final_df)
            
    elif Required_calcs == 'Anomaly Detection':
        # --- Anomaly Detection UI ---
        st.subheader("Select Target Variable(s)")
        st.info("The anomaly detection process will run individually for each target variable you select.")
        target_features_ad = st.multiselect("Select one or more target variables", options=numeric_columns, default=numeric_columns[-1:] if numeric_columns else [], key="target_features_ad")
        
        if 'run_ad_button' in st.session_state and st.session_state.run_ad_button:
            if not target_features_ad:
                st.error("Please select at least one target variable for anomaly detection.")
            else:
                # Use all numeric columns for feature-based methods
                all_numeric_features = df.select_dtypes(include=np.number).columns.tolist()
                
                # Loop through each selected target and run the analysis
                for target in target_features_ad:
                    ad_params = {
                        'z_threshold': st.session_state.ad_z,
                        'rolling_window': st.session_state.ad_roll,
                        'iso_contamination': st.session_state.ad_iso,
                        'lof_neighbors': st.session_state.ad_lof_n,
                        'lof_contamination': st.session_state.ad_lof_c,
                        'pca_components': st.session_state.ad_pca,
                        'recon_error_percentile': st.session_state.ad_recon,
                        'vote_threshold': st.session_state.ad_vote
                    }
                    methods_to_run = [method for method, selected in st.session_state.methods_selected.items() if selected]

                    with st.spinner(f"Running anomaly detection for '{target}'..."):
                        run_anomaly_detection(
                            df=df,
                            features=all_numeric_features,
                            target_col=target, # Pass a single target string
                            lag=st.session_state.ad_lag,
                            methods_to_run=methods_to_run,
                            params=ad_params
                        )
                st.success("Anomaly detection process complete for all selected targets.")
    
    elif Required_calcs == 'EDA':
        if st.session_state.get('examine_lags', False):
            st.header("Lag Correlation Analysis")
            st.subheader("Select Target Variable(s)")
            target_for_lag = st.selectbox("Select the Target Variable for Lag Analysis", options=numeric_columns, index=len(numeric_columns)-1 if len(numeric_columns) > 0 else 0, key="target_feature_lag")
            num_lags = st.slider("Number of Lags to Examine", min_value=1, max_value=48, value=24, key="num_lags_eda")
            
            if st.toggle("Run Lag Analysis", key="run_lag_analysis_button"):
                if not target_for_lag:
                    st.error("Please select a target variable.")
                else:
                    run_lag_correlation_analysis(df, target_for_lag, num_lags)

        if basic_EDA:
            perform_basic_eda(df)

        if examine_3D_feature:
            st.header("3D Feature Visualization")
            st.write("Select four numeric variables to create an interactive 3D scatter plot.")
            
            if len(numeric_columns) >= 4:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    x_col_3d = st.selectbox("Select X-axis variable", options=numeric_columns, index=0, key="3d_x")
                with col2:
                    y_col_3d = st.selectbox("Select Y-axis variable", options=numeric_columns, index=1, key="3d_y")
                with col3:
                    z_col_3d = st.selectbox("Select Z-axis variable", options=numeric_columns, index=2, key="3d_z")
                with col4:
                    color_col_3d = st.selectbox("Select Color variable", options=numeric_columns, index=3, key="3d_color")

                if st.button("Generate 3D Plot", key="run_3d_plot_button", type="primary"):
                    with st.spinner("Generating 3D plot..."):
                        fig_3d = plot_3d_plotly(df, x_col=x_col_3d, y_col=y_col_3d, z_col=z_col_3d, color_col=color_col_3d,
                                                title=f'3D Scatter Plot: {x_col_3d} vs {y_col_3d} vs {z_col_3d}')
                        st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("At least 4 numeric columns are required to generate a 3D plot.")
    

    else:
        st.info("Awaiting for a CSV file to be uploaded.")
