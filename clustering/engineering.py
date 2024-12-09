

##### Feature Engineering #####
# Calculate mean and standard deviation per participant (ID) for each ilr column
for col in ilr_resid_columns:
    df[f'{col}_mean'] = df.groupby('ID')[col].transform('mean')
    df[f'{col}_std'] = df.groupby('ID')[col].transform('std')
    df[f'{col}_min'] = df.groupby('ID')[col].transform('min')
    df[f'{col}_max'] = df.groupby('ID')[col].transform('max')

# Additional time-based feature: time point relative to the total duration for each participant
df['relative_time'] = df.groupby('ID')['time'].transform(lambda x: x / x.max())

# Deviation from group
# for col in ilr_columns:
#     df[f'{col}_deviation'] = df[col] - df.groupby(['Experiment', 'time'])[col].transform('mean')

# Interaction Terms: Control Variables with Time
# for col in control_columns:
#     df[f'{col}_time_interaction'] = df[col] * df['time']

# Residualizing ilr variables
# from statsmodels.regression.mixed_linear_model import MixedLM

# for col in ilr_columns:
#     model = MixedLM.from_formula(
#         f"{col} ~ excluded + injustice + personal + violence",
#         groups="Experiment", 
#         # re_formula="1|condition",
#         data=df
#     )
#     result = model.fit()
#     df[f'{col}_residual'] = df[col] - result.fittedvalues

# Moving Average or Exponential Moving Average (EMA)
window_size = 2
for col in ilr_resid_columns:
    df[f'{col}_moving_avg'] = df.groupby('ID')[col].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    df[f'{col}_ema'] = df.groupby('ID')[col].transform(lambda x: x.ewm(span=window_size, adjust=False).mean())
ilr_moving_avg_cols = ['ilr1_residual_moving_avg', 'ilr2_residual_moving_avg', 'ilr3_residual_moving_avg']
ilr_ema_cols = ['ilr1_residual_ema', 'ilr2_residual_ema', 'ilr3_residual_ema']

# Fourier transformation
fourier_components = 1
# Compute Fourier transform for each 'ilr' variable per participant
for column_group in [ilr_moving_avg_cols, ilr_ema_cols]:
    for col in column_group:
        def compute_fourier(x):
            fft_vals = np.fft.fft(x)
            # Retain only the first N components (real and imaginary parts)
            return np.hstack([fft_vals.real[:fourier_components], fft_vals.imag[:fourier_components]])

        # Apply Fourier transformation and store as separate columns
        fourier_df = (
            df.groupby('ID')[col]
            .apply(lambda x: compute_fourier(x.values))
            .apply(pd.Series)
            .rename(columns=lambda i: f'{col}_fourier_{i+1}')
        )

        # Add Fourier features back to the dataframe
        df = df.join(fourier_df, on='ID')


##### Prepare Time Series Data for Clustering #####
time_series_data = []
grouped = df.groupby("ID")
feature_columns = [
    # 'ilr1', 'ilr2', 'ilr3', 

    # 'condition',
    # 'gender', 'age',
    # 'excluded', 'injustice', 'personal', 'violence',

    # 'ilr1_mean', 'ilr2_mean', 'ilr3_mean',
    # 'ilr1_std', 'ilr2_std', 'ilr3_std',
    # 'ilr1_min', 'ilr2_min', 'ilr3_min',
    # 'ilr1_max', 'ilr2_max', 'ilr3_max',

    'relative_time',

    'ilr1_residual_mean', 'ilr2_residual_mean', 'ilr3_residual_mean',
    'ilr1_residual_std', 'ilr2_residual_std', 'ilr3_residual_std',
    'ilr1_residual_min', 'ilr2_residual_min', 'ilr3_residual_min',
    'ilr1_residual_max', 'ilr2_residual_max', 'ilr3_residual_max',

    # 'ilr1_deviation', 'ilr2_deviation', 'ilr3_deviation',
    # 'excluded_time_interaction', 'injustice_time_interaction', 'personal_time_interaction', 'violence_time_interaction',
    # 'ilr1_moving_avg', 'ilr2_moving_avg', 'ilr3_moving_avg', 
    # 'ilr1_ema', 'ilr2_ema', 'ilr3_ema',
    #'ilr1_residual', 'ilr2_residual', 'ilr3_residual',
    #'ilr1_fourier_1', 'ilr2_fourier_1', 'ilr3_fourier_1',
    # 'ilr1_fourier_2', 'ilr2_fourier_2', 'ilr3_fourier_2'
    # 'ilr1_moving_avg_fourier_1', 'ilr2_moving_avg_fourier_1', 'ilr3_moving_avg_fourier_1',
    # 'ilr1_moving_avg_fourier_2', 'ilr2_moving_avg_fourier_2', 'ilr3_moving_avg_fourier_2',
    # 'ilr1_residual_fourier_1', 'ilr2_residual_fourier_1', 'ilr3_residual_fourier_1',

    # 'ilr1_moving_avg_residual_fourier_1', 'ilr2_moving_avg_residual_fourier_1', 'ilr3_moving_avg_residual_fourier_1',
    # 'ilr1_moving_avg_residual_fourier_2', 'ilr2_moving_avg_residual_fourier_2', 'ilr3_moving_avg_residual_fourier_2'

    'ilr1_residual_moving_avg_fourier_1', 'ilr2_residual_moving_avg_fourier_1', 'ilr3_residual_moving_avg_fourier_1',
    'ilr1_residual_moving_avg_fourier_2', 'ilr2_residual_moving_avg_fourier_2', 'ilr3_residual_moving_avg_fourier_2',

    # 'ilr1_residual_ema_fourier_1', 'ilr2_residual_ema_fourier_1', 'ilr3_residual_ema_fourier_1',
    # 'ilr1_residual_ema_fourier_2', 'ilr2_residual_ema_fourier_2', 'ilr3_residual_ema_fourier_2'
]
print(f"{len(feature_columns)} features used:")
print(feature_columns)

