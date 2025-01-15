# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to safely load datasets with error handling
def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip', comment='#')
        print(f"Loaded {filepath} successfully.")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# Load datasets
male_file = "C:/Users/Shravani/Downloads/nhanes_adult_male_bmx_2020.csv"
female_file = "C:/Users/Shravani/Downloads/nhanes_adult_female_bmx_2020.csv"

male_df = load_dataset(male_file)
female_df = load_dataset(female_file)

if male_df is not None and female_df is not None:
    # Convert DataFrames to numpy arrays
    male = male_df.to_numpy()
    female = female_df.to_numpy()

    # Confirm shapes and preview data
    print(f"Male data shape: {male.shape}")
    print(f"Female data shape: {female.shape}")
    print("Male data preview:\n", male[:5])
    print("Female data preview:\n", female[:5])

    # 2. Histograms of Male and Female Weights
    plt.figure(figsize=(10, 6))

    # Female Weights Histogram
    plt.subplot(2, 1, 1)
    plt.hist(female[:, 0], bins=20, color='pink', edgecolor='black')
    plt.title('Female Weights')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Frequency')

    # Male Weights Histogram
    plt.subplot(2, 1, 2)
    plt.hist(male[:, 0], bins=20, color='blue', edgecolor='black')
    plt.title('Male Weights')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Frequency')

    # Set identical x-axis limits
    plt.xlim(30, 150)
    plt.tight_layout()
    plt.show()

    # 3. Box-and-Whisker Plot for Weights
    plt.figure(figsize=(8, 6))
    plt.boxplot([female[:, 0], male[:, 0]], labels=['Female', 'Male'], patch_artist=True)
    plt.title('Boxplot of Weights')
    plt.ylabel('Weight (kg)')
    plt.grid(True)
    plt.show()

    # 4. Compute Basic Numerical Aggregates
    def compute_aggregates(data):
        return {
            'Mean': np.mean(data),
            'Median': np.median(data),
            'Variance': np.var(data),
            'Standard Deviation': np.std(data),
            'Skewness': pd.Series(data).skew(),
            'Kurtosis': pd.Series(data).kurt()
        }

    male_agg = compute_aggregates(male[:, 0])
    female_agg = compute_aggregates(female[:, 0])

    print("Male Weight Statistics:", male_agg)
    print("Female Weight Statistics:", female_agg)

    # 5. Add BMI Column for Females
    female_bmi = female[:, 0] / ((female[:, 1] / 100) ** 2)
    female = np.hstack((female, female_bmi.reshape(-1, 1)))

    print(f"Female data with BMI shape: {female.shape}")

    # 6. Standardize Female Data (z-scores)
    zfemale = (female - female.mean(axis=0)) / female.std(axis=0)
    print("Standardized Female Data:\n", zfemale[:5])

    # 7. Scatterplot Matrix
    selected_columns = [0, 1, 6, 5, -1]  # Weight, Height, Waist, Hip, BMI
    zfemale_subset = zfemale[:, selected_columns]

    # Convert to DataFrame for pairplot
    df = pd.DataFrame(zfemale_subset, columns=['Weight', 'Height', 'Waist', 'Hip', 'BMI'])
    sns.pairplot(df)
    plt.show()

    # Compute Correlations
    pearson_corr = df.corr(method='pearson')
    spearman_corr = df.corr(method='spearman')
    print("Pearson Correlation:\n", pearson_corr)
    print("Spearman Correlation:\n", spearman_corr)

    # 8. Compute Ratios and Add Columns
    female_ratios = np.vstack((female[:, 6] / female[:, 1], female[:, 6] / female[:, 5])).T
    male_ratios = np.vstack((male[:, 6] / male[:, 1], male[:, 6] / male[:, 5])).T

    female = np.hstack((female, female_ratios))
    male = np.hstack((male, male_ratios))

    # 9. Boxplot of Ratios
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [female[:, -2], male[:, -2], female[:, -1], male[:, -1]],
        labels=['Female Waist/Height', 'Male Waist/Height', 'Female Waist/Hip', 'Male Waist/Hip'],
        patch_artist=True
    )
    plt.title('Boxplot of Ratios')
    plt.ylabel('Ratio')
    plt.grid(True)
    plt.show()

    # 10. Print Extremes of BMI
    lowest_bmi_indices = np.argsort(female[:, -1])[:5]
    highest_bmi_indices = np.argsort(female[:, -1])[-5:]

    print("Lowest BMI Participants:\n", zfemale[lowest_bmi_indices])
    print("Highest BMI Participants:\n", zfemale[highest_bmi_indices])

else:
    print("Data loading failed. Please check the file paths or content.")
