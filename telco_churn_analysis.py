# ============================================================
# 6001CMD Machine Learning
# Telco Customer Churn Analysis
# Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE


# ============================================================
# 1.4 DATASET CHARACTERISTICS
# ============================================================

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display dataset dimensions and first five records
print("\n===== 1.4 Dataset Characteristics =====")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()


# ============================================================
# 3.0 CRITICAL ANALYSIS OF DATA QUALITY
# ============================================================


# ============================================================
# 3.1 MISSING VALUES AND DATA-TYPE CONSISTENCY
# ============================================================

print("\n===== 3.1 Missing Values and Data-Type Consistency =====")

# Convert TotalCharges from string to numerical format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3.2 DUPLICATE RECORDS
# ============================================================

print("\n===== 3.2 Duplicate Records =====")

# Check complete duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Check duplicate customer identifiers
print(
    "Duplicate customer IDs:",
    df["customerID"].duplicated().sum()
)


# ============================================================
# 3.3 FEATURE DISTRIBUTIONS AND SKEWNESS
# ============================================================

print("\n===== 3.3 Feature Distributions and Skewness =====")

numeric_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

# Descriptive statistics
print("\nDescriptive statistics:")
print(df[numeric_cols].describe())

# Skewness
print("\nSkewness:")
print(df[numeric_cols].skew())

# Histograms
df[numeric_cols].hist(
    figsize=(10, 6),
    bins=30
)

plt.tight_layout()
plt.show()


# ============================================================
# 3.4 OUTLIER ANALYSIS
# ============================================================

print("\n===== 3.4 Outlier Analysis =====")

# Generate boxplots for numerical attributes
for col in numeric_cols:
    plt.figure(figsize=(7, 3))

    plt.boxplot(
        df[col].dropna(),
        orientation="horizontal"
    )

    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)

    plt.tight_layout()
    plt.show()


# ============================================================
# 3.5 CORRELATION ANALYSIS
# ============================================================

print("\n===== 3.5 Correlation Analysis =====")

numeric_df = df[
    [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]
]

# Calculate correlation matrix
corr = numeric_df.corr()

print("\nCorrelation matrix:")
print(corr)

# Visualise correlation matrix
plt.figure(figsize=(7, 5))

plt.imshow(
    corr,
    cmap="coolwarm",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=45
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 3.6 CLASS IMBALANCE
# ============================================================

print("\n===== 3.6 Class Imbalance =====")

# Count target classes
churn_counts = df["Churn"].value_counts()

# Calculate target percentages
churn_percentage = (
    df["Churn"].value_counts(normalize=True) * 100
)

print("\nChurn counts:")
print(churn_counts)

print("\nChurn percentages:")
print(churn_percentage)

# Visualise target distribution
churn_counts.plot(kind="bar")

plt.title("Distribution of Churn")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# 3.7 NOISE AND DATA INCONSISTENCY
# ============================================================

print("\n===== 3.7 Noise and Data Inconsistency =====")

# Identify categorical attributes
categorical_cols = df.select_dtypes(
    include=["object", "string"]
).columns

# Display unique values in categorical attributes
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].unique())


# ============================================================
# 4.0 DATA PREPROCESSING STRATEGY EVALUATION
# ============================================================


# ============================================================
# 4.1 DATA CLEANING AND MISSING-VALUE TREATMENT
# ============================================================

print("\n===== 4.1 Missing-Value Treatment =====")

# Split dataset before learning the imputation value
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

# Create median imputer
imputer = SimpleImputer(
    strategy="median"
)

# Learn median using training data only
train_df[["TotalCharges"]] = imputer.fit_transform(
    train_df[["TotalCharges"]]
)

# Apply the learned median to testing data
test_df[["TotalCharges"]] = imputer.transform(
    test_df[["TotalCharges"]]
)

print(
    "Median learned from training data:",
    imputer.statistics_[0]
)

print(
    "Training missing values:",
    train_df["TotalCharges"].isnull().sum()
)

print(
    "Testing missing values:",
    test_df["TotalCharges"].isnull().sum()
)


# ============================================================
# 4.2 DUPLICATE REMOVAL AND NOISE REDUCTION
# ============================================================

print("\n===== 4.2 Duplicate Removal and Noise Reduction =====")

# No removal is performed because Section 3.2 identified
# no duplicate rows or duplicate customer IDs.

print(
    "Duplicate removal not required:",
    df.duplicated().sum() == 0
)


# ============================================================
# 4.3 IDENTIFIER REMOVAL AND FEATURE SELECTION
# ============================================================

print("\n===== 4.3 Identifier Removal and Feature Selection =====")

# Remove identifier and target from predictors
X = df.drop(
    columns=["customerID", "Churn"]
)

# Define target
y = df["Churn"]

print("Predictor shape:", X.shape)
print("Target shape:", y.shape)

print(
    "customerID in predictors:",
    "customerID" in X.columns
)

print(
    "Churn in predictors:",
    "Churn" in X.columns
)


# ============================================================
# 4.4 CATEGORICAL ENCODING
# ============================================================

print("\n===== 4.4 Categorical Encoding =====")

# Identify categorical predictor attributes
categorical_cols = X.select_dtypes(
    include=["object", "string"]
).columns

# Create one-hot encoder
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Fit encoder using training data only
encoder.fit(
    train_df[categorical_cols]
)

# Transform training categorical attributes
X_train_encoded = encoder.transform(
    train_df[categorical_cols]
)

print(
    "Number of categorical attributes:",
    len(categorical_cols)
)

print(
    "Encoded feature count:",
    X_train_encoded.shape[1]
)

print("\nExample encoded features:")

print(
    encoder.get_feature_names_out(
        categorical_cols
    )[:10]
)


# ============================================================
# 4.5 STANDARDISATION AND NORMALISATION
# ============================================================

print("\n===== 4.5 Standardisation =====")

numeric_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

# Create standard scaler
scaler = StandardScaler()

# Fit and transform training numerical attributes
X_train_scaled = scaler.fit_transform(
    train_df[numeric_cols]
)

print("\nMeans after standardisation:")
print(
    X_train_scaled.mean(axis=0)
)

print("\nStandard deviations after standardisation:")
print(
    X_train_scaled.std(axis=0)
)


# ============================================================
# 4.6 LOG AND POWER TRANSFORMATIONS
# ============================================================

print("\n===== 4.6 Log Transformation Evaluation =====")

# Calculate original skewness
original_skew = train_df[
    "TotalCharges"
].skew()

# Apply logarithmic transformation for evaluation
total_charges_log = np.log1p(
    train_df["TotalCharges"]
)

# Calculate transformed skewness
log_skew = total_charges_log.skew()

print(
    "Original TotalCharges skewness:",
    original_skew
)

print(
    "Log-transformed TotalCharges skewness:",
    log_skew
)


# ============================================================
# 4.7 CLASS BALANCING
# ============================================================

print("\n===== 4.7 Class Balancing with SMOTE =====")

# Prepare predictors and target
X_smote = df.drop(
    columns=["customerID", "Churn"]
)

y_smote = df["Churn"]

# Encode categorical variables for SMOTE evaluation
X_smote = pd.get_dummies(
    X_smote,
    drop_first=False
)

# Stratified train-test split
X_train_smote, X_test_smote, y_train_smote, y_test_smote = (
    train_test_split(
        X_smote,
        y_smote,
        test_size=0.20,
        random_state=42,
        stratify=y_smote
    )
)

# Impute missing TotalCharges using training data only
smote_imputer = SimpleImputer(
    strategy="median"
)

X_train_smote[["TotalCharges"]] = (
    smote_imputer.fit_transform(
        X_train_smote[["TotalCharges"]]
    )
)

X_test_smote[["TotalCharges"]] = (
    smote_imputer.transform(
        X_test_smote[["TotalCharges"]]
    )
)

# Apply SMOTE only to training data
smote = SMOTE(
    random_state=42
)

X_train_balanced, y_train_balanced = (
    smote.fit_resample(
        X_train_smote,
        y_train_smote
    )
)

print("\nTraining class distribution before SMOTE:")
print(
    y_train_smote.value_counts()
)

print("\nTraining class distribution after SMOTE:")
print(
    y_train_balanced.value_counts()
)

print("\nTesting class distribution:")
print(
    y_test_smote.value_counts()
)


# ============================================================
# 4.8 PCA AND DIMENSIONALITY REDUCTION
# ============================================================

print("\n===== 4.8 PCA Evaluation =====")

# PCA is not applied as a mandatory preprocessing step.
# The encoded feature space remains manageable and retaining
# original features provides greater interpretability.

print(
    "PCA not applied: dimensionality remains manageable."
)


# ============================================================
# 4.9 PROPOSED COMPLETE PREPROCESSING PIPELINE
# ============================================================

print("\n===== 4.9 Complete Preprocessing Pipeline =====")

# Separate predictors and binary target
X = df.drop(
    columns=["customerID", "Churn"]
)

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Define feature groups
numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

# Numerical preprocessing pipeline
numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

# Categorical preprocessing pipeline
categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

# Combine numerical and categorical transformations
preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numeric_features
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features
    )
])

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Fit preprocessing transformations on training data only
X_train_processed = preprocessor.fit_transform(
    X_train
)

# Apply learned transformations to testing data
X_test_processed = preprocessor.transform(
    X_test
)

# Verify final preprocessing results
print(
    "Original training shape:",
    X_train.shape
)

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Original testing shape:",
    X_test.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)

print(
    "Missing values in processed training data:",
    np.isnan(X_train_processed).sum()
)

print(
    "Missing values in processed testing data:",
    np.isnan(X_test_processed).sum()
)

print(
    "\nPipeline execution completed successfully."
)