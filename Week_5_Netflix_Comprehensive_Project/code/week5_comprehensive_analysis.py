# ============================================================
# WEEK 5
# COMPREHENSIVE DATA SCIENCE PROJECT REPORTING
# Dataset: Netflix Movies and TV Shows
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CREATE FOLDERS
# ============================================================

os.makedirs("../visualizations", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

file_path = "../data/netflix_titles.csv"

df = pd.read_csv(
    file_path,
    encoding="latin1"
)

print("=" * 60)
print("NETFLIX DATASET LOADED")
print("=" * 60)

print("Dataset shape:", df.shape)


# ============================================================
# 3. INITIAL DATA INSPECTION
# ============================================================

print("\nFirst 5 records:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())


# ============================================================
# 4. DATA CLEANING
# ============================================================

original_rows = len(df)

duplicate_count = df.duplicated().sum()

df = df.drop_duplicates()

print("\nDuplicates removed:", duplicate_count)


# Convert date_added

df["date_added"] = pd.to_datetime(
    df["date_added"],
    errors="coerce"
)


# Convert release year

df["release_year"] = pd.to_numeric(
    df["release_year"],
    errors="coerce"
)


# Missing value handling

df["director"] = df["director"].fillna(
    "Unknown"
)

df["country"] = df["country"].fillna(
    "Unknown"
)

df["cast"] = df["cast"].fillna(
    "Not Available"
)

df["rating"] = df["rating"].fillna(
    df["rating"].mode()[0]
)

df["duration"] = df["duration"].fillna(
    "Unknown"
)

df["listed_in"] = df["listed_in"].fillna(
    "Unknown"
)


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

# Year content was added

df["added_year"] = df[
    "date_added"
].dt.year


# Number of countries

df["country_count"] = df[
    "country"
].apply(
    lambda x: len(str(x).split(","))
)


# Number of genres

df["genre_count"] = df[
    "listed_in"
].apply(
    lambda x: len(str(x).split(","))
)


# Number of cast members

df["cast_count"] = df[
    "cast"
].apply(
    lambda x: len(str(x).split(","))
)


# ============================================================
# 6. BASIC DATASET SUMMARY
# ============================================================

total_titles = len(df)

movie_count = (
    df["type"] == "Movie"
).sum()

tv_count = (
    df["type"] == "TV Show"
).sum()

movie_percentage = (
    movie_count / total_titles
) * 100

tv_percentage = (
    tv_count / total_titles
) * 100


print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("Total titles:", total_titles)
print("Movies:", movie_count)
print("TV Shows:", tv_count)
print(f"Movie percentage: {movie_percentage:.2f}%")
print(f"TV Show percentage: {tv_percentage:.2f}%")


# ============================================================
# 7. TOP COUNTRIES
# ============================================================

country_data = (
    df["country"]
    .str.split(",")
    .explode()
    .str.strip()
)

top_countries = (
    country_data
    .value_counts()
    .head(10)
)

print("\nTop 10 countries:")
print(top_countries)


# ============================================================
# 8. TOP GENRES
# ============================================================

genre_data = (
    df["listed_in"]
    .str.split(",")
    .explode()
    .str.strip()
)

top_genres = (
    genre_data
    .value_counts()
    .head(10)
)

print("\nTop 10 genres:")
print(top_genres)


# ============================================================
# 9. TOP RATINGS
# ============================================================

top_ratings = (
    df["rating"]
    .value_counts()
    .head(10)
)

print("\nTop ratings:")
print(top_ratings)


# ============================================================
# 10. YEARLY CONTENT TREND
# ============================================================

yearly_content = (
    df.groupby("release_year")
    .size()
    .reset_index(
        name="title_count"
    )
)

yearly_content = yearly_content[
    yearly_content["release_year"] >= 2000
]


# ============================================================
# 11. VISUALIZATION 1
# MOVIES VS TV SHOWS
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="type"
)

plt.title(
    "Distribution of Movies and TV Shows"
)

plt.xlabel("Content Type")
plt.ylabel("Number of Titles")

plt.tight_layout()

plt.savefig(
    "../visualizations/01_content_type_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. VISUALIZATION 2
# RELEASE YEAR TREND
# ============================================================

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=yearly_content,
    x="release_year",
    y="title_count"
)

plt.title(
    "Netflix Content Distribution by Release Year"
)

plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.tight_layout()

plt.savefig(
    "../visualizations/02_release_year_trend.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. VISUALIZATION 3
# TOP COUNTRIES
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)

plt.title(
    "Top 10 Countries by Number of Netflix Titles"
)

plt.xlabel("Number of Titles")
plt.ylabel("Country")

plt.tight_layout()

plt.savefig(
    "../visualizations/03_top_countries.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. VISUALIZATION 4
# TOP GENRES
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_genres.values,
    y=top_genres.index
)

plt.title(
    "Top 10 Netflix Genres"
)

plt.xlabel("Number of Titles")
plt.ylabel("Genre")

plt.tight_layout()

plt.savefig(
    "../visualizations/04_top_genres.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. VISUALIZATION 5
# RATINGS
# ============================================================

plt.figure(figsize=(10, 6))

rating_order = (
    df["rating"]
    .value_counts()
    .head(10)
    .index
)

sns.countplot(
    data=df,
    y="rating",
    order=rating_order
)

plt.title(
    "Netflix Content Rating Distribution"
)

plt.xlabel("Number of Titles")
plt.ylabel("Rating")

plt.tight_layout()

plt.savefig(
    "../visualizations/05_rating_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. WEEK 3
# HYPOTHESIS TESTING
# ============================================================

movies = df[
    df["type"] == "Movie"
]["release_year"].dropna()

tv_shows = df[
    df["type"] == "TV Show"
]["release_year"].dropna()


# Welch's t-test

t_statistic, t_p_value = stats.ttest_ind(
    movies,
    tv_shows,
    equal_var=False
)


print("\n" + "=" * 60)
print("WELCH T-TEST")
print("=" * 60)

print("T-statistic:", t_statistic)
print("P-value:", t_p_value)


# ============================================================
# 17. CHI-SQUARE TEST
# ============================================================

rating_table = pd.crosstab(
    df["type"],
    df["rating"]
)

chi2, chi_p_value, degrees_of_freedom, expected = (
    stats.chi2_contingency(
        rating_table
    )
)


print("\n" + "=" * 60)
print("CHI-SQUARE TEST")
print("=" * 60)

print("Chi-square:", chi2)
print("P-value:", chi_p_value)
print("Degrees of freedom:", degrees_of_freedom)


# ============================================================
# 18. WEEK 3 VISUALIZATION
# TYPE VS RATING
# ============================================================

plt.figure(figsize=(12, 6))

sns.heatmap(
    rating_table,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Content Type vs Content Rating"
)

plt.xlabel("Rating")
plt.ylabel("Content Type")

plt.tight_layout()

plt.savefig(
    "../visualizations/06_type_rating_heatmap.png",
    dpi=300
)

plt.close()


# ============================================================
# 19. WEEK 4
# MACHINE LEARNING MODEL
# ============================================================

df["target"] = df[
    "type"
].map({
    "Movie": 0,
    "TV Show": 1
})


features = [
    "release_year",
    "added_year",
    "rating",
    "country_count",
    "genre_count",
    "cast_count"
]

X = df[features]

y = df["target"]


# Remove missing target

valid_rows = y.notna()

X = X[
    valid_rows
]

y = y[
    valid_rows
]


# ============================================================
# 20. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


numeric_features = [
    "release_year",
    "added_year",
    "country_count",
    "genre_count",
    "cast_count"
]

categorical_features = [
    "rating"
]


# ============================================================
# 21. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 22. LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# Train

pipeline.fit(
    X_train,
    y_train
)


# Predictions

y_pred = pipeline.predict(
    X_test
)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 23. ML PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 60)
print("MACHINE LEARNING PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")


# ============================================================
# 24. VISUALIZATION 7
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Movie",
        "TV Show"
    ],
    yticklabels=[
        "Movie",
        "TV Show"
    ]
)

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "../visualizations/07_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 25. VISUALIZATION 8
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC={auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title(
    "ROC Curve"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "../visualizations/08_roc_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# 26. VISUALIZATION 9
# MODEL METRIC COMPARISON
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]
})


plt.figure(figsize=(9, 5))

bars = sns.barplot(
    data=metrics,
    x="Metric",
    y="Score"
)

plt.ylim(
    0,
    1
)

plt.title(
    "Machine Learning Model Performance"
)

plt.ylabel(
    "Score"
)

for bar, value in zip(
    bars.patches,
    metrics["Score"]
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "../visualizations/09_model_metrics.png",
    dpi=300
)

plt.close()


# ============================================================
# 27. STRATEGIC INSIGHTS
# ============================================================

top_country_name = (
    top_countries.index[0]
)

top_country_count = (
    top_countries.iloc[0]
)

top_genre_name = (
    top_genres.index[0]
)

top_genre_count = (
    top_genres.iloc[0]
)

top_rating_name = (
    top_ratings.index[0]
)

top_rating_count = (
    top_ratings.iloc[0]
)


# ============================================================
# 28. CREATE INSIGHTS TABLE
# ============================================================

insights = pd.DataFrame({
    "Area": [
        "Catalog Composition",
        "International Presence",
        "Genre",
        "Rating",
        "Statistical Analysis",
        "Machine Learning"
    ],

    "Finding": [
        f"Movies represent {movie_percentage:.2f}% "
        f"and TV Shows represent {tv_percentage:.2f}% "
        "of the catalog.",

        f"{top_country_name} has the highest "
        f"number of associated titles among the "
        "countries represented in the dataset.",

        f"{top_genre_name} is the most frequently "
        f"listed genre/category.",

        f"{top_rating_name} is the most frequently "
        f"observed content rating.",

        f"Welch t-test p-value = "
        f"{t_p_value:.6f}; Chi-square p-value = "
        f"{chi_p_value:.6f}.",

        f"Logistic Regression achieved "
        f"accuracy={accuracy:.4f}, "
        f"F1={f1:.4f}, "
        f"ROC-AUC={auc:.4f}."
    ]
})


insights.to_csv(
    "../outputs/strategic_insights.csv",
    index=False
)


# ============================================================
# 29. CREATE SUMMARY RESULTS
# ============================================================

summary_results = pd.DataFrame({
    "Metric": [
        "Total Titles",
        "Movies",
        "TV Shows",
        "Movie Percentage",
        "TV Show Percentage",
        "T-Test Statistic",
        "T-Test P-Value",
        "Chi-Square Statistic",
        "Chi-Square P-Value",
        "Chi-Square Degrees of Freedom",
        "ML Accuracy",
        "ML Precision",
        "ML Recall",
        "ML F1 Score",
        "ML ROC-AUC"
    ],

    "Value": [
        total_titles,
        movie_count,
        tv_count,
        movie_percentage,
        tv_percentage,
        t_statistic,
        t_p_value,
        chi2,
        chi_p_value,
        degrees_of_freedom,
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]
})


summary_results.to_csv(
    "../outputs/project_summary_results.csv",
    index=False
)


# ============================================================
# 30. STRATEGIC RECOMMENDATIONS
# ============================================================

recommendations = pd.DataFrame({

    "Recommendation": [

        "Monitor the balance between Movies and TV Shows "
        "when evaluating catalog composition.",

        "Use country-level catalog analysis to identify "
        "regional content opportunities.",

        "Track high-frequency genres and evaluate whether "
        "content investment aligns with audience demand.",

        "Use rating distributions to support audience "
        "segmentation and content categorization.",

        "Combine statistical findings with descriptive "
        "analysis rather than relying on a single metric.",

        "Compare multiple machine learning algorithms "
        "before selecting a production model.",

        "Improve the model using additional features, "
        "cross-validation and hyperparameter tuning."
    ],

    "Reason": [

        "Catalog composition can change over time "
        "and should be monitored.",

        "Country-level patterns can reveal geographic "
        "concentration and potential expansion areas.",

        "Genre patterns provide a measurable view of "
        "catalog composition.",

        "Ratings provide information about the type "
        "of audience a title may target.",

        "Multiple analytical methods provide a broader "
        "evidence base.",

        "A single baseline model may not capture all "
        "patterns in the dataset.",

        "Additional modeling can improve robustness "
        "and generalization."
    ]
})


recommendations.to_csv(
    "../outputs/strategic_recommendations.csv",
    index=False
)


# ============================================================
# 31. FINAL REPORT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("WEEK 5 PROJECT COMPLETED")
print("=" * 60)

print("\nGenerated visualizations:")

for i in range(1, 10):
    print(
        f"{i}. ../visualizations/"
        f"{str(i).zfill(2)}_*.png"
    )

print("\nGenerated output files:")

print(
    "1. ../outputs/strategic_insights.csv"
)

print(
    "2. ../outputs/project_summary_results.csv"
)

print(
    "3. ../outputs/strategic_recommendations.csv"
)

print("\nUse these outputs in the final DOC report.")

print("\nAnalysis completed successfully!")