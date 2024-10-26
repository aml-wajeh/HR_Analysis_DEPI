import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging
from datetime import datetime
from sqlalchemy import create_engine
import scipy.sparse as sp
import sklearn
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_CONFIG = {
    "DRIVER": "{SQL Server}",
    "SERVER": ".",
    "DATABASE": "Aml",
    "Trusted_Connection": "yes",
    "MultipleActiveResultSets": "True",
    "TrustServerCertificate": "true"
}

TABLES = ['EducationLevel', 'SatisfiedLevel', 'RatingLevel', 'Employee', 'PerformanceRating']

# Function to connect to the database using SQLAlchemy
def connect_to_database():
    try:
        conn_str = ';'.join(f"{k}={v}" for k, v in DB_CONFIG.items())
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={conn_str}")
        return engine
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

# Function to fetch data from a table
def fetch_table_data(engine, table_name):
    query = f"SELECT * FROM {table_name}"
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error(f"Error fetching data from {table_name}: {e}")
        raise


# Function to preprocess the Employee table
def preprocess_employee(df):
    logger.info("Preprocessing Employee data")
    
    # Convert boolean columns properly
    df['Attrition'] = df['Attrition'].astype(int).astype(bool)
    df['OverTime'] = df['OverTime'].astype(int).astype(bool)
    
    # Convert date columns
    df['HireDate'] = pd.to_datetime(df['HireDate'])
    
    # Calculate derived features
    current_date = datetime.now()
    df['TenureYears'] = (current_date - df['HireDate']).dt.days / 365.25
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    df['TotalExperience'] = df['YearsAtCompany'] + df['YearsInMostRecentRole'] + df['YearsSinceLastPromotion']
    
    # Create job level without using salary (to avoid correlation issues)
    df['JobLevel'] = pd.qcut(df['YearsAtCompany'], q=5, labels=['Entry', 'Junior', 'Mid', 'Senior', 'Executive'])
    
    return df

# Function to preprocess the PerformanceRating table
def preprocess_performance(df):
    logger.info("Preprocessing PerformanceRating data")
    
    # Convert date columns to datetime
    df['ReviewDate'] = pd.to_datetime(df['ReviewDate'])
    
    # Create a new feature for review year
    df['ReviewYear'] = df['ReviewDate'].dt.year
    
    # Calculate the ratio of training opportunities taken
    df['TrainingParticipationRatio'] = df['TrainingOpportunitiesTaken'] / df['TrainingOpportunitiesWithinYear']
    
    # Create a composite satisfaction score
    satisfaction_columns = ['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
    df['CompositeSatisfactionScore'] = df[satisfaction_columns].mean(axis=1)
    
    return df

def merge_tables(tables):
    logger.info("Merging tables")
    
    try:
        employee = tables['Employee'].copy()
        performance = tables['PerformanceRating'].copy()
        education = tables['EducationLevel'].copy()
        satisfied = tables['SatisfiedLevel'].copy()
        rating = tables['RatingLevel'].copy()
        
        # Ensure proper data types
        employee['EmployeeID'] = employee['EmployeeID'].astype(str)
        performance['EmployeeID'] = performance['EmployeeID'].astype(str)
        
        # Convert foreign key columns to integers for proper joining
        employee['Education'] = employee['Education'].astype(int)
        performance['EnvironmentSatisfaction'] = performance['EnvironmentSatisfaction'].astype(int)
        performance['SelfRating'] = performance['SelfRating'].astype(int)
        
        # Perform merges
        merged = pd.merge(employee, performance, on='EmployeeID', how='left')
        merged = pd.merge(merged, education, left_on='Education', right_on='EducationLevelID', how='left')
        merged = pd.merge(merged, satisfied, left_on='EnvironmentSatisfaction', right_on='SatisfactionID', how='left')
        merged = pd.merge(merged, rating, left_on='SelfRating', right_on='RatingID', how='left')
        
        # Remove duplicate columns and reset index
        merged = merged.loc[:,~merged.columns.duplicated()].reset_index(drop=True)
        
        return merged
        
    except Exception as e:
        logger.error(f"Error during table merge: {e}")
        raise

# Function to handle missing values and encode categorical variables
def preprocess_data(df):
    logger.info("Preprocessing merged data")
    logger.info(f"Input data shape: {df.shape}")
    
    # Separate 'Attrition' column
    attrition = df['Attrition']
    
    # Remove ID and name columns before preprocessing
    columns_to_drop = df.columns[df.columns.str.contains('ID|Name', case=False)].tolist()
    df_without_attrition = df.drop(columns=['Attrition'] + columns_to_drop)
    
    # Identify numeric and categorical columns
    numeric_features = df_without_attrition.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df_without_attrition.select_dtypes(include=['object', 'category', 'bool']).columns
    
    logger.info(f"Numeric features: {numeric_features.tolist()}")
    logger.info(f"Categorical features: {categorical_features.tolist()}")
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Check scikit-learn version and create appropriate OneHotEncoder
    sklearn_version = sklearn.__version__
    logger.info(f"scikit-learn version: {sklearn_version}")
    
    if sklearn.__version__ >= '0.23':
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', onehot_encoder)
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the data
    processed_data = preprocessor.fit_transform(df_without_attrition)
    logger.info(f"Processed data shape: {processed_data.shape}")
    
    # Get feature names
    onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    if hasattr(onehot_encoder, 'get_feature_names_out'):
        cat_feature_names = onehot_encoder.get_feature_names_out(categorical_features).tolist()
    else:
        cat_feature_names = onehot_encoder.get_feature_names(categorical_features).tolist()
    
    feature_names = numeric_features.tolist() + cat_feature_names
    logger.info(f"Total number of features after preprocessing: {len(feature_names)}")
    
    # Create a new dataframe with processed data
    if sp.issparse(processed_data):
        processed_data = processed_data.toarray()
    
    processed_df = pd.DataFrame(processed_data, columns=feature_names, index=df.index)
    
    # Add 'Attrition' back to the processed dataframe
    processed_df['Attrition'] = attrition
    
    logger.info(f"Final processed dataframe shape: {processed_df.shape}")
    
    return processed_df, preprocessor

# Function to create visualizations
def create_visualizations(df):
    logger = logging.getLogger(__name__)
    logger.info("Creating visualizations")
    
    # Create output directory
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Use a different style setting
        plt.style.use('default')
        sns.set_theme()  # Set seaborn default theme
        
        # 1. Department Distribution
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='Department')
        plt.title('Employee Count by Department', pad=20)
        plt.xlabel('Department', labelpad=10)
        plt.ylabel('Count', labelpad=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/department_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Salary Distribution by Department
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df, x='Department', y='Salary')
        plt.title('Salary Distribution by Department', pad=20)
        plt.xlabel('Department', labelpad=10)
        plt.ylabel('Salary', labelpad=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/salary_by_department.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Age Distribution by Gender
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='Age', hue='Gender', multiple='stack', bins=30)
        plt.title('Age Distribution by Gender', pad=20)
        plt.xlabel('Age', labelpad=10)
        plt.ylabel('Count', labelpad=10)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/age_by_gender.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Correlation Heatmap
        plt.figure(figsize=(16, 12))
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                    fmt='.2f', square=True, linewidths=0.5)
        plt.title('Correlation Heatmap of Numerical Features', pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Years at Company vs Salary
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='YearsAtCompany', y='Salary', hue='Department', 
                    alpha=0.6)
        plt.title('Years at Company vs Salary by Department', pad=20)
        plt.xlabel('Years at Company', labelpad=10)
        plt.ylabel('Salary', labelpad=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/years_vs_salary.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 6. Attrition Analysis
        plt.figure(figsize=(12, 6))
        attrition_by_dept = df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack()
        attrition_by_dept.plot(kind='bar', stacked=True)
        plt.title('Attrition Rate by Department', pad=20)
        plt.xlabel('Department', labelpad=10)
        plt.ylabel('Percentage', labelpad=10)
        plt.legend(title='Attrition', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(f'{output_dir}/attrition_by_department.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualizations saved to {output_dir}/")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        raise

    finally:
        plt.close('all') 


# Function to perform statistical tests
def perform_statistical_tests(df):
    logger.info("Performing statistical tests")

    # T-test: Salary difference between attrition groups
    attrition_yes = df[df['Attrition'] == True]['Salary']
    attrition_no = df[df['Attrition'] == False]['Salary']
    t_stat, p_value = stats.ttest_ind(attrition_yes, attrition_no)
    logger.info(f"T-test for salary difference between attrition groups: t-statistic = {t_stat}, p-value = {p_value}")

    # Chi-square test: Association between OverTime and Attrition
    contingency_table = pd.crosstab(df['OverTime'], df['Attrition'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    logger.info(f"Chi-square test for association between OverTime and Attrition: chi2 = {chi2}, p-value = {p_value}")

    # ANOVA: Job satisfaction across departments
    departments = df['Department'].unique()
    satisfaction_by_dept = [df[df['Department'] == dept]['JobSatisfaction'] for dept in departments]
    f_stat, p_value = stats.f_oneway(*satisfaction_by_dept)
    logger.info(f"ANOVA test for job satisfaction across departments: F-statistic = {f_stat}, p-value = {p_value}")

    # Correlation test: Age and Salary
    corr, p_value = stats.pearsonr(df['Age'], df['Salary'])
    logger.info(f"Correlation test between Age and Salary: correlation = {corr}, p-value = {p_value}")

# Function to test hypotheses
def test_hypotheses(df):
    logger = logging.getLogger(__name__)
    logger.info("Testing hypotheses")
    
    results = []  # Store hypothesis test results
    
    # Hypothesis 1: Gender pay gap with sample size check
    male_salary = df[df['Gender'] == 'Male']['Salary']
    female_salary = df[df['Gender'] == 'Female']['Salary']
    
    results.append("\nHypothesis 1: Gender pay gap")
    if len(male_salary) > 30 and len(female_salary) > 30:
        t_stat, p_value = stats.ttest_ind(male_salary, female_salary)
        results.extend([
            f"T-statistic: {t_stat:.4f}",
            f"P-value: {p_value:.4f}",
            f"Male average salary: ${male_salary.mean():.2f}",
            f"Female average salary: ${female_salary.mean():.2f}",
            f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis"
        ])
    else:
        results.append("Insufficient sample size for analysis")

    # Hypothesis 2: Education-Salary correlation
    correlation, p_value = stats.pearsonr(df['Education'].astype(float), df['Salary'])
    results.extend([
        "\nHypothesis 2: Education-Salary correlation",
        f"Correlation coefficient: {correlation:.4f}",
        f"P-value: {p_value:.4f}",
        f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis"
    ])

    # Hypothesis 3: Impact of overtime on manager ratings
    try:
        overtime_rating = df[df['OverTime']]['ManagerRating']
        no_overtime_rating = df[~df['OverTime']]['ManagerRating']
        
        if len(overtime_rating) > 30 and len(no_overtime_rating) > 30:
            t_stat, p_value = stats.ttest_ind(overtime_rating, no_overtime_rating)
            results.extend([
                "\nHypothesis 3: Overtime impact on manager ratings",
                f"T-statistic: {t_stat:.4f}",
                f"P-value: {p_value:.4f}",
                f"Average rating (overtime): {overtime_rating.mean():.2f}",
                f"Average rating (no overtime): {no_overtime_rating.mean():.2f}",
                f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis"
            ])
        else:
            results.append("\nHypothesis 3: Insufficient sample size for overtime analysis")
    except KeyError:
        results.append("\nHypothesis 3: Required data not available")

    # Hypothesis 4: Years at company vs Job satisfaction
    try:
        correlation, p_value = stats.spearmanr(df['YearsAtCompany'], df['JobSatisfaction'])
        results.extend([
            "\nHypothesis 4: Years at company vs Job satisfaction",
            f"Spearman correlation coefficient: {correlation:.4f}",
            f"P-value: {p_value:.4f}",
            f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis"
        ])
    except KeyError:
        results.append("\nHypothesis 4: Required data not available")

    # Hypothesis 5: Work-life balance across departments
    try:
        departments = df['Department'].unique()
        wlb_by_dept = [df[df['Department'] == dept]['WorkLifeBalance'] for dept in departments]
        
        if all(len(group) > 30 for group in wlb_by_dept):
            f_statistic, p_value = stats.f_oneway(*wlb_by_dept)
            results.extend([
                "\nHypothesis 5: Work-life balance across departments",
                f"F-statistic: {f_statistic:.4f}",
                f"P-value: {p_value:.4f}",
                f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis"
            ])
        else:
            results.append("\nHypothesis 5: Insufficient sample size for analysis")
    except KeyError:
        results.append("\nHypothesis 5: Required data not available")

    # Print all results
    for result in results:
        print(result)

    return results


# Function to create additional visualizations
def create_additional_visualizations(df):
    logger.info("Creating additional visualizations")
    
    # Create numeric-only dataframe for correlation
    numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
    
    # Remove ID columns from correlation analysis
    id_columns = numeric_df.columns[numeric_df.columns.str.contains('ID', case=False)]
    numeric_df = numeric_df.drop(columns=id_columns)
    
    # 1. Correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    # 2. Pairplot of selected features (excluding ID columns)
    selected_features = ['Age', 'Salary', 'TotalExperience', 'TenureYears']
    if 'CompositeSatisfactionScore' in df.columns:
        selected_features.append('CompositeSatisfactionScore')
    selected_features.append('Attrition')
    
    sns.pairplot(df[selected_features], hue='Attrition')
    plt.suptitle('Pairplot of Selected Features', y=1.02)
    plt.tight_layout()
    plt.savefig('pairplot_selected_features.png')

    # 3. Distribution of Attrition
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Attrition', data=df)
    plt.title('Distribution of Attrition')
    plt.tight_layout()
    plt.savefig('attrition_distribution.png')

    # 4. Boxplot of Salary by Job Level
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='JobLevel', y='Salary', data=df)
    plt.title('Salary Distribution by Job Level')
    plt.tight_layout()
    plt.savefig('salary_by_job_level.png')

    # 5. Violin plot of Total Experience by Attrition
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Attrition', y='TotalExperience', data=df)
    plt.title('Total Experience Distribution by Attrition')
    plt.tight_layout()
    plt.savefig('experience_by_attrition.png')

    # 6. Bar plot of Education Level
    plt.figure(figsize=(10, 6))
    sns.countplot(x='EducationLevel', data=df)
    plt.title('Education Level Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('education_level_distribution.png')

    # 7. Scatter plot of Age vs. Salary colored by Attrition
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Salary', hue='Attrition', data=df)
    plt.title('Age vs. Salary (colored by Attrition)')
    plt.tight_layout()
    plt.savefig('age_vs_salary_attrition.png')

    # 8. KDE plot of Age by Attrition
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Age', hue='Attrition', shade=True)
    plt.title('Age Distribution by Attrition')
    plt.tight_layout()
    plt.savefig('age_distribution_by_attrition.png')



def train_and_evaluate_model(X, y):
    logger = logging.getLogger(__name__)
    logger.info("Training and evaluating the model")
    
    try:
        # Create output directory for model results
        output_dir = 'model_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Split the data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define the model with class weight to handle imbalance
        rf_model = RandomForestClassifier(
            random_state=42,
            class_weight='balanced'
        )
        
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Perform grid search with stratified k-fold
        grid_search = GridSearchCV(
            rf_model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        logger.info("Training model...")
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        classification_rep = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Print results
        print("\nModel Evaluation Results:")
        print("\nClassification Report:")
        print(classification_rep)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
        
        # Plot top 20 feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=feature_importance.head(20),
            x='importance',
            y='feature'
        )
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save model performance metrics
        with open(f'{output_dir}/model_performance.txt', 'w') as f:
            f.write("Model Performance Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write("Classification Report:\n")
            f.write(classification_rep)
            f.write(f"\nROC AUC Score: {roc_auc:.4f}\n")
            f.write(f"\nBest Parameters:\n{str(grid_search.best_params_)}")
        
        logger.info(f"Model evaluation results saved to {output_dir}/")
        return best_model, feature_importance
        
    except Exception as e:
        logger.error(f"Error in model training and evaluation: {e}")
        raise
    
    finally:
        plt.close('all')


# Main function to run the data preprocessing and analysis
def main():
    try:
        # Connect to the database
        engine = connect_to_database()
        
        # Fetch data from all tables
        tables = {table: fetch_table_data(engine, table) for table in TABLES}
        
        # Close the database connection
        engine.dispose()
        
        # Preprocess individual tables
        tables['Employee'] = preprocess_employee(tables['Employee'])
        tables['PerformanceRating'] = preprocess_performance(tables['PerformanceRating'])
        
        # Merge all tables
        merged_data = merge_tables(tables)
        logger.info(f"Merged data shape: {merged_data.shape}")
        
        # Preprocess the merged data
        processed_data, preprocessor = preprocess_data(merged_data)
        
        # Save the preprocessed data to a CSV file
        processed_data.to_csv('preprocessed_hr_data.csv', index=False)
        logger.info("Preprocessed data saved to 'preprocessed_hr_data.csv'")
        
        # Create visualizations
        create_visualizations(merged_data)
        logger.info("Basic visualizations saved as PNG files")
        
        # Perform statistical tests
        perform_statistical_tests(merged_data)
        
        # Test hypotheses
        test_hypotheses(merged_data)
        
        # Create additional visualizations
        create_additional_visualizations(merged_data)
        logger.info("Additional visualizations saved as PNG files")
        
        # Prepare data for modeling
        X = processed_data.drop('Attrition', axis=1)
        y = processed_data['Attrition']
        
        # Train and evaluate the model
        best_model, feature_importance = train_and_evaluate_model(X, y)
        
        # Save feature importance to CSV
        feature_importance.to_csv('feature_importance.csv', index=False)
        logger.info("Feature importance saved to 'feature_importance.csv'")
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception("Exception details:")

if __name__ == "__main__":
    main()