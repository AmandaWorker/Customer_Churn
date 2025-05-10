import os
import logging
import shutil

import pytest

import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from churn_library_solution import import_data, perform_eda, encoder_helper, perform_feature_engineering
from churn_library_solution import classification_report_image
from churn_library_solution import model_plots, train_models

# logging.basicConfig(
#     filename='./logs/churn_library.log',
#     level = logging.INFO,
#     filemode='w',
#     format='%(name)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler("./results.log")
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': ['A', 'B', 'C']
    })

@pytest.fixture
def sample_df2():
    return pd.DataFrame({
        'category': ['A', 'B', 'A', 'C'],
        'churn': [1, 0, 1, 1]
    })



def test_import_data(tmp_path):
    """
    Test the `import_data` function to ensure it correctly reads a CSV file into a DataFrame.

    Parameters:
    -----------
    tmp_path : pathlib.Path
        A temporary directory provided by pytest for creating test files.

    Process:
    --------
    - Creates a temporary CSV file with known data.
    - Uses `import_data` to read the CSV into a DataFrame.
    - Compares the imported DataFrame to the expected DataFrame.

    Asserts:
    --------
    - That the resulting DataFrame is not empty.
    - That the DataFrame has the correct column names.
    - That the contents of the DataFrame exactly match the expected data.

    Logs:
    -----
    - Confirms successful import and validation of the CSV content.

    Raises:
    -------
    AssertionError:
        If the DataFrame is empty, columns are incorrect, or content does not match the expected result.
    """
    # Create a temporary CSV file
    test_csv_path = tmp_path / "test_data.csv"
    data = {
        'column1': [1, 2, 3],
        'column2': ['A', 'B', 'C']
    }
    expected_df = pd.DataFrame(data)
    expected_df.to_csv(test_csv_path, index=False)

    # Call the function
    df = import_data(test_csv_path)
    logger.info("import_data completed successfully")

    # Assertions
    assert not df.empty, "DataFrame is empty"
    assert list(df.columns) == ['column1', 'column2'], "Incorrect columns"
    pd.testing.assert_frame_equal(df, expected_df)
    logger.info("import_data output is as expected")


    
def test_perform_eda(sample_df):
    """
    Test the `perform_eda` function to ensure it generates the correct EDA plots and handles new column logic.

    Parameters:
    -----------
    sample_df : pd.DataFrame
        A sample DataFrame used to verify EDA plot generation and optional column creation.

    Process:
    --------
    - Applies a custom logic to create a new column.
    - Runs EDA plotting functions for specified categorical and quantitative columns.
    - Verifies that all expected plot image files are generated in the 'images' directory.

    Asserts:
    --------
    - That each expected EDA image file is successfully created.

    Logs:
    -----
    - Confirms successful execution of `perform_eda`.
    - Logs the presence of all expected output files.

    Raises:
    -------
    AssertionError:
        If any of the expected image files are missing.
    """

    test_new_col_logic = ('New', lambda row: 0 if row['column2'] == 'A' else 1)

    perform_eda(
            sample_df,
            cat_columns=['column2'],
            quant_columns=['column1'],
            new_col_logic=test_new_col_logic)

    logger.info("perform_eda completed successfully")

    expected_files = [
            'New_hist.png',
            'column1_hist.png',
            'column1_kde.png',
            'column2_bar.png',
            'correlation_heatmap.png']
    
    for filename in expected_files:
            file_path = os.path.join('images', filename)
            assert os.path.exists(
                file_path), f"Expected file not found: {file_path}"

    logger.info("All expected files found")
    logger.info("Test perform_eda completed")
    
    
    
def test_encoder_helper():
    """
    Test the `encoder_helper` function to verify that it correctly adds mean-encoded columns for categorical variables.

    Process:
    --------
    - Creates a sample DataFrame with a categorical column and a numeric target.
    - Applies `encoder_helper` to encode the categorical column with the mean of the response variable.
    - Validates that the new encoded column is added and that its values match the expected group means.

    Asserts:
    --------
    - That the new encoded column (e.g., 'category_churn') exists in the output DataFrame.
    - That each encoded value corresponds to the correct group mean within a small relative tolerance.

    Logs:
    -----
    - Confirms successful execution and correctness of the `encoder_helper` output.

    Raises:
    -------
    AssertionError:
        If the expected column is missing or if any encoded value deviates from the correct mean.
    """
    # Sample DataFrame
    data = {
        'category': ['A', 'B', 'A', 'B', 'C', 'C', 'A'],
        'churn': [0, 1, 0, 7, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Run encoder_helper
    df_encoded = encoder_helper(df.copy(), cat_columns=['category'], response='churn')
    logger.info("encoder_helper completed successfully")

    # Check if new column was added
    new_col = 'category_churn'
    assert new_col in df_encoded.columns, f"Expected column not found: {new_col}"

    # Check values are approximately correct
    expected_means = df.groupby('category')['churn'].mean().to_dict()
    for i, row in df.iterrows():
        actual = df_encoded.loc[i, new_col]
        expected = expected_means[row['category']]
        assert pytest.approx(actual, rel=1e-5) == expected, (
            f"Incorrect encoding for row {i}: expected {expected}, got {actual}"
        )

    logger.info("encoder_helper output is as expected")


    
    
def test_encoder_helper_with_known_values(sample_df2):
    """
    Test the `encoder_helper` function to verify it encodes categorical columns correctly using the target mean.

    Parameters:
    -----------
    sample_df2 : pd.DataFrame
        A sample DataFrame containing a categorical column and a target column.

    Asserts:
    --------
    - That the DataFrame returned by `encoder_helper` matches the expected DataFrame with the encoded values.

    Logs:
    -----
    - Confirms that the encoder helper output matches the expected result.

    Raises:
    -------
    AssertionError:
        If the encoded DataFrame does not exactly match the expected DataFrame.
    """
    expected_df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'C'],
        'churn': [1, 0, 1, 1],
        'category_churn': [1, 0, 1, 1]
    })

    df_encoded = encoder_helper(sample_df2.copy(), ['category'], 'churn')

    pd.testing.assert_frame_equal(df_encoded, expected_df)
    logger.info("encoder_helper output is as expected")
    
    
    
@pytest.fixture
def sample_df_fe():
    return pd.DataFrame({
        'feature1': [10, 20, 30, 40, 50, 60],
        'feature2': [1, 0, 1, 0, 1, 0],
        'churn':    [0, 1, 0, 1, 0, 1]
    })


def test_perform_feature_engineering(sample_df_fe):
    """
    Test the `perform_feature_engineering` function to ensure it processes the data correctly and splits it into train/test sets.

    Parameters:
    -----------
    sample_df_fe : pd.DataFrame
        A sample DataFrame containing the features and target column for testing.
    
    Asserts:
    --------
    - That all four outputs (X_train, X_test, y_train, y_test) are not None.
    - That the outputs have the expected number of rows: 4 for training and 2 for testing.

    Logs:
    -----
    - Confirms successful validation of output shapes from the feature engineering function.

    Raises:
    -------
    AssertionError:
        If any output is None or does not match the expected shape.
    """
    response = 'churn'
    columns = ['feature1', 'feature2']

    X_train, X_test, y_train, y_test = perform_feature_engineering(sample_df_fe, response, columns)

    # Check that all outputs are created
    assert X_train is not None, "X_train not created"
    assert X_test is not None, "X_test not created"
    assert y_train is not None, "y_train not created"
    assert y_test is not None, "y_test not created"

    # Check expected shapes
    assert X_train.shape[0] == 4, f"Expected 4 rows in X_train, got {X_train.shape[0]}"
    assert X_test.shape[0] == 2, f"Expected 2 rows in X_test, got {X_test.shape[0]}"
    assert y_train.shape[0] == 4, f"Expected 4 rows in y_train, got {y_train.shape[0]}"
    assert y_test.shape[0] == 2, f"Expected 2 rows in y_test, got {y_test.shape[0]}"

    logger.info("perform_feature_engineering output shapes are as expected")

    
    

@pytest.fixture
def dummy_predictions():
    # Simulated binary classification example
    y_train = pd.Series([0, 1, 0, 1, 1, 0])
    y_test = pd.Series([1, 0, 1])
    y_train_preds_lr = y_train.copy()
    y_train_preds_rf = y_train.copy()
    y_test_preds_lr = y_test.copy()
    y_test_preds_rf = y_test.copy()

    return y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf



def test_classification_report_image(dummy_predictions):
    """
    Test the `classification_report_image` function to ensure it generates and saves classification report images.

    Parameters:
    -----------
    dummy_predictions : tuple
        A tuple containing true labels and predicted labels in the following order:
        (y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)

    Asserts:
    --------
    - That the expected report image files are created in the 'images/results' directory.

    Logs:
    -----
    - Confirms the generation of each expected image file.
    - Logs a success message after verifying all classification report images.

    Raises:
    -------
    AssertionError:
        If any of the expected report image files are not found in the specified directory.
    """
    y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = dummy_predictions
    output_dir = 'images/results'

    classification_report_image(
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf,
            output_path=output_dir
        )

    expected_files = [
            'random_forest_train_report.png',
            'random_forest_test_report.png',
            'logistic_regression_train_report.png',
            'logistic_regression_test_report.png'
        ]

    for file in expected_files:
        assert os.path.exists(os.path.join(output_dir, file)), f"Missing report: {file}"
        
    logger.info("All classification report images generated successfully.")

    
    

@pytest.fixture
def dummy_data(request):
    # Generate simple dummy classification data
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    output_path = Path("./models")

    # Ensure the output directory is clean (this is now handled automatically by pytest)
    output_path.mkdir(parents=True, exist_ok=True)

    # Add a finalizer to delete the output files after the test finishes
    def cleanup():
        for file in ['rfc_model.pkl', 'logistic_model.pkl']:
            file_path = output_path / file
            if file_path.exists():
                os.remove(file_path)
                logger.info(f"Deleted model file: {file_path}")

    # Register the finalizer with pytest
    request.addfinalizer(cleanup)

    return X_train, X_test, y_train, y_test, output_path


def test_train_models(dummy_data):
    """
    Test the `train_models` function to ensure it trains models and saves them correctly.

    Parameters:
    -----------
    dummy_data : tuple
        A tuple containing training and testing datasets and an output path in the following order:
        (X_train, X_test, y_train, y_test, output_path)

    Asserts:
    --------
    - That the expected model files ('rfc_model.pkl', 'logistic_model.pkl') are created in the output path.
    
    Logs:
    -----
    - Confirms the existence of each expected model file.
    - Logs a success message once all files are verified.

    Raises:
    -------
    AssertionError:
        If any of the expected model files are not found at the specified location.
    """
    X_train, X_test, y_train, y_test, output_path = dummy_data

    train_models(X_train, X_test, y_train, y_test, output_path)
    
   # Check if model files are created
    expected_files = ['rfc_model.pkl', 'logistic_model.pkl']
    for file in expected_files:
        file_path = output_path / file  

        assert file_path.exists(), f"Model file missing: {file}"
        logger.info(f"Model file exists: {file_path}")

    logger.info("All model files generated and verified successfully.")
    
    
    
def test_train_models_load(dummy_data):
    """
    Test the `train_models` function to ensure that model files are correctly created and can be loaded.

    Parameters:
    -----------
    dummy_data : tuple
        A tuple containing training and testing data along with the output path:
        (X_train, X_test, y_train, y_test, output_path)

    Process:
    --------
    - Calls `train_models` to generate and save model files.
    - Verifies that each expected model file exists in the specified output path.
    - Attempts to load each model using `joblib` to ensure files are not corrupted or empty.

    Asserts:
    --------
    - That each expected model file is present.
    - That each model file can be successfully loaded and is not None.

    Logs:
    -----
    - Confirms successful loading of each model file.

    Raises:
    -------
    AssertionError:
        If any model file is missing or cannot be loaded properly.
    """
    X_train, X_test, y_train, y_test, output_path = dummy_data

    train_models(X_train, X_test, y_train, y_test, output_path)

    # Check if model files are created and can be loaded
    expected_files = ['rfc_model.pkl', 'logistic_model.pkl']
    for file in expected_files:
        file_path = os.path.join(output_path, file)
        assert os.path.exists(file_path), f"Model file missing: {file}"
        
        model = joblib.load(file_path)
        assert model is not None, f"Failed to load model: {file}"

    logger.info("All model files loaded successfully.")
    
    
        
@pytest.fixture
def sample_models_and_data(tmp_path):
    # Generate dummy classification data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rfc_model = RandomForestClassifier(random_state=42)
    rfc_model.fit(X_train, y_train)

    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    # Use a temporary directory for outputs
    output_path = tmp_path / "results"
    output_path.mkdir(parents=True, exist_ok=True)

    return rfc_model, lr_model, X_test, y_test, output_path
        

    
def test_model_plots(sample_models_and_data):
    """
    Test the `model_plots` function to ensure it generates and saves the ROC curve plot correctly.

    Parameters:
    -----------
    sample_models_and_data : tuple
        A tuple containing two trained models (random forest and logistic regression),
        the test features and labels, and the output directory:
        (rfc_model, lr_model, X_test, y_test, output_path)

    Process:
    --------
    - Calls `model_plots` to generate a ROC curve plot comparing the two models.
    - Checks whether the expected output file ('roc_curve.png') is created in the specified directory.

    Asserts:
    --------
    - That the ROC curve plot file is successfully created and exists at the expected location.

    Logs:
    -----
    - Confirms the successful generation and presence of the ROC curve plot.

    Raises:
    -------
    AssertionError:
        If the ROC curve image file is not found after running the function.
    """
    rfc_model, lr_model, X_test, y_test, output_path = sample_models_and_data

    # Call the function to generate ROC plot
    model_plots(rfc_model, lr_model, X_test, y_test, output_path)

    expected_file = output_path / "roc_curve.png"

    # Check if the ROC curve image was created
    assert expected_file.exists(), "ROC curve plot was not created as expected."

    logger.info("model_plots output is as expected.")
    
    


# if __name__ == "__main__":
# 	pass
