# library doc string


# import libraries
import shap
import joblib
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report, RocCurveDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


os.environ['QT_QPA_PLATFORM']='offscreen'


# It seems unnecessary to create this function when pd.read_csv is so well established and useful
def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    
    return df
    


def perform_eda(df, output_path, cat_columns=None, quant_columns=None, new_col_logic=None):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    print("The shape of the dataframe is:", df.shape)
    print("The number of null values per columns is:", df.isnull().sum())
    print("Here is a description of the dataframe:", df.describe())
    
    
    # Optional column creation
    if new_col_logic:
        new_col_name, func = new_col_logic
        df[new_col_name] = df.apply(func, axis=1)
        print(f"New column '{new_col_name}' created.")
        
        plt.figure(figsize=(20,10)) 
        df[new_col_name].hist();
        plt.title(f'Distribution of {new_col_name}')
        
        filename = os.path.join(output_path, f"{new_col_name}_hist.png")
        plt.savefig(filename)
        plt.close()
        
    if quant_columns:
        for col in quant_columns:
            plt.figure(figsize=(20, 10))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            
            filename = os.path.join(output_path, f"{col}_hist.png")
            plt.savefig(filename)
            plt.close()
            
        for col in quant_columns:
            plt.figure(figsize=(20,10))
            sns.histplot(df[col], stat='density', kde=True)
            plt.title(f'kde plot of {col}')
            filename = os.path.join(output_path, f"{col}_kde.png")
            plt.savefig(filename)
            plt.close()
            
    if cat_columns:
        for col in cat_columns:
            plt.figure(figsize=(20, 10))
            df[col].value_counts('normalize').plot(kind='bar')
            plt.title(f'Bar plot of {col}')
            plt.xticks(rotation=45)
            filename = os.path.join(output_path, f"{col}_bar.png")
            plt.savefig(filename)
            plt.close()
            
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    
    filename = os.path.join(output_path, "correlation_heatmap.png")
    plt.savefig(filename)
    plt.close()
    


def encoder_helper(df, cat_columns, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    
    for col in cat_columns:
        col_list = []
        col_groups = df.groupby(col)[response].mean()
        
        for val in df[col]:
            col_list.append(col_groups.loc[val])
        df[f"{col}_{response}"] = col_list
        
    return df



def perform_feature_engineering(df, response, columns):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df[response]
    X = df[columns]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test





def classification_report_image(y_train,
                                 y_test,
                                 y_train_preds_lr,
                                 y_train_preds_rf,
                                 y_test_preds_lr,
                                 y_test_preds_rf,
                                 output_path):
    '''
    Generates and saves classification reports as images for both models.

    input:
            y_train: true labels for training set
            y_test: true labels for test set
            y_train_preds_lr: logistic regression train preds
            y_test_preds_lr: logistic regression test preds
            y_train_preds_rf: random forest train preds
            y_test_preds_rf: random forest test preds
            output_path: directory to save images

    output:
            None (images saved to disk)
    '''

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    report_data = {
        'random_forest_train': classification_report(y_train, y_train_preds_rf, output_dict=True),
        'random_forest_test':  classification_report(y_test, y_test_preds_rf, output_dict=True),
        'logistic_regression_train': classification_report(y_train, y_train_preds_lr, output_dict=True),
        'logistic_regression_test':  classification_report(y_test, y_test_preds_lr, output_dict=True)
    }

    for key, report_dict in report_data.items():
        df_report = pd.DataFrame(report_dict).transpose()

        plt.figure(figsize=(10, 4))
        plt.title(key.replace('_', ' ').title(), fontsize=14)
        plt.axis('off')
        
        table = plt.table(
            cellText=df_report.round(2).values,
            colLabels=df_report.columns,
            rowLabels=df_report.index,
            loc='center',
            cellLoc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.tight_layout()

        filename = os.path.join(output_path, f"{key}_report.png")
        plt.savefig(filename)
        plt.close()
        


    
def model_plots(rfc_model, lr_model, X_test, y_test, output_path):
    '''
    Creates and stores the ROC curve comparing both models.

    Inputs:
        rfc_model: trained random forest model
        lrc_model: trained logistic regression model
        X_test: test feature set
        y_test: true labels for test set
        output_pth: path to save the ROC curve plot

    Output:
        None
    '''

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    lr_plot = plot_roc_curve(lr_model, X_test, y_test)
    
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lr_plot.plot(ax=ax, alpha=0.8)

    plt.title('ROC Curve Comparison')
    plt.tight_layout()

    # Save the figure
    filename = os.path.join(output_path, "roc_curve.png")
    plt.savefig(filename)
    plt.close()
    

    
    
def feature_importance_plot(model, X_data, output_path):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90);
    
    # Save the figure
    filename = os.path.join(output_path, "feature_importance.png")
    plt.savefig(filename)
    plt.close()
    
    # Plot the explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
#     shap.summary_plot(shap_values, X_data, plot_type="bar")
    
    fig = shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    filename = os.path.join(output_path, "shap_summary.png")
    plt.savefig(filename)
    
    
    
def train_models(X_train, X_test, y_train, y_test, output_path):
    '''
    train, store model results: stored models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    
#     output_path = Path(output_path)  # Ensure output_path is a Path object
        
    # Ensure the subdirectories exist for results and models
    results_path = Path(os.path.join(output_path, "images/results"))
    model_path = Path(os.path.join(output_path, "models"))


    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4, 5, 100],
        'criterion' : ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    

    # Call classification report function
    classification_report_image(y_train,
                                 y_test,
                                 y_train_preds_lr,
                                 y_train_preds_rf,
                                 y_test_preds_lr,
                                 y_test_preds_rf,
                                 results_path)
    
    # Call the model_plots function
    model_plots(cv_rfc.best_estimator_, lrc, X_test, y_test, results_path)
    
    # Call feature importance function
    feature_importance_plot(cv_rfc.best_estimator_, X_test, results_path)
    
    # save best model
    joblib.dump(cv_rfc.best_estimator_, model_path / 'rfc_model.pkl')
    joblib.dump(lrc, model_path / 'logistic_model.pkl')
    
    