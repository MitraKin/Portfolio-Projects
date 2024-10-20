

<h1>Titanic - Machine Learning from Disaster</h1>

<p>This repository contains a solution to the famous <a href="https://www.kaggle.com/c/titanic">Kaggle Titanic competition</a>, where the goal is to predict whether a passenger survived the sinking of the Titanic using various machine learning techniques. The solution uses Python, data preprocessing, Exploratory Data Analysis (EDA), and a Random Forest Classifier with hyperparameter tuning using GridSearchCV.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#objective">Objective</a></li>
    <li><a href="#approach">Approach</a></li>
    <ul>
        <li><a href="#1-data-preprocessing">1. Data Preprocessing</a></li>
        <li><a href="#2-exploratory-data-analysis-eda">2. Exploratory Data Analysis (EDA)</a></li>
        <li><a href="#3-feature-engineering">3. Feature Engineering</a></li>
        <li><a href="#4-model-building">4. Model Building</a></li>
        <li><a href="#5-hyperparameter-tuning">5. Hyperparameter Tuning</a></li>
        <li><a href="#6-prediction-and-submission">6. Prediction and Submission</a></li>
    </ul>
    <li><a href="#libraries-used">Libraries Used</a></li>
    <li><a href="#how-to-run-the-code">How to Run the Code</a></li>
</ul>

<h2 id="overview">Overview</h2>
<p>The dataset is based on the sinking of the Titanic in 1912, one of the deadliest commercial maritime disasters in history. We are provided with information about a subset of the passengers onboard, including whether they survived the disaster. The objective is to predict survival based on passenger attributes like age, gender, class, and more.</p>
<p>This project uses the <strong>Random Forest Classifier</strong> to make predictions on the test set, and hyperparameters are optimized using <strong>GridSearchCV</strong> for better accuracy.</p>

<h2 id="dataset">Dataset</h2>
<p>Two CSV files are provided in this project:</p>
<ul>
    <li><strong>train.csv</strong>: This file contains the training data, including the labels (<code>Survived</code>), and features such as <code>Pclass</code>, <code>Name</code>, <code>Sex</code>, <code>Age</code>, <code>SibSp</code>, <code>Parch</code>, <code>Ticket</code>, <code>Fare</code>, <code>Cabin</code>, and <code>Embarked</code>.</li>
    <li><strong>test.csv</strong>: This file contains the test data without the <code>Survived</code> column, and our task is to predict survival for the passengers in this file.</li>
</ul>
<p>The target variable for the prediction is <code>Survived</code>:</p>
<ul>
    <li><code>0</code> = Passenger did not survive.</li>
    <li><code>1</code> = Passenger survived.</li>
</ul>

<h2 id="objective">Objective</h2>
<p>The primary goal of this project is to:</p>
<ul>
    <li>Preprocess the dataset to handle missing values and irrelevant features.</li>
    <li>Perform Exploratory Data Analysis (EDA) to understand data distribution and relationships.</li>
    <li>Engineer features that help improve the model's performance.</li>
    <li>Build a machine learning model using the <strong>Random Forest Classifier</strong>.</li>
    <li>Fine-tune the model using <strong>GridSearchCV</strong>.</li>
    <li>Generate predictions for the test dataset and create a submission file that complies with the competition format.</li>
</ul>

<h2 id="approach">Approach</h2>

<h3 id="1-data-preprocessing">1. Data Preprocessing</h3>
<p>We start by loading both the training and test datasets and perform the following preprocessing steps:</p>

<h4>Handling Missing Values</h4>
<ul>
    <li><strong>Embarked</strong>: Missing values in the <code>Embarked</code> column are filled with the mode (<code>'S'</code>, the most frequent port of embarkation).</li>
    <li><strong>Fare</strong>: Missing values in <code>Fare</code> are filled with the median of the column.</li>
    <li><strong>Age</strong>: Missing <code>Age</code> values are imputed by the median age of passengers based on their title (e.g., Mr, Mrs, Miss).</li>
</ul>

<h4>Feature Extraction and Transformation</h4>
<ul>
    <li><strong>Title Extraction</strong>: The passenger’s title (Mr, Mrs, Miss, etc.) is extracted from the <code>Name</code> column and used as a separate feature.</li>
    <li><strong>Label Encoding</strong>: Categorical variables like <code>Sex</code>, <code>Embarked</code>, and <code>Title</code> are encoded using <code>LabelEncoder</code>.</li>
    <li><strong>Dropping Unnecessary Columns</strong>: We remove columns like <code>PassengerId</code>, <code>Cabin</code>, <code>Ticket</code>, and <code>Name</code> since they do not contribute directly to the model's performance and might cause overfitting.</li>
</ul>

<h3 id="2-exploratory-data-analysis-eda">2. Exploratory Data Analysis (EDA)</h3>
<p>We perform basic exploratory data analysis (EDA) to understand data distribution, correlations between features, and patterns that may help with prediction. Visualizations and insights are generally done using libraries like <code>matplotlib</code> and <code>seaborn</code>, but are omitted in this version of the script for simplicity. The focus is on understanding:</p>
<ul>
    <li>Survival rates based on gender, class, and age.</li>
    <li>Relationships between family size (<code>SibSp</code> and <code>Parch</code>) and survival.</li>
    <li>The distribution of fare prices across different classes.</li>
</ul>

<h3 id="3-feature-engineering">3. Feature Engineering</h3>
<p>We create new features from the existing ones to improve model performance:</p>
<ul>
    <li><strong>Title</strong>: Extracted from the <code>Name</code> column, which provides a more meaningful representation of social status or age grouping than the <code>Name</code> field alone.</li>
    <li><strong>Family Size</strong>: Combining <code>SibSp</code> and <code>Parch</code> to create a feature representing the total number of family members onboard.</li>
</ul>

<h3 id="4-model-building">4. Model Building</h3>
<p>We use <strong>Random Forest Classifier</strong> as our predictive model. Random Forest is a powerful ensemble learning method based on decision trees. It reduces overfitting and increases predictive performance by averaging the results of multiple decision trees.</p>

<h3 id="5-hyperparameter-tuning">5. Hyperparameter Tuning</h3>
<p>To optimize the Random Forest model, we use <strong>GridSearchCV</strong> to tune the following hyperparameters:</p>
<ul>
    <li><code>n_estimators</code>: The number of trees in the forest (values tested: 100, 200, 300).</li>
    <li><code>max_depth</code>: The maximum depth of the trees (values tested: <code>None</code>, 10, 20, 30).</li>
    <li><code>min_samples_split</code>: The minimum number of samples required to split an internal node (values tested: 2, 5, 10).</li>
    <li><code>min_samples_leaf</code>: The minimum number of samples required to be at a leaf node (values tested: 1, 2, 4).</li>
</ul>
<p>We run the grid search with cross-validation (<code>cv=5</code>) to identify the best hyperparameters that maximize performance.</p>

<h3 id="6-prediction-and-submission">6. Prediction and Submission</h3>
<p>Once the best model is identified through GridSearchCV, we use this model to predict the <code>Survived</code> values for the test dataset. We then prepare a submission file that contains:</p>
<ul>
    <li><code>PassengerId</code>: The ID of the passenger from the test dataset.</li>
    <li><code>Survived</code>: The predicted value (either <code>0</code> or <code>1</code>).</li>
</ul>
<p>This file is saved as <code>titanic_submission.csv</code> in the correct format with exactly two columns: <code>PassengerId</code> and <code>Survived</code>.</p>

<h2 id="libraries-used">Libraries Used</h2>
<ul>
    <li><strong>pandas</strong>: For data manipulation and handling missing values.</li>
    <li><strong>numpy</strong>: For numerical computations.</li>
    <li><strong>matplotlib</strong> & <strong>seaborn</strong>: For visualizing data (optional in this implementation).</li>
    <li><strong>scikit-learn</strong>: For building the Random Forest model, preprocessing, hyperparameter tuning, and evaluation.</li>
    <li><strong>GridSearchCV</strong>: For hyperparameter optimization.</li>
</ul>

<h2 id="how-to-run-the-code">How to Run the Code</h2>
<ol>
    <li>Clone this repository:
        <pre><code>git clone https://github.com/yourusername/titanic-prediction.git
cd titanic-prediction
        </code></pre>
    </li>
    <li>Install the required libraries:
        <pre><code>pip install pandas numpy scikit-learn matplotlib seaborn
        </code></pre>
    </li>
    <li>Download the Titanic dataset from <a href="https://www.kaggle.com/c/titanic/data">Kaggle</a> and place the <code>train.csv</code> and <code>test.csv</code> files in the same directory as the script.</li>
    <li>Run the script:
        <pre><code>python titanic_prediction.py
        </code></pre>
    </li>
    <li>After running the script, a CSV file named <code>titanic_submission.csv</code> will be generated in the working directory. This file is in the correct format for submission to Kaggle.</li>
</ol>

<h2 id="conclusion">Conclusion</h2>
<p>This project provides a robust solution to the Titanic survival prediction problem using a Random Forest model with hyperparameter tuning. The preprocessing steps like handling missing values, encoding categorical variables, and feature engineering play a crucial role in enhancing model accuracy.</p>
<p>Feel free to fork and experiment with different algorithms or more advanced feature engineering techniques!</p>

</body>
</html>
