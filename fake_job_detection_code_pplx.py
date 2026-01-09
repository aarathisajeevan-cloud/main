
"""
FAKE JOB POSTING DETECTION USING MACHINE LEARNING WITH NLP-NLTK
This script demonstrates how to detect fraudulent job postings using Natural Language Processing
and machine learning algorithms. The pipeline includes text preprocessing, feature extraction, 
model training, and evaluation.

Dataset: Employment Scam Aegean Dataset (EMSCAD) - 18,000 job postings
Models Supported: Naive Bayes, Random Forest, MLP Classifier
"""

# ============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================================

# Data manipulation and numerical computing
import pandas as pd
import numpy as np

# NLTK (Natural Language Toolkit) for NLP tasks
import nltk
from nltk.tokenize import word_tokenize  # Splits text into individual words/tokens
from nltk.corpus import stopwords  # Common words to remove (a, the, is, etc.)
from nltk.stem import WordNetLemmatizer  # Converts words to their base form
import re  # Regular expressions for text cleaning

# Scikit-learn for machine learning
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text to numerical features
from sklearn.model_selection import train_test_split  # Splits data into train/test sets
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron (Neural Network)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score  # More metrics

# Warnings suppression
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 2: DOWNLOAD NLTK RESOURCES (Run once)
# ============================================================================

# These downloads are required for NLTK to function properly
nltk.download('punkt')  # Tokenization data
nltk.download('stopwords')  # Stopwords list
nltk.download('wordnet')  # Lemmatization data
nltk.download('averaged_perceptron_tagger')  # POS tagging

# ============================================================================
# STEP 3: DATA LOADING
# ============================================================================

def load_dataset(filepath):
    """
    Load the fake job postings dataset from CSV file.

    The dataset typically contains columns like:
    - job_id: Unique identifier for the job posting
    - title: Job title
    - description: Detailed job description
    - company_profile: Information about the company
    - requirements: Job requirements
    - benefits: Benefits offered
    - fraudulent: Target variable (0 = Real, 1 = Fake)

    Args:
        filepath (str): Path to the CSV file containing job postings

    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(r"C:\Users\aarat\Desktop\Bvoc IT\sem6\proj_s6\main\emscad_cleaned_excel.xlsx")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Fraudulent jobs: {(df['fraudulent'] == 1).sum()} ({(df['fraudulent'] == 1).sum()/len(df)*100:.2f}%)")
    print(f"Real jobs: {(df['fraudulent'] == 0).sum()} ({(df['fraudulent'] == 0).sum()/len(df)*100:.2f}%)")
    return df

# Example: load_data = load_dataset('fake_job_postings.csv')

# ============================================================================
# STEP 4: TEXT PREPROCESSING
# ============================================================================

class TextPreprocessor:
    """
    Preprocesses text data by cleaning and normalizing job descriptions.
    This step is crucial for removing noise and preparing text for ML models.
    """

    def __init__(self):
        """Initialize the preprocessor with NLTK tools."""
        # Load English stopwords (common words like 'the', 'a', 'is', etc.)
        self.stop_words = set(stopwords.words('english'))

        # Initialize lemmatizer (converts words to base form: running -> run)
        self.lemmatizer = WordNetLemmatizer()

    def remove_html_tags(self, text):
        """
        Remove HTML tags from text (e.g., <br>, <p>, etc.)

        Args:
            text (str): Text potentially containing HTML tags

        Returns:
            str: Text without HTML tags
        """
        # Regular expression to match and remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', str(text))
        return clean_text

    def remove_special_characters(self, text):
        """
        Remove special characters and keep only alphanumeric and spaces.

        Args:
            text (str): Raw text

        Returns:
            str: Text with only alphanumeric characters
        """
        # Keep only letters, numbers, and spaces
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
        return clean_text

    def convert_to_lowercase(self, text):
        """
        Convert text to lowercase for uniformity.
        This prevents treating 'Job' and 'job' as different words.

        Args:
            text (str): Text in mixed case

        Returns:
            str: Text in lowercase
        """
        return str(text).lower()

    def tokenize_text(self, text):
        """
        Break text into individual words (tokens).
        This is fundamental for NLTK-based processing.

        Example: "I love programming" -> ["I", "love", "programming"]

        Args:
            text (str): Complete text

        Returns:
            list: List of word tokens
        """
        # word_tokenize splits text into tokens while preserving punctuation
        tokens = word_tokenize(text)
        return tokens

    def remove_stopwords(self, tokens):
        """
        Remove common English stopwords that don't add much meaning.
        Stopwords include: a, an, the, is, are, I, you, etc.

        This reduces noise in data and improves model efficiency.

        Args:
            tokens (list): List of word tokens

        Returns:
            list: Tokens with stopwords removed
        """
        # Keep only tokens that are not in the stopwords set
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        return filtered_tokens

    def lemmatize_tokens(self, tokens):
        """
        Convert words to their base/lemma form.
        Examples:
        - running, runs, ran -> run
        - programming, programs -> program
        - better -> good

        This helps the model recognize similar words as the same concept.

        Args:
            tokens (list): List of word tokens

        Returns:
            list: Lemmatized tokens
        """
        # lemmatizer.lemmatize() converts each word to its base form
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized

    def preprocess(self, text):
        """
        Execute the complete preprocessing pipeline on a single text.

        Pipeline order:
        1. Remove HTML tags
        2. Remove special characters
        3. Convert to lowercase
        4. Tokenization (word splitting)
        5. Remove stopwords
        6. Lemmatization
        7. Join tokens back to a single string

        Args:
            text (str): Raw job description text

        Returns:
            str: Fully preprocessed text ready for feature extraction
        """
        # Step 1: Remove HTML tags
        text = self.remove_html_tags(text)

        # Step 2: Remove special characters
        text = self.remove_special_characters(text)

        # Step 3: Convert to lowercase
        text = self.convert_to_lowercase(text)

        # Step 4: Tokenization
        tokens = self.tokenize_text(text)

        # Step 5: Remove stopwords
        tokens = self.remove_stopwords(tokens)

        # Step 6: Lemmatization
        tokens = self.lemmatize_tokens(tokens)

        # Step 7: Join tokens back into a single string for feature extraction
        processed_text = ' '.join(tokens)

        return processed_text

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Example usage:
# raw_job_desc = "We are looking for experienced <b>Python developers</b>! 
#                  Running and managing databases is essential."
# cleaned_desc = preprocessor.preprocess(raw_job_desc)
# print(cleaned_desc)
# Output: "look experienced python develop run manage databas essenti"

# ============================================================================
# STEP 5: PREPROCESSING DATASET
# ============================================================================

def preprocess_dataset(df, text_column='description'):
    """
    Apply preprocessing to all job descriptions in the dataset.

    Args:
        df (pd.DataFrame): Dataset with job postings
        text_column (str): Name of column containing job descriptions

    Returns:
        pd.DataFrame: Dataset with preprocessed descriptions
    """
    print("Preprocessing job descriptions...")
    # Apply preprocessing function to each description
    df['processed_description'] = df[text_column].apply(preprocessor.preprocess)
    print("Preprocessing completed!")
    return df

# Example: processed_data = preprocess_dataset(load_data)

# ============================================================================
# STEP 6: FEATURE EXTRACTION USING TF-IDF
# ============================================================================

def extract_tfidf_features(train_texts, test_texts):
    """
    Convert text descriptions into numerical features using TF-IDF.

    TF-IDF (Term Frequency-Inverse Document Frequency) measures how important 
    each word is in each document relative to the entire corpus.

    - TF (Term Frequency): How often a word appears in a document
    - IDF (Inverse Document Frequency): How rare a word is across all documents

    Words that appear frequently in a document but rarely across all documents
    get higher weights, making them more distinctive for classification.

    Args:
        train_texts (list): Job descriptions for training
        test_texts (list): Job descriptions for testing

    Returns:
        tuple: (X_train, X_test, vectorizer)
        - X_train: TF-IDF feature matrix for training data (sparse matrix)
        - X_test: TF-IDF feature matrix for test data (sparse matrix)
        - vectorizer: Fitted TfidfVectorizer object (can be saved for later use)
    """
    print("Extracting TF-IDF features...")

    # Create TF-IDF vectorizer with specific parameters
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Use only top 5000 most important features
        min_df=2,  # Ignore words appearing in less than 2 documents
        max_df=0.8,  # Ignore words appearing in more than 80% of documents
        ngram_range=(1, 2),  # Use unigrams and bigrams (single and double words)
        stop_words='english',  # Additional stopwords removal
        sublinear_tf=True  # Apply sublinear TF scaling
    )

    # Fit vectorizer on training data and transform it
    X_train = vectorizer.fit_transform(train_texts)

    # Transform test data using the fitted vectorizer
    X_test = vectorizer.transform(test_texts)

    print(f"Training features shape: {X_train.shape}")  # (samples, features)
    print(f"Test features shape: {X_test.shape}")

    return X_train, X_test, vectorizer

# Example usage:
# X_train, X_test, vectorizer = extract_tfidf_features(
#     df_train['processed_description'].values,
#     df_test['processed_description'].values
# )

# ============================================================================
# STEP 7: DATA SPLITTING
# ============================================================================

def split_data(df, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets.

    - Training set (80%): Used to train the model
    - Testing set (20%): Used to evaluate model performance on unseen data

    Args:
        df (pd.DataFrame): Full dataset with preprocessed descriptions
        test_size (float): Proportion of data to use for testing (default: 0.2)
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
        - X_train, X_test: Preprocessed job descriptions
        - y_train, y_test: Target labels (0 = Real, 1 = Fake)
    """
    # Split features (X) and target (y)
    X = df['processed_description'].values
    y = df['fraudulent'].values

    # Perform train-test split with stratification
    # Stratification ensures both sets have similar distribution of fake/real jobs
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class balance in both sets
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    print(f"Training set fake jobs: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")
    print(f"Testing set fake jobs: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.2f}%)")

    return X_train, X_test, y_train, y_test

# Example usage:
# X_train, X_test, y_train, y_test = split_data(processed_data)

# ============================================================================
# STEP 8: MODEL TRAINING
# ============================================================================

class FakeJobDetectionModel:
    """
    Machine learning models for fake job posting detection.
    Supports multiple algorithms: Naive Bayes, Random Forest, MLP.
    """

    def __init__(self, model_type='mlp'):
        """
        Initialize model with specified algorithm.

        Args:
            model_type (str): 'naive_bayes', 'random_forest', or 'mlp'
        """
        self.model_type = model_type
        self.model = self._create_model()

    def _create_model(self):
        """Create the specified machine learning model."""
        if self.model_type == 'naive_bayes':
            # Multinomial Naive Bayes: Great for text classification
            # Assumes feature independence; fast and effective for sparse data
            return MultinomialNB(alpha=1.0)

        elif self.model_type == 'random_forest':
            # Random Forest: Ensemble of decision trees
            # Handles non-linear patterns; robust to outliers
            return RandomForestClassifier(
                n_estimators=100,  # Number of trees in the forest
                max_depth=20,  # Maximum depth of each tree
                min_samples_split=5,  # Minimum samples to split a node
                random_state=42
            )

        elif self.model_type == 'mlp':
            # Multi-layer Perceptron: Neural network
            # Best for capturing complex patterns; can learn intricate relationships
            return MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),  # Three hidden layers
                activation='relu',  # Activation function
                learning_rate='adaptive',  # Adaptive learning rate
                max_iter=200,  # Maximum iterations
                random_state=42
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train):
        """
        Train the model on training data.

        Args:
            X_train: TF-IDF feature matrix for training
            y_train: Target labels for training (0 = Real, 1 = Fake)
        """
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        print(f"{self.model_type} model training completed!")

    def predict(self, X_test):
        """
        Make predictions on test data.

        Args:
            X_test: TF-IDF feature matrix for testing

        Returns:
            np.array: Predicted labels (0 or 1)
        """
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Get probability predictions (confidence scores).

        Args:
            X_test: TF-IDF feature matrix for testing

        Returns:
            np.array: Probability of each class [prob_real, prob_fake]
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_test)
        else:
            # Random Forest and MLP support predict_proba
            return self.model.predict_proba(X_test)

# Example training:
# model = FakeJobDetectionModel(model_type='mlp')
# model.train(X_train, y_train)

# ============================================================================
# STEP 9: MODEL EVALUATION
# ============================================================================

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate model performance using multiple metrics.

    Metrics explained:
    - Accuracy: Percentage of correct predictions
    - Precision: Of predicted fake jobs, how many were actually fake (minimizes false alarms)
    - Recall: Of actual fake jobs, how many did we catch (minimizes missed frauds)
    - F1-Score: Harmonic mean of precision and recall (overall performance)
    - ROC-AUC: Area under the ROC curve (how well model distinguishes classes)

    Args:
        y_true: Actual labels
        y_pred: Predicted labels
        y_pred_proba: Probability predictions (optional)

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    metrics = {}

    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Precision: TP / (TP + FP) - How precise are our fake job predictions?
    metrics['precision'] = precision_score(y_true, y_pred)

    # Recall: TP / (TP + FN) - How many fake jobs do we catch?
    metrics['recall'] = recall_score(y_true, y_pred)

    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    metrics['f1_score'] = f1_score(y_true, y_pred)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = tn  # Correctly classified real jobs
    metrics['false_positives'] = fp  # Real jobs classified as fake (Type I error)
    metrics['false_negatives'] = fn  # Fake jobs classified as real (Type II error)
    metrics['true_positives'] = tp  # Correctly classified fake jobs

    # ROC-AUC
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])

    return metrics

def print_evaluation_results(metrics):
    """
    Print evaluation metrics in a readable format.

    Args:
        metrics (dict): Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")

    if 'roc_auc' in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(f"True Negatives (Real jobs correctly classified):  {metrics['true_negatives']}")
    print(f"False Positives (Real jobs wrongly as fake):      {metrics['false_positives']}")
    print(f"False Negatives (Fake jobs wrongly as real):      {metrics['false_negatives']}")
    print(f"True Positives (Fake jobs correctly classified):  {metrics['true_positives']}")
    print("="*60 + "\n")

# ============================================================================
# STEP 10: COMPLETE PIPELINE FUNCTION
# ============================================================================

def complete_pipeline(csv_file, model_type='mlp'):
    """
    Execute the complete fake job detection pipeline.

    This function orchestrates all steps:
    1. Load dataset
    2. Preprocess text
    3. Extract TF-IDF features
    4. Split into train/test
    5. Train model
    6. Make predictions
    7. Evaluate model

    Args:
        csv_file (str): Path to the CSV file with job postings
        model_type (str): Type of model ('naive_bayes', 'random_forest', 'mlp')

    Returns:
        tuple: (model, vectorizer, metrics)
    """
    print("\n" + "="*60)
    print("FAKE JOB POSTING DETECTION PIPELINE")
    print("="*60 + "\n")

    # Step 1: Load data
    print("Step 1: Loading dataset...")
    df = load_dataset(csv_file)

    # Step 2: Preprocess
    print("\nStep 2: Preprocessing text...")
    df = preprocess_dataset(df)

    # Step 3: Split data
    print("\nStep 3: Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 4: Feature extraction
    print("\nStep 4: Extracting TF-IDF features...")
    X_train, X_test, vectorizer = extract_tfidf_features(X_train, X_test)

    # Step 5: Train model
    print(f"\nStep 5: Training {model_type} model...")
    model = FakeJobDetectionModel(model_type=model_type)
    model.train(X_train, y_train)

    # Step 6: Make predictions
    print("\nStep 6: Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Step 7: Evaluate
    print("\nStep 7: Evaluating model performance...")
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    print_evaluation_results(metrics)

    return model, vectorizer, metrics

# ============================================================================
# STEP 11: PREDICT ON NEW JOB POSTING
# ============================================================================

def predict_job_posting(job_description, model, vectorizer):
    """
    Predict whether a new job posting is fake or real.

    Args:
        job_description (str): Raw job description text
        model: Trained FakeJobDetectionModel
        vectorizer: Fitted TfidfVectorizer

    Returns:
        dict: Prediction result with probability
    """
    # Preprocess the job description
    processed_text = preprocessor.preprocess(job_description)

    # Convert to TF-IDF features
    features = vectorizer.transform([processed_text])

    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    result = {
        'prediction': 'FAKE JOB' if prediction == 1 else 'REAL JOB',
        'confidence': max(probability) * 100,
        'prob_real': probability[0] * 100,
        'prob_fake': probability[1] * 100
    }

    return result

# Example usage:
# new_job = "Send money now to get a job! Quick cash opportunity!"
# result = predict_job_posting(new_job, trained_model, fitted_vectorizer)
# print(f"Prediction: {result['prediction']}")
# print(f"Confidence: {result['confidence']:.2f}%")

print("Code prepared successfully! Ready for execution.")
print("\nTo run this code, ensure you have the dataset CSV file and call:")
print("model, vectorizer, metrics = complete_pipeline('fake_job_postings.csv', model_type='mlp')")
