import os
from multiguard.paths import MODELS_DIR
import re
import json
import numpy as np
import nltk
import torch
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
import importlib.util
from nltk.tokenize import word_tokenize, sent_tokenize

# 自定义分词和句子切分函数，避免使用NLTK依赖
def custom_word_tokenize(text):
    """简单的分词函数，不依赖NLTK资源"""
    # 首先用标点符号将文本分割
    text = re.sub(r'([.,!?;:"])', r' \1 ', text)
    # 然后分割空白字符
    words = [word.strip() for word in text.split() if word.strip()]
    return words

def custom_sent_tokenize(text):
    """简单的句子切分函数，不依赖NLTK资源"""
    # 使用常见的句子结束标记进行分割
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# 添加NLTK资源下载函数
def ensure_nltk_resources():
    """确保所有必要的NLTK资源都被下载并正确配置"""
    resources = ['punkt', 'wordnet']
    
    for resource in resources:
        try:
            # 检查资源是否已经下载
            nltk.data.find(f'tokenizers/{resource}')
            print(f"NLTK资源 '{resource}' 已存在")
        except LookupError:
            # 如果没有下载，则下载
            print(f"下载NLTK资源 '{resource}'...")
            nltk.download(resource, quiet=True)
    
    print("完成NLTK资源设置")

    # 在模块加载时确保资源已下载
    try:
        ensure_nltk_resources()
    except Exception as e:
        print(f"警告: NLTK资源初始化失败: {str(e)}. 将使用自定义分词方法。")

class MLDetector:
    def __init__(self):
        """Initialize the machine learning detector with necessary models."""
        self.models_path = str(MODELS_DIR)
        
        # Initialize transformer model for text analysis
        self.transformer_model_name = os.path.join(self.models_path, "distilbert-base-uncased")
        self.tokenizer = None
        self.transformer_model = None
        
        # Initialize classifiers
        self.rf_classifier = None
        self.tfidf_vectorizer = None
        self.scaler = None
        
        # Load pre-trained models if they exist
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models if they exist, otherwise initialize them."""
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        
        # Load transformer model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name)
            self.transformer_model = AutoModel.from_pretrained(self.transformer_model_name)
        except:
            print("Warning: Could not load transformer model. Will use when needed.")
        
        # Load random forest classifier if it exists
        try:
            self.rf_classifier = joblib.load(os.path.join(self.models_path, 'rf_classifier.pkl'))
            self.tfidf_vectorizer = joblib.load(os.path.join(self.models_path, 'tfidf_vectorizer.pkl'))
            self.scaler = joblib.load(os.path.join(self.models_path, 'scaler.pkl'))
            print("Machine learning models loaded successfully.")
        except:
            print("No pre-trained models found. Using rule-based analysis with ML enhancement.")
            # Initialize for later training
            self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.tfidf_vectorizer = TfidfVectorizer(max_features=100)
            self.scaler = StandardScaler()
    
    def get_text_embedding(self, text):
        """
        Generate embeddings for text using transformer model.
        
        Args:
            text (str): Input text to generate embeddings for
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        if not text or len(text.strip()) == 0:
            # Return zero vector if text is empty
            return np.zeros(768)
        
        if self.tokenizer is None or self.transformer_model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name)
                self.transformer_model = AutoModel.from_pretrained(self.transformer_model_name)
            except:
                print("Warning: Could not load transformer model. Using zero vector.")
                return np.zeros(768)
        
        # Tokenize and get model output
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.transformer_model(**inputs)
        
        # Use CLS token embedding (first token) as the sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
        return embedding
    
    def extract_features(self, user_data):
        """
        Extract and normalize features from user data for ML model.
        
        Args:
            user_data (dict): User data including username, email, bio, etc.
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        # Username features
        username = user_data.get('username', '')
        features['username_length'] = len(username)
        features['username_digit_ratio'] = sum(c.isdigit() for c in username) / max(1, len(username))
        features['username_alpha_ratio'] = sum(c.isalpha() for c in username) / max(1, len(username))
        features['username_special_char_count'] = len(re.findall(r'[^a-zA-Z0-9]', username))
        
        # Email features
        email = user_data.get('email', '')
        features['email_length'] = len(email)
        features['email_has_digit'] = int(any(c.isdigit() for c in email))
        domain = email.split('@')[-1] if '@' in email else ''
        features['email_domain_common'] = int(any(d in domain.lower() for d in ['gmail', 'yahoo', 'hotmail', 'outlook']))
        features['email_domain_length'] = len(domain)
        
        # Bio features
        bio = user_data.get('bio', '')

        try:
            words = word_tokenize(bio)
        except LookupError:
            nltk.download("punkt")
            nltk.download("punkt_tab")
            words = word_tokenize(bio)
        features['bio_word_count'] = len(words)
        features['bio_avg_word_length'] = np.mean([len(w) for w in words]) if words else 0

        try:
            sentences = sent_tokenize(bio)
        except LookupError:
            nltk.download("punkt")
            sentences = sent_tokenize(bio)
        features['bio_sentence_count'] = len(sentences)
        
        features['bio_special_char_ratio'] = len(re.findall(r'[^a-zA-Z0-9\s]', bio)) / max(1, len(bio))
        
        # Get transformer embeddings for text
        bio_embedding = self.get_text_embedding(bio)
        for i, val in enumerate(bio_embedding):
            features[f'bio_embedding_{i}'] = val
        
        # Time and interaction features
        try:
            features['time_to_fill'] = float(user_data.get('time_to_fill', 0))
        except:
            features['time_to_fill'] = 0
            
        # Typing pattern features
        typing_pattern = user_data.get('typing_pattern', '')
        if typing_pattern:
            try:
                intervals = [float(i) for i in typing_pattern.split(',')]
                features['typing_pattern_count'] = len(intervals)
                features['typing_pattern_mean'] = np.mean(intervals) if intervals else 0
                features['typing_pattern_std'] = np.std(intervals) if len(intervals) > 1 else 0
                features['typing_pattern_min'] = min(intervals) if intervals else 0
                features['typing_pattern_max'] = max(intervals) if intervals else 0
            except:
                features['typing_pattern_count'] = 0
                features['typing_pattern_mean'] = 0
                features['typing_pattern_std'] = 0
                features['typing_pattern_min'] = 0
                features['typing_pattern_max'] = 0
        else:
            features['typing_pattern_count'] = 0
            features['typing_pattern_mean'] = 0
            features['typing_pattern_std'] = 0
            features['typing_pattern_min'] = 0
            features['typing_pattern_max'] = 0
        
        # Navigation features
        navigation = user_data.get('navigation', '0')
        features['navigation_count'] = int(navigation) if navigation.isdigit() else 0
        
        return features
    
    def predict_bot_probability(self, user_data):
        """
        Predict the probability that the user is a bot using ML models.
        If models are not trained, it will fallback to rule-based methods with ML enhancements.
        
        Args:
            user_data (dict): User data dictionary
            
        Returns:
            float: Probability that the user is a bot (0-1 scale)
            list: Evidence or reasoning for the prediction
        """
        evidence = []
        features = self.extract_features(user_data)
        
        # Convert features to a format suitable for ML models
        feature_names = sorted(features.keys())
        feature_vector = np.array([features[f] for f in feature_names]).reshape(1, -1)
        
        # Check if we have a trained model
        if self.rf_classifier is not None and hasattr(self.rf_classifier, 'classes_'):
            # We have a trained model, use it for prediction
            try:
                # Scale features
                scaled_features = self.scaler.transform(feature_vector)
                
                # Make prediction
                bot_probability = self.rf_classifier.predict_proba(scaled_features)[0][1]
                
                # Get feature importances for evidence
                importances = self.rf_classifier.feature_importances_
                sorted_indices = np.argsort(importances)[::-1]
                
                # Add top features to evidence
                for i in range(min(5, len(sorted_indices))):
                    idx = sorted_indices[i]
                    feature_name = feature_names[idx]
                    feature_value = features[feature_name]
                    importance = importances[idx]
                    evidence.append(f"Feature '{feature_name}' with value {feature_value:.4f} contributed {importance:.2%} to the prediction")
                
                evidence.append(f"Machine learning model predicts a bot probability of {bot_probability:.2%}")
                return bot_probability, evidence
                
            except Exception as e:
                evidence.append(f"Error using ML model: {str(e)}. Falling back to enhanced rule-based analysis.")
        
        # Fallback to enhanced rule-based methods
        return self._enhanced_rule_based_analysis(user_data, features, evidence)
    
    def _enhanced_rule_based_analysis(self, user_data, features, evidence):
        """
        Fallback method that uses rule-based analysis enhanced with ML features.
        
        Args:
            user_data (dict): User data
            features (dict): Extracted features
            evidence (list): List to append evidence to
            
        Returns:
            float: Probability that the user is a bot (0-1 scale)
            list: Evidence with additional items
        """
        # Start with base probability
        bot_probability = 0.5
        
        # Analyze username patterns with ML features
        username = user_data.get('username', '')
        if features['username_digit_ratio'] > 0.5:
            bot_probability += 0.1
            evidence.append(f"Username '{username}' has a high digit ratio ({features['username_digit_ratio']:.2f}), suggesting automated generation")
        elif features['username_special_char_count'] > 2:
            bot_probability += 0.05
            evidence.append(f"Username '{username}' has {features['username_special_char_count']} special characters, unusual for human users")
        else:
            bot_probability -= 0.1
            evidence.append(f"Username '{username}' has natural characteristics with balanced alphabet and number usage")
        
        # Analyze email patterns
        email = user_data.get('email', '')
        if features['email_domain_common'] == 1:
            bot_probability -= 0.15
            evidence.append(f"Email '{email}' uses a common provider domain, typical of human users")
        else:
            bot_probability += 0.1
            evidence.append(f"Email '{email}' uses an uncommon domain, potential sign of automated account")
        
        # Analyze bio complexity using NLP features
        bio = user_data.get('bio', '')
        if features['bio_word_count'] < 5:
            bot_probability += 0.15
            evidence.append("Bio text is unusually short, potential sign of minimal bot input")
        elif features['bio_sentence_count'] > 1 and features['bio_avg_word_length'] > 4:
            bot_probability -= 0.2
            evidence.append("Bio shows complex sentence structure and vocabulary typical of human writing")
        
        # Analyze typing patterns
        if features['typing_pattern_count'] > 0:
            if features['typing_pattern_std'] < 50:
                bot_probability += 0.25
                evidence.append(f"Typing rhythm shows unusually consistent intervals ({features['typing_pattern_std']:.2f}ms), suggesting automated input")
            elif 50 <= features['typing_pattern_std'] <= 500:
                bot_probability -= 0.25
                evidence.append(f"Typing rhythm shows natural human variation ({features['typing_pattern_std']:.2f}ms)")
        
        # Time to fill analysis
        if features['time_to_fill'] < 5:
            bot_probability += 0.3
            evidence.append(f"Form completed extremely quickly ({features['time_to_fill']:.2f}s), strongly suggesting automation")
        elif 5 <= features['time_to_fill'] <= 15:
            bot_probability += 0.1
            evidence.append(f"Form completed very quickly ({features['time_to_fill']:.2f}s), potential bot behavior")
        elif 15 < features['time_to_fill'] <= 180:
            bot_probability -= 0.2
            evidence.append(f"Form completion time ({features['time_to_fill']:.2f}s) is within normal human range")
        
        # Ensure probability is between 0 and 1
        bot_probability = max(0, min(1, bot_probability))
        
        evidence.append(f"Enhanced analysis predicts a bot probability of {bot_probability:.2%}")
        return bot_probability, evidence
    
    def train(self, data, labels):
        """
        Train the ML models using labeled data.
        
        Args:
            data (list): List of user_data dictionaries
            labels (list): List of binary labels (1 for bot, 0 for human)
            
        Returns:
            float: Training accuracy
        """
        print(f"Training ML models with {len(data)} samples...")
        
        # Extract features from all samples
        X = []
        for user_data in data:
            features = self.extract_features(user_data)
            feature_vector = [features[f] for f in sorted(features.keys())]
            X.append(feature_vector)
        
        X = np.array(X)
        y = np.array(labels)
        
        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train random forest classifier
        self.rf_classifier.fit(X_scaled, y)
        
        # Save models
        joblib.dump(self.rf_classifier, os.path.join(self.models_path, 'rf_classifier.pkl'))
        joblib.dump(self.tfidf_vectorizer, os.path.join(self.models_path, 'tfidf_vectorizer.pkl'))
        joblib.dump(self.scaler, os.path.join(self.models_path, 'scaler.pkl'))
        
        # Return training accuracy
        accuracy = self.rf_classifier.score(X_scaled, y)
        print(f"Training completed with accuracy: {accuracy:.2%}")
        return accuracy

# Initialize global detector instance
ml_detector = MLDetector() 