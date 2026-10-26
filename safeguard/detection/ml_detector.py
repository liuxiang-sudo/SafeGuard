"""Feature extraction and a reusable detector with offline rule fallback."""
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import warnings
from threading import RLock

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import StandardScaler

from safeguard.data.validation import normalize_user_data, validate_training_data
from safeguard.paths import MODELS_DIR

logger = logging.getLogger(__name__)


def _load_saved_model(path):
    # Version-mismatched pickle estimators can load but fail later at prediction.
    with warnings.catch_warnings():
        warnings.simplefilter('error', InconsistentVersionWarning)
        return joblib.load(path)

try:
    from nltk.tokenize import word_tokenize, sent_tokenize
except ImportError:
    word_tokenize = sent_tokenize = None

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

class MLDetector:
    def __init__(self, models_path=None, use_transformer=True):
        self.models_path = str(models_path if models_path is not None else MODELS_DIR)
        self.transformer_model_name = str(Path(self.models_path) / "distilbert-base-uncased")
        self.tokenizer = None
        self.transformer_model = None
        self._transformer_attempted = False
        self._allow_transformer = use_transformer
        self._requires_transformer = False
        self._lock = RLock()
        self.feature_names = None
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self._load_models()

    def _load_models(self):
        """Load one coherent model bundle; keep support for legacy local models."""
        bundle_path = Path(self.models_path) / 'detector.joblib'
        try:
            if bundle_path.is_file():
                bundle = _load_saved_model(bundle_path)
                classifier, scaler = bundle['classifier'], bundle['scaler']
                self.feature_names = bundle['feature_names']
                self._requires_transformer = bundle['uses_transformer']
                self._allow_transformer = self._allow_transformer and bundle['uses_transformer']
            else:
                classifier_path = Path(self.models_path) / 'rf_classifier.pkl'
                scaler_path = Path(self.models_path) / 'scaler.pkl'
                if not (classifier_path.is_file() and scaler_path.is_file()):
                    return
                classifier, scaler = _load_saved_model(classifier_path), _load_saved_model(scaler_path)
            if set(classifier.classes_) != {0, 1}:
                raise ValueError('Saved classifier must contain both classes')
            self.rf_classifier, self.scaler = classifier, scaler
        except Exception as exc:
            self.feature_names = None
            self._requires_transformer = False
            logger.warning('Cannot load local classifier; using rules: %s', exc)

    def _load_transformer(self):
        """Attempt the optional local model once; never download on a request."""
        if self._transformer_attempted or not self._allow_transformer:
            return
        self._transformer_attempted = True
        if not (Path(self.transformer_model_name) / 'config.json').is_file():
            return
        try:
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name, local_files_only=True)
            model = AutoModel.from_pretrained(self.transformer_model_name, local_files_only=True)
            model.eval()
            self.tokenizer, self.transformer_model = tokenizer, model
        except Exception as exc:
            logger.info('Local text model unavailable; using zero embeddings: %s', exc)

    def get_text_embedding(self, text):
        with self._lock:
            self._load_transformer()
            if not text.strip() or self.transformer_model is None:
                return np.zeros(768)
            import torch
            inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
            return outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().flatten()

    def extract_features(self, user_data):
        """
        Extract and normalize features from user data for ML model.
        
        Args:
            user_data (dict): User data including username, email, bio, etc.
            
        Returns:
            dict: Dictionary of extracted features
        """
        user_data = normalize_user_data(user_data)
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
            words = word_tokenize(bio) if word_tokenize else custom_word_tokenize(bio)
        except LookupError:
            words = custom_word_tokenize(bio)
        features['bio_word_count'] = len(words)
        features['bio_avg_word_length'] = np.mean([len(w) for w in words]) if words else 0

        try:
            sentences = sent_tokenize(bio) if sent_tokenize else custom_sent_tokenize(bio)
        except LookupError:
            sentences = custom_sent_tokenize(bio)
        features['bio_sentence_count'] = len(sentences)
        
        features['bio_special_char_ratio'] = len(re.findall(r'[^a-zA-Z0-9\s]', bio)) / max(1, len(bio))
        
        # Get transformer embeddings for text
        bio_embedding = self.get_text_embedding(bio)
        for i, val in enumerate(bio_embedding):
            features[f'bio_embedding_{i}'] = val
        
        # Time and interaction features
        try:
            value = float(user_data.get('time_to_fill', 0))
            features['time_to_fill'] = value if math.isfinite(value) and value >= 0 else 0
        except:
            features['time_to_fill'] = 0
            
        # Typing pattern features
        typing_pattern = user_data.get('typing_pattern', '')
        if typing_pattern:
            try:
                intervals = [float(i) for i in typing_pattern.split(',')]
                if not all(math.isfinite(value) and value >= 0 for value in intervals):
                    raise ValueError('Invalid typing interval')
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
        try:
            value = float(navigation)
            features['navigation_count'] = int(value) if math.isfinite(value) and value >= 0 and value.is_integer() else 0
        except (TypeError, ValueError, OverflowError):
            features['navigation_count'] = 0
        
        return features
    
    def predict_bot_probability(self, user_data):
        with self._lock:
            return self._predict_bot_probability(normalize_user_data(user_data))

    def _predict_bot_probability(self, user_data):
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
        feature_names = self.feature_names or sorted(features.keys())
        
        # Check if we have a trained model
        if self.rf_classifier is not None and hasattr(self.rf_classifier, 'classes_'):
            # We have a trained model, use it for prediction
            try:
                if set(feature_names) != set(features):
                    raise ValueError('The saved model uses a different feature schema')
                feature_vector = np.array([features[f] for f in feature_names]).reshape(1, -1)
                # Scale features
                scaled_features = self.scaler.transform(feature_vector)
                
                # Make prediction
                if self._requires_transformer and self.transformer_model is None:
                    raise ValueError('The text model used for training is unavailable')
                bot_index = list(self.rf_classifier.classes_).index(1)
                bot_probability = float(self.rf_classifier.predict_proba(scaled_features)[0][bot_index])
                
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
        if user_data.get('time_to_fill') and features['time_to_fill'] < 5:
            bot_probability += 0.3
            evidence.append(f"Form completed extremely quickly ({features['time_to_fill']:.2f}s), strongly suggesting automation")
        elif user_data.get('time_to_fill') and 5 <= features['time_to_fill'] <= 15:
            bot_probability += 0.1
            evidence.append(f"Form completed very quickly ({features['time_to_fill']:.2f}s), potential bot behavior")
        elif user_data.get('time_to_fill') and 15 < features['time_to_fill'] <= 180:
            bot_probability -= 0.2
            evidence.append(f"Form completion time ({features['time_to_fill']:.2f}s) is within normal human range")
        
        # Ensure probability is between 0 and 1
        bot_probability = max(0, min(1, bot_probability))
        
        evidence.append(f"Enhanced analysis predicts a bot probability of {bot_probability:.2%}")
        return bot_probability, evidence
    
    def train(self, data, labels):
        """Validate and fit a candidate; publish it only after a successful save.

        The returned accuracy is measured on the training set, not held-out data.
        """
        data, labels = validate_training_data(data, labels)
        with self._lock:
            self._load_transformer()
            features = [self.extract_features(sample) for sample in data]
            names = sorted(features[0])
            X = np.asarray([[sample[name] for name in names] for sample in features])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(X_scaled, labels)
            accuracy = float(classifier.score(X_scaled, labels))
            bundle = {'classifier': classifier, 'scaler': scaler,
                      'feature_names': names,
                      'uses_transformer': self.transformer_model is not None}
            Path(self.models_path).mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=self.models_path, suffix='.joblib')
            os.close(fd)
            try:
                joblib.dump(bundle, temporary)
                os.replace(temporary, Path(self.models_path) / 'detector.joblib')
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self.rf_classifier, self.scaler = classifier, scaler
            self.feature_names = names
            self._requires_transformer = bundle['uses_transformer']
            return accuracy


# The web and training modules share this instance so uploaded training is live.
ml_detector = MLDetector()
