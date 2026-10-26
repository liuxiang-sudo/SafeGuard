"""Behavioral regressions for offline detection, training, and Web uploads."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from multiguard.data.validation import validate_training_data
from multiguard.detection.ml_detector import MLDetector
from multiguard.detection import ml_detector as detector_module
from multiguard.detection import platform_detector
from multiguard.training import train_model as training
from multiguard.web.app import app


def dataset():
    samples, labels = [], []
    for label in (0, 1):
        for index in range(6):
            samples.append({
                'username': f'bot{index}999999' if label else f'alice_{index}',
                'email': 'bot@automated.example' if label else 'alice@gmail.com',
                'bio': 'Buy now!' if label else 'I enjoy reading books and meeting my friends.',
                'time_to_fill': 2 if label else 60,
                'typing_pattern': '20,20,21' if label else '150,450,250',
                'navigation': index,
            })
            labels.append(label)
    return samples, labels


class TemporaryDetectorTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.detector = MLDetector(self.directory / 'models', use_transformer=False)
        self.samples, self.labels = dataset()


class DetectorTests(TemporaryDetectorTest):
    def test_missing_nltk_data_uses_offline_tokenizers(self):
        with patch.object(detector_module, 'word_tokenize', side_effect=LookupError), \
             patch.object(detector_module, 'sent_tokenize', side_effect=LookupError):
            features = self.detector.extract_features(self.samples[0])
            probability, evidence = self.detector.predict_bot_probability(self.samples[0])
        self.assertGreater(features['bio_word_count'], 0)
        self.assertTrue(np.isfinite(probability))
        self.assertTrue(0 <= probability <= 1)
        self.assertTrue(evidence)

    def test_numeric_null_and_malformed_features_remain_finite(self):
        for navigation in (2, '2.0', None, '²', 'nan'):
            with self.subTest(navigation=navigation):
                features = self.detector.extract_features({
                    'username': 123, 'email': None, 'bio': None,
                    'navigation': navigation, 'time_to_fill': 'inf',
                    'typing_pattern': '1,nan,-2',
                })
                self.assertTrue(all(np.isfinite(value) for value in features.values()))

    def test_optional_transformer_failure_is_attempted_only_once(self):
        import builtins
        original_import = builtins.__import__
        model_dir = self.directory / 'optional-models'
        (model_dir / 'distilbert-base-uncased').mkdir(parents=True)
        (model_dir / 'distilbert-base-uncased/config.json').write_text('{}')
        detector = MLDetector(model_dir)
        attempts = []

        def offline_import(name, *args, **kwargs):
            if name == 'transformers':
                attempts.append(name)
                raise ImportError('Optional dependency is absent')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=offline_import):
            first = detector.get_text_embedding('first')
            second = detector.get_text_embedding('second')
        self.assertEqual(attempts, ['transformers'])
        self.assertTrue(np.array_equal(first, np.zeros(768)))
        self.assertTrue(np.array_equal(first, second))

    def test_training_round_trip_predicts_with_the_trained_classifier(self):
        self.detector.train(self.samples, self.labels)
        probability, evidence = self.detector.predict_bot_probability(self.samples[-1])
        self.assertTrue(any('Machine learning model predicts' in item for item in evidence))
        reloaded = MLDetector(self.detector.models_path, use_transformer=False)
        actual, reloaded_evidence = reloaded.predict_bot_probability(self.samples[-1])
        self.assertEqual(probability, actual)
        self.assertTrue(any('Machine learning model predicts' in item for item in reloaded_evidence))

    def test_failed_save_preserves_the_active_and_saved_model(self):
        self.detector.train(self.samples, self.labels)
        classifier, scaler = self.detector.rf_classifier, self.detector.scaler
        bundle_path = Path(self.detector.models_path) / 'detector.joblib'
        original = bundle_path.read_bytes()
        with patch.object(detector_module.joblib, 'dump', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.detector.train(self.samples, self.labels)
        self.assertIs(self.detector.rf_classifier, classifier)
        self.assertIs(self.detector.scaler, scaler)
        self.assertEqual(bundle_path.read_bytes(), original)
        self.assertEqual(len(list(bundle_path.parent.glob('*.joblib'))), 1)

    def test_invalid_labels_do_not_create_a_model(self):
        for labels in ([0] * len(self.samples), self.labels[:-1], [2] * len(self.samples)):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                self.detector.train(self.samples, labels)
        self.assertFalse(Path(self.detector.models_path).exists())


class TrainingTests(TemporaryDetectorTest):
    def test_supported_json_and_csv_formats_share_one_loader(self):
        samples, labels = self.samples[:1] + self.samples[-1:], [0, 1]
        variants = (
            {'samples': samples, 'labels': labels},
            {'data': samples, 'labels': labels},
            [{'data': sample, 'is_bot': bool(label)} for sample, label in zip(samples, labels)],
        )
        for index, content in enumerate(variants):
            path = self.directory / f'dataset-{index}.json'
            path.write_text(json.dumps(content, ensure_ascii=False), encoding='utf-8-sig')
            loaded, actual_labels = training.load_data_from_file(path)
            self.assertEqual(actual_labels, labels)
            self.assertEqual(loaded[0]['username'], samples[0]['username'])
        path = self.directory / 'dataset.csv'
        path.write_text('username,email,label\n用户,user@example.com,0\nbot,bot@example.com,1\n',
                        encoding='utf-8-sig')
        loaded, actual_labels = training.load_data_from_file(path)
        self.assertEqual(loaded[0]['username'], '用户')
        self.assertEqual(actual_labels, labels)

    def test_mismatched_nonbinary_and_nonfinite_labels_are_rejected(self):
        for labels in ([0], [0, 2], [0, 'nan'], [0, None]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                validate_training_data(self.samples[:2], labels)

    def test_invalid_cli_path_does_not_fall_back_to_synthetic_training(self):
        with patch.object(sys, 'argv', ['train', '--data', str(self.directory / 'missing.json')]), \
             patch.object(training, 'generate_synthetic_data') as generate, \
             patch.object(training.ml_detector, 'train') as train, \
             redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                training.main()
        self.assertEqual(error.exception.code, 2)
        generate.assert_not_called()
        train.assert_not_called()

    def test_cli_can_train_directly_on_upload_format(self):
        path = self.directory / 'input.json'
        path.write_text(json.dumps({'samples': self.samples, 'labels': self.labels}), encoding='utf-8')
        with patch.object(sys, 'argv', ['train', '--data', str(path)]), \
             patch.object(training, 'ml_detector', self.detector), \
             patch.object(training, 'PROCESSED_DIR', self.directory), \
             patch.object(training, 'ensure_runtime_dirs'), \
             redirect_stdout(io.StringIO()) as output:
            training.main()
        self.assertTrue((self.directory / 'real_dataset.json').is_file())
        self.assertIn('训练集准确率', output.getvalue())
        self.assertTrue(hasattr(self.detector.rf_classifier, 'classes_'))

    def test_synthetic_data_requires_at_least_two_samples(self):
        for size in (0, 1, -1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                training.generate_synthetic_data(size)
        with redirect_stdout(io.StringIO()):
            samples, labels = training.generate_synthetic_data(5)
        self.assertEqual(len(samples), 5)
        self.assertEqual(set(labels), {0, 1})


class EvaluationTests(TemporaryDetectorTest):
    def test_small_balanced_dataset_can_be_evaluated_without_fixed_five_folds(self):
        try:
            from multiguard.evaluation import enhanced_evaluation as evaluation
        except ImportError:
            self.skipTest('Optional evaluation dependencies are not installed')
        samples = self.samples[:3] + self.samples[-3:]
        labels = [0, 0, 0, 1, 1, 1]
        with patch.object(evaluation, 'load_twibot20_data', return_value=(samples, labels)), \
             patch.object(evaluation, 'MLDetector', return_value=self.detector), \
             patch.object(evaluation, 'REPORTS_DIR', self.directory), \
             redirect_stdout(io.StringIO()):
            results = evaluation.comprehensive_evaluation()
        self.assertTrue(np.isfinite(results['cv_mean']))
        self.assertEqual(np.asarray(results['confusion_matrix']).shape, (2, 2))
        self.assertTrue((self.directory / 'evaluation_results.json').is_file())


class WebTests(TemporaryDetectorTest):
    def setUp(self):
        super().setUp()
        original = {name: app.config[name] for name in ('TESTING', 'SECRET_KEY', 'UPLOAD_FOLDER')}
        self.addCleanup(app.config.update, original)
        app.config.update(TESTING=True, SECRET_KEY='test-only-key',
                          UPLOAD_FOLDER=str(self.directory / 'uploads'))
        self.client = app.test_client()
        for target, value in (
                ('multiguard.training.train_model.ml_detector', self.detector),
                ('multiguard.detection.platform_detector.ml_detector', self.detector),
                ('multiguard.training.train_model.TRAINING_SAMPLES_DIR', self.directory / 'samples')):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_pages_static_assets_and_all_platforms_work_without_models(self):
        for path in ('/', '/upload', '/static/js/script.js', '/static/css/style.css'):
            with self.subTest(path=path):
                with self.client.get(path) as response:
                    self.assertEqual(response.status_code, 200)
        for platform in ('general', 'weibo', 'douyin', 'tieba'):
            with self.subTest(platform=platform):
                response = self.client.post('/analyze', data={
                    'platform': platform, 'username': 'sample', 'bio': 'Hello world.',
                    'screen_size': 'invalidx900', 'navigation': '2',
                })
                self.assertEqual(response.status_code, 200)

    def test_chinese_filenames_and_repeated_uploads_train_the_live_detector(self):
        for _ in range(2):
            content = json.dumps({'samples': self.samples, 'labels': self.labels}).encode('utf-8')
            with redirect_stdout(io.StringIO()):
                response = self.client.post('/upload', data={
                    'dataset_file': (io.BytesIO(content), '训练数据.json'),
                })
            self.assertEqual(response.status_code, 302)
        uploaded = list((self.directory / 'uploads').glob('*.json'))
        self.assertEqual(len(uploaded), 2)
        self.assertTrue(hasattr(self.detector.rf_classifier, 'classes_'))
        _, evidence = platform_detector.ml_detector.predict_bot_probability(self.samples[-1])
        self.assertTrue(any('Machine learning model predicts' in item for item in evidence))

    def test_invalid_platform_is_rejected_before_upload(self):
        response = self.client.post('/upload', data={
            'platform': '../../outside',
            'dataset_file': (io.BytesIO(b'{}'), 'data.json'),
        })
        self.assertEqual(response.status_code, 302)
        self.assertFalse((self.directory / 'uploads').exists())
        self.assertEqual(self.client.post('/analyze', data={'platform': 'unknown'}).status_code, 400)
        self.assertEqual(self.client.get('/download_template/unknown').status_code, 404)

    def test_single_class_upload_reports_failure_without_replacing_model(self):
        content = json.dumps({'samples': self.samples[:2], 'labels': [0, 0]}).encode()
        response = self.client.post('/upload', data={
            'dataset_file': (io.BytesIO(content), 'one-class.json'),
        })
        self.assertEqual(response.status_code, 302)
        with self.client.session_transaction() as session:
            messages = session['_flashes']
        self.assertTrue(any('两类样本' in message for _, message in messages))
        self.assertFalse(hasattr(self.detector.rf_classifier, 'classes_'))

    def test_bad_dates_and_screen_sizes_do_not_break_rule_fallback(self):
        score, evidence = platform_detector.analyze_account_age('2999-01-01')
        self.assertEqual(score, 0.5)
        result = platform_detector.analyze_platform_user({
            'screen_size': 'not-a-numberx900', 'bio': None, 'user_agent': None,
        })
        self.assertTrue(0 <= result['probability_bot'] <= 1)


if __name__ == '__main__':
    unittest.main()
