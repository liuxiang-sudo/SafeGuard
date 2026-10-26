"""Regression checks for moving code and resources into packages.

These checks need only the standard library; no models or network are used.
"""
import ast
from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from safeguard.paths import PROJECT_ROOT, WEB_DIR


class StructureTests(unittest.TestCase):
    def test_resource_paths_from_another_working_directory(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-c',
                 'from safeguard.paths import PROJECT_ROOT, WEB_DIR, MODELS_DIR; '
                 'assert (WEB_DIR / "templates/index.html").is_file(); '
                 'assert MODELS_DIR == PROJECT_ROOT / "models"; '
                 'print(PROJECT_ROOT)'],
                cwd=directory, env=env, text=True, capture_output=True,
                encoding='utf-8', errors='replace',
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_python_syntax_and_internal_import_targets(self):
        sources = [PROJECT_ROOT / 'app.py']
        for directory in ('safeguard', 'examples', 'tests'):
            sources.extend((PROJECT_ROOT / directory).rglob('*.py'))
        old_modules = {'ml_detector', 'platform_detector', 'detector',
                       'train_model', 'data_processing'}
        for source in sources:
            with self.subTest(source=str(source.relative_to(PROJECT_ROOT))):
                tree = ast.parse(source.read_text(encoding='utf-8'))
                package = '.'.join(source.relative_to(PROJECT_ROOT).parts[:-1])
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    if node.level:
                        module = importlib.util.resolve_name(
                            '.' * node.level + (node.module or ''), package)
                    else:
                        module = node.module or ''
                        self.assertNotIn(module, old_modules)
                    if module.startswith('safeguard.'):
                        path = PROJECT_ROOT.joinpath(*module.split('.'))
                        self.assertTrue(path.with_suffix('.py').is_file()
                                        or (path / '__init__.py').is_file(), module)

    def test_moved_page_resources_exist(self):
        app_source = (WEB_DIR / 'app.py').read_text(encoding='utf-8')
        for template in re.findall(r"render_template\(['\"]([^'\"]+)", app_source):
            self.assertTrue((WEB_DIR / 'templates' / template).is_file(), template)
        for template in (WEB_DIR / 'templates').glob('*.html'):
            text = template.read_text(encoding='utf-8')
            for asset in re.findall(
                    r"url_for\(['\"]static['\"],\s*filename=['\"]([^'\"]+)", text):
                self.assertTrue((WEB_DIR / 'static' / asset).is_file(), asset)

    def test_converters_read_raw_data_and_write_processed_output(self):
        fixture = [{
            'ID': '1', 'profile': {'screen_name': 'sample_user',
                                  'description': 'A sample profile'},
            'tweet': ['A sample post'], 'domain': 'general',
        }]
        converters = (
            ('convert_twibot20_data', 'convert_twibot20_to_system_format'),
            ('simple_twibot20_converter', 'convert_twibot20_simple'),
        )
        for module_name, function_name in converters:
            with self.subTest(converter=module_name), tempfile.TemporaryDirectory() as tmp:
                raw, processed = Path(tmp) / 'raw', Path(tmp) / 'processed'
                raw.mkdir()
                processed.mkdir()
                (raw / 'TwiBot-20_sample.json').write_text(
                    json.dumps(fixture), encoding='utf-8')
                module = importlib.import_module(f'safeguard.data.{module_name}')
                with patch.object(module, 'TWIBOT_DIR', raw), \
                     patch.object(module, 'PROCESSED_DIR', processed), \
                     patch.object(module, 'ensure_runtime_dirs'), \
                     redirect_stdout(io.StringIO()):
                    output = Path(getattr(module, function_name)())
                self.assertEqual(output.parent, processed)
                data = json.loads(output.read_text(encoding='utf-8'))
                self.assertEqual(len(data['samples']), 1)
                self.assertEqual(len(data['labels']), 1)
                self.assertEqual(data['samples'][0]['username'], 'sample_user')


if __name__ == '__main__':
    unittest.main()
