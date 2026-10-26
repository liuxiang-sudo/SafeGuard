"""Shared validation for uploaded and command-line training data."""
import math


def normalize_user_data(sample):
    if not isinstance(sample, dict):
        raise ValueError('每个样本必须是对象')
    normalized = {}
    for key, value in sample.items():
        if value is None:
            normalized[key] = ''
        elif isinstance(value, bool):
            normalized[key] = str(value).lower()
        elif isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'字段 {key} 不能包含 NaN 或 Infinity')
            normalized[key] = str(value)
        else:
            raise ValueError(f'字段 {key} 必须为文本或数值')
    return normalized


def validate_training_data(samples, labels, require_both_classes=True):
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError('数据集必须包含非空样本列表')
    if not isinstance(labels, (list, tuple)) or len(samples) != len(labels):
        raise ValueError('样本数和标签数必须一致')
    clean_labels = []
    for value in labels:
        if isinstance(value, str) and value.strip().lower() in {'true', 'false'}:
            value = int(value.strip().lower() == 'true')
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError('标签只能为 0（人类）或 1（机器人）') from None
        if not math.isfinite(number) or number not in (0, 1):
            raise ValueError('标签只能为 0（人类）或 1（机器人）')
        clean_labels.append(int(number))
    if require_both_classes and set(clean_labels) != {0, 1}:
        raise ValueError('训练数据必须同时包含人类和机器人两类样本')
    return [normalize_user_data(sample) for sample in samples], clean_labels
