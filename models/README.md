# 本地模型

该目录保存模型资源，权重和训练产物不纳入 Git。

`MLDetector` 从 `models/distilbert-base-uncased/` 读取 DistilBERT，包括 config.json、tokenizer_config.json、tokenizer.json、vocab.txt 和 pytorch_model.bin（或兼容权重）。

如需完整文本特征，请自行获取与 distilbert-base-uncased 对应的 Hugging Face 模型和 tokenizer，并保存至该目录。已有本地模型可继续使用，无须上传压缩包。

训练脚本生成 rf_classifier.pkl、scaler.pkl 和 tfidf_vectorizer.pkl。这些文件应由同一环境、同一次训练生成；仅加载可信来源的序列化模型。

缺少 Transformer 时，现有实现使用零向量替代文本嵌入；缺少分类器时也会尝试降级处理。需要完整 ML 推理时，应先准备文本模型，再训练分类器。
