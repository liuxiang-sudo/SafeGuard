# 本地模型

模型权重和训练产物均不上传 GitHub。

## 规则与随机森林

不提供文本模型也可以运行。程序使用规则降级分析，或使用合成 / 自备数据训练随机森林：

```bash
python -m multiguard.training.train_model --synth 1000
```

训练会原子写入 models/detector.joblib，包含分类器、缩放器、特征顺序和文本模型使用信息。Web 上传训练完成后，当前进程中的检测器立即使用新模型。多个独立服务进程仍需分别重载或重启。

已有 rf_classifier.pkl 和 scaler.pkl 仍支持读取；scikit-learn 版本不兼容时会回退到规则分析，建议在当前环境重新训练。仅加载可信来源的序列化模型。

## 可选文本模型

需要完整文本嵌入时，先安装额外依赖：

```bash
python -m pip install -r requirements-transformer.txt
```

自行将 distilbert-base-uncased 模型保存到 models/distilbert-base-uncased/，包含 config.json、tokenizer 文件、vocab.txt 和兼容模型权重。

检测器只读取本地模型，首次需要时加载一次，不会在请求中下载模型或 NLTK 资源。缺少文本模型时使用零向量；使用文本模型训练的分类器在文本模型不可用时会降级为规则分析。
