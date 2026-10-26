# 代码结构与迁移说明

## 模块职责

| 模块 | 职责 | 主要依赖 |
| --- | --- | --- |
| safeguard.web.app | 页面、表单分析、数据上传与训练路由 | detection、training、paths |
| safeguard.detection.platform_detector | 各平台规则与 ML 结果整合 | ml_detector |
| safeguard.detection.detector | 通用规则和检测工具 | ml_detector |
| safeguard.detection.ml_detector | 文本与行为特征、随机森林、模型保存和预测 | paths、PyTorch、Transformers、scikit-learn |
| safeguard.training.train_model | 数据加载、合成训练数据和训练入口 | MLDetector、paths |
| safeguard.data | 下载、预处理和 TwiBot-20 转换 | paths；增强数据时调用 training |
| safeguard.evaluation | 指标评估、报告与图表 | MLDetector、paths、绘图库 |
| safeguard.paths | 仓库根路径和运行目录 | Python 标准库 |

模板与静态资源由 web 模块管理。业务代码通过包导入引用彼此，不向 sys.path 临时添加目录，也不通过改变工作目录寻找模型。

## 旧文件位置

| 原位置 | 新位置 |
| --- | --- |
| app.py 中的路由 | safeguard/web/app.py；根 app.py 保留启动入口 |
| detector.py、platform_detector.py、ml_detector.py | safeguard/detection/ |
| train_model.py | safeguard/training/train_model.py |
| data_processing.py、两个 TwiBot 转换脚本、download_real_datasets.py | safeguard/data/ |
| download_and_process_real_dataset.py | safeguard/data/process_local_dataset.py |
| enhanced_evaluation.py | safeguard/evaluation/enhanced_evaluation.py |
| auto_script_weibo.py | examples/weibo_demo.py |
| test.py | examples/tokenization_demo.py（演示，不参与测试发现） |
| test_twibot20_integration.py | tests/twibot20_integration.py（手动运行） |
| templates/、static/ | safeguard/web/ 下对应目录 |
| datasets/、training_samples/、TwiBot-20/ | data/datasets/、data/training_samples/、data/raw/TwiBot-20/ |
| 根目录生成的 JSON | data/processed/ |
| sample_training_data.json、twitter_template.json | examples/data/ |
| evaluation_results.*、result_*.html | reports/ |
| README.pdf | docs/legacy/README.pdf |
| hbdenv/、__MACOSX/、旧 __pycache__/ | local/ 下对应归档目录 |

## 运行与维护

从根目录用 `python -m 包.模块` 执行工具。演示脚本和本地 CSV 处理脚本仅在作为命令运行时执行操作，导入时不提交请求、打开浏览器或读写数据。

资源路径统一维护在 safeguard/paths.py。新增工具应引用这些路径；用户传入的文件路径则保留命令行语义。

原 data_processing.py 末尾绕过了已有 main()，固定处理某个 JSON 文件。整理后接回 main()，使 --samples、--dataset、--type 和 --output 等既有参数生效。两个 TwiBot 转换器原先引用不存在的 BotRGCN 目录，已改为 data/raw/TwiBot-20/。

训练数据统一校验后交给共享检测器，支持网页与命令行使用同一份 JSON / CSV。模型通过单一 bundle 原子保存，单进程上传训练立即生效；保存失败保留原模型。文本模型按需加载且仅访问本地资源，缺少 NLTK 分词资源时离线回退。

回归测试覆盖迁移路径、数据格式、非法标签、模型保存失败、训练与重载预测、无模型网页分析和上传训练。完整文本模型推理使用本机 dag_gpu 环境验证；GPU 加速未作调整。
