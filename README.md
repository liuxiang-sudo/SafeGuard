# SafeGuard

基于 Flask 的多平台人机检测应用，结合账号资料、交互行为、平台规则和机器学习模型判断用户是否为机器人。

## 功能

- 支持通用场景、微博、抖音和贴吧的账号分析。
- 分析用户名、邮箱、简介和表单交互特征。
- 展示人类 / 机器人 / 不确定分类、概率及判断依据。
- 支持上传 JSON / CSV 数据集进行模型训练。
- 提供数据处理、模型训练和评估工具。

## 运行

建议使用 Python 3.11，在项目根目录执行：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

浏览器访问 <http://127.0.0.1:5000>，按 `Ctrl+C` 停止服务。

仓库不包含数据集和模型文件。缺少模型时程序会尝试降级分析；完整模型准备见 [模型说明](models/README.md)。检测结果用于实验与展示。

## 训练与测试

在已安装依赖的 Python 环境中执行（Windows 可将 `python` 替换为 `.\.venv\Scripts\python.exe`）：

```bash
# 使用合成数据训练模型
python -m multiguard.training.train_model --synth 1000

# 运行目录迁移回归测试
python -m unittest discover -s tests -v
```
