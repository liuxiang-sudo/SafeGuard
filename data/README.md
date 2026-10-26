# 数据目录

- raw/TwiBot-20/：原始 TwiBot-20 数据和第三方说明，转换器读取 TwiBot-20_sample.json。
- datasets/：下载、上传的 JSON / CSV；本地 CSV 处理器读取 twitter_real_data.csv。
- processed/：转换与增强结果、预处理输出、训练导出的 real_dataset.json。
- training_samples/：上传训练时生成的样本摘要。

实际数据默认不纳入 Git。小型示例也仅在本地 examples/data/ 中保留。网页上传接受 samples（或 data）与 labels 字段；命令行训练的 --data 使用由 data、is_bot 组成的记录列表。

现有 TwiBot 转换逻辑包含规则推断标签和模拟行为特征，其输出不能直接视为真实行为观测或原数据集标签。
