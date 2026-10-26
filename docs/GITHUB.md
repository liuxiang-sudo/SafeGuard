# GitHub 同步

目标仓库：https://github.com/liuxiang-sudo/SafeGuard

GitHub 使用独立的源码初始提交，避免上传原历史中的虚拟环境和大型模型。原 master 历史与 Gitee origin 保留。

本地发布分支为 codex/github-upload，对应 GitHub 的 main。后续从该分支提交并推送：

```bash
git add -A
git diff --cached --stat
git commit -m "Describe your changes"
git push github HEAD:main
```

.gitignore 排除本地环境、模型、全部实际数据、示例 JSON、缓存、IDE 配置和生成报告。第三方许可证保存在 docs/third_party/，目录中的说明文件仍可提交。

每次提交前可用 `git ls-files -ci --exclude-standard` 检查是否误跟踪了忽略文件。原 master 历史含大型文件，请勿将该历史合并至发布分支。
