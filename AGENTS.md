# 轮腿项目工作约束

每项项目任务必须应用 [wheelleg-project-guard](.agents/skills/wheelleg-project-guard/SKILL.md)，包括分析、修改、回归、查新与“继续”。优先复用当前上下文中的同一版本，避免重复读取。
用户最新明确要求优先；仅按需读取计划对应小节。通过验收即结束，不扩展无关工作。

## 持久执行硬约束（用户要求，2026-09-21）

- 每轮项目任务开始先按需读取根目录 `PROJECT_MEMORY.md` 的当前状态和未完成项；较大的对话结束前更新决策、变更、验证证据、失败和下一步，不能只依赖聊天上下文。
- GPU 新基线位于 `wheelleg_warp/`，采用 MuJoCo Warp；原 `wheelleg_ppo/` CPU 代码及历史结果为效果校对基准。未通过配对校对不得替换原基线，物理吞吐不得表述成 PPO 端到端训练加速。
- 每次实质变更完成后及时提交，提交标题和说明使用中文，说明“改了什么、为何修改、验证结果/未完成项”；同时暂存 `PROJECT_MEMORY.md`。禁止绕过 `.githooks/commit-msg`，新克隆执行 `git config core.hooksPath .githooks`。
- 修改前停止 `systemctl --user stop wheelleg-git-sync.service`，工作收尾、更新记忆并提交后重新启动；或全程持有 `.git/project-write.lock` 的排他锁。不可让守护进程把一轮未完成编辑拆成验收提交。
- GitHub 同步守护 `wheelleg-git-sync.service` 每30秒检查；稳定落盘的非忽略文件自动生成中文快照提交、更新记忆、推送当前上游分支。显式提交也自动推送。瞬间编辑及尚未稳定写完的文件不逐次提交，忽略文件不上传。
- 尊重他人暂存内容，不强推、不重写历史、不自动解决远程冲突。推送失败保留本地提交并在 journal 中报告，不能声称已同步；任务结束须检查远程 HEAD 与本地一致及守护服务 active。
