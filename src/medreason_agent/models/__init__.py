"""模型 backend 接口。

实验脚本通过这里的 backend 层调用 mock 或真实 Qwen，避免把具体模型细节散落到
每个实验脚本里。
"""
