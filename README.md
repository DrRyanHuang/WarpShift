# WarpShift

在已配置 SGLang 运行依赖的 CUDA 环境中使用。差分模式还需 NumPy、Triton 和 `nvidia.nvcomp`。

## 准备

```bash
git clone --recurse-submodules https://github.com/DrRyanHuang/WarpShift.git
cd WarpShift
```

创建 `models.local.json`，填入两份同结构、同 tokenizer 的 BF16 模型路径：

```json
{
  "Teacher1": "/models/teacher1",
  "Teacher2": "/models/teacher2"
}
```

## 启动

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh \
  --models models.local.json --source delta -- --tp-size 1
```

多卡时同步调整 `CUDA_VISIBLE_DEVICES` 和 `--tp-size`。
`--source` 可选 `delta`（差分）、`pinned`（锁页内存）、`pageable`（普通内存）、`disk`（文件加载）；更换方式需重启服务。

## 切换

在另一终端进入仓库目录，服务就绪后执行：

```bash
python -m warpshift prepare
python -m warpshift switch --model Teacher2
python -m warpshift status
```

切回时将 `Teacher2` 改为 `Teacher1`。服务重启后重新执行 `prepare`。

## 推理

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -H 'X-WarpShift-Model: Teacher2' \
  -d '{"text":"你好","sampling_params":{"max_new_tokens":64}}'
```

请求头中的模型名应与当前模型一致。
