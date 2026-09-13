# GLM-5.3-Flash-BF16 on gfx1100: experimental PD branch

本目录随适配分支管理代码、构建和部署。当前范围是 gfx1100、BF16 权重和 KV cache、
eager、最大长度 8192、并发 1、无推测解码。不是百万上下文或生产性能承诺。
CPU 测试不能证明 GPU/PD 正确性；完成下方验收后才能扩大范围。

## 分支与上游

`main` 只同步上游；`gfx1100/glm53-bf16-pd` 保存适配。`origin` 是自己的 fork，
`upstream` 是 `https://github.com/vllm-project/vllm.git`。
确认适配变更已提交、工作区干净后，使用 merge 保留适配历史：

```bash
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
git switch gfx1100/glm53-bf16-pd
git merge main
```

如果 `--ff-only` 失败，检查分歧，不对 main 强制 reset。每次合并后重跑数值测试和
GPU/PD 验收，再构建新镜像。提交应遵循仓库 AGENTS.md 的署名及 AI 辅助说明。

## 实现边界

一个开关 `VLLM_ROCM_GFX1100_GLM53=1` 控制适配；未启用时保留上游行为。
平台配置检查模型、dtype、eager、长度、并发和 KV 类型，选择 native RMSNorm 与 Triton MoE。
硬件分派还检查实际 gfx1100，不伪装 gfx950，也不宣称 AITER 支持 gfx1100。

- `vllm/utils/rocm_gfx1100.py`：EP 路由补齐、过滤非本地专家的 FP32 求和、相对索引 top-k、BF16 MLA 缓存。
- 索引器使用已有 ROCm kpool 压缩和 logits 路径，top-k 改走 Torch 回退。
- MoE 权重关闭额外 padding；普通线性层使用 PyTorch GEMM；MHC 保留 Torch/Triton 回退。
- FP8 分组量化使用已有 Triton fallback，避开发生过段错误的编译算子。
  BF16 模型的索引器仍使用 FP8 数据，这不等于开启 FP8 模型权重量化。
- 当前上游已有 rope-free 稀疏 MLA 路径，入口也已取消 gfx950 限制；无需旧镜像的导入补丁。
- rope-free MLA 的 KV 元数据初始化不加载 AITER；索引就绪接口在同一 stream 下为无操作。
- gfx1100 稀疏 MLA 保留 2048 个历史 token 和最多 3 个未满池尾部 token，
  压紧有效索引后使用最多 2051 宽度的 Triton 路径。负索引和越界页保持无效，
  不会变成对缓存第 0 格的注意力。该修改不扩展 FP8 权重支持。
- NIXL 独立 kpool 尾部按有效数据长度和实际块间距分别构建传输区域，
  不把填充算进数据长度，也不丢弃尾部。该状态按完整副本传输。
  NIXL connector 协议版本为 12，P/D 四台必须一起更新，拒绝混用旧版本。

没有移植旧 FP8 权重实验的 GEMM 补丁。新增 native/HIP 算子如果再次失败，要依据新栈定位。

## 构建一次，分发同一镜像

在 Linux ROCm 构建机的这个仓库根目录执行。Docker build 使用本地源码，
显式编译 gfx1100，复用上游 Dockerfile 的 native 编译缓存和 NIXL/UCX 构建阶段。
不把最新 Python 文件覆盖到旧 `glm53-flash` 镜像的 `.so` 上；两者接口已变化。

```bash
export COMMIT=$(git rev-parse HEAD)
export GFX1100_IMAGE=local/vllm:glm53-gfx1100
docker buildx bake -f docker/docker-bake-rocm.hcl -f docker/gfx1100-glm53.hcl glm53-gfx1100 --print
docker buildx bake -f docker/docker-bake-rocm.hcl -f docker/gfx1100-glm53.hcl glm53-gfx1100 --set '*.args.max_jobs=8'
```

基础依赖仍由上游 `docker/Dockerfile.rocm` / `Dockerfile.rocm_base` 管理。
确认其 PyTorch/ROCm/Triton/NIXL 版本与当前 checkout 匹配，PyTorch 包含 gfx1100。
可通过 `--set 'glm53-gfx1100.args.BASE_IMAGE=你的基础镜像@sha256:...'` 固定基础镜像。
上游需要的 NIC 专用依赖可覆盖 `NIC_BACKEND`；本配置不安装 AINIC/Broadcom 专用驱动。
构建后记录实际镜像 ID/digest、源码提交和依赖版本；禁止把未提交工作区标成可复现发布版本。

```bash
docker image inspect "$GFX1100_IMAGE" --format '{{.Id}}'
docker save "$GFX1100_IMAGE" -o /data/services/glm53-gfx1100.tar
```

将 tar 分发给其余三台并执行 `docker load -i /data/services/glm53-gfx1100.tar`。
检查四台镜像 ID 一致。模型在每台宿主机上准备好；P/D 引擎必须使用同一 checkpoint。

## 四节点启动

| 角色 | IP | TP ranks | API | rendezvous | NIXL side channel |
| --- | --- | --- | --- | --- | --- |
| P0 | 10.5.10.36 | 0–7 | 8001 | .36:29501 | .36:5557 |
| P1 | 10.5.10.3 | 8–15 | headless | .36:29501 | 通过 P0 交换元数据 |
| D0 | 10.5.10.55 | 0–7 | 8002 | .55:29502 | .55:5657 |
| D1 | 10.5.10.56 | 8–15 | headless | .55:29502 | 通过 D0 交换元数据 |

这是一个跨两节点的 P 引擎和一个跨两节点的 D 引擎。各自 TP16/EP，节点各 8 张卡。
不是四个独立 API 实例。容器使用 host network，网卡名称必须使用各宿主机实际接口。
四台内网之间需允许分布式通信和 NIXL/UCX 所用的连接，包括动态端口，不能只放行 29501。
先确认上一轮的 Worker 已退出、每卡可用显存满足 0.90 的启动预算。脚本不自动清理其他容器。

在各自宿主机从仓库根目录执行对应一条：

```bash
bash examples/disaggregated/glm53_gfx1100/launch-node.sh p0
```

```bash
bash examples/disaggregated/glm53_gfx1100/launch-node.sh p1
```

```bash
bash examples/disaggregated/glm53_gfx1100/launch-node.sh d0
```

```bash
bash examples/disaggregated/glm53_gfx1100/launch-node.sh d1
```

P0 启动后立即启动 P1，D0 启动后立即启动 D1，不要等主节点就绪才启动从节点。
启动命令前可设置 `IMAGE`、`MODEL_DIR`、`IFACE_NAME`、`CONTAINER_NAME`、`LOG_DIR`。
默认模型路径 `/data/models/zai-org/GLM-5.3-Flash-BF16`；日志 `/data/logs/glm53-src-角色/latest.log`。
脚本可复制到 `/data/services`，运行不依赖源码工作目录。
它是前台 Docker + tee，保留终端或使用 tmux。退出后检查容器状态；重跑前自行检查并删除
已经停止的同名容器。停止某节点请用 `docker stop --time 30 glm53-src-p0` 等精确容器名。

查看完整命令而不启动：

```bash
IFACE_NAME=bond0 bash examples/disaggregated/glm53_gfx1100/launch-node.sh p0 --dry-run
```

## 验收顺序

以下步骤均在 Linux/ROCm 服务器执行；本地 Mac 只修改代码和做静态检查。

1. 在已按仓库贡献指南安装 vLLM 及测试依赖的虚拟环境中，先验证回退函数的语义：

```bash
.venv/bin/python -m pytest --confcutdir=tests/standalone_tests tests/standalone_tests/test_rocm_gfx1100.py -q
```

1. ROCm 构建/测试环境完成 vLLM 可编辑安装后，用同一测试文件验证 GPU：

```bash
GFX1100_TEST_DEVICE=cuda .venv/bin/python -m pytest --confcutdir=tests/standalone_tests tests/standalone_tests/test_rocm_gfx1100.py -q
VLLM_ROCM_GFX1100_GLM53=1 .venv/bin/python -m pytest tests/kernels/moe/test_moe_align_block_size.py -q
.venv/bin/python -m pytest tests/v1/attention/test_rocm_glm5next_sparse.py -q
```

1. P0 与 D0 `/health` 均返回 200 后，通过上游 NIXL 集成测试代理验证真实传输。
在可访问四节点的、已经安装 vLLM 及测试依赖的环境中，从本仓库执行：

```bash
.venv/bin/python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --host 127.0.0.1 --port 8000 --prefiller-hosts 10.5.10.36 --prefiller-ports 8001 --decoder-hosts 10.5.10.55 --decoder-ports 8002
```

这是测试代理，不是生产网关。通过其 `/v1/completions` 或 `/v1/chat/completions` 请求，
确认 P 端产生 `kv_transfer_params`、D 端成功加载远端缓存，没有本地重算掩盖失败。
检查 NIXL 传输日志/指标以及实际输出；请求成功不能单独证明缓存传输成功。

1. 对固定提示、temperature=0，对比同 checkpoint 非 PD 基线的输出/token/logprob。
覆盖短提示、跨 kpool=4 边界、跨 KV block 边界、接近 8192 的长度、重复提示和多轮对话。
图像/视频也需要独立验收，当前未关闭视觉输入，但没有完成视觉端到端验证。
保留日志、GPU/CPU 峰值内存、首 token 延迟及 decode 吞吐，再评估扩大长度和并发。

本地 macOS 没有 gfx1100，也没有四节点环境。因此 GPU 内核编译、模型评测和真实 NIXL
传输必须在服务器完成，不能将本分支的 CPU 检查当成部署验收。

## 当前验证记录

2026-09-14：本地 macOS 独立测试环境完成 `test_rocm_gfx1100.py` 的 CPU 数值和
配置测试（31 项），包括 EP 非本地专家过滤、top-k、BF16 缓存写入，以及短行和满历史
情况下的 kpool 尾部保留。该环境使用 CPU Torch 2.14.0；服务器为 ROCm Torch 2.13.0，
不能用本地通过代替 GPU 验证。新增 ROCm 回归用例覆盖 2051 宽度的分页映射和无效索引。
另外用隔离方法调用完成 metadata 初始化及 2176 对齐缓冲区到 2051 逻辑索引的
5 项 CPU 检查；未运行整个 attention 测试文件。修改文件的全部适用 pre-commit 钩子通过，
包括 Ruff、mypy 和 Markdown 检查。

链路审查包含 BF16 投影/MoE/MHC、KDA 卷积与递归状态、索引缓存和尾部缓存、
稀疏 MLA prefill/decode、NIXL 分组注册及 P/D 状态交接。与固定的 NIXL 1.4.0 源码
核对了 connector 调用的 16 个 Python 方法，均存在；这只证明方法存在，不证明传输成功。

服务器已报告四节点模块加载和 UCX agent 初始化通过，尚未完成四节点模型推理或
真实 P/D 缓存传输验收。RDMA 显存注册仍有已知失败，初次部署沿用 TCP/ROCm transport。
主机原生安装、日志和逐条启动命令由外层项目的 `deployment/` 文档管理；
本目录的 Docker 命令是另外一种构建方式，不要混用容器与原生环境。

后续服务器日志暴露了此前审查遗漏的独立带填充尾部视图：
`shape=(1145,2,4,128)`、`stride=(6336,512,128,1)`、BF16，
每块有效数据 2048 字节、块间距 12672 字节。已补充这一精确几何的注册回归，
以及 P/D 块间距不同情况下的描述符拷贝检查，验证 padding 和其他块不被覆盖。
这些是 CPU 描述符/内存模拟检查，不是实际 UCX/GPU 网络传输验收。

服务器随后在 `compile_or_warm_up_model` 报告
`'function' object has no attribute 'compile'`。GLM decoder 构造时此前无条件按
`enable_jit_warmup` 注册 TileLang MHC，而 gfx1100 运行时已选择 Torch/Triton MHC。
ROCm 的延迟 `tilelang_jit` 装饰器返回普通函数，没有 `.compile()`，因此未使用的
TileLang 预热仍会使启动失败。现将这四组注册与运行时 `HAS_TILELANG_MHC` 条件统一；
KDA 的 Triton 状态散布预热保留，也不关闭全局 JIT 预热。

新增测试隔离执行模型源码的注册块，覆盖 TileLang 可用性与 JIT 预热开关的四种组合。
旧条件在“不使用 TileLang 但开启预热”时失败；修复后独立 CPU 测试共 35 项通过。
这未执行真实 TileLang/GPU 编译或多节点启动，服务器仍需按原命令重启验收。
