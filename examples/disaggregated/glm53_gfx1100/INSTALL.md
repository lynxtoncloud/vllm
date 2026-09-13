# gfx1100 GLM-5.3-Flash-BF16 四节点 PD 统一安装手册

本手册用于分支 `gfx1100/glm53-bf16-pd`，所有服务器命令在 **Linux 宿主机 root Shell**
执行。Mac 只负责 SSH 登录和维护源码，不构建 ROCm 镜像。当前为实验适配：BF16、
eager、每引擎并发 1、最大长度 8192；GPU、模型输出和真实 NIXL 传输仍须验收。

## 1. 节点及目录标准

| 角色 | 宿主机 IP | 职责 | API | 分布式主节点 | NIXL 侧信道 |
| --- | --- | --- | --- | --- | --- |
| P0 | 10.5.10.36 | Prefill 主节点，构建与分发入口 | 8001 | 10.5.10.36:29501 | 5557 |
| P1 | 10.5.10.3 | Prefill 从节点，headless | 无 | 10.5.10.36:29501 | 5557 |
| D0 | 10.5.10.55 | Decode 主节点 | 8002 | 10.5.10.55:29502 | 5657 |
| D1 | 10.5.10.56 | Decode 从节点，headless | 无 | 10.5.10.55:29502 | 5657 |

每台 8 张 gfx1100；P0+P1 为一个 TP16/EP 引擎，D0+D1 为另一个 TP16/EP 引擎。
四台均从 GitHub 拉取源码，使用同一提交、同一模型和同一构建镜像。

| 宿主机路径 | 用途 |
| --- | --- |
| `/data/vllm` | GitHub fork 的源码 checkout |
| `/data/models/zai-org/GLM-5.3-Flash-BF16` | 完整 BF16 checkpoint |
| `/data/cache/huggingface` | Hugging Face 缓存 |
| `/data/services/glm53` | 启动入口、发布提交、镜像信息 |
| `/data/images` | 镜像 tar 和校验文件 |
| `/data/logs/glm53-build` | 构建日志 |
| `/data/logs/glm53-src-p0` 等 | 各节点服务日志 |
| `/data/logs/glm53-proxy` | PD 测试代理日志 |

容器中模型为 `/models/GLM-5.3-Flash-BF16`，节点日志为 `/app/logs`。
Docker 自身的数据目录保持已有配置；本手册不迁移已有容器和 Docker 数据。

## 2. 经 P0 登录其他节点

### 2.1 两段 SSH：沿用已经验证可用的密钥

先从管理终端登录 `.36`。以下按 `.36` 也使用 21985、root 和该密钥举例；
如果 `.36` 的外部入口地址或凭据不同，第一条使用你已经能登录 `.36` 的命令。

```bash
ssh -p 21985 -i ~/.ssh/rebond.pem root@10.5.10.36
```

然后在 **P0 宿主机**分别进入其他节点（密钥位于 P0 的 `/root/.ssh/rebond.pem`）：

```bash
ssh -p 21985 -i ~/.ssh/rebond.pem root@10.5.10.3
```

```bash
ssh -p 21985 -i ~/.ssh/rebond.pem root@10.5.10.55
```

```bash
ssh -p 21985 -i ~/.ssh/rebond.pem root@10.5.10.56
```

每个会话先执行 `hostname; ip -br -4 addr` 确认所在机器。退出内层 SSH 只回到 P0。
首次连接核对主机指纹，不关闭 host key 校验。私钥不放进 `/data`、源码或镜像。

### 2.2 可选：管理电脑直接使用 ProxyJump

仅当管理电脑上的密钥也能认证目标节点时，在其 `~/.ssh/config` 中合并以下配置，
不要覆盖已有配置。`glm-p0` 的 HostName 可换为实际可达的 `.36` 外部地址。
ProxyJump 在管理电脑使用目标密钥，不会自动使用 `.36` 上的私钥。
参见 [OpenSSH ProxyJump 文档](https://man.openbsd.org/ssh_config#ProxyJump)。

```sshconfig
Host glm-p0
    HostName 10.5.10.36
    User root
    Port 21985
    IdentityFile ~/.ssh/rebond.pem
    IdentitiesOnly yes

Host glm-p1 glm-d0 glm-d1
    User root
    Port 21985
    IdentityFile ~/.ssh/rebond.pem
    IdentitiesOnly yes
    ProxyJump glm-p0

Host glm-p1
    HostName 10.5.10.3
Host glm-d0
    HostName 10.5.10.55
Host glm-d1
    HostName 10.5.10.56
```

之后可直接运行 `ssh glm-p1`、`ssh glm-d0`、`ssh glm-d1`。不需要开启 agent forwarding。

## 3. 四台宿主机准备

以下在 **每台宿主机**执行。本节 apt 命令适用于 Ubuntu；先确认系统和磁盘。

```bash
cat /etc/os-release
uname -r
ip -br -4 addr
df -h /data /var/lib
ls -l /dev/kfd /dev/dri/renderD*
cat /sys/module/amdgpu/version
```

GPU 驱动必须已工作。宿主机 ROCm 与容器用户态版本分别记录，不能用宿主机版本
推断容器版本；本手册不自动升级或替换正在使用的驱动。
磁盘需同时容纳完整模型、镜像、tar 及构建缓存，按实际文件大小留出空间。

```bash
apt-get update
apt-get install -y ca-certificates curl git rsync tmux iproute2 netcat-openbsd
mkdir -p /data/models/zai-org/GLM-5.3-Flash-BF16
mkdir -p /data/cache/huggingface /data/services/glm53 /data/images
mkdir -p /data/logs/glm53-build /data/logs/glm53-proxy
```

### 3.1 Docker 29.8.0

先检查；已有正常工作的 Docker 29.8.0 和 buildx 时跳过安装。

```bash
docker version
docker buildx version
docker info --format '{{.DockerRootDir}}'
```

未安装的 Ubuntu 节点按下列步骤配置官方 apt 源并选择仓库提供的 **29.8.0** 包版本。
若存在 docker.io、podman-docker 或独立 containerd/runc，先检查其占用服务，再按
[Docker Ubuntu 安装说明](https://docs.docker.com/engine/install/ubuntu/)处理冲突包。
这里不自动卸载已有运行环境。

```bash
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
printf 'Types: deb\nURIs: https://download.docker.com/linux/ubuntu\nSuites: %s\nComponents: stable\nArchitectures: %s\nSigned-By: /etc/apt/keyrings/docker.asc\n' \
  "${UBUNTU_CODENAME:-$VERSION_CODENAME}" "$(dpkg --print-architecture)" \
  > /etc/apt/sources.list.d/docker.sources
apt-get update
apt-cache madison docker-ce
```

检查下面打印的版本非空且为 29.8.0；仓库没有该版本时停止，不自动换成最新版本。

```bash
DOCKER_VERSION=$(apt-cache madison docker-ce | awk '$3 ~ /^5:29\.8\.0-/ {print $3; exit}')
printf 'Docker package: %s\n' "$DOCKER_VERSION"
```

```bash
test -n "$DOCKER_VERSION" && apt-get install -y \
  "docker-ce=$DOCKER_VERSION" "docker-ce-cli=$DOCKER_VERSION" \
  containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
docker version
docker buildx version
docker run --rm hello-world
```

Engine 和 CLI 固定 29.8.0；其余组件实际版本可用 `dpkg-query -W` 留档。
后续安全更新由维护者安排，不执行无差别升级或降级。

### 3.2 网卡和 UFW

P0 已确认 `bond0`；其他节点以实际承载各自 `10.5.10.*` 地址的接口为准。
容器采用 `--network host`，无需 `-p`。先运行 `ufw status verbose`；
如果已启用 UFW，在每台宿主机执行下面四条，允许四个指定内网来源的节点通信。
需要覆盖 RCCL/Gloo/UCX 动态端口，不能只放行 rendezvous 端口。

```bash
ufw allow from 10.5.10.36 to any comment 'GLM PD P0 internal'
ufw allow from 10.5.10.3 to any comment 'GLM PD P1 internal'
ufw allow from 10.5.10.55 to any comment 'GLM PD D0 internal'
ufw allow from 10.5.10.56 to any comment 'GLM PD D1 internal'
ufw status numbered
```

保留现有 21985 SSH 放行规则。不要为此重置 UFW 或在 SSH 会话中直接启用未配置的防火墙。
这些规则允许指定来源访问本机所有端口，只用于这四台可信内网节点；交换机/云安全组也需允许相应双向流量。
分布式通信默认不提供安全隔离，见[仓库安全说明](../../../docs/usage/security.md)。

## 4. 源码和发布版本

### 4.1 P0 从 GitHub 拉取并固定提交

首次安装，在 **P0** 执行；`/data/vllm` 已存在时使用后面的更新步骤，不重复 clone。

```bash
git clone --branch gfx1100/glm53-bf16-pd https://github.com/lynxtoncloud/vllm.git /data/vllm
cd /data/vllm
git status --short
git rev-parse HEAD > /data/services/glm53/source.commit
cat /data/services/glm53/source.commit
```

已有 checkout 更新时，先确认 `git status --short` 为空：

```bash
cd /data/vllm
git fetch origin
git switch gfx1100/glm53-bf16-pd
git pull --ff-only origin gfx1100/glm53-bf16-pd
git rev-parse HEAD > /data/services/glm53/source.commit
```

发布期间不再移动这个文件；如要换提交，应重新构建、分发及验收。

### 4.2 P0 分发提交号，其他三台各自从 GitHub 拉取

以下在 **P0** 执行（其他节点已完成第 3 节建目录）：

```bash
scp -P 21985 -i ~/.ssh/rebond.pem /data/services/glm53/source.commit root@10.5.10.3:/data/services/glm53/
scp -P 21985 -i ~/.ssh/rebond.pem /data/services/glm53/source.commit root@10.5.10.55:/data/services/glm53/
scp -P 21985 -i ~/.ssh/rebond.pem /data/services/glm53/source.commit root@10.5.10.56:/data/services/glm53/
```

以下在 **P1、D0、D1 各执行一次**；首次 clone，已有仓库则跳过第一条。

```bash
git clone --branch gfx1100/glm53-bf16-pd https://github.com/lynxtoncloud/vllm.git /data/vllm
cd /data/vllm
git status --short
git fetch origin
git switch --detach "$(cat /data/services/glm53/source.commit)"
test "$(git rev-parse HEAD)" = "$(cat /data/services/glm53/source.commit)"
```

存在本地改动时先处理，不使用 `reset --hard`。部署节点 detached HEAD 是固定发布版本的预期状态。

## 5. 模型分发与校验

P0 的完整模型必须位于 `/data/models/zai-org/GLM-5.3-Flash-BF16`，包含 config、tokenizer、
全部权重分片及 index。若原模型在其他目录，先确认路径再移动或链接，不重复下载覆盖。
以下在 **P0** 运行；rsync 可以重跑续传，不删除目标已有文件。

```bash
rsync -aL --partial --info=progress2 -e 'ssh -p 21985 -i /root/.ssh/rebond.pem' \
  /data/models/zai-org/GLM-5.3-Flash-BF16/ root@10.5.10.3:/data/models/zai-org/GLM-5.3-Flash-BF16/
```

```bash
rsync -aL --partial --info=progress2 -e 'ssh -p 21985 -i /root/.ssh/rebond.pem' \
  /data/models/zai-org/GLM-5.3-Flash-BF16/ root@10.5.10.55:/data/models/zai-org/GLM-5.3-Flash-BF16/
```

```bash
rsync -aL --partial --info=progress2 -e 'ssh -p 21985 -i /root/.ssh/rebond.pem' \
  /data/models/zai-org/GLM-5.3-Flash-BF16/ root@10.5.10.56:/data/models/zai-org/GLM-5.3-Flash-BF16/
```

`-L` 将模型快照符号链接转换为真实文件，避免链接指向其他机器不存在的 HF cache。
P0 生成完整文件校验表，耗时取决于权重大小和磁盘速度：

```bash
cd /data/models/zai-org/GLM-5.3-Flash-BF16
find -L . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > /data/services/glm53/model.sha256
scp -P 21985 -i ~/.ssh/rebond.pem /data/services/glm53/model.sha256 root@10.5.10.3:/data/services/glm53/
scp -P 21985 -i ~/.ssh/rebond.pem /data/services/glm53/model.sha256 root@10.5.10.55:/data/services/glm53/
scp -P 21985 -i ~/.ssh/rebond.pem /data/services/glm53/model.sha256 root@10.5.10.56:/data/services/glm53/
```

在 **每台宿主机**执行，所有文件必须为 `OK`：

```bash
cd /data/models/zai-org/GLM-5.3-Flash-BF16
sha256sum -c /data/services/glm53/model.sha256
```

## 6. P0 构建源码镜像

仅在 **P0** 构建。此流程会访问镜像仓库、GitHub 及构建依赖源；SSH 跳板不自动为其他
进程提供互联网代理。基础镜像中的 PyTorch/ROCm/Triton 也必须支持 gfx1100。
不使用旧 `glm53-flash` 镜像覆盖 Python 文件的方式部署本分支。
先执行 `tmux new -s glm53-build`，在新会话中继续以下命令。

```bash
cd /data/vllm
export COMMIT=$(cat /data/services/glm53/source.commit)
test "$(git rev-parse HEAD)" = "$COMMIT"
git status --short
export GFX1100_IMAGE="local/vllm:glm53-gfx1100-$COMMIT"
docker pull rocm/vllm-dev:base
BASE_IMAGE=$(docker image inspect rocm/vllm-dev:base --format '{{index .RepoDigests 0}}')
printf '%s\n' "$BASE_IMAGE" > /data/services/glm53/base-image.txt
```

检查工作区干净、基础镜像 digest 非空，然后打印构建参数：

```bash
docker buildx bake -f docker/docker-bake-rocm.hcl -f docker/gfx1100-glm53.hcl \
  glm53-gfx1100 --set "glm53-gfx1100.args.BASE_IMAGE=$BASE_IMAGE" \
  --set '*.args.max_jobs=8' --print
```

在 tmux 会话内执行构建并保留日志。重新进入 Shell 时需重新设置上一段的环境变量。

```bash
set -o pipefail
docker buildx bake -f docker/docker-bake-rocm.hcl -f docker/gfx1100-glm53.hcl \
  glm53-gfx1100 --set "glm53-gfx1100.args.BASE_IMAGE=$BASE_IMAGE" \
  --set '*.args.max_jobs=8' 2>&1 | tee "/data/logs/glm53-build/$COMMIT.log"
```

构建失败就停在本步骤，保留首个错误；不要运行同名的旧镜像。成功后记录并导出：

```bash
docker image inspect "$GFX1100_IMAGE" --format '{{.Id}}' > /data/services/glm53/image.id
printf '%s\n' "$GFX1100_IMAGE" > /data/services/glm53/image.tag
docker image inspect "$GFX1100_IMAGE" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
docker save "$GFX1100_IMAGE" -o /data/images/glm53-gfx1100.tar
cd /data/images
sha256sum glm53-gfx1100.tar > glm53-gfx1100.tar.sha256
```

revision 必须等于 `source.commit`。本 overlay 编译 vLLM gfx1100 目标，不能补救
基础镜像中不支持 gfx1100 的 PyTorch 库；下一节的真实 GPU 检查仍是必要步骤。

## 7. 分发镜像与安装启动入口

在 **P0** 逐台执行；`/data/services/glm53/` 中只放本部署的发布资料，不放私钥。

```bash
rsync -a --partial --info=progress2 -e 'ssh -p 21985 -i /root/.ssh/rebond.pem' /data/images/glm53-gfx1100.tar* root@10.5.10.3:/data/images/
rsync -a -e 'ssh -p 21985 -i /root/.ssh/rebond.pem' /data/services/glm53/ root@10.5.10.3:/data/services/glm53/
```

```bash
rsync -a --partial --info=progress2 -e 'ssh -p 21985 -i /root/.ssh/rebond.pem' /data/images/glm53-gfx1100.tar* root@10.5.10.55:/data/images/
rsync -a -e 'ssh -p 21985 -i /root/.ssh/rebond.pem' /data/services/glm53/ root@10.5.10.55:/data/services/glm53/
```

```bash
rsync -a --partial --info=progress2 -e 'ssh -p 21985 -i /root/.ssh/rebond.pem' /data/images/glm53-gfx1100.tar* root@10.5.10.56:/data/images/
rsync -a -e 'ssh -p 21985 -i /root/.ssh/rebond.pem' /data/services/glm53/ root@10.5.10.56:/data/services/glm53/
```

在 **P1、D0、D1** 校验并加载：

```bash
cd /data/images
sha256sum -c glm53-gfx1100.tar.sha256 && docker load -i glm53-gfx1100.tar
```

在 **四台宿主机**核对源码和镜像，并复制对应版本的启动入口：

```bash
cd /data/vllm
export IMAGE=$(cat /data/services/glm53/image.tag)
test "$(git rev-parse HEAD)" = "$(cat /data/services/glm53/source.commit)"
test "$(docker image inspect "$IMAGE" --format '{{.Id}}')" = "$(cat /data/services/glm53/image.id)"
test "$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')" = "$(cat /data/services/glm53/source.commit)"
install -m 0755 examples/disaggregated/glm53_gfx1100/launch-node.sh /data/services/glm53/launch-node.sh
```

每条 `test` 必须退出码为 0；失败时停止，不能只看最后一条 install 成功。

## 8. GPU 和回退函数检查

服务尚未启动时，在 **四台宿主机**运行。先确认能看到八卡且架构均为 gfx1100。
检查容器临时虚拟环境复用镜像依赖，不改动镜像中的 PyTorch：

```bash
export IMAGE=$(cat /data/services/glm53/image.tag)
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --entrypoint /bin/bash "$IMAGE" -lc \
  'uv venv --system-site-packages /tmp/.venv && /tmp/.venv/bin/python -c '\''import torch; print("Torch:", torch.__version__, "HIP:", torch.version.hip); print([torch.cuda.get_device_properties(i).gcnArchName for i in range(torch.cuda.device_count())]); assert torch.cuda.device_count() == 8'\'''
```

然后执行回退函数 GPU 测试；测试是启动前的必要检查，不能替代完整模型验收。
下面将测试文件单独挂载，避免宿主源码遮蔽镜像中的已编译包。

```bash
set -o pipefail
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host \
  -e GFX1100_TEST_DEVICE=cuda -e VLLM_ROCM_USE_AITER=0 \
  --mount type=bind,src=/data/vllm/tests/standalone_tests,dst=/checks,readonly \
  --entrypoint /bin/bash "$IMAGE" -lc \
  'uv venv --system-site-packages /tmp/.venv && uv pip install --python /tmp/.venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple pytest && /tmp/.venv/bin/python -m pytest --confcutdir=/checks /checks/test_rocm_gfx1100.py -q' \
  2>&1 | tee /data/logs/glm53-gpu-check.log
```

这一文件使用 cuda:0；八卡枚举不等于八卡算子测试。需要逐卡验收时分别设置
容器的 `HIP_VISIBLE_DEVICES=0` 到 `7` 重跑。失败时停止部署并保存栈。
更多 MoE、稀疏 MLA 和模型验证见 [README 验收顺序](README.md#验收顺序)。

## 9. 四节点手工启动

每个节点独立保留一个 tmux 会话：先运行 `tmux new -s glm53`，再粘贴该角色的命令。
`Ctrl-b d` 离开会话而保留服务；`tmux attach -t glm53` 返回。脚本前台运行并 tee 日志。
先检查 `docker ps` 及 GPU 显存，确保本轮没有遗留 Worker；不批量停止无关容器。

P0 启动后立即启动 P1；D0 启动后立即启动 D1。主节点会等待同组从节点，不能等主节点
health 成功才启动从节点。下面省略 `IFACE_NAME`，脚本按该机 IP 自动识别接口。

### P0：10.5.10.36

```bash
export IMAGE=$(cat /data/services/glm53/image.tag)
bash /data/services/glm53/launch-node.sh p0 --dry-run
bash /data/services/glm53/launch-node.sh p0
```

### P1：10.5.10.3

```bash
export IMAGE=$(cat /data/services/glm53/image.tag)
bash /data/services/glm53/launch-node.sh p1 --dry-run
bash /data/services/glm53/launch-node.sh p1
```

### D0：10.5.10.55

```bash
export IMAGE=$(cat /data/services/glm53/image.tag)
bash /data/services/glm53/launch-node.sh d0 --dry-run
bash /data/services/glm53/launch-node.sh d0
```

### D1：10.5.10.56

```bash
export IMAGE=$(cat /data/services/glm53/image.tag)
bash /data/services/glm53/launch-node.sh d1 --dry-run
bash /data/services/glm53/launch-node.sh d1
```

默认模型及缓存目录已统一，无需额外 export。`--dry-run` 中检查本机 IP、master、rank、
producer/consumer、网卡和模型路径。不要恢复旧命令的 max-num-seqs=512、百万上下文、
AITER MoE、CUDA graph 或旧的 GLM_GFX1100_* 补丁开关。

## 10. 就绪、PD 请求和日志

在 **P0 的另一终端**检查两个主节点：

```bash
curl --noproxy '*' -i --max-time 10 http://10.5.10.36:8001/health
curl --noproxy '*' -i --max-time 10 http://10.5.10.55:8002/health
```

两者都应 HTTP 200。P1、D1 是 headless，无需也不应等待其 API 端口监听。
`Model loading took` 只说明权重加载阶段结束；`Poller timed out` 本身不能判定死锁。

在 **P0** 的新 tmux 会话内启动仓库自带的 NIXL 测试代理，绑定回环地址：

```bash
export IMAGE=$(cat /data/services/glm53/image.tag)
set -o pipefail
docker run --rm --name glm53-pd-proxy --network host \
  --mount type=bind,src=/data/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py,dst=/proxy.py,readonly \
  --entrypoint /bin/bash "$IMAGE" -lc \
  'uv venv --system-site-packages /tmp/.venv && /tmp/.venv/bin/python /proxy.py --host 127.0.0.1 --port 8000 --prefiller-hosts 10.5.10.36 --prefiller-ports 8001 --decoder-hosts 10.5.10.55 --decoder-ports 8002' \
  2>&1 | tee /data/logs/glm53-proxy/latest.log
```

P0 另一个终端发出短请求：

```bash
curl --noproxy '*' --fail-with-body --max-time 300 \
  http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/GLM-5.3-Flash-BF16","prompt":"The capital of France is","temperature":0,"max_tokens":32,"stream":false}'
```

这只是 smoke test。还需确认 P 产生 `kv_transfer_params`、D 成功加载远端 KV，
并用 NIXL 日志/指标排除 D 本地重算；请求返回文本不是 PD 已验证的充分条件。
对比同一 checkpoint 的非 PD 基线输出及 logprob，再扩大输入长度和测试图像/视频。

每台在自己的日志目录排错，例如 P0：

```bash
tail -n 100 /data/logs/glm53-src-p0/latest.log
grep -n -i -m 8 -B 8 -A 35 -E 'Traceback|ERROR|Fatal Python|Segmentation|out of memory' /data/logs/glm53-src-p0/latest.log
docker inspect glm53-src-p0 --format 'Status={{.State.Status}} Exit={{.State.ExitCode}} OOM={{.State.OOMKilled}}'
docker exec glm53-src-p0 ps -eo pid,ppid,stat,pcpu,etime,args
```

P1、D0、D1 把 `p0` 换成相应角色。检查 rendezvous 要在服务启动后执行，
未监听前 `nc -vz` 失败不等于防火墙未放通：

```bash
nc -vz -w 3 10.5.10.36 29501
nc -vz -w 3 10.5.10.55 29502
```

## 11. 停止、更新和回滚

先停止代理接收新请求，再在四台各自停止本部署的容器。例如 P0：

```bash
docker stop --time 30 glm53-src-p0
docker inspect glm53-src-p0 --format '{{.State.Status}}'
```

确认停止后才移除该容器壳以便重用名字；日志、源码、模型均是宿主机文件，会保留。

```bash
docker rm glm53-src-p0
```

其他角色使用自己的容器名。更新前把 `/data/services/glm53` 的发布资料另存到带提交号的目录，
保留旧镜像 tag/ID 和对应源码提交。然后重新执行源码固定、构建、分发、校验与成组启动。
回滚时四台恢复同一套 `source.commit`、`image.tag`、`image.id`、启动入口和模型校验表；
不要让一个 TP 组混用新旧镜像。部署节点不直接 merge upstream；在开发分支合并并验证后发布。

## 12. 验收记录

每次发布保存：四台源码 SHA、镜像 ID、模型校验结果、Docker/驱动/容器 ROCm/PyTorch
版本、GPU 测试日志、启动日志、NIXL 传输证据以及与非 PD 基线的比较。
本手册在 Mac 编写并进行静态检查，未代替你登录服务器执行安装，也未宣称四节点部署已通过。
