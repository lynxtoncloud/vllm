# 四节点后台全量测试（在P1管理，不用systemd）

拓扑是P0+P1一个TP16 Prefill引擎，D0+D1一个TP16 Decode引擎。
管理脚本在P1运行，通过密钥SSH执行远端启停；测试客户端也在P1。
每个任务独立session、stdin关闭、stdout/stderr落盘并忽略SIGHUP。
关闭终端不会停止任务；主机重启不会自动恢复，需手工start。

## 一次准备

四台使用同一个 `gfx1100/glm5next-int8` 分支、提交和模型文件。
现有服务若由旧命令启动，先用原方式停止并核实显存释放；本管理器不接管、
不批量杀死旧进程。P1需有 `/root/.ssh/rebond.pem`，且能登录另外三台。
以下命令由用户在P1执行，不新建clone/worktree：

```bash
cd /data/vllm
git switch gfx1100/glm5next-int8
git pull --ff-only
for host in 10.5.10.36 10.5.10.55 10.5.10.56; do
  ssh -p 21985 -i ~/.ssh/rebond.pem "root@$host" \
    'cd /data/vllm && git switch gfx1100/glm5next-int8 && git pull --ff-only && git log -1 --oneline' || break
done
git log -1 --oneline
apt-get update && apt-get install -y ffmpeg
.venv/bin/python -c 'import httpx, PIL, pybase64, regex, transformers; print("client dependencies OK")'
ffmpeg -hide_banner -encoders 2>/dev/null | grep libx264
```

`ffmpeg/libx264`用于P1生成视频；监控只需Python标准库、Linux/proc/sysfs。
可在各节点安装 `ethtool rdma-core ipmitool` 丰富网卡/BMC信息，缺失命令会
记录为不可用，不把缺失功耗当0。BMC不支持DCMI时，需另行记录机柜电表功耗。
核对各节点时钟同步，否则跨节点窗口对齐不可信。

配置在 `assessment_config.json`。默认1M上下文、batch512、eager，关闭诊断，
启用已验证的两项DMA-BUF设置，保留INT8大workspace指针修复。
不设置额外 `max-num-seqs`。可修改配置中的批处理/graph参数，但每次配置或
代码变化必须使用新的RUN_ID，不能与原结果混算。
配置中的LD_LIBRARY_PATH会覆盖继承值；如有额外依赖路径请明确填入配置。
代理绑定P0内网地址10.5.10.36:8000，供P1访问。

## 启动、状态、停止

所有命令都在P1执行。每次重新登录后，重新设置下面三个变量即可：

```bash
cd /data/vllm
export GLM_ASSESS_RUN=full-20260915-a
export GLM_ASSESS_JOB=examples/disaggregated/glm5next_int8/assessment_jobs.py
export GLM_ASSESS_ROOT=/data/logs/glm53-assessment/$GLM_ASSESS_RUN

# 四节点采样和引擎、P0代理均后台运行，命令返回不代表模型已加载完成。
.venv/bin/python "$GLM_ASSESS_JOB" start monitors --run-id "$GLM_ASSESS_RUN"
.venv/bin/python "$GLM_ASSESS_JOB" start engines --run-id "$GLM_ASSESS_RUN"
.venv/bin/python "$GLM_ASSESS_JOB" start proxy --run-id "$GLM_ASSESS_RUN"
.venv/bin/python "$GLM_ASSESS_JOB" status all --run-id "$GLM_ASSESS_RUN"

# 两引擎与代理均就绪后再开始；/healthcheck只代表代理进程存活。
curl --noproxy '*' --fail --max-time 30 http://10.5.10.36:8001/health
curl --noproxy '*' --fail --max-time 30 http://10.5.10.55:8002/health
curl --noproxy '*' --fail --max-time 30 http://10.5.10.36:8000/healthcheck
.venv/bin/python "$GLM_ASSESS_JOB" start test --run-id "$GLM_ASSESS_RUN"
```

使用同一RUN_ID重复 `start test` 会跳过已通过阶段，失败/未完成阶段重测，
旧attempt原始结果保留。正在运行时重复start会拒绝创建副本。
引擎身份与代码配置应保持一致；调整过服务启动参数就新建RUN_ID。
暂停或异常后续跑前，确认P/D的Running和Waiting归零；HTTP断开不保证旧请求
已经在后端撤销。如仍有残留负载，手工停止并重启本轮引擎后再续跑，避免混入旧流量。

```bash
# 查看当前阶段、进程和退出码。
.venv/bin/python "$GLM_ASSESS_JOB" status all --run-id "$GLM_ASSESS_RUN"
tail -n 80 -f "$GLM_ASSESS_ROOT/p1/test/console.log"

# 暂停压测，保留服务与监控。
.venv/bin/python "$GLM_ASSESS_JOB" stop test --run-id "$GLM_ASSESS_RUN"

# 测试完成后停监控，方便收集稳定快照；引擎可继续服务。
.venv/bin/python "$GLM_ASSESS_JOB" stop monitors --run-id "$GLM_ASSESS_RUN"

# 需要整组停止时：测试→代理→引擎→监控。
.venv/bin/python "$GLM_ASSESS_JOB" stop all --run-id "$GLM_ASSESS_RUN"
```

stop核对PID、进程启动tick、boot ID、session后，仅向本脚本启动的进程组发
TERM，再清理未退出的同组进程。不会使用pkill或重置GPU。
如果停止期间SSH失联，会报告失败节点；恢复网络后重试stop/status。

## 测试矩阵及失败处理

- 文本：约2K、32K、128K、512K、1M，头/中/尾三处随机密钥检索。
  1M实际输入1047551 tokens，功能输出预算1024，不超过1048576总长度。
- 视觉：448单图、1344单图、多图顺序、随机编码OCR、柱状图比较。
- 视频：JPEG帧序列、MP4颜色时序；图片+视频混合；32K文本+图片。
- 混合负载：短文本、单图、MP4按请求轮换，检验共享调度。
- 先做每个场景3次已知答案功能验证，再做通过场景的性能测试。
- 每个性能场景默认并发1/2/4/8/16/32，每档max(16,4×并发)请求，2次热身，
  固定256输出、ignore_eos、独立nonce/图像像素，避免简单重复缓存。
  未限制引擎并发；这些数值仅是测量档位。
- 准备token/媒体在计时外；计时包含HTTP提交、P排队与预填充、KV转移、D生成。
  客户端每请求的TTFT从获得并发槽位后算起，不包含客户端等待槽位的时间。
- 文本性能使用已知答案检索语料，与此前random dataset并非同一个数据集，
  不直接把两组吞吐差异归因于代码优化；调整配置时复用本套矩阵比较。
- 逐请求结果立即写JSONL；阶段写result.json和P/D前后metrics。
  服务健康检查失败会取消当前阶段并保留已完成请求；没有自动重启服务或降级。
  答案错误但服务健康时继续其他功能场景，该场景性能被标为阻塞。
- 功能测试是基础正确性验证，不是全面精度评测；性能阶段强制256输出，
  单独保留answer_matches，不将强制输出的HTTP成功当成质量认证。

**默认全矩阵可能持续数天。** 仅1M的六个并发档就有272个正式请求，若仍按
此前32K约895输入tok/s粗估，光输入处理就约88小时；这只是安排时长的量级，
不代表1M实际吞吐。单请求总超时为86400秒，可在配置中修改。
如果要先完成一轮覆盖，可显式把concurrency改为[1,2,4]并使用新RUN_ID；
脚本不会自行跳过1M或降低并发。

## 采样及报告

每节点每10秒记录GPU忙碌/显存/温度/功耗、CPU/内存/压力/磁盘计数、各NIC
收发/丢包/错误、各RDMA端口数据量与硬件计数。每分钟记录进程、socket、
近期内核信息和本机BMC功耗（若支持）。不调用torch，不占用GPU推理显存。
P1同时是TP节点和压测客户端，客户端CPU负载也会记录；若P1先饱和，后续
应将客户端移到独立机器复核，不把客户端瓶颈归因于GPU。

```bash
# 在P1收集其余三台；P1数据直接使用原目录。运行中收集可能不完整，应标为中间报告。
.venv/bin/python "$GLM_ASSESS_JOB" collect all --run-id "$GLM_ASSESS_RUN"
.venv/bin/python examples/disaggregated/glm5next_int8/assessment_report.py \
  "$GLM_ASSESS_ROOT" --hours 10 --input-price 0.8 --output-price 2.8
cat "$GLM_ASSESS_ROOT/report.md"
```

如果实际电价0.8元/度，再加 `--electricity-price 0.8`。只有四节点整机BMC
采样覆盖都达到95%，才合计整机能耗/每日电费。GPU功耗始终单列，不能替代
整机电费。 report.json保留P50/P95/P99、每节点RDMA速率、监控覆盖及财务参数。
结算仍需电表、待机/制冷/折旧/运维数据，不根据不完整阶段推测收益。

不要仅据NIXL平均MB/s判断25G链路利用率；该数值不等于集群网卡墙钟吞吐。
检查RDMA端口与NIC每方向利用率、错误/PFC等原始计数，再与P/D忙碌、TTFT、
TPOT联动分析。报告不把bond、物理口、RDMA重复相加。

## 链路基线（可选，和模型压测分开执行）

在装有iperf3的节点上测TCP基线可发现明显链路异常，但不能证明ROCm RDMA正常。
例如D0前台运行 `iperf3 -s -1 -p 5201`，P1运行
`iperf3 -c 10.5.10.55 -p 5201 -P 4 -t 30 -J > iperf-p1-d0.json`。
测试反向时重新启动一次服务端，客户端加 `-R`。
不要在正式模型压测期间混跑iperf3，也不要据TCP结果替代NIXL/GPU转移结果。

这些脚本本地测试的是控制逻辑、结果归档和采样计算；四机ROCm、真实模型与
后台session的现场行为仍需在上述服务器执行验证。
