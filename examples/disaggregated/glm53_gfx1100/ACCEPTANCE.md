# GLM-5.3-Flash-BF16 / W7900D 商业运营验收

版本日期：2026-09-14。使用建议门槛，业务负责人可在测试前调整并签字冻结。
测试对象是四节点、两个 TP16/EP 引擎，通过 NIXL TCP/ROCm 分离 P/D。
当前已证明一条请求生成成功且发生真实缓存传输；这不是商业上线结论。

## 范围、状态和执行规则

| 项目 | 当前值 |
| --- | --- |
| P 组 | P0 `.36:8001` + P1 `.3`，headless |
| D 组 | D0 `.55:8002` + D1 `.56`，headless |
| 验证代理 | P0 `127.0.0.1:8000`，仓库 toy_proxy_server.py |
| 配置 | BF16 权重/BF16 KV、eager、纯文本、max-model-len=2048、max-num-seqs=1 |
| 基础修复版本 | `5ac6cd80e`；正式报告还需记录实际测试提交 |
| 已观察 | 两组 HTTP 200；一次 5 输入/16 输出 token；16 次 NIXL 传输、298409984 字节、失败计数 0 |
| 尚未验收 | 长上下文、业务质量、工具、流式、故障、容量、稳定性、安全与运营 |

主机初始化仍由外层项目 deployment 文档管理。本文件及测试进入 fork，供服务器拉取。
不创建多主机批量操作脚本。HTTP 用例只在 P0 执行，不改变服务配置、不重启主机、不注入故障。
每个命令单独执行；不要 pytest-xdist，并暂停其他客户端流量，避免污染 NIXL 增量和缓存测试。
测试失败必须保留，不能通过扩大数值容差、忽略错误或跳过用例把结果变为通过。

## 建议上线门槛

以下是拟定验收标准，不是对 W7900D 的性能承诺。先测串行基线再冻结可售套餐。

| 维度 | 建议门槛 | 测量口径 |
| --- | --- | --- |
| 功能/接口 | 本文必需用例 100% 通过，无未关闭 P0/P1 缺陷 | 不支持的功能必须在产品明确禁用，不能继续宣传支持 |
| PD 正确性 | 新请求 NIXL 字节/次数增长；失败与通知失败增量均为 0 | 空闲服务前后快照，计数下降表示重启，不能当通过 |
| 与本地 prefill 一致 | 固定 seed、temperature=0，token 文本一致；所选 token logprob 绝对差 ≤0.02 | 有分歧则保存完整输出分析；通过不代表与参考 GPU 数学等价 |
| 暖态短请求延迟 | 128 输入/64 输出，串行：TTFT p95 ≤15s，TPOT p95 ≤250ms，E2E p95 ≤45s | 冷 JIT 单列；至少 200 请求，不从单条请求推算 p99 |
| 抖动 | ITL p99 ≤500ms，记录最长停顿 | SSE 到达间隔；一个 chunk 可能包含多个 token，不能冒充逐 token 内核时间 |
| 压力 | 额定负载成功率 ≥99.9%，负载高于额度时明确 429/503，无无界排队 | 额定负载先实测，再取满足延迟门槛容量的 ≤70% |
| 稳定性 | 24h、至少 10000 请求，成功率 ≥99.9%，OOM/worker 崩溃/传输失败为 0 | 不含主动故障注入窗口；全部失败仍单独报告 |
| 内存恢复 | 暖态固定形状循环后，空闲活跃请求为 0，缓存可回收；显存无持续增长 | 比较同一静置窗口；建议非缓存增长 ≤512MiB/GPU，结合分配器统计解释 |
| 恢复 | 故障检测 ≤30s、恢复接流 ≤10min、取消后资源释放 ≤30s | 从注入到监控/服务状态时间，作为首版目标 |
| 产品可用性 | 初始内部目标 99.9%/月 | 30 天约 43.2 分钟错误预算；实验 24h 不等于月 SLA |
| 租户和数据 | 越权、串租户、泄漏内部地址/密钥、错误计费：0 | 外部 API 入口验证，不以仅绑定 localhost 代替鉴权 |

故障级别：P0=串租户/错误计费/缓存污染或静默错误；P1=生成失败、错误码不符、资源泄漏、SLO 不达标；
P2=不影响已承诺能力的可用性/展示问题。业务负责人、研发和运维共同签署放行。

## 自动用例和场景矩阵

自动测试：`tests/standalone_tests/test_glm53_pd_acceptance.py`。
仅依赖 Python 标准库、pytest 和 regex，不加载 vLLM/GPU 扩展。记录每次请求/响应、状态、耗时到 JSONL。
只使用合成数据；真实客户数据进入报告前需脱敏。API key 不写入报告。

| ID / 优先级 | 场景与操作 | 预期与证据 | 执行方式 |
| --- | --- | --- | --- |
| ENV-01 / P1 | 四机核对分支提交、权重清单/hash、Torch/ROCm/NIXL、启动参数 | 四机一致，环境清单归档 | 手工，逐台 |
| API-01 / P1 | P0/D0 health、models | 200，模型名一致 | TestHealth |
| FUN-01 / P1 | 英文、中文、算术、订单资料问答 | 基础语义、finish_reason、usage 正确 | TestFunctional |
| FUN-02 / P1 | 多轮纠错 AX731→BY942 | 遵循最后纠正，不残留旧编号 | TestFunctional |
| FUN-03 / P1 | 严格 JSON schema 抽取订单/数量 | JSON 可解析、字段和值正确 | TestFunctional |
| FUN-04 / P1 | required 工具调用→合成工具结果→总结 | 参数和 tool_call_id 正确；不执行任何真实工具 | TestFunctional |
| ISO-01 / P0 | A/B/A 相似前缀、不同 marker | 无其他请求 marker，重复 A 保持正确 | TestFunctional；不等于多租户安全验证 |
| PD-01 / P0 | 唯一 nonce 新请求，D metrics 前后快照 | 字节/次数严格增，失败不增；不要求次数恰为16 | TestTransfer |
| CTX-01 / P0 | 精确 token ID 输入 1、3/4/5/6/7 | 池边界和余数0–3可生成、usage 长度准确 | TestBoundary |
| CTX-02 / P0 | 127/128/129、511–515、1023/1024/1025 | 长短索引路径无崩溃/错误长度 | TestBoundary |
| CTX-03 / P0 | 2044/2045/2046/2047 + 合法生成预算 | 总长度不超过2048，边缘请求成功 | TestBoundary |
| NUM-01 / P0 | 同输入分别经 PD、直接 D 本地 prefill | token/logprob/finish_reason 一致 | TestParity |
| API-02 / P1 | 错误模型、负 max_tokens/temperature | 404/400 + JSON error，不出现200或500 | TestGateway |
| API-03 / P1 | 输入2049；短输入请求输出4096 | 当前2048配置返回400，不断流、不崩溃 | TestGateway |
| API-04 / P1 | SSE 响应头、JSON事件、正文、完成标记 | text/event-stream，结束 [DONE]，finish_reason正确 | TestGateway |
| CACHE-01 / P0 | D 先直接预热前缀，再 PD 同前缀/扩展前缀/不同前缀 | 输出对照通过，不能用预期PD零传输简单判错 | 复用上游 test_edge_cases.py，见下文 |
| CACHE-02 / P0 | 按实际 KV block 大小 B 测 B−1/B/B+1、2B±1、末块 padding | 正确映射，非目标块不变，PD/本地一致 | 手工取配置 B 后扩展 NUM-01；128不是所有层页大小 |
| NUM-02 / P0 | 同 checkpoint 与受支持参考平台比较 | 200条以上领域固定集，无系统性质量下降 | 外部参考环境，不能把直接 D 当硬件参考 |
| BUS-01 / P1 | 中文客服、代码解释、表格抽取、摘要、翻译各40题 | 关键事实准确≥95%，格式遵循≥99%，领域负责人复核 | 200条带答案/评分标准的业务集 |
| BUS-02 / P0 | 上下文未给出的订单/价格；资料内恶意指令 | 不捏造交易结果；不越权调用工具；无敏感泄漏 | 合成对抗集，至少50题，人工双审 |
| CAP-01 / P1 | 暖态串行 128入/64出，200次 | 基线延迟、吞吐、NIXL带宽/CPU全部记录 | vllm bench serve |
| CAP-02 / P1 | 客户端并发2/4/8，混合短长请求 | 排队受控，无静默错误；容量是客户端并发，不改engine max-num-seqs | 分阶段单条 bench，超SLO停止增加 |
| CAP-03 / P1 | 2倍额定到达速率突发60s再回落 | 429/503与Retry-After一致，恢复≤30s，无长尾积压 | 商业网关和限流配置准备后 |
| SOAK-01 / P1 | 24h混合长短/多轮/JSON请求，至少10000次 | 成功率、显存/RSS/FD/线程/缓存趋势达标 | 独立压测时段，见稳定性操作 |
| CANCEL-01 / P1 | SSE输出2个chunk后断开客户端 | P和D均取消/释放，随后新请求成功，无孤儿请求 | 手工单次，观察两组metrics和日志 |
| FAIL-01 / P1 | P1单worker退出；随后另一次D1单worker退出 | 同TP组停止接流，网关不返回伪成功，告警≤30s | 隔离验收环境注入，不自动执行 |
| FAIL-02 / P0 | 只隔离NIXL传输路径，不影响HTTP；保留fail策略 | 有明确传输失败，不静默改成本地重算；修复后新请求成功 | 运维短时网络故障，有恢复手段 |
| FAIL-03 / P1 | 重启P组后D组，旧remote engine metadata失效 | 新握手正常，旧请求不重放计费、无永久缓存占用 | 维护窗口手工重启 |
| FAIL-04 / P1 | D加载中/尚未就绪时请求代理 | 有界等待或503，不200后断流；恢复后ready | 隔离环境 |
| FAIL-05 / P1 | 磁盘接近阈值、日志轮转、监控中断 | 预警，生成不因日志塞满系统盘崩溃 | 使用专用测试配额，不填满生产/data |
| SEC-01 / P0 | 无key、错key、过期key访问商业网关 | 401/403；不得从外部直连P/D绕过鉴权 | 商业网关准备后 |
| SEC-02 / P0 | 两租户相同前缀、不同私有marker、并发请求 | 不串会话、日志按权限隔离，缓存策略经审核 | 两测试租户，禁止真实敏感数据 |
| SEC-03 / P0 | 外部提交kv_transfer_params伪造remote_host/block_ids | 网关拒绝/移除内部字段，禁止任意内网目标 | 只使用预定测试地址，不扫描内网 |
| SEC-04 / P1 | 超大HTTP body、非法JSON、异常Unicode、图片/视频 | 有界内存，400/413/明确不支持，正常请求可恢复 | 商业网关准备后 |
| OPS-01 / P1 | request ID从网关到P/D、异常日志与指标 | 同一请求可追踪，错误原因不暴露地址/栈，告警有人接收 | 手工观察 |
| BILL-01 / P0 | 正常/流式/取消/超时/重试的usage和账单 | 规则事先定义；内部P生成不重复向客户计费 | 对照合成账单，重复请求无重复扣款 |
| REL-01 / P1 | 灰度升级和回滚，NIXL协议版本不一致 | 无混合版本错误传输；回滚含代码/配置/依赖清单 | 蓝绿或维护窗口，当前四机不具备冗余容量 |
| CTX-1M / P0 | 目标1048576总上下文，分阶段扩容 | 见1M专项；当前不可执行并标BLOCKED | 容量适配后单独验收 |

自动语义题是冒烟测试，不能替代领域质量集。若思考内容耗尽128输出预算，应记录实际
reasoning/content用量并单独定义产品思考模式，不靠修改期望文本掩盖失败。

## P0 逐条执行自动测试

先保持四台服务和代理运行，拉取本次测试提交。测试工具安装在独立环境，不修改服务环境。

```bash
git -C /data/vllm pull --ff-only origin gfx1100/glm53-bf16-pd
```

```bash
uv venv --python /data/vllm/.venv/bin/python /data/venvs/glm53-acceptance
```

```bash
uv pip install --python /data/venvs/glm53-acceptance/bin/python --default-index https://pypi.tuna.tsinghua.edu.cn/simple pytest regex
```

```bash
mkdir -p /data/logs/glm53-acceptance
```

```bash
export GLM53_PD_TEST_URL=http://127.0.0.1:8000
export GLM53_PREFILL_URL=http://10.5.10.36:8001
export GLM53_DECODE_URL=http://10.5.10.55:8002
export GLM53_TEST_RESULTS=/data/logs/glm53-acceptance/requests.jsonl
```

日志追加写入，保留每次响应；每一轮应使用独立结果目录或归档旧文件。下面用例固定对应2048上下文。

```bash
cd /data/vllm
```

第一组：健康、真实PD计数和功能。

```bash
/data/venvs/glm53-acceptance/bin/python -m pytest --confcutdir=tests/standalone_tests tests/standalone_tests/test_glm53_pd_acceptance.py -k 'TestHealth or TestTransfer or TestFunctional' -v --junitxml=/data/logs/glm53-acceptance/functional.xml
```

第二组：精确边界与本地prefill对照。直接D请求不携带kv_transfer_params，沿用上游的对照方式；
这验证PD没有改变同一后端结果，不证明GPU后端本身绝对正确。

```bash
/data/venvs/glm53-acceptance/bin/python -m pytest --confcutdir=tests/standalone_tests tests/standalone_tests/test_glm53_pd_acceptance.py -k 'TestBoundary or TestParity' -v --junitxml=/data/logs/glm53-acceptance/boundary-parity.xml
```

第三组：商业API契约。当前toy代理在上游4xx处理、流式Content-Type上存在已知缺口，
这些测试预计暴露失败，不能把失败跳过后宣布可商业运营。

```bash
/data/venvs/glm53-acceptance/bin/python -m pytest --confcutdir=tests/standalone_tests tests/standalone_tests/test_glm53_pd_acceptance.py -k TestGateway -v --junitxml=/data/logs/glm53-acceptance/gateway.xml
```

上游前缀缓存场景可复用 `tests/v1/kv_connector/nixl_integration/test_edge_cases.py` 的三个场景；
它需要openai客户端及上游测试依赖，属于后续扩展，不必为上述独立测试安装整套CUDA测试依赖。

## 容量测试：一次只运行一个阶段

在P0用现有服务环境运行。先5次预热，再正式200次；短请求强制64输出token。
模型tokenizer从本地路径读取，不下载模型。客户端并发数不等于引擎同时执行的序列数。

```bash
NO_PROXY=127.0.0.1,10.5.10.36,10.5.10.55 /data/vllm/.venv/bin/vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions --model zai-org/GLM-5.3-Flash-BF16 --tokenizer /data/models/zai-org/GLM-5.3-Flash-BF16 --dataset-name random --random-input-len 128 --random-output-len 64 --num-prompts 5 --max-concurrency 1 --request-rate inf --ignore-eos
```

```bash
NO_PROXY=127.0.0.1,10.5.10.36,10.5.10.55 /data/vllm/.venv/bin/vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions --model zai-org/GLM-5.3-Flash-BF16 --tokenizer /data/models/zai-org/GLM-5.3-Flash-BF16 --dataset-name random --random-input-len 128 --random-output-len 64 --num-prompts 200 --max-concurrency 1 --request-rate inf --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 --goodput ttft:15000 tpot:250 e2el:45000 --save-result --save-detailed --result-dir /data/logs/glm53-acceptance --result-filename capacity-c1.json
```

串行通过后，每次手工重复正式命令，将 `--max-concurrency` 改为2、4、8，并分别保存
`capacity-c2.json`、`capacity-c4.json`、`capacity-c8.json`。不要修改engine的max-num-seqs=1。
短长混合需固定50% 128/64、30% 512/128、20% 1536/256的合成请求集，记录实际token长度，
三个独立基准结果不能当作混合负载结果。随机输入只测性能，不能评估语义质量。

稳态后用容量允许的固定到达速率运行24h混合负载，至少10000请求；另测突发和取消。
每分钟采集每台GPU显存/利用率、CPU/RSS、FD、bond0吞吐、NIXL失败、队列和KV缓存指标。
容量接近上限、OOM、失败计数增加或延迟持续越界时停止升压，保留现场而非自动重启擦除证据。
每个阶段开始/结束保存P/D metrics快照，注明进程重启以免误读计数器。

## 故障注入的手工流程

仅在隔离验收环境或已安排维护窗口执行，每次只注入一种故障：

1. 保存版本、启动命令、metrics、进程与日志；确认SSH管理通道和恢复命令可用。
2. 启动一条合成请求并记录时间/request ID；运维选择本次目标worker或网络规则。
3. 注入FAIL-01/02等单一故障，记录客户端状态、完整流式结束方式、告警时间、P/D资源。
4. 恢复网络或按原命令重启整个受影响TP组；确认health，再执行TestTransfer和TestFunctional。
5. 对照账单，确认取消/失败/重试规则，没有重复计费；归档故障和恢复时间线。

本测试文件不包含kill、iptables、删除缓存或自动重启操作。当前两组各仅一个分布式实例，
任何一个TP worker故障都可能影响整组，商业高可用需要额外冗余，不能凭四台机器称为四副本。

## 1M 上下文专项（当前阻塞）

目标按1048576 token总预算定义：输入token（包括chat模板/工具定义）+最大输出token≤1048576。
`max_tokens=4096`只调整输出预算，不能扩展服务的max-model-len。
当前2048服务对5+4096必须返回400；用户已观察toy代理将此类错误表现为curl(18)，这是API门槛失败。

当前gfx1100 profile在 `vllm/utils/rocm_gfx1100.py` 明确限制max-model-len≤8192；
不得仅删除检查或宣称checkpoint配置写1M就已支持。1M属于新适配和容量项目，须完成：

- 四台采集实际权重常驻、Available KV cache、各KV group spec/block size/每token字节数，
  按rank计算1M的峰值。索引器/MLA中存在TP复制数据，不能把16卡剩余显存简单相加。
- 分开预算BF16 MLA、压缩索引缓存、raw tail、KDA状态、临时logits和NIXL注册描述符；
  现有Torch top-k和pool级logits随上下文增长，需实测chunked prefill的显存与时间。
- 先在隔离实例测8K→32K→128K→512K→1M；每阶段覆盖N−1/N/N+1总长度、池/页尾、
  needle位于开头/中间/结尾、相同前缀不同后缀、PD/本地结果一致、取消后的资源回收。
- 保持max-num-batched-tokens为已测chunk预算，不把它一并改成1M。
  8K以上需先实现对应profile支持并通过容量检查，现有命令不能直接运行这些阶段。
- 每阶段先记录可用KV token容量、每rank显存余量、加载/预热时间、TTFT与传输字节。
  总长度声明必须由最弱rank容量支撑，不能通过CPU swap/offload未测组合掩盖不够。

1M用例在报告中记BLOCKED，不记SKIP后放行，不对外售卖1M套餐。保留已通过的2048服务，
另开适配窗口验证，避免同时重启四台丢失可用基线。

## 报告模板与放行

每条用例记录：ID、时间、commit、模型hash、配置、执行者、输入token/输出token、期望、实际、
PASS/FAIL/BLOCKED/NOT_RUN、响应文件、P/D日志、前后metrics、缺陷链接和复测版本。
SKIP/NOT_RUN/BLOCKED不计入已通过；报告必须列出计划总数、执行数、成功数、失败数。

性能报告同时给出attempted/succeeded/failed、HTTP错误、流中断、超时、TTFT/TPOT/ITL/E2E
的p50/p95/p99及样本数、实际输入输出token、goodput、资源峰值；不只展示吞吐。

最终签署：研发（正确性和回归）、运维（容量/恢复/监控）、业务（语义和套餐）、
产品负责人（鉴权/配额/计费/承诺范围）。目前状态：功能冒烟通过，商业验收待执行。

## 本次交付验证记录

本地Mac收集到44个在线用例和4个离线用例。4个离线用例通过，验证指标解析、SSE解析、
上游400记录和传输中断记录；未配置在线地址，44个在线用例均未执行（pytest显示SKIPPED）。
在线功能、商业网关和性能门槛均不得据此标记通过。代码和文档通过适用pre-commit检查。
