# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join measured load windows with four-host counters and estimate production."""

import argparse
import json
import statistics
import tarfile
from pathlib import Path

import regex as re


def json_lines(lines):
    for line in lines:
        try:
            yield json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue


def read_samples(root, role):
    path = root / role / "monitor/data/samples.jsonl"
    if path.exists():
        with path.open() as handle:
            return sorted(json_lines(handle), key=lambda s: s["time"])
    archive = root / "collected" / f"{role}.tar"
    if archive.exists():
        with tarfile.open(archive) as tar:
            for member in tar:
                if member.name.removeprefix("./") == "monitor/data/samples.jsonl":
                    return sorted(
                        json_lines(tar.extractfile(member)), key=lambda s: s["time"]
                    )
    return []


def counter_rate(a, b, seconds):
    if a is None or b is None or b < a or seconds <= 0:
        return None
    return (b - a) / seconds


def gpu_watts(sample):
    values = []
    for gpu in sample.get("gpu", {}).values():
        sensors = gpu.get("sensors", {})
        power = sensors.get("power1_average")
        if power is None:
            power = sensors.get("power1_input")
        if power is None:
            return None
        values.append(power / 1e6)
    return sum(values) if values else None


def host_watts(sample):
    text = sample.get("host_power", {}).get("stdout", "")
    match = re.search(r"Instantaneous power reading\s*:\s*(\d+)\s+Watts", text, re.I)
    return float(match[1]) if match else None


def window(samples, start, end, iface):
    duration = end - start
    rx, tx, rdma_tx, busy, vram, cpu = [], [], [], [], [], []
    gpu_ws = host_ws = covered = gpu_covered = host_covered = 0.0
    drops = 0
    counter_resets = 0
    last_power = None
    last_power_time = 0
    for a, b in zip(samples, samples[1:]):
        if host_watts(a) is not None:
            last_power, last_power_time = host_watts(a), a["time"]
        dt = b["time"] - a["time"]
        overlap = max(0, min(b["time"], end) - max(a["time"], start))
        if not overlap or dt <= 0 or dt > 60:
            continue
        covered += overlap
        power = gpu_watts(a)
        if power is not None:
            gpu_ws += power * overlap
            gpu_covered += overlap
        if last_power is not None and b["time"] - last_power_time <= 120:
            host_ws += last_power * overlap
            host_covered += overlap
        for direction, rates in (("rx_bytes", rx), ("tx_bytes", tx)):
            first = (
                a.get("network", {}).get(iface, {}).get("counters", {}).get(direction)
            )
            last = (
                b.get("network", {}).get(iface, {}).get("counters", {}).get(direction)
            )
            value = counter_rate(first, last, dt)
            if value is not None:
                rates.append(value * 8 / 1e9)
        for field in ("rx_dropped", "tx_dropped", "rx_errors", "tx_errors"):
            first = a.get("network", {}).get(iface, {}).get("counters", {}).get(field)
            last = b.get("network", {}).get(iface, {}).get("counters", {}).get(field)
            if first is not None and last is not None:
                if last < first:
                    counter_resets += 1
                else:
                    drops += last - first
        # Expose each RDMA port separately; never add bond and physical counters.
        for port, data in a.get("rdma", {}).items():
            value = counter_rate(
                data.get("counters", {}).get("port_xmit_data"),
                b.get("rdma", {})
                .get(port, {})
                .get("counters", {})
                .get("port_xmit_data"),
                dt,
            )
            if value is not None:
                rdma_tx.append((port, value * 4 * 8 / 1e9))
        for gpu in a.get("gpu", {}).values():
            if gpu.get("gpu_busy_percent") is not None:
                busy.append(gpu["gpu_busy_percent"])
            if gpu.get("mem_info_vram_used") is not None:
                vram.append(gpu["mem_info_vram_used"] / 2**30)
        try:
            ca = list(map(int, a["proc"]["stat"].splitlines()[0].split()[1:9]))
            cb = list(map(int, b["proc"]["stat"].splitlines()[0].split()[1:9]))
            diff = [y - x for x, y in zip(ca, cb)]
            if sum(diff) > 0 and min(diff) >= 0:
                cpu.append(100 * (1 - (diff[3] + diff[4]) / sum(diff)))
        except (KeyError, TypeError, ValueError, IndexError):
            pass

    def mean(values):
        return statistics.mean(values) if values else None

    return {
        "coverage": covered / duration,
        "gpu_power_coverage": gpu_covered / duration,
        "host_power_coverage": host_covered / duration,
        "gpu_energy_kwh": gpu_ws / 3.6e6 if gpu_covered else None,
        "host_energy_kwh": host_ws / 3.6e6 if host_covered else None,
        "gpu_busy_mean_pct": mean(busy),
        "cpu_busy_mean_pct": mean(cpu),
        "vram_max_gib_per_card": max(vram) if vram else None,
        "nic_rx_mean_gbps": mean(rx),
        "nic_tx_mean_gbps": mean(tx),
        "nic_tx_max_gbps": max(tx) if tx else None,
        "nic_errors_drops_delta": drops,
        "counter_resets": counter_resets,
        "rdma_port_tx_mean_gbps": {
            port: mean([v for p, v in rdma_tx if p == port])
            for port in sorted({p for p, v in rdma_tx})
        },
    }


def fmt(value, digits=2):
    return "未测" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Run directory on P0 after collect")
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument(
        "--input-price", type=float, default=0.8, help="CNY per million input tokens"
    )
    parser.add_argument(
        "--output-price", type=float, default=2.8, help="CNY per million output tokens"
    )
    parser.add_argument("--electricity-price", type=float, help="Actual CNY per kWh")
    args = parser.parse_args()
    suite = args.root / "p0/test/data"
    cfg = json.loads((suite / "manifest.json").read_text())["config"]
    samples = {role: read_samples(args.root, role) for role in cfg["nodes"]}
    stages = []
    lines = [
        "# PD全量测试报告",
        "",
        "功能通过、性能吞吐和主机指标分别记录；HTTP成功不代表质量评测通过。",
        "缺失监控不按零处理；未执行、失败阶段不得用于收益估算。",
        "",
        (
            "| 阶段 | 状态 | 完成/失败 | 输入tok/s | 输出tok/s | "
            "TTFT均值/P95秒 | TPOT均值/P95毫秒 | 日产值元 |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    from assessment_suite import plan

    for mode, case, c in plan(cfg):
        key = f"{mode}-{case}-c{c}"
        path = suite / key / "result.json"
        if not path.exists():
            lines.append(f"| {key} | 未完成/未执行 | — | — | — | — | — | — |")
            continue
        stage = json.loads(path.read_text())
        if "started" not in stage:
            lines.append(f"| {key} | {stage.get('status')} | — | — | — | — | — | — |")
            continue
        stage["hosts"] = {
            role: window(data, stage["started"], stage["ended"], cfg["interface"])
            for role, data in samples.items()
        }
        revenue = None
        if stage.get("passed") and mode == "perf":
            revenue = (
                (
                    stage["input_tokens_per_s"] * args.input_price
                    + stage["output_tokens_per_s"] * args.output_price
                )
                * args.hours
                * 3600
                / 1e6
            )
        stage["daily_gross_cny"] = revenue
        measured = all(
            h["host_power_coverage"] >= 0.95 for h in stage["hosts"].values()
        )
        energy = (
            sum(h["host_energy_kwh"] or 0 for h in stage["hosts"].values())
            if measured
            else None
        )
        stage["cluster_host_energy_kwh"] = energy
        stage["daily_electricity_cny"] = (
            energy / stage["duration_s"] * args.hours * 3600 * args.electricity_price
            if energy is not None and args.electricity_price is not None
            else None
        )
        ttft, tpot = stage["ttft_s"], stage["tpot_s"]

        def ms(value):
            return None if value is None else value * 1000

        status = "通过" if stage.get("passed") else "失败/KV未通过"
        lines.append(
            f"| {key} | {status} | {stage['completed']}/{stage['failed']} | "
            f"{fmt(stage['input_tokens_per_s'])} | "
            f"{fmt(stage['output_tokens_per_s'])} | "
            f"{fmt(ttft['mean'])}/{fmt(ttft['p95'])} | "
            f"{fmt(ms(tpot['mean']))}/{fmt(ms(tpot['p95']))} | {fmt(revenue)} |"
        )
        stages.append(stage)
    lines += [
        "",
        "## 各节点监控（与每个阶段的负载时间窗口对齐）",
        "",
        (
            "| 阶段/节点 | 覆盖率 | GPU忙碌% | CPU忙碌% | NIC RX/TX Gbps | "
            "显存峰值GiB/卡 | GPU kWh | 整机kWh |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in stages:
        for role, host in stage["hosts"].items():
            lines.append(
                f"| {stage['mode']}-{stage['case']}-c{stage['concurrency']}/{role} | "
                f"{host['coverage']:.0%} | {fmt(host['gpu_busy_mean_pct'])} | "
                f"{fmt(host['cpu_busy_mean_pct'])} | "
                f"{fmt(host['nic_rx_mean_gbps'])}/{fmt(host['nic_tx_mean_gbps'])} | "
                f"{fmt(host['vram_max_gib_per_card'])} | "
                f"{fmt(host['gpu_energy_kwh'], 4)} | "
                f"{fmt(host['host_energy_kwh'], 4)} |"
            )
    lines += [
        "",
        "## 能耗及收入（仅性能通过阶段）",
        "",
        "| 场景 | 日产值元 | 每日电费元 | 扣电费后元（未扣其他成本） |",
        "|---|---:|---:|---:|",
    ]
    for stage in stages:
        if stage["daily_gross_cny"] is None:
            continue
        revenue = stage["daily_gross_cny"]
        cost = stage["daily_electricity_cny"]
        lines.append(
            f"| {stage['case']}-c{stage['concurrency']} | {fmt(revenue)} | "
            f"{fmt(cost)} | {fmt(revenue - cost if cost is not None else None)} |"
        )
    lines += [
        "",
        "## 口径",
        "",
        f"每天{args.hours}小时维持该阶段吞吐，输入{args.input_price}元/百万token、输出{args.output_price}元/百万token。各场景不能相加。",
        "默认单价参考2026-09-15智谱直供原价：https://help.aliyun.com/en/model-studio/glm-5-3-flash-by-zhipu；可用参数更新。",
        "日产值是假设所有产出可售的API等价产值；未包含质量折价、空闲时段、税费、折旧、机房及运维。",
        "GPU传感器功耗不是整机功耗，只有四节点BMC功耗覆盖均达到95%才合计整机能耗。采样估算不替代电表。",
        "NIC速率按收/发分别统计，不叠加bond与物理口；RDMA可能绕过Linux网卡计数，逐RDMA端口速率另见report.json。",
        "CPU忙碌不含idle/iowait；原始vmstat、pressure、diskstats、错误/丢包及内核记录在各节点samples.jsonl。",
        "先核对四台时钟同步；每阶段监控可能包含同机其他任务。KV核验是阶段级增量验证，不能证明每个请求的传输或全面模型精度。",
        "1M指1047551输入token加最多1024输出，留在1048576总上下文内。图文32K另含模板和视觉token，以服务端usage为准。",
    ]
    destination = args.root / "report.md"
    destination.write_text("\n".join(lines) + "\n")
    (args.root / "report.json").write_text(
        json.dumps(
            {"assumptions": vars(args) | {"root": str(args.root)}, "stages": stages},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(destination)


if __name__ == "__main__":
    main()
