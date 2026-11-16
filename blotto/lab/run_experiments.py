"""Run the first exact Game 1 audit and exploratory population/response study."""

from fractions import Fraction
from pathlib import Path
from time import perf_counter
import argparse
import csv
import hashlib
import json
import platform
import sys

from blotto_lab import __version__
from blotto_lab.core import (
    DEFAULT_GAME, Population, best_response, constructive_counter,
    exhaustive_best_response, iter_allocations, mix_tables, score_twice, security_of,
)
from blotto_lab.populations import initial_scenarios
from blotto_lab.arena import ResponseArchive, advance_history, fraction_record, latest_only

ROOT = Path(__file__).resolve().parent


def exact(value):
    return fraction_record(value)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def response_record(response):
    return {"allocation": list(response.allocation), "expected_score": exact(response.expected_score),
            "optimum_count": response.optimum_count, "evaluations": response.evaluations,
            "method": response.method}


def write_csv(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def progress(record):
    step = record["step"]
    if step % 1000 == 0:
        print(f"History step {step}: security={record['security']['decimal']:.6f}, "
              f"exploitability={record['exploitability']['decimal']:.6f}, "
              f"distinct={record['archive_size']}", flush=True)


def make_plot(output, history, forgetting):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    ax = axes[0]
    ax.semilogx([r["history_mass"] for r in history],
                [r["security"]["decimal"] for r in history], color="#166e80", linewidth=2)
    ax.axhline(7.5, color="#526678", linestyle="--", label="Mixed minimax value: 7.5")
    ax.axhline(5, color="#b67927", linestyle=":", label="Best pure guarantee: 5")
    ax.set(title="Full-history response pool: guarantee for its randomized mixture",
           xlabel="Retained observations (log scale)", ylabel="Worst-case expected points", ylim=(0,8))
    ax.grid(alpha=.18)
    ax.legend(loc="lower right", frameon=False)
    ax = axes[1]
    ax.plot([r["step"] for r in forgetting["trace"]],
            [r["current_security"]["decimal"] for r in forgetting["trace"]],
            color="#a04b48", linewidth=1.5, marker="o", markersize=3)
    ax.axhline(5, color="#b67927", linestyle=":")
    ax.set(title="Latest-only responses: guarantee of each current fixed allocation",
           xlabel="Response step", ylabel="Worst-case points", ylim=(0,8))
    ax.grid(alpha=.18)
    fig.suptitle("Exact Game 1 adaptation diagnostics", fontsize=15, fontweight="bold")
    fig.savefig(output/"adaptation.png", dpi=170)
    plt.close(fig)
    return True


def build_report(output, results):
    models = results["scenarios"]
    optimal = results["scenario_optima"]
    final = results["history_trace"][-1]
    cycle = results["latest_only"]
    lines = [
        "**Game 1 foundation: verified first results**", "",
        "The exact scorer, population evaluator, best-response optimizer, independent enumerator, "
        "and resumable response archive are implemented. This study uses mathematical controls "
        "and explicit synthetic hypotheses. It does not estimate actual applicant proportions or select final submissions.", "",
        "**Correctness evidence**", "",
        f"The full-space audits covered {results['full_scans'][0]['evaluations']:,} allocations per selected model. "
        f"Dynamic programming and enumeration agreed on score, vector, and number of optima for "
        f"all {len(results['full_scans'])} selected models.", "",
        "The uniform-vector optimum is (0,12,23,30,35), unique, with exact expected score "
        "90904115/9196252. Other checks cover exact ties, duplicate weights, self-exclusion, "
        "tiny-instance exhaustive comparisons, analytic controls, and deterministic checkpoint resumption.", "",
        (f"The production constructive-counter routine was checked on {results['counter_audit']['checked']:,} "
        "allocations; every checked counter was legal and scored at least 10 points.")
        if results['counter_audit']['enabled'] else "The optional all-allocation counter audit was skipped in this run.", "",
        "**How population changes the best response**", "",
        "Each row optimizes a different stated population. Scores in this table are exact for that population; "
        "a high score against a toy hypothesis is not an estimate of a competition score.", "",
        "| Population | Exact best-response vector | Expected points | Number of tied optima |",
        "| --- | --- | ---: | ---: |",
    ]
    for model in models:
        item = optimal[model["key"]]
        lines.append(f"| {model['title']} | {tuple(item['allocation'])} | "
                     f"{item['expected_score']['decimal']:.6f} | {item['optimum_count']:,} |")
    lines += ["", "The first three populations have analytic distributions. The final three are saved "
              "finite samples from unfitted generators; their scores are exact conditional on those samples. "
              "No default mixture over these six scenarios has been assigned.", "",
              "**Cross-population performance**", "",
              "Rows below use each scenario's chosen best response. Columns score it against a different "
              "population. This exposes specialization; the column optima above are the corresponding benchmarks.", "",
              "| Candidate source | " + " | ".join(m["key"] for m in models) + " |",
              "| --- | " + " | ".join("---:" for _ in models) + " |"]
    for model in models:
        key = model["key"]
        lines.append("| " + key + " | " + " | ".join(
            f"{results['cross_scores'][key][m['key']]['decimal']:.3f}" for m in models) + " |")
    lines += ["", "Detailed exact scores and scenario regret are saved in results.json. "
              "Sensitivity to a mixture of the two random controls is in mixture-sensitivity.csv. "
              "The weights there are deliberately varied experiment inputs, not estimates of human behavior.", "",
              "**What happens during adaptation**", "",
              f"The full-history experiment added {final['step']:,} exact best responses, retaining "
              f"{final['history_mass']:,} observations across {final['archive_size']:,} distinct allocations. "
              f"Its final empirical mixture has a full-space exploitability of "
              f"{final['exploitability']['decimal']:.6f} points and guarantees "
              f"{final['security']['decimal']:.6f} expected points against any legal opponent.", "",
              "That guarantee is computed by an exact best response over the entire action space. "
              "It describes the distribution obtained by drawing from the retained history. "
              "It is not a guarantee for any particular member, and its frequencies are not an applicant forecast. "
              "Progress need not be monotonic, and iteration count is not a convergence certificate.", ""]
    if cycle["cycle_length"] is not None:
        lines.append(f"The latest-only baseline entered a deterministic cycle of length {cycle['cycle_length']} "
                     f"starting at step {cycle['cycle_start']}. Every current fixed allocation remained counterable.")
    else:
        lines.append(f"The latest-only baseline found no repeated allocation within {len(cycle['trace'])} steps. "
                     "Every current fixed allocation still had a profitable exact counter.")
    lines += ["", "The plots compare a mixture's security with individual fixed allocations' security. "
              "Those are different output objects. They illustrate why evolving a single champion and "
              "constructing a hard-to-exploit distribution are different tasks.", ""]
    if results["plot_created"]:
        lines.append("![Exact adaptation diagnostics](adaptation.png)")
    lines += ["", "**What this establishes and what remains**", "",
              "The foundation can optimize a specified Game 1 population exactly, expose counters, "
              "and quantify the security of an adapting archive. The next stage is to build and challenge "
              "population hypotheses, including transparent use of historical behavioral features and "
              "held-out generators. Robust scenario selection and final pure-vector choice remain separate "
              "from this exploratory demonstration. Game 2 has not been developed in this milestone.", "",
              "Run configuration, code hashes, exact results, sampled supports, checkpoint, and timing "
              "measurements are saved alongside this report. All assumptions are recorded in scenarios.json."]
    (output/"report.md").write_text("\n".join(lines) + "\n")


def run(config, output, resume=False):
    for key in ("seed", "finite_population_size", "history_steps", "record_every",
                "checkpoint_every", "latest_only_max_steps"):
        if type(config[key]) is not int or config[key] < (0 if key in ("seed", "history_steps") else 1):
            raise ValueError(f"Invalid integer configuration: {key}")
    if type(config["check_every_constructive_counter"]) is not bool:
        raise ValueError("Counter audit setting must be boolean")
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(ROOT.glob("blotto_lab/*.py"))}
    source_hashes["run_experiments.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {"version": __version__, "python": sys.version, "platform": platform.platform(),
                "config": config, "code_sha256": source_hashes,
                "semantics": "Game 1 exact; synthetic exploration; no applicant prevalence inferred"}
    checkpoint = output/"history-checkpoint.json"
    if resume:
        previous_manifest = json.loads((output/"manifest.json").read_text())
        if previous_manifest["code_sha256"] != source_hashes:
            raise ValueError("Cannot silently resume after source changes; preserve the old run and start a new one")
        old_config = {k:v for k,v in previous_manifest['config'].items() if k != 'history_steps'}
        new_config = {k:v for k,v in config.items() if k != 'history_steps'}
        if old_config != new_config:
            raise ValueError("Only history_steps may change when resuming this experiment")
        state = ResponseArchive.load(checkpoint)
        if state.game != DEFAULT_GAME or state.steps > config['history_steps']:
            raise ValueError("Checkpoint game or target step count does not match")
        manifest['previous_targets'] = previous_manifest.get('previous_targets',[])+[previous_manifest['config']['history_steps']]
    else:
        state = ResponseArchive.start()
    save_json(output/"manifest.json", manifest)
    models = initial_scenarios(config["seed"],config["finite_population_size"])
    if set(config['full_scan_scenarios'])-set(m.key for m in models):
        raise ValueError("Unknown full-scan scenario")
    save_json(output/"scenarios.json", [m.metadata() for m in models])
    save_json(output/"finite-populations.json", {m.key: m.population.to_dict() for m in models if m.population})
    optima = {m.key: best_response(m.table) for m in models}
    results = {"scenarios": [m.metadata() for m in models],
               "scenario_optima": {k: response_record(v) for k,v in optima.items()},
               "full_scans": [], "timings_seconds": {}}
    for model in models:
        if model.key not in config["full_scan_scenarios"]:
            continue
        start = perf_counter()
        scan = exhaustive_best_response(model.table)
        duration = perf_counter()-start
        dp = optima[model.key]
        if (scan.allocation,scan.expected_score,scan.optimum_count) != (dp.allocation,dp.expected_score,dp.optimum_count):
            raise AssertionError("Independent enumeration and DP disagree")
        item = response_record(scan)
        item.update(scenario=model.key,seconds=duration)
        results["full_scans"].append(item)
        print(f"Full scan passed: {model.key}, {scan.evaluations:,} actions",flush=True)
    if not results["full_scans"]:
        raise ValueError("Select at least one full-scan scenario for this audit report")

    start, checked = perf_counter(), 0
    if config["check_every_constructive_counter"]:
        for x in iter_allocations():
            y = constructive_counter(x)
            if score_twice(y,x) < 20:
                raise AssertionError("Constructive counter failed")
            checked += 1
            if checked % 1000000 == 0:
                print(f"Constructive counter check: {checked:,} allocations",flush=True)
    results["counter_audit"] = {"enabled":config["check_every_constructive_counter"],
                                "checked": checked,"seconds":perf_counter()-start}

    results["cross_scores"], results["scenario_regret"] = {}, {}
    for source, response in optima.items():
        results["cross_scores"][source] = {m.key:exact(m.table.expected_score(response.allocation)) for m in models}
        results["scenario_regret"][source] = {m.key:exact(optima[m.key].expected_score-m.table.expected_score(response.allocation)) for m in models}
    sensitivity = []
    for tenth in range(11):
        weight = Fraction(tenth,10)
        table = mix_tables([models[0].table,models[1].table],[weight,1-weight])
        response = best_response(table)
        sensitivity.append([str(weight),*response.allocation,float(response.expected_score),
                            float(models[0].table.expected_score(response.allocation)),
                            float(models[1].table.expected_score(response.allocation))])
    write_csv(output/"mixture-sensitivity.csv",["uniform_vector_weight","castle1","castle2","castle3","castle4","castle5","mixture_score","uniform_score","soldier_random_score"],sensitivity)
    progress_path = output/'arena-progress.jsonl'
    old = []
    if resume and progress_path.exists():
        lines = progress_path.read_text().splitlines()
        for i,line in enumerate(lines):
            try:
                record=json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines)-1:
                    break
                raise
            if record['step'] < state.steps:
                old.append(record)
    with progress_path.open('w') as handle:
        for record in old:
            handle.write(json.dumps(record)+'\n')
    def durable_progress(record):
        with progress_path.open('a') as handle:
            handle.write(json.dumps(record)+'\n')
            handle.flush()
        progress(record)
    start = perf_counter()
    trace = advance_history(state,config["history_steps"],config["record_every"],
        checkpoint,config["checkpoint_every"],durable_progress)
    results["timings_seconds"]["history_run"] = perf_counter()-start
    results["history_trace"] = [r for r in old if r["step"] < trace[0]["step"]]+trace
    results["latest_only"] = latest_only(max_steps=config["latest_only_max_steps"])
    archive = state.population()
    save_json(output/"arena-mixture.json",archive.to_dict())
    results["history_mixture_scenario_scores"] = {
        m.key:exact(sum((mass*m.table.expected_score(x) for x,mass in archive.entries),Fraction(0))/archive.total_mass)
        for m in models}
    results["plot_created"] = make_plot(output,results["history_trace"],results["latest_only"])
    save_json(output/"results.json",results)
    write_csv(output/"arena-progress.csv",["step","history_mass","archive_size","exploitability","security"],
        [[r["step"],r["history_mass"],r["archive_size"],r["exploitability"]["decimal"],r["security"]["decimal"]] for r in results["history_trace"]])
    build_report(output,results)
    print(f"Report saved: {output/'report.md'}",flush=True)
    return results


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/first-pass.json")
    parser.add_argument("--output",type=Path,default=ROOT/"results/first-pass")
    parser.add_argument("--resume",action="store_true",help="Continue saved history to the configured total step count")
    args=parser.parse_args()
    run(json.loads(args.config.read_text()),args.output.resolve(),args.resume)
