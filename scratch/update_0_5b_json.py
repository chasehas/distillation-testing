import json

with open("docs/benchmark_results_0_5b.json", "r", encoding="utf-8") as f:
    b05 = json.load(f)

with open("docs/math_scale_150.json", "r", encoding="utf-8") as f:
    m150 = json.load(f)

b05["math_reasoning"] = m150["math_reasoning"]

with open("docs/benchmark_results_0_5b.json", "w", encoding="utf-8") as f:
    json.dump(b05, f, indent=2)

print("Successfully updated docs/benchmark_results_0_5b.json with N=150 math results.")
