import glob
import json
import os

def generate_dashboard():
    # Only read completed or valid structured log files (with valid seed)
    files = glob.glob('models/*_s[0-9]*_results.json')
    
    results = []
    for f in files:
        with open(f, 'r') as file:
            try:
                results.append(json.load(file))
            except Exception:
                pass

    # Sort results to be predictable
    results.sort(key=lambda x: (x.get('mode', ''), x.get('seed', 0)))

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PentaNet vs BitNet - NeurIPS Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #0f172a;
            --surface-color: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --accent: #38bdf8;
        }
        body { font-family: 'Inter', system-ui, sans-serif; background-color: var(--bg-color); color: var(--text-main); margin: 0; padding: 40px 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 2.5rem; margin: 0 0 10px 0; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .header p { color: var(--text-muted); font-size: 1.1rem; }
        .card { background-color: var(--surface-color); border-radius: 16px; padding: 30px; margin-bottom: 30px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3); border: 1px solid var(--border); }
        .card h2 { margin-top: 0; margin-bottom: 20px; font-size: 1.4rem; display: flex; align-items: center; gap: 10px; }
        .chart-container { position: relative; height: 400px; width: 100%; }
        table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
        th, td { padding: 16px; text-align: left; border-bottom: 1px solid var(--border); }
        th { color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em; }
        tr:last-child td { border-bottom: none; }
        .badge { background: #334155; padding: 4px 8px; border-radius: 6px; font-size: 0.85rem; font-weight: bold; }
        .badge.pentanet { background: rgba(56, 189, 248, 0.2); color: #38bdf8; }
        .badge.bitnet { background: rgba(244, 63, 94, 0.2); color: #f43f5e; }
        .controls { display: flex; flex-wrap: wrap; gap: 25px; margin-bottom: 20px; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; align-items: center; }
        .controls strong { margin-right: 10px; color: var(--accent); }
        .controls label { margin-right: 15px; cursor: pointer; user-select: none; font-size: 0.95rem; }
        .controls input[type="checkbox"] { accent-color: var(--accent); margin-right: 5px; transform: scale(1.1); cursor: pointer; }
        select { background-color: #0f172a; color: var(--text-main); border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px; font-size: 1rem; cursor: pointer; }
        select:focus { outline: none; border-color: var(--accent); }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>PentaNet Metrics Dashboard</h1>
        <p>Live interactive visualization of Model Training.</p>
    </div>

    <!-- Summary Table -->
    <div class="card">
        <h2>📊 Training Results Summary</h2>
        <table id="summaryTable">
            <thead>
                <tr>
                    <th>Model Architecture</th>
                    <th>Seed</th>
                    <th>Parameters</th>
                    <th>Total Time</th>
                    <th>Best Perplexity (PPL)</th>
                    <th>Best Val Loss</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>

    <!-- PPL Chart -->
    <div class="card">
        <h2>📉 Perplexity Convergence</h2>
        <div class="controls" id="pplControls">
            <div>
                <strong>Algorithms:</strong>
                <label><input type="checkbox" class="model-filter" value="pentanet" checked> PentaNet</label>
                <label><input type="checkbox" class="model-filter" value="bitnet" checked> BitNet</label>
            </div>
            <div id="seedFilters">
                <strong>Seeds:</strong>
                <!-- Populated by JS -->
            </div>
        </div>
        <div class="chart-container">
            <canvas id="pplChart"></canvas>
        </div>
    </div>

    <!-- Weights Chart -->
    <div class="card">
        <h2>⚖️ PentaNet Weight Entropy (Evolution of {-2 .. 2})</h2>
        <div class="controls">
            <div>
                <strong>PentaNet Explorer: </strong>
                <select id="weightSeedSelect"></select>
            </div>
        </div>
        <div class="chart-container">
            <canvas id="weightChart"></canvas>
        </div>
    </div>
</div>

<script>
    const rawData = DATA_PLACEHOLDER;
    
    // --- 1. Populate Table ---
    const tbody = document.querySelector('#summaryTable tbody');
    rawData.forEach(run => {
        if (!run.log || run.log.length === 0) return;
        const ppl = Math.exp(run.best_val_loss).toFixed(2);
        
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><span class="badge ${run.mode}">${run.mode.toUpperCase()}</span></td>
            <td>${run.seed}</td>
            <td>${run.params_M.toFixed(1)}M</td>
            <td>${Math.round(run.total_time)}s</td>
            <td style="font-weight: bold; color: ${run.mode === 'pentanet' ? '#38bdf8' : '#f43f5e'}">${ppl}</td>
            <td>${run.best_val_loss.toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
    });

    // Extract unique seeds from all data
    const allSeeds = [...new Set(rawData.map(r => r.seed))].sort((a,b)=>a-b);
    const seedFiltersDiv = document.getElementById('seedFilters');
    allSeeds.forEach(seed => {
        seedFiltersDiv.innerHTML += `<label><input type="checkbox" class="seed-filter" value="${seed}" checked> ${seed}</label>`;
    });

    // Extract pentanet runs for the dropdown
    const weightSelect = document.getElementById('weightSeedSelect');
    const pentanetRuns = rawData.filter(r => r.mode === 'pentanet' && r.log && r.log.length > 2);
    pentanetRuns.forEach(run => {
        weightSelect.innerHTML += `<option value="${run.seed}">Seed ${run.seed}</option>`;
    });

    // --- 2. Chart: PPL Convergence ---
    const ctxPpl = document.getElementById('pplChart').getContext('2d');
    
    const pplDatasetsAll = [];
    const knownColors = {
        'pentanet': ['#38bdf8', '#0ea5e9', '#6366f1', '#a855f7'],
        'bitnet': ['#f43f5e', '#e11d48', '#be123c', '#9f1239']
    };
    
    let colorIndex = {'pentanet': 0, 'bitnet': 0};
    let maxItersPpl = [];

    rawData.forEach(run => {
        if (!run.log || run.log.length < 2) return;
        
        const logs = run.log.slice(1);
        const dataPoints = logs.map(l => l.ppl);
        const iterPoints = logs.map(l => l.iter);
        
        if (iterPoints.length > maxItersPpl.length) maxItersPpl = iterPoints;
        
        const color = knownColors[run.mode][colorIndex[run.mode] % knownColors[run.mode].length];
        colorIndex[run.mode]++;
        
        pplDatasetsAll.push({
            _meta_mode: run.mode,
            _meta_seed: String(run.seed),
            label: `${run.mode.toUpperCase()} (Seed ${run.seed})`,
            data: dataPoints,
            borderColor: color,
            backgroundColor: `${color}22`,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            tension: 0.3,
            fill: false
        });
    });

    const pplChart = new Chart(ctxPpl, {
        type: 'line',
        data: { labels: maxItersPpl, datasets: [...pplDatasetsAll] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: { type: 'logarithmic', title: { display: true, text: 'Validation PPL (Log)' }, grid: { color: '#334155' } },
                x: { title: { display: true, text: 'Iterations' }, grid: { color: '#334155', display: false } }
            },
            plugins: { legend: { labels: { color: '#f8fafc' } } }
        }
    });

    // Update Chart Logic for Toggles
    function updatePplChart() {
        const activeModels = Array.from(document.querySelectorAll('.model-filter:checked')).map(cb => cb.value);
        const activeSeeds = Array.from(document.querySelectorAll('.seed-filter:checked')).map(cb => cb.value);
        
        pplChart.data.datasets = pplDatasetsAll.filter(ds => 
            activeModels.includes(ds._meta_mode) && activeSeeds.includes(ds._meta_seed)
        );
        pplChart.update();
    }

    document.querySelectorAll('.model-filter, .seed-filter').forEach(cb => {
        cb.addEventListener('change', updatePplChart);
    });

    // --- 3. Chart: Weight Distribution ---
    const ctxWeights = document.getElementById('weightChart').getContext('2d');
    const bucketColors = ['#e11d48', '#38bdf8', '#22c55e', '#0284c7', '#be123c'];
    const weightKeys = ['-2', '-1', '0', '1', '2'];
    
    let weightChart;

    function renderWeightChart(seed) {
        const run = pentanetRuns.find(r => String(r.seed) === String(seed));
        if (!run) return;

        const logs = run.log;
        const iters = logs.map(l => l.iter);
        
        const datasets = weightKeys.map((key, i) => {
            const dataPts = logs.map(l => {
                const w = l.weights;
                const total = Object.values(w).reduce((a,b)=>a+b, 0);
                return (w[key] / total) * 100;
            });
            return {
                label: `Weight Bucket [${key}]`,
                data: dataPts,
                borderColor: bucketColors[i],
                backgroundColor: bucketColors[i],
                fill: false,
                tension: 0.3,
                pointRadius: 0,
                pointHoverRadius: 5,
                borderWidth: 3
            };
        });

        if (weightChart) weightChart.destroy();
        
        weightChart = new Chart(ctxWeights, {
            type: 'line',
            data: { labels: iters, datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                scales: {
                    y: { min: 0, max: 100, title: { display: true, text: 'Percentage (%)' }, grid: { color: '#334155' } },
                    x: { title: { display: true, text: 'Iterations' }, grid: { color: '#334155', display: false } }
                },
                plugins: {
                    legend: { labels: { color: '#f8fafc' } },
                    tooltip: { callbacks: { label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(1)}%` } }
                }
            }
        });
    }

    if (pentanetRuns.length > 0) {
        renderWeightChart(pentanetRuns[0].seed);
    }

    weightSelect.addEventListener('change', (e) => {
        renderWeightChart(e.target.value);
    });

</script>
</body>
</html>
"""

    html_content = html_template.replace("DATA_PLACEHOLDER", json.dumps(results))
    
    with open('dashboard.html', 'w') as f:
        f.write(html_content)
        
    print("✅ Interactive dashboard written to dashboard.html")

if __name__ == "__main__":
    generate_dashboard()
