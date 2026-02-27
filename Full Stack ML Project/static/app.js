/* ═══════════════════════════════════════════════════════
   DYNAMIC ML STUDIO — App Logic
   ═══════════════════════════════════════════════════════ */
const API = '';
let currentStep = 1;
let selectedDataset = null;
let edaData = null;
let selectedTask = null;
let trainResult = null;

// ── Step Navigation ─────────────────────────────
function goToStep(step) {
    if (step < 1 || step > 5) return;
    currentStep = step;
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`panel-${step}`).classList.add('active');
    document.querySelectorAll('.step-btn').forEach(b => {
        const s = +b.dataset.step;
        b.classList.remove('active', 'done');
        if (s === step) b.classList.add('active');
        else if (s < step) b.classList.add('done');
        if (s <= step) b.disabled = false;
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

document.querySelectorAll('.step-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (!btn.disabled) goToStep(+btn.dataset.step);
    });
});

// ── Load Datasets ───────────────────────────────
async function loadDatasets() {
    try {
        const res = await fetch(`${API}/api/datasets`);
        const datasets = await res.json();
        const grid = document.getElementById('dataset-grid');
        if (datasets.length === 0) {
            grid.innerHTML = '<div class="loading-placeholder">No CSV files found in the datasets/ folder.</div>';
            return;
        }
        grid.innerHTML = datasets.map(d => `
            <div class="dataset-card" data-path="${d.path}" onclick="selectDataset(this, '${d.path}', '${d.name}')">
                <div class="ds-icon">📄</div>
                <div class="ds-name">${d.name}</div>
                <div class="ds-meta">${d.size_kb} KB</div>
                <div class="ds-folder">📁 ${d.folder === '.' ? 'root' : d.folder}</div>
            </div>
        `).join('');
    } catch (e) {
        document.getElementById('dataset-grid').innerHTML =
            '<div class="loading-placeholder">❌ Failed to load datasets. Is the server running?</div>';
    }
}

// ── Select Dataset ──────────────────────────────
async function selectDataset(el, path, name) {
    document.querySelectorAll('.dataset-card').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
    selectedDataset = { path, name };
    showToast(`Selected: ${name}`);

    // Auto-analyze
    showToast('Analyzing dataset...');
    try {
        const res = await fetch(`${API}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
        edaData = await res.json();
        if (edaData.error) { showToast('Error: ' + edaData.error); return; }
        renderEDA(edaData);
        goToStep(2);
    } catch (e) {
        showToast('Analysis failed: ' + e.message);
    }
}

// ── Render EDA ──────────────────────────────────
function renderEDA(data) {
    document.getElementById('eda-filename').textContent = data.info.filename;
    document.getElementById('eda-subtitle').textContent =
        `${data.info.rows.toLocaleString()} rows × ${data.info.columns} columns • ${data.info.memory_kb} KB`;

    // Stats tiles
    document.getElementById('eda-stats').innerHTML = `
        <div class="metric-tile"><div class="value purple">${data.info.rows.toLocaleString()}</div><div class="label">Rows</div></div>
        <div class="metric-tile"><div class="value">${data.info.columns}</div><div class="label">Columns</div></div>
        <div class="metric-tile"><div class="value gold">${data.numeric_cols.length}</div><div class="label">Numeric</div></div>
        <div class="metric-tile"><div class="value red">${data.categorical_cols.length}</div><div class="label">Categorical</div></div>
        <div class="metric-tile"><div class="value">${data.info.memory_kb} KB</div><div class="label">Memory</div></div>
        <div class="metric-tile"><div class="value purple">${data.columns.reduce((s, c) => s + c.missing, 0)}</div><div class="label">Missing Values</div></div>
    `;

    // Column details table
    const colTable = document.getElementById('col-table');
    colTable.querySelector('thead').innerHTML = '<tr><th>#</th><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Stats / Top Values</th></tr>';
    colTable.querySelector('tbody').innerHTML = data.columns.map((c, i) => {
        let stats = '';
        if (c.is_numeric) stats = `Mean: ${c.mean ?? '—'} | Std: ${c.std ?? '—'} | Min: ${c.min ?? '—'} | Max: ${c.max ?? '—'}`;
        else if (c.top_values) stats = Object.entries(c.top_values).map(([k, v]) => `${k} (${v})`).join(', ');
        return `<tr><td>${i + 1}</td><td><strong>${c.name}</strong></td><td>${c.dtype}</td>
            <td style="color:${c.missing > 0 ? 'var(--red)' : 'var(--green)'}">${c.missing} (${c.missing_pct}%)</td>
            <td>${c.unique}</td><td>${stats}</td></tr>`;
    }).join('');

    // Sample table
    const sampleTable = document.getElementById('sample-table');
    sampleTable.querySelector('thead').innerHTML = '<tr>' + data.sample_cols.map(c => `<th>${c}</th>`).join('') + '</tr>';
    sampleTable.querySelector('tbody').innerHTML = data.sample.map(row =>
        '<tr>' + row.map(v => `<td>${v}</td>`).join('') + '</tr>'
    ).join('');

    // EDA plots
    const plotsEl = document.getElementById('eda-plots');
    plotsEl.innerHTML = data.plots.map(p => `
        <div class="plot-container" onclick="openLightbox(this)">
            <img src="${p.image}" alt="${p.title}" loading="lazy">
            <div class="plot-title">${p.title}</div>
        </div>
    `).join('');

    // Update task filename
    document.getElementById('task-filename').textContent = data.info.filename;
}

// ── Select Task ─────────────────────────────────
function selectTask(task) {
    selectedTask = task;
    document.querySelectorAll('.task-card').forEach(c => c.classList.remove('selected'));
    document.getElementById(`task-${task}`).classList.add('selected');
    renderConfigForm(task);
}

function renderConfigForm(task) {
    const configCard = document.getElementById('config-card');
    configCard.style.display = 'block';
    const form = document.getElementById('config-form');

    if (task === 'clustering') {
        const numCols = edaData.columns.filter(c => c.is_numeric).map(c => c.name);
        form.innerHTML = `
            <div class="form-group" style="grid-column:1/-1;">
                <label>Select Features for Clustering (numeric only)</label>
                <div class="checkbox-group" id="feature-checkboxes">
                    ${numCols.map(c => `<label><input type="checkbox" value="${c}" checked> ${c}</label>`).join('')}
                </div>
            </div>
        `;
    } else {
        const allCols = edaData.columns.map(c => c.name);
        const numericCols = edaData.columns.filter(c => c.is_numeric).map(c => c.name);
        const targetOptions = task === 'regression' ? numericCols : allCols;
        form.innerHTML = `
            <div class="form-group">
                <label>Target Column (what to predict)</label>
                <select id="target-select">
                    ${targetOptions.map(c => `<option value="${c}">${c}</option>`).join('')}
                </select>
            </div>
            <div class="form-group" style="grid-column:1/-1;">
                <label>Feature Columns (inputs — leave unchecked to auto-select)</label>
                <div class="checkbox-group" id="feature-checkboxes">
                    ${allCols.map(c => `<label><input type="checkbox" value="${c}" checked> ${c}</label>`).join('')}
                </div>
            </div>
        `;
        // Uncheck target from features when target changes
        document.getElementById('target-select').addEventListener('change', () => {
            const target = document.getElementById('target-select').value;
            document.querySelectorAll('#feature-checkboxes input').forEach(cb => {
                if (cb.value === target) cb.checked = false;
            });
        });
    }
    configCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ── Train Model ─────────────────────────────────
async function trainModel() {
    const btn = document.getElementById('btn-train');
    btn.innerHTML = '<span class="spinner"></span> Training...';
    btn.disabled = true;

    const features = [...document.querySelectorAll('#feature-checkboxes input:checked')].map(cb => cb.value);
    const target = selectedTask !== 'clustering' ? document.getElementById('target-select').value : null;

    // Remove target from features if present
    const cleanFeatures = features.filter(f => f !== target);

    try {
        const res = await fetch(`${API}/api/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                path: selectedDataset.path,
                task: selectedTask,
                target: target,
                features: cleanFeatures,
            })
        });
        trainResult = await res.json();
        if (trainResult.error) { showToast('Error: ' + trainResult.error); return; }

        renderResults(trainResult);
        goToStep(4);
    } catch (e) {
        showToast('Training failed: ' + e.message);
    } finally {
        btn.innerHTML = '🤖 Train Model';
        btn.disabled = false;
    }
}

// ── Render Results ──────────────────────────────
function renderResults(result) {
    const area = document.getElementById('results-area');
    let html = '';

    // Steps
    result.steps.forEach(step => {
        html += `<div class="step-card">
            <div class="step-title">${step.title}</div>
            <div class="step-explanation">${step.explanation}</div>
            <div>${step.details.map(d => `<div class="step-detail">${d}</div>`).join('')}</div>
        </div>`;
    });

    // Metrics with explanations
    html += `<div class="card"><div class="card-title">📊 Model Evaluation Metrics</div>
        <div class="card-subtitle">Each metric explained in plain language</div>`;
    for (const [name, m] of Object.entries(result.metrics)) {
        html += `<div class="metric-explain">
            <div class="me-name">${name}</div>
            <div class="me-value">${m.value}</div>
            <div class="me-text">${m.explanation}</div>
        </div>`;
    }
    html += '</div>';

    // Parameters
    html += `<div class="card"><div class="card-title">⚙️ Model Parameters</div>
        <div class="card-subtitle">All hyperparameters and settings used during training</div>
        <div class="param-grid">`;
    for (const [k, v] of Object.entries(result.parameters)) {
        const val = Array.isArray(v) ? v.join(', ') : v;
        html += `<div class="param-item"><div class="pk">${k}</div><div class="pv">${val}</div></div>`;
    }
    html += '</div></div>';

    // Plots
    if (result.plots && result.plots.length > 0) {
        html += `<div class="card"><div class="card-title">📈 Evaluation Plots</div>
            <div class="plots-grid">`;
        result.plots.forEach(p => {
            html += `<div class="plot-container" onclick="openLightbox(this)">
                <img src="${p.image}" alt="${p.title}" loading="lazy">
                <div class="plot-title">${p.title}</div>
                <div class="plot-explanation">${p.explanation || ''}</div>
            </div>`;
        });
        html += '</div></div>';
    }

    area.innerHTML = html;

    // Update auto-summary for step 5
    renderAutoSummary();
}

// ── Auto Summary ────────────────────────────────
function renderAutoSummary() {
    const summary = document.getElementById('auto-summary');
    if (!trainResult || !selectedDataset) return;

    const metricsHtml = Object.entries(trainResult.metrics)
        .map(([k, v]) => `<li><strong>${k}:</strong> ${v.value}</li>`).join('');

    summary.innerHTML = `
        <h3>🤖 Auto-Generated Summary</h3>
        <ul>
            <li><strong>Dataset:</strong> ${selectedDataset.name} (${edaData.info.rows.toLocaleString()} rows × ${edaData.info.columns} columns)</li>
            <li><strong>Task:</strong> ${trainResult.task.charAt(0).toUpperCase() + trainResult.task.slice(1)}</li>
            <li><strong>Algorithm:</strong> ${trainResult.parameters.Algorithm || 'N/A'}</li>
            ${metricsHtml}
            <li><strong>Features Used:</strong> ${(trainResult.parameters['Features Used'] || []).join(', ')}</li>
        </ul>
    `;
}

// ── Download Report ─────────────────────────────
function downloadReport() {
    const notes = document.getElementById('comment-box').value;
    let report = '════════════════════════════════════════\n';
    report += '  DYNAMIC ML STUDIO — Analysis Report\n';
    report += '════════════════════════════════════════\n\n';
    report += `Date: ${new Date().toLocaleString()}\n`;
    report += `Dataset: ${selectedDataset?.name || 'N/A'}\n`;
    report += `Task: ${trainResult?.task || 'N/A'}\n\n`;

    if (trainResult) {
        report += '── Steps ──────────────────────────────\n';
        trainResult.steps.forEach(s => {
            report += `\n${s.title}\n${s.explanation}\n`;
            s.details.forEach(d => report += `  • ${d}\n`);
        });

        report += '\n── Metrics ────────────────────────────\n';
        for (const [k, v] of Object.entries(trainResult.metrics)) {
            report += `${k}: ${v.value}\n  → ${v.explanation}\n`;
        }

        report += '\n── Parameters ─────────────────────────\n';
        for (const [k, v] of Object.entries(trainResult.parameters)) {
            report += `${k}: ${Array.isArray(v) ? v.join(', ') : v}\n`;
        }
    }

    report += '\n── Your Notes ─────────────────────────\n';
    report += notes || '(No notes added)';
    report += '\n\n════════════════════════════════════════\n';

    const blob = new Blob([report], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `ml_report_${(selectedDataset?.name || 'analysis').replace('.csv', '')}_${Date.now()}.txt`;
    a.click();
    showToast('Report downloaded!');
}

function copyReport() {
    const notes = document.getElementById('comment-box').value;
    const summary = document.getElementById('auto-summary').innerText;
    const text = summary + '\n\nNotes:\n' + (notes || '(No notes)');
    navigator.clipboard.writeText(text).then(() => showToast('Copied to clipboard!'));
}

// ── Reset ───────────────────────────────────────
function resetAll() {
    selectedDataset = null;
    edaData = null;
    selectedTask = null;
    trainResult = null;
    document.querySelectorAll('.dataset-card').forEach(c => c.classList.remove('selected'));
    document.querySelectorAll('.task-card').forEach(c => c.classList.remove('selected'));
    document.getElementById('config-card').style.display = 'none';
    document.getElementById('results-area').innerHTML = '';
    document.getElementById('auto-summary').innerHTML = '';
    document.getElementById('comment-box').value = '';
    goToStep(1);
}

// ── Lightbox ────────────────────────────────────
function openLightbox(container) {
    const img = container.querySelector('img');
    document.getElementById('lightbox-img').src = img.src;
    document.getElementById('lightbox').classList.add('show');
}
function closeLightbox() { document.getElementById('lightbox').classList.remove('show'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ── Toast ───────────────────────────────────────
function showToast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 3000);
}

// ── Init ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', loadDatasets);
