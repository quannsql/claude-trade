const state = {
    currentSymbol: null,
    chartCleanups: [],
    lastUpdateAt: null,
    terminalAutoScroll: true,
};

const palette = {
    ink: '#171a20',
    muted: '#737b89',
    line: '#e4e7ec',
    cyan: '#0891b2',
    cyanSoft: 'rgba(8, 145, 178, 0.16)',
    green: '#15803d',
    greenSoft: 'rgba(21, 128, 61, 0.14)',
    red: '#dc2626',
    redSoft: 'rgba(220, 38, 38, 0.14)',
    violet: '#6d5bd0',
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function numberValue(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
}

function formatCurrency(value, digits = 2) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    }).format(numberValue(value));
}

function formatNumber(value, digits = 2) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    }).format(numberValue(value));
}

function formatPercent(value) {
    return `${formatNumber(value, 2)}%`;
}

function formatAddress(address) {
    if (!address) return 'Not configured';
    if (address.length <= 14) return address;
    return `${address.slice(0, 6)}…${address.slice(-6)}`;
}

function formatDateTime(value) {
    if (!value) return '--';
    const date = typeof value === 'number' ? new Date(value) : new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString('en-GB', {
        day: '2-digit',
        month: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
    });
}

function setText(id, value) {
    const element = $(id);
    if (element) element.textContent = value;
}

function setSignedClass(element, value) {
    if (!element) return;
    element.classList.toggle('positive-text', value > 0);
    element.classList.toggle('negative-text', value < 0);
}

function sideBadge(isLong, label) {
    const cls = isLong === null ? 'badge-neutral' : isLong ? 'badge-long' : 'badge-short';
    return `<span class="badge ${cls}">${escapeHtml(label)}</span>`;
}

function tableEmpty(tbody, colspan, message) {
    tbody.innerHTML = `<tr class="muted-row"><td colspan="${colspan}">${escapeHtml(message)}</td></tr>`;
}

function fetchJson(url) {
    return fetch(url, { cache: 'no-store' }).then((response) => {
        if (!response.ok) {
            throw new Error(`Request failed: ${response.status}`);
        }
        return response.json();
    });
}

function setStatus(online, message) {
    const pill = $('status-pill');
    if (pill) pill.classList.toggle('offline', !online);
    setText('live-status', message);
}

/* ─────────── Live state (positions / orders / fills / margin) ─────────── */

function renderPositions(positions) {
    const tbody = $('positions-tbody');
    if (!tbody) return;

    setText('positions-count', String(positions?.length ?? 0));

    if (!positions || positions.length === 0) {
        tableEmpty(tbody, 6, 'No open positions');
        return;
    }

    tbody.innerHTML = positions.map((posInfo) => {
        const p = posInfo.position || posInfo;
        const szi = numberValue(p.szi);
        const isLong = szi > 0;
        const upnl = numberValue(p.unrealizedPnl);
        const upnlClass = upnl > 0 ? 'buy-text' : upnl < 0 ? 'sell-text' : '';
        return `
            <tr>
                <td><strong>${escapeHtml(p.coin)}</strong></td>
                <td>${sideBadge(isLong, isLong ? 'Long' : 'Short')}</td>
                <td>${escapeHtml(p.szi)}</td>
                <td>${formatNumber(p.entryPx, 4)}</td>
                <td>${formatNumber(p.markPx, 4)}</td>
                <td class="${upnlClass}">${upnl > 0 ? '+' : ''}${formatCurrency(upnl, 4)}</td>
            </tr>
        `;
    }).join('');
}

function renderOrders(orders) {
    const tbody = $('orders-tbody');
    if (!tbody) return;

    setText('orders-count', String(orders?.length ?? 0));

    if (!orders || orders.length === 0) {
        tableEmpty(tbody, 4, 'No active orders');
        return;
    }

    tbody.innerHTML = orders.map((order) => {
        const isBuy = order.side === 'B' || String(order.side).toLowerCase().includes('buy');
        return `
            <tr>
                <td><strong>${escapeHtml(order.coin)}</strong></td>
                <td>${sideBadge(isBuy, isBuy ? 'Long' : 'Short')}</td>
                <td>${escapeHtml(order.limitPx ?? order.px ?? '--')}</td>
                <td>${escapeHtml(order.sz ?? '--')}</td>
            </tr>
        `;
    }).join('');
}

function renderFills(fills) {
    const tbody = $('history-tbody');
    if (!tbody) return;

    if (!fills || fills.length === 0) {
        tableEmpty(tbody, 5, 'No live fills yet');
        return;
    }

    tbody.innerHTML = fills.slice(0, 50).map((fill) => {
        const fee = numberValue(fill.fee);
        const pnl = numberValue(fill.closedPnl) - fee; // net sau phí
        const pnlClass = pnl > 0 ? 'buy-text' : pnl < 0 ? 'sell-text' : '';
        const action = String(fill.dir || fill.side || '--');
        const isLong = action.toLowerCase().includes('long');
        return `
            <tr>
                <td>${formatDateTime(fill.time)}</td>
                <td><strong>${escapeHtml(fill.coin ?? '--')}</strong></td>
                <td>${sideBadge(isLong, action)}</td>
                <td>${escapeHtml(fill.px ?? '--')}</td>
                <td class="${pnlClass}">${pnl > 0 ? '+' : ''}${formatNumber(pnl, 4)}</td>
            </tr>
        `;
    }).join('');
}

function renderMarginGauge(marginInfo) {
    const bar = $('margin-usage-bar');
    const text = $('margin-usage-text');
    if (!bar || !text) return;

    const total = numberValue(marginInfo.accountValue);
    const used = numberValue(marginInfo.totalMarginUsed);
    const pct = total > 0 ? Math.min((used / total) * 100, 100) : 0;

    bar.style.width = `${pct.toFixed(1)}%`;
    bar.classList.toggle('warn', pct >= 50 && pct < 70);
    bar.classList.toggle('danger', pct >= 70);
    text.textContent = `${pct.toFixed(1)}%`;
}

async function updateState() {
    try {
        const data = await fetchJson('/api/state');

        if (data.error) {
            setStatus(false, data.error);
            renderOrders([]);
            renderFills([]);
            renderPositions([]);
            return;
        }

        setStatus(true, 'Live feed');
        setText('wallet-address', formatAddress(data.address));

        const marginInfo = data.margin_summary || {};
        const totalValue = numberValue(marginInfo.accountValue);
        const marginUsed = numberValue(marginInfo.totalMarginUsed);
        const withdrawable = numberValue(marginInfo.withdrawable);
        const available = withdrawable || Math.max(totalValue - marginUsed, 0);

        setText('val-total', formatCurrency(totalValue));
        setText('val-margin', formatCurrency(available));
        renderMarginGauge(marginInfo);
        renderOrders(data.open_orders || []);
        renderFills(data.fills || []);
        renderPositions(data.positions || []);

        state.lastUpdateAt = Date.now();
    } catch (error) {
        setStatus(false, 'API offline');
        console.error('Failed to fetch state:', error);
    }
}

/* ─────────── Charts ─────────── */

function clearCharts() {
    state.chartCleanups.forEach((cleanup) => cleanup());
    state.chartCleanups = [];
}

function makeGradient(ctx, hexSoft) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 240);
    gradient.addColorStop(0, hexSoft);
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
    return gradient;
}

function renderCharts(series) {
    clearCharts();

    if (!window.Chart) {
        ['equity-chart', 'drawdown-chart', 'pnl-chart', 'cum-pnl-chart'].forEach((id) => {
            const container = $(id);
            if (container) {
                container.innerHTML = '<div class="empty-state">Chart library unavailable.</div>';
            }
        });
        return;
    }

    const getLabels = (data) => (data || []).map(d => formatDateTime(d.time * 1000));
    const getValues = (data) => (data || []).map(d => d.value);

    const tooltipStyle = {
        backgroundColor: 'rgba(23, 26, 32, 0.94)',
        titleColor: '#9aa3b2',
        bodyColor: '#f5f7fb',
        titleFont: { family: '"JetBrains Mono", monospace', size: 10 },
        bodyFont: { family: '"JetBrains Mono", monospace', size: 12, weight: '700' },
        padding: 10,
        cornerRadius: 8,
        displayColors: false,
        caretSize: 5,
    };

    const baseOptions = (format) => ({
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 500, easing: 'easeOutQuart' },
        plugins: {
            legend: { display: false },
            tooltip: {
                ...tooltipStyle,
                callbacks: {
                    label: (item) => format(item.parsed.y),
                },
            },
        },
        scales: {
            x: { display: false },
            y: {
                position: 'right',
                grid: { color: 'rgba(228, 231, 236, 0.55)', drawTicks: false },
                border: { display: false },
                ticks: {
                    color: palette.muted,
                    padding: 6,
                    maxTicksLimit: 5,
                    font: { family: '"JetBrains Mono", monospace', size: 10 },
                    callback: (value) => format(value),
                },
            },
        },
        interaction: { intersect: false, mode: 'index' },
    });

    const fmtUsd = (v) => `$${formatNumber(v, 2)}`;
    const fmtPct = (v) => `${formatNumber(v, 2)}%`;

    const makeChartJs = (containerId, type, buildData, options) => {
        const container = $(containerId);
        if (!container) return;
        container.innerHTML = `<canvas id="${containerId}-canvas"></canvas>`;
        const ctx = document.getElementById(`${containerId}-canvas`).getContext('2d');

        const chart = new Chart(ctx, {
            type,
            data: buildData(ctx),
            options,
        });
        state.chartCleanups.push(() => chart.destroy());
    };

    makeChartJs('equity-chart', 'line', (ctx) => ({
        labels: getLabels(series.equity),
        datasets: [{
            data: getValues(series.equity),
            borderColor: palette.cyan,
            backgroundColor: makeGradient(ctx, palette.cyanSoft),
            borderWidth: 2,
            fill: true,
            tension: 0.25,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: palette.cyan,
        }],
    }), baseOptions(fmtUsd));

    makeChartJs('drawdown-chart', 'line', (ctx) => ({
        labels: getLabels(series.drawdown),
        datasets: [{
            data: getValues(series.drawdown),
            borderColor: palette.red,
            backgroundColor: makeGradient(ctx, palette.redSoft),
            borderWidth: 2,
            fill: true,
            tension: 0.25,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: palette.red,
        }],
    }), baseOptions(fmtPct));

    const pnlData = series.pnl || [];
    makeChartJs('pnl-chart', 'bar', () => ({
        labels: getLabels(pnlData),
        datasets: [{
            data: getValues(pnlData),
            backgroundColor: pnlData.map(d => d.color || palette.cyan),
            borderRadius: 3,
            maxBarThickness: 14,
        }],
    }), baseOptions(fmtUsd));

    makeChartJs('cum-pnl-chart', 'line', (ctx) => ({
        labels: getLabels(series.cum_pnl),
        datasets: [{
            data: getValues(series.cum_pnl),
            borderColor: palette.green,
            backgroundColor: makeGradient(ctx, palette.greenSoft),
            borderWidth: 2,
            fill: true,
            tension: 0.25,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: palette.green,
        }],
    }), baseOptions(fmtUsd));
}

/* ─────────── Metrics ─────────── */

function renderMetrics(metrics) {
    const pnl = numberValue(metrics.net_pnl_usd);
    const pnlElement = $('metric-pnl');
    const pnlCard = pnlElement?.closest('.metric-card');
    setText('metric-pnl', formatCurrency(pnl, 4));
    setSignedClass(pnlElement, pnl);
    pnlCard?.classList.toggle('positive', pnl > 0);
    pnlCard?.classList.toggle('negative', pnl < 0);

    setText('metric-final-equity', `Final equity ${formatCurrency(metrics.final_equity_usd, 4)}`);
    setText('metric-win-rate', formatPercent(metrics.win_rate_pct));
    setText('metric-trades', `${metrics.total_trades || 0} trades · ${formatNumber(metrics.trades_per_day, 1)}/day`);
    setText('metric-drawdown', formatPercent(metrics.max_drawdown_pct));
    setText('metric-profit-factor', metrics.profit_factor === null || metrics.profit_factor === undefined ? '--' : formatNumber(metrics.profit_factor, 2));
    setText('metric-expectancy', `Expectancy ${formatCurrency(metrics.expectancy_usd, 4)}`);

    // Fee metrics — fee ăn vào PnL bao nhiêu
    const fees = numberValue(metrics.total_fees_usd);
    setText('metric-fees', formatCurrency(fees, 2));
    const gross = pnl + fees; // gross trước phí
    const feesElement = $('metric-fees');
    if (fees > 0 && gross > 0) {
        setText('metric-fee-ratio', `${formatNumber((fees / gross) * 100, 0)}% of gross`);
    } else if (fees > 0) {
        setText('metric-fee-ratio', 'gross ≤ 0');
    } else {
        setText('metric-fee-ratio', '-- of gross');
    }
    setSignedClass(feesElement, fees > 0 ? -1 : 0);
}

/* ─────────── Backtest / LIVE table ─────────── */

function renderBacktestTable(trades) {
    const tbody = $('backtest-tbody');
    if (!tbody) return;

    if (!trades || trades.length === 0) {
        tableEmpty(tbody, 8, 'No trades found');
        return;
    }

    tbody.innerHTML = trades.map((trade) => {
        const pnl = numberValue(trade.net_pnl);
        const pnlClass = pnl > 0 ? 'buy-text' : pnl < 0 ? 'sell-text' : '';
        const direction = String(trade.direction || '--');
        const isLong = direction.toLowerCase().includes('long');
        return `
            <tr>
                <td>${trade.index}</td>
                <td>${sideBadge(isLong, direction)}</td>
                <td>${escapeHtml(trade.exit_reason || '--')}</td>
                <td>${escapeHtml(trade.score ?? '--')}</td>
                <td>${formatNumber(trade.entry_price, 2)}</td>
                <td>${formatNumber(trade.exit_price, 2)}</td>
                <td class="${pnlClass}">${pnl > 0 ? '+' : ''}${formatCurrency(pnl, 4)}</td>
                <td>${formatCurrency(trade.equity_after, 4)}</td>
            </tr>
        `;
    }).join('');
}

function populateSymbolSelect(symbols, selected) {
    const select = $('symbol-select');
    if (!select) return;

    let list = symbols || [];
    if (!list.includes('LIVE')) list = ['LIVE', ...list];

    const current = list.map(s => `sym:${s}`).join('|');
    if (select.dataset.loaded === current) {
        if (select.value !== selected) select.value = selected;
        return;
    }

    select.innerHTML = list
        .map(s => `<option value="${escapeHtml(s)}">${escapeHtml(s === 'LIVE' ? 'LIVE' : s)}</option>`)
        .join('');
    select.dataset.loaded = current;
    if (selected) select.value = selected;
}

async function updateBacktest(symbol = state.currentSymbol) {
    try {
        const isLive = symbol === 'LIVE' || !symbol;
        const endpoint = isLive ? '/api/live_chart' : '/api/backtest';
        const url = new URL(endpoint, window.location.origin);
        if (symbol && symbol !== 'LIVE') url.searchParams.set('symbol', symbol);

        const data = await fetchJson(url.toString());
        state.currentSymbol = data.selected_symbol || 'LIVE';

        populateSymbolSelect(data.symbols || [], state.currentSymbol);
        renderMetrics(data.metrics || {});
        renderCharts(data.series || {});
        renderBacktestTable(data.trades || []);

        const isLiveMode = state.currentSymbol === 'LIVE';
        setText('equity-note', isLiveMode ? 'Live database' : 'CSV backtest');

        const tradesTitle = document.querySelector('#trades .panel-header strong');
        if (tradesTitle) {
            tradesTitle.textContent = isLiveMode ? 'Recent Live Fills' : 'Recent Simulated Trades';
        }

        const tradesSpan = document.querySelector('#trades .panel-header span');
        if (tradesSpan) {
            tradesSpan.textContent = isLiveMode ? 'Live' : 'Backtest';
        }

        state.lastUpdateAt = Date.now();
    } catch (error) {
        console.error('Failed to fetch backtest:', error);
        setStatus(false, 'API offline');
    }
}

/* ─────────── "Updated Xs ago" ticker ─────────── */

function tickLastUpdate() {
    if (!state.lastUpdateAt) return;
    const seconds = Math.max(0, Math.round((Date.now() - state.lastUpdateAt) / 1000));
    const label = seconds < 3 ? 'just now'
        : seconds < 60 ? `${seconds}s ago`
        : `${Math.floor(seconds / 60)}m ${seconds % 60}s ago`;
    setText('last-update', label);
}

/* ─────────── Terminal ─────────── */

function initTerminal() {
    const terminal = $('terminal');
    if (!terminal) return;

    // Autoscroll thông minh: dừng khi user cuộn lên đọc log cũ
    terminal.addEventListener('scroll', () => {
        const nearBottom = terminal.scrollHeight - terminal.scrollTop - terminal.clientHeight < 40;
        state.terminalAutoScroll = nearBottom;
    });

    const clearButton = $('terminal-clear');
    if (clearButton) {
        clearButton.addEventListener('click', () => {
            terminal.innerHTML = '';
            state.terminalAutoScroll = true;
        });
    }

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/logs`);

    ws.onopen = () => {
        appendTerminal('Connected to log stream.', 'ok');
        window.setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) ws.send('ping');
        }, 30000);
    };

    ws.onmessage = (event) => appendTerminal(event.data);
    ws.onerror = () => appendTerminal('Log stream error.', 'error');
    ws.onclose = () => appendTerminal('Log stream disconnected.', 'error');
}

function appendTerminal(message, forcedClass = '') {
    const terminal = $('terminal');
    if (!terminal) return;

    const div = document.createElement('div');
    const match = String(message).match(/^(\d{2}:\d{2}:\d{2})\s-\s(.*)/);
    const body = match ? match[2] : String(message);
    let className = forcedClass;

    if (!className) {
        if (/(error|failed|reject|exception|traceback|insufficient|invalid|orphan)/i.test(body)) {
            className = 'error';
        } else if (/(warning|timeout|cancel|skip|block|cooldown|circuit)/i.test(body)) {
            className = 'warn';
        } else if (/(connected|filled|closed|saved|started|ready|placed|recorded|win=true)/i.test(body)) {
            className = 'ok';
        }
    }

    if (className) div.classList.add(className);

    if (match) {
        div.innerHTML = `<span class="time">${escapeHtml(match[1])}</span>${escapeHtml(body)}`;
    } else {
        div.textContent = body;
    }

    terminal.appendChild(div);
    while (terminal.childElementCount > 140) {
        terminal.removeChild(terminal.firstElementChild);
    }
    if (state.terminalAutoScroll) {
        terminal.scrollTop = terminal.scrollHeight;
    }
}

/* ─────────── Boot ─────────── */

function bindEvents() {
    const select = $('symbol-select');
    if (select) {
        select.addEventListener('change', (event) => {
            state.currentSymbol = event.target.value;
            updateBacktest(state.currentSymbol);
        });
    }

    const refreshButton = $('refresh-button');
    if (refreshButton) {
        refreshButton.addEventListener('click', () => {
            updateState();
            updateBacktest(state.currentSymbol);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    bindEvents();
    initTerminal();
    updateState();
    updateBacktest();
    window.setInterval(updateState, 5000);
    window.setInterval(() => updateBacktest(state.currentSymbol), 30000);
    window.setInterval(tickLastUpdate, 1000);
});
