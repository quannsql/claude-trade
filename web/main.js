const state = {
    currentSymbol: null,
    chartCleanups: [],
};

const palette = {
    ink: '#171a20',
    muted: '#737b89',
    line: '#e4e7ec',
    cyan: '#0891b2',
    cyanSoft: 'rgba(8, 145, 178, 0.18)',
    green: '#15803d',
    greenSoft: 'rgba(21, 128, 61, 0.16)',
    red: '#dc2626',
    redSoft: 'rgba(220, 38, 38, 0.16)',
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
    return `${address.slice(0, 6)}...${address.slice(-6)}`;
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

function renderStorage(storage) {
    const backend = storage?.backend || storage?.storage || 'sqlite';
    const ready = storage?.ready ?? storage?.storage_ready;
    const text = ready === false ? `${backend} unavailable` : `${backend} ready`;
    setText('storage-status', text);
}

function renderOrders(orders) {
    const tbody = $('orders-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!orders || orders.length === 0) {
        tableEmpty(tbody, 4, 'No active orders');
        return;
    }

    tbody.innerHTML = orders.map((order) => {
        const isBuy = order.side === 'B' || String(order.side).toLowerCase().includes('buy');
        const sideText = isBuy ? 'LONG' : 'SHORT';
        const sideClass = isBuy ? 'side-buy' : 'side-sell';
        return `
            <tr>
                <td>${escapeHtml(order.coin)}</td>
                <td class="${sideClass}">${sideText}</td>
                <td>${escapeHtml(order.limitPx ?? order.px ?? '--')}</td>
                <td>${escapeHtml(order.sz ?? '--')}</td>
            </tr>
        `;
    }).join('');
}

function renderFills(fills) {
    const tbody = $('history-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!fills || fills.length === 0) {
        tableEmpty(tbody, 5, 'No live fills yet');
        return;
    }

    tbody.innerHTML = fills.slice(0, 50).map((fill) => {
        const pnl = numberValue(fill.closedPnl);
        const pnlClass = pnl > 0 ? 'buy-text' : pnl < 0 ? 'sell-text' : '';
        const action = fill.dir || fill.side || '--';
        const actionClass = String(action).toLowerCase().includes('long') ? 'buy-text' : 'sell-text';
        return `
            <tr>
                <td>${formatDateTime(fill.time)}</td>
                <td>${escapeHtml(fill.coin ?? '--')}</td>
                <td class="${actionClass}">${escapeHtml(action)}</td>
                <td>${escapeHtml(fill.px ?? '--')}</td>
                <td class="${pnlClass}">${pnl > 0 ? '+' : ''}${formatNumber(pnl, 4)}</td>
            </tr>
        `;
    }).join('');
}

async function updateState() {
    try {
        const data = await fetchJson('/api/state');

        if (data.error) {
            setText('live-status', data.error);
            if (data.storage) renderStorage(data.storage);
            renderOrders([]);
            renderFills([]);
            return;
        }

        setText('live-status', 'Live feed');
        setText('wallet-address', formatAddress(data.address));

        const marginInfo = data.margin_summary || {};
        const totalValue = numberValue(marginInfo.accountValue);
        const marginUsed = numberValue(marginInfo.totalMarginUsed);
        const withdrawable = numberValue(marginInfo.withdrawable);
        const available = withdrawable || Math.max(totalValue - marginUsed, 0);

        setText('val-total', formatCurrency(totalValue));
        setText('val-margin', formatCurrency(available));
        renderOrders(data.open_orders || []);
        renderFills(data.fills || []);
        renderStorage(data.storage);
    } catch (error) {
        setText('live-status', 'API offline');
        console.error('Failed to fetch state:', error);
    }
}

function chartOptions(container) {
    return {
        width: Math.max(container.clientWidth, 320),
        height: Math.max(container.clientHeight, 220),
        layout: {
            background: { type: 'solid', color: 'transparent' },
            textColor: palette.muted,
            fontFamily: '"Google Sans", Arial, sans-serif',
            fontSize: 12,
        },
        grid: {
            vertLines: { color: 'rgba(115, 123, 137, 0.12)' },
            horzLines: { color: 'rgba(115, 123, 137, 0.12)' },
        },
        rightPriceScale: {
            borderColor: palette.line,
        },
        timeScale: {
            borderColor: palette.line,
            timeVisible: true,
            secondsVisible: false,
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
    };
}

function clearCharts() {
    state.chartCleanups.forEach((cleanup) => cleanup());
    state.chartCleanups = [];
}

function makeChart(containerId) {
    const container = $(containerId);
    if (!container || !window.LightweightCharts) return null;
    container.innerHTML = '';

    const chart = LightweightCharts.createChart(container, chartOptions(container));
    const resizeObserver = new ResizeObserver((entries) => {
        if (!entries.length) return;
        const rect = entries[0].contentRect;
        chart.applyOptions({
            width: Math.max(Math.floor(rect.width), 320),
            height: Math.max(Math.floor(rect.height), 220),
        });
    });
    resizeObserver.observe(container);

    state.chartCleanups.push(() => {
        resizeObserver.disconnect();
        chart.remove();
    });

    return chart;
}

function renderCharts(series) {
    clearCharts();

    if (!window.LightweightCharts) {
        ['equity-chart', 'drawdown-chart', 'pnl-chart'].forEach((id) => {
            const container = $(id);
            if (container) {
                container.innerHTML = '<div class="empty-state">Chart library unavailable.</div>';
            }
        });
        return;
    }

    const equityChart = makeChart('equity-chart');
    if (equityChart) {
        const equitySeries = equityChart.addAreaSeries({
            lineColor: palette.cyan,
            topColor: palette.cyanSoft,
            bottomColor: 'rgba(8, 145, 178, 0.01)',
            lineWidth: 2,
            priceFormat: { type: 'price', precision: 4, minMove: 0.0001 },
        });
        equitySeries.setData(series.equity || []);
        equityChart.timeScale().fitContent();
    }

    const drawdownChart = makeChart('drawdown-chart');
    if (drawdownChart) {
        const drawdownSeries = drawdownChart.addAreaSeries({
            lineColor: palette.red,
            topColor: 'rgba(220, 38, 38, 0.02)',
            bottomColor: palette.redSoft,
            lineWidth: 2,
            priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
        });
        drawdownSeries.setData(series.drawdown || []);
        drawdownChart.timeScale().fitContent();
    }

    const pnlChart = makeChart('pnl-chart');
    if (pnlChart) {
        const pnlSeries = pnlChart.addHistogramSeries({
            priceFormat: { type: 'price', precision: 4, minMove: 0.0001 },
        });
        pnlSeries.setData(series.pnl || []);
        pnlChart.timeScale().fitContent();
    }

    const cumPnlChart = makeChart('cum-pnl-chart');
    if (cumPnlChart) {
        const cumPnlSeries = cumPnlChart.addAreaSeries({
            lineColor: palette.green,
            topColor: palette.greenSoft,
            bottomColor: 'rgba(21, 128, 61, 0.01)',
            lineWidth: 2,
            priceFormat: { type: 'price', precision: 4, minMove: 0.0001 },
        });
        cumPnlSeries.setData(series.cum_pnl || []);
        cumPnlChart.timeScale().fitContent();
    }
}

function renderSymbolOptions(symbols, selected) {
    const select = $('symbol-select');
    if (!select) return;

    if (!symbols || symbols.length === 0) {
        select.innerHTML = '<option>No results</option>';
        select.disabled = true;
        return;
    }

    select.disabled = false;
    select.innerHTML = symbols.map((symbol) => (
        `<option value="${escapeHtml(symbol)}"${symbol === selected ? ' selected' : ''}>${escapeHtml(symbol)}</option>`
    )).join('');
}

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
    setText('metric-trades', `${metrics.total_trades || 0} trades`);
    setText('metric-drawdown', formatPercent(metrics.max_drawdown_pct));
    setText('metric-profit-factor', metrics.profit_factor === null ? '--' : formatNumber(metrics.profit_factor, 2));
    setText('metric-expectancy', `Expectancy ${formatCurrency(metrics.expectancy_usd, 4)}`);
}

function renderExitReasons(counts) {
    const container = $('exit-reasons');
    if (!container) return;

    const entries = Object.entries(counts || {});
    const total = entries.reduce((sum, [, count]) => sum + Number(count), 0);
    if (!entries.length || total === 0) {
        container.innerHTML = '<div class="empty-state">No exit reason data yet.</div>';
        return;
    }

    container.innerHTML = entries.map(([reason, count]) => {
        const percent = (Number(count) / total) * 100;
        return `
            <div class="reason-row">
                <header>
                    <span>${escapeHtml(reason)}</span>
                    <span>${count} / ${formatNumber(percent, 1)}%</span>
                </header>
                <div class="reason-track">
                    <div class="reason-fill" style="width:${Math.max(percent, 3)}%"></div>
                </div>
            </div>
        `;
    }).join('');
}

function renderBacktestTable(trades) {
    const tbody = $('backtest-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!trades || trades.length === 0) {
        tableEmpty(tbody, 8, 'No backtest trades found');
        return;
    }

    tbody.innerHTML = trades.map((trade) => {
        const pnl = numberValue(trade.net_pnl);
        const pnlClass = pnl > 0 ? 'buy-text' : pnl < 0 ? 'sell-text' : '';
        const sideClass = String(trade.direction).toLowerCase() === 'long' ? 'buy-text' : 'sell-text';
        return `
            <tr>
                <td>${trade.index}</td>
                <td class="${sideClass}">${escapeHtml(trade.direction || '--')}</td>
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

function renderReportImage(chartImage) {
    const image = $('result-chart-image');
    const empty = $('report-empty');
    if (!image || !empty) return;

    if (chartImage) {
        image.src = `${chartImage}?v=${Date.now()}`;
        image.style.display = 'block';
        empty.style.display = 'none';
        return;
    }

    image.removeAttribute('src');
    image.style.display = 'none';
    empty.style.display = 'block';
}

async function updateBacktest(symbol = state.currentSymbol) {
    try {
        const isLive = symbol === 'LIVE' || !symbol;
        const endpoint = isLive ? '/api/live_chart' : '/api/backtest';
        const url = new URL(endpoint, window.location.origin);
        if (symbol && symbol !== 'LIVE') url.searchParams.set('symbol', symbol);

        const data = await fetchJson(url.toString());
        state.currentSymbol = data.selected_symbol || 'LIVE';
        
        let symbolsList = data.symbols || [];
        if (!symbolsList.includes('LIVE')) {
            symbolsList = ['LIVE', ...symbolsList];
        }

        renderSymbolOptions(symbolsList, state.currentSymbol);
        renderMetrics(data.metrics || {});
        renderCharts(data.series || {});
        renderExitReasons(data.exit_reason_counts || {});
        renderBacktestTable(data.trades || []);
        renderReportImage(data.chart_image);

        const isLiveMode = state.currentSymbol === 'LIVE';
        setText('chart-source', isLiveMode ? 'Live Database' : (data.selected_symbol ? `${data.selected_symbol}_trades.csv` : 'No result CSV'));
        setText('equity-note', isLiveMode ? 'Live Database' : 'CSV backtest');
        
        const tradesTitle = document.querySelector('#trades .panel-header strong');
        if (tradesTitle) {
            tradesTitle.textContent = isLiveMode ? 'Recent Live Fills' : 'Recent Simulated Trades';
        }
        
        const tradesSpan = document.querySelector('#trades .panel-header span');
        if (tradesSpan) {
            tradesSpan.textContent = isLiveMode ? 'Live' : 'Backtest';
        }

        setText('last-update', new Date().toLocaleTimeString('en-GB'));
    } catch (error) {
        console.error('Failed to fetch backtest:', error);
        setText('chart-source', 'API offline');
    }
}

function initTerminal() {
    const terminal = $('terminal');
    if (!terminal) return;

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
    const lowered = body.toLowerCase();
    let className = forcedClass;

    if (!className) {
        if (/(error|failed|reject|exception|traceback|insufficient|invalid)/i.test(lowered)) {
            className = 'error';
        } else if (/(connected|filled|closed|saved|started|ready|placed)/i.test(lowered)) {
            className = 'ok';
        }
    }

    if (className) div.classList.add(className);

    if (match) {
        div.innerHTML = `<span class="time">[${escapeHtml(match[1])}]</span>${escapeHtml(body)}`;
    } else {
        div.textContent = body;
    }

    terminal.appendChild(div);
    while (terminal.childElementCount > 140) {
        terminal.removeChild(terminal.firstElementChild);
    }
    terminal.scrollTop = terminal.scrollHeight;
}

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
});
