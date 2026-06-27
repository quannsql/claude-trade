// Terminal WebSocket
const terminal = document.getElementById('terminal');
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/logs`);

ws.onmessage = function(event) {
    const msg = event.data;
    const div = document.createElement('div');
    
    // Parse the log string if it matches our formatter
    // Format: "HH:MM:SS - message"
    const match = msg.match(/^(\d{2}:\d{2}:\d{2})\s-\s(.*)/);
    if (match) {
        div.innerHTML = `<span class="time">[${match[1]}]</span> <span class="msg">${match[2]}</span>`;
        if (match[2].includes('❌')) {
            div.style.color = '#ff3366';
        } else if (match[2].includes('✅') || match[2].includes('🚀')) {
            div.style.color = '#00ff66';
        }
    } else {
        div.textContent = msg;
    }

    terminal.appendChild(div);
    // Auto scroll to bottom
    if (terminal.childElementCount > 100) {
        terminal.removeChild(terminal.firstChild);
    }
    terminal.scrollTop = terminal.scrollHeight;
};

// Fetch API State
async function updateState() {
    try {
        const res = await fetch('/api/state');
        const data = await res.json();
        
        if (data.error) {
            console.error(data.error);
            return;
        }

        document.getElementById('wallet-address').textContent = 
            data.address.substring(0, 6) + '...' + data.address.substring(38);

        // Update Balance
        const marginInfo = data.margin_summary || {};
        const totalVal = parseFloat(marginInfo.accountValue || 0).toFixed(2);
        const marginVal = parseFloat(marginInfo.totalMarginUsed || 0).toFixed(2);
        const avail = (parseFloat(totalVal) - parseFloat(marginVal)).toFixed(2);
        
        document.getElementById('val-total').textContent = `$${totalVal}`;
        document.getElementById('val-margin').textContent = `$${avail}`;

        // Update Orders
        const tbody = document.getElementById('orders-tbody');
        tbody.innerHTML = '';
        
        const orders = data.open_orders || [];
        if (orders.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color: #8b8b9a">No active orders</td></tr>';
        } else {
            orders.forEach(o => {
                const isBuy = o.side === 'B';
                const sideClass = isBuy ? 'buy-text' : 'sell-text';
                const sideText = isBuy ? 'LONG' : 'SHORT';
                
                tbody.innerHTML += `
                    <tr>
                        <td>${o.coin}</td>
                        <td class="${sideClass}">${sideText}</td>
                        <td>${o.limitPx}</td>
                        <td>${o.sz}</td>
                    </tr>
                `;
            });
        }
    } catch (err) {
        console.error("Failed to fetch state:", err);
    }
}

// Periodic updates (moved to top to prevent chart errors blocking it)
setInterval(updateState, 5000);
updateState();

try {
    // Lightweight Charts Mock Initialization
    const chartProperties = {
        layout: {
            background: { type: 'solid', color: '#111115' },
            textColor: '#8b8b9a',
        },
        grid: {
            vertLines: { color: '#22222a' },
            horzLines: { color: '#22222a' },
        },
        rightPriceScale: {
            borderVisible: false,
        },
        timeScale: {
            borderVisible: false,
            timeVisible: true,
            secondsVisible: false,
        },
    };

    const chartContainer = document.getElementById('chart-container');
    const chart = LightweightCharts.createChart(chartContainer, chartProperties);
    const areaSeries = chart.addAreaSeries({
        lineColor: '#00f0ff',
        topColor: 'rgba(0, 240, 255, 0.4)',
        bottomColor: 'rgba(0, 240, 255, 0.0)',
        lineWidth: 2,
    });

    // Mock Data for Equity Curve
    let mockEquity = 100;
    let currentTime = Math.floor(Date.now() / 1000) - 3600; // 1 hour ago
    const data = [];
    for(let i=0; i<60; i++) {
        mockEquity += (Math.random() - 0.45) * 2; // slightly upward bias
        data.push({ time: currentTime + i * 60, value: mockEquity });
    }
    areaSeries.setData(data);

    // Handle window resize
    new ResizeObserver(entries => {
        if (entries.length === 0 || entries[0].target !== chartContainer) { return; }
        const newRect = entries[0].contentRect;
        chart.applyOptions({ height: newRect.height, width: newRect.width });
    }).observe(chartContainer);
} catch (e) {
    console.error("Chart rendering error:", e);
}
