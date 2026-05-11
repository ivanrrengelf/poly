document.addEventListener("DOMContentLoaded", async () => {

    // Configuración del gráfico (Chart.js)
    Chart.defaults.color = '#9ca3af';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    let portfolioChart = null;

    async function fetchSimulationData() {
        try {
            const response = await fetch('/api/simulation');
            if (!response.ok) throw new Error('Error fetching data');
            const data = await response.json();

            if (data.error) {
                console.error(data.error);
                alert("Error en simulación: " + data.error);
                return;
            }

            // Ocultar loader y mostrar contenido
            document.getElementById('loader').classList.add('hidden');
            document.getElementById('content').classList.remove('hidden');

            updateMetrics(data.metrics);
            updateChart(data.chart_data);
            updateTable(data.recent_trades);
            updateOpenPositions(data.open_positions || []);
            updateJournal(data.trade_journal || []);

        } catch (error) {
            console.error(error);
            document.getElementById('loader').innerHTML = `<p style="color:var(--loss-color)">Error conectando con el backend.</p>`;
        }
    }

    function formatCurrency(value) {
        return new Intl.NumberFormat('es-ES', { style: 'currency', currency: 'EUR' }).format(value);
    }

    function getMarketLabel(trade) {
        return trade.market ?? trade.question ?? trade.market_id ?? 'Unknown market';
    }

    function updateMetrics(metrics) {
        const formatPct = (val) => (val > 0 ? '+' : '') + val.toFixed(2) + '%';

        document.getElementById('val-capital').innerText = formatCurrency(metrics.final_capital);
        document.getElementById('val-initial').innerText = formatCurrency(metrics.initial_capital);
        document.getElementById('val-winrate').innerText = metrics.win_rate.toFixed(1) + '%';
        document.getElementById('val-trades').innerText = metrics.closed_trades ?? metrics.total_trades;

        const roiBadge = document.getElementById('val-roi');
        roiBadge.innerText = formatPct(metrics.roi_pct);
        if (metrics.roi_pct >= 0) {
            roiBadge.className = 'badge positive';
        } else {
            roiBadge.className = 'badge negative';
        }
    }

    function updateChart(chartData) {
        const ctx = document.getElementById('portfolioChart').getContext('2d');

        const labels = chartData.map(d => {
            const date = new Date(d.time);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        });
        const values = chartData.map(d => d.value);

        // Crear gradiente para el gráfico
        let gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(0, 240, 255, 0.5)');
        gradient.addColorStop(1, 'rgba(0, 240, 255, 0.0)');

        if (portfolioChart) portfolioChart.destroy();

        portfolioChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Portfolio Value ($)',
                    data: values,
                    borderColor: '#00f0ff',
                    backgroundColor: gradient,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(10, 12, 18, 0.9)',
                        titleColor: '#fff',
                        bodyColor: '#00f0ff',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        padding: 10
                    }
                },
                scales: {
                    x: {
                        grid: { display: false, drawBorder: false },
                        ticks: { maxTicksLimit: 8 }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
                        ticks: {
                            callback: function (value) {
                                return '$' + value;
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    function updateTable(trades) {
        const tbody = document.getElementById('trades-body');
        tbody.innerHTML = '';

        trades.slice().reverse().forEach(trade => {
            const tr = document.createElement('tr');

            const entryDate = new Date(trade.entry_time ?? trade.time);
            const exitDate = new Date(trade.exit_time ?? trade.time);
            const entryStr = entryDate.toLocaleDateString() + ' ' + entryDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const exitStr = exitDate.toLocaleDateString() + ' ' + exitDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            const statusClass = trade.status && trade.status.startsWith('WIN') ? 'status-win' : 'status-loss';
            const pnlSign = trade.pnl >= 0 ? '+' : '';
            const marketLabel = getMarketLabel(trade);

            tr.innerHTML = `
                <td>${entryStr}</td>
                <td>${exitStr}</td>
                <td style="max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${marketLabel}">${marketLabel}</td>
                <td>${((trade.probability ?? trade.prob) * 100).toFixed(1)}%</td>
                <td>${formatCurrency(trade.entry_price)}</td>
                <td>${formatCurrency(trade.exit_price)}</td>
                <td>${formatCurrency(trade.bet_size)}</td>
                <td class="${statusClass}">${pnlSign}${formatCurrency(trade.pnl)}</td>
                <td>${trade.holding_period ?? '-'}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    function updateOpenPositions(positions) {
        const tbody = document.getElementById('open-positions-body');
        if (!tbody) return;

        tbody.innerHTML = '';

        if (!positions.length) {
            const tr = document.createElement('tr');
            tr.innerHTML = '<td colspan="6" class="empty-state">No open positions</td>';
            tbody.appendChild(tr);
            return;
        }

        positions.slice().reverse().forEach(position => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${position.trade_id}</td>
                <td style="max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${position.question}">${position.question}</td>
                <td>${formatCurrency(position.entry_price)}</td>
                <td>${formatCurrency(position.bet_size)}</td>
                <td>${(position.probability * 100).toFixed(1)}%</td>
                <td>${position.edge.toFixed(4)}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    function updateJournal(journal) {
        const tbody = document.getElementById('journal-body');
        if (!tbody) return;

        tbody.innerHTML = '';

        journal.slice().reverse().forEach(trade => {
            const tr = document.createElement('tr');
            const marketLabel = getMarketLabel(trade);
            tr.innerHTML = `
                <td>${trade.trade_id}</td>
                <td>${trade.market_id ?? 'Unknown'}</td>
                <td style="max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${marketLabel}">${marketLabel}</td>
                <td>${trade.status}</td>
                <td>${formatCurrency(trade.entry_price)}</td>
                <td>${formatCurrency(trade.exit_price)}</td>
                <td>${formatCurrency(trade.pnl)}</td>
                <td>${trade.holding_period}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Inicializar
    fetchSimulationData();

});
