document.addEventListener("DOMContentLoaded", async () => {
    
    // Configuración del gráfico (Chart.js)
    Chart.defaults.color = '#9ca3af';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    let portfolioChart = null;

    async function fetchSimulationData() {
        try {
            const response = await fetch('/api/live');
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
            
            // Unir active y recent trades para que el usuario pueda ver lo que está pasando AHORA
            const allTrades = [...(data.active_trades || []), ...(data.recent_trades || [])];
            updateTable(allTrades);

        } catch (error) {
            console.error(error);
            document.getElementById('loader').innerHTML = `<p style="color:var(--loss-color)">Error conectando con el backend.</p>`;
        }
    }

    function formatCurrency(value) {
        return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
    }

    function updateMetrics(metrics) {
        const formatPct = (val) => (val > 0 ? '+' : '') + val.toFixed(2) + '%';
        
        document.getElementById('val-capital').innerText = formatCurrency(metrics.final_capital);
        document.getElementById('val-initial').innerText = formatCurrency(metrics.initial_capital);
        document.getElementById('val-winrate').innerText = metrics.win_rate.toFixed(1) + '%';
        document.getElementById('val-trades').innerText = metrics.total_trades;

        const roiBadge = document.getElementById('val-roi');
        roiBadge.innerText = formatPct(metrics.roi_pct);
        if (metrics.roi_pct >= 0) {
            roiBadge.className = 'badge positive';
        } else {
            roiBadge.className = 'badge negative';
        }
    }

    function updateChart(chartData) {
        if (!chartData || chartData.length === 0) {
            // No chart data available in live mode yet
            return;
        }
        
        const ctx = document.getElementById('portfolioChart').getContext('2d');
        
        const labels = chartData.map(d => {
            const date = new Date(d.time);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
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
                            callback: function(value) {
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

        if (!trades) return;

        // Invertir para mostrar los más recientes arriba
        trades.reverse().forEach(trade => {
            const tr = document.createElement('tr');
            
            const tradeTime = trade.entry_time || trade.time;
            const date = new Date(tradeTime);
            const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            const pnl = parseFloat(trade.pnl || 0);
            const statusClass = pnl > 0 ? 'status-win' : (pnl < 0 ? 'status-loss' : '');
            const pnlSign = pnl > 0 ? '+' : '';
            
            const typeColor = trade.type === 'LONG' ? 'color: var(--accent-neon); font-weight: 600;' : 'color: #f472b6; font-weight: 600;';
            const edge = parseFloat(trade.edge || 0);
            const edgeSign = edge > 0 ? '+' : '';
            
            const marketName = trade.question || trade.market || 'Unknown Market';
            const prob = parseFloat(trade.predicted_prob || trade.prob || 0);
            
            tr.innerHTML = `
                <td>${dateStr}</td>
                <td style="max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${marketName}">${marketName}</td>
                <td style="${typeColor}">${trade.type}</td>
                <td>${edgeSign}${edge.toFixed(1)}%</td>
                <td>${(prob * 100).toFixed(1)}%</td>
                <td>${formatCurrency(trade.bet_size)}</td>
                <td class="${statusClass}">${pnlSign}${formatCurrency(pnl)}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Inicializar
    fetchSimulationData();

});
