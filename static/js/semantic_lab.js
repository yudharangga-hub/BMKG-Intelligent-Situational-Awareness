// Semantic Lab JS (Word2Vec Slang Thesaurus)
document.addEventListener('DOMContentLoaded', function() {
    const searchBtn = document.getElementById('w2vSearchBtn');
    const input = document.getElementById('w2vInput');
    const tableBody = document.querySelector('#w2vTable tbody');
    const tablePlaceholder = document.getElementById('w2vTablePlaceholder');
    const graphPlaceholder = document.getElementById('w2vGraphPlaceholder');
    const graphCanvas = document.getElementById('w2vGraph');

    function renderTable(words) {
        tableBody.innerHTML = '';
        if (!words || words.length === 0) {
            tablePlaceholder.style.display = '';
            return;
        }
        tablePlaceholder.style.display = 'none';
        words.forEach(w => {
            let tr = document.createElement('tr');
            tr.innerHTML = `<td>${w.word}</td><td>${w.score.toFixed(3)}</td>`;
            tableBody.appendChild(tr);
        });
    }

    function renderGraph(words, query) {
        if (!words || words.length === 0) {
            graphPlaceholder.style.display = '';
            graphCanvas.style.display = 'none';
            return;
        }
        graphPlaceholder.style.display = 'none';
        graphCanvas.style.display = '';
        // vis-network: center node + neighbors
        const nodes = [ { id: query, label: query, color: '#0d47a1', font: { color: '#fff', size: 20 }, size: 30 } ];
        const edges = [];
        words.forEach(w => {
            nodes.push({ id: w.word, label: w.word, color: '#00b894', font: { color: '#222', size: 16 }, size: 20 });
            edges.push({ from: query, to: w.word, label: w.score.toFixed(2), color: { color: '#b2bec3' }, width: 2 + w.score * 5 });
        });
        // Destroy previous network if exists
        if (graphCanvas._network) { graphCanvas._network.destroy(); }
        const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
        const options = {
            nodes: { shape: 'dot', borderWidth: 2 },
            edges: { font: { align: 'middle' }, smooth: { type: 'dynamic' } },
            physics: { enabled: true, barnesHut: { gravitationalConstant: -2000 } },
            layout: { improvedLayout: true }
        };
        graphCanvas._network = new vis.Network(graphCanvas, data, options);
    }

    searchBtn.onclick = async function() {
        const word = input.value.trim();
        if (!word) return alert('Masukkan kata!');
        tablePlaceholder.style.display = 'none';
        graphPlaceholder.style.display = 'none';
        tableBody.innerHTML = '';
        graphCanvas.style.display = 'none';
        // Fetch ke backend
        try {
            const res = await fetch('/api/word2vec?word=' + encodeURIComponent(word));
            const data = await res.json();
            if (data.similar && Array.isArray(data.similar)) {
                renderTable(data.similar);
                renderGraph(data.similar, word);
            } else {
                tablePlaceholder.style.display = '';
                tablePlaceholder.innerText = 'Gagal mengambil data.';
            }
        } catch(e) {
            tablePlaceholder.style.display = '';
            tablePlaceholder.innerText = 'Gagal mengambil data.';
        }
    };
});
