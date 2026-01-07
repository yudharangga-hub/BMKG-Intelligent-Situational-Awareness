// Floating Chatbot Widget JS
// Show/hide widget, handle chat input, send to backend (BMKG chatbot)
document.addEventListener('DOMContentLoaded', function() {
    const widget = document.getElementById('chatbot-widget');
    const openBtn = document.getElementById('chatbot-open-btn');
    const closeBtn = document.getElementById('chatbot-close-btn');
    const form = document.getElementById('chatbot-form');
    const input = document.getElementById('chatbot-input');
    const chatBody = document.getElementById('chatbot-body');

    openBtn.onclick = () => widget.classList.add('open');
    closeBtn.onclick = () => widget.classList.remove('open');

    form.onsubmit = async function(e) {
        e.preventDefault();
        const userMsg = input.value.trim();
        if (!userMsg) return;
        chatBody.innerHTML += `<div class='chat-msg user'><b>Anda:</b> ${userMsg}</div>`;
        input.value = '';
        chatBody.scrollTop = chatBody.scrollHeight;

        // Kirim ke backend chatbot
        chatBody.innerHTML += `<div class='chat-msg bot'><b>Bot:</b> <span class='text-muted'><i class='fas fa-spinner fa-spin'></i> Memproses...</span></div>`;
        chatBody.scrollTop = chatBody.scrollHeight;
        try {
            const res = await fetch('/api/chatbot', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: userMsg})
            });
            const data = await res.json();
            // Hapus spinner, tampilkan jawaban
            chatBody.lastElementChild.innerHTML = `<b>Bot:</b> ${data.reply}`;
            chatBody.scrollTop = chatBody.scrollHeight;
        } catch(e) {
            chatBody.lastElementChild.innerHTML = `<b>Bot:</b> <span class='text-danger'>Gagal mengambil jawaban.</span>`;
        }
    };
});
