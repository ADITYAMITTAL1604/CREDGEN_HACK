// frontend/chat.js - Standalone Chat Widget Logic (No iframe embed support)
// Uses UI provided by widget-ui.js

(function () {
    'use strict';

    const API_BASE = window.location.origin || '';

    // Session handling
    function getOrCreateSession() {
        const key = 'CREDGEN_SESSION_ID';
        let id = localStorage.getItem(key);

        if (!id) {
            id = 'web-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
            localStorage.setItem(key, id);
        }
        return id;
    }

    let SESSION_ID = getOrCreateSession();

    function resetSession() {
        const key = 'CREDGEN_SESSION_ID';
        localStorage.removeItem(key);
        SESSION_ID = getOrCreateSession();
    }

    window.resetChatSession = resetSession;

    // Utility DOM getter
    function UI() {
        return {
            chatbox: document.getElementById('credgen-chatbox'),
            messages: document.getElementById('credgen-messages'),
            form: document.getElementById('credgen-form'),
            input: document.getElementById('credgen-input'),
            attachments: document.getElementById('credgen-attachments'),
        };
    }

    // Render messages
    function appendMessage(text, sender = 'bot', files = null) {
        const ui = UI();
        if (!ui.messages) return;

        if (files && files.length > 0) {
            const fileMessage = document.createElement('div');
            fileMessage.className = `file-attachment-message ${sender}`;
            fileMessage.innerHTML = createFileAttachmentMessage(files);
            ui.messages.appendChild(fileMessage);
        }

        if (text && text.trim()) {
            const msg = document.createElement('div');
            msg.className = sender === 'user' ? 'user-message' : 'bot-message';

            const content = document.createElement('div');
            content.textContent = text;

            const ts = document.createElement('div');
            ts.className = 'message-timestamp';
            ts.textContent = new Date().toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
                hour12: true
            });

            msg.appendChild(content);
            msg.appendChild(ts);
            ui.messages.appendChild(msg);
        }

        ui.messages.scrollTop = ui.messages.scrollHeight;
    }

    // Build file preview HTML
    function createFileAttachmentMessage(files) {
        const totalSize = files.reduce((s, f) => s + f.size, 0);
        const fileList = files.map(file => `
            <div class="file-list-item">
                <div class="file-list-icon">${window.getFileIcon(window.getFileType(file.type))}</div>
                <div class="file-list-name">${file.name}</div>
                <div class="file-list-size">${window.formatBytes(file.size)}</div>
            </div>
        `).join('');

        return `
            <div class="file-message-header">
                <i class="fas fa-paperclip"></i>
                <span>${files.length} file(s) attached (${window.formatBytes(totalSize)})</span>
            </div>
            <div class="file-list">${fileList}</div>
        `;
    }

    // Typing Indicator Logic
    function showTypingIndicator() {
        const ui = UI();
        if (!ui.messages) return;

        // Remove existing if any (safety)
        hideTypingIndicator();

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.id = 'credgen-typing-indicator';
        indicator.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;

        ui.messages.appendChild(indicator);
        ui.messages.scrollTop = ui.messages.scrollHeight;
    }

    function hideTypingIndicator() {
        const indicator = document.getElementById('credgen-typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    // Sending state
    function setSending(state) {
        const ui = UI();
        const btn = ui.form.querySelector('button[type="submit"]');

        ui.input.disabled = state;
        btn.disabled = state;

        if (state) {
            btn.textContent = 'Sending…';
            btn.style.opacity = '0.7';
        } else {
            btn.textContent = 'Send';
            btn.style.opacity = '1';
        }
    }

    // API wrappers
    async function apiCall(path, body = {}) {
        const res = await fetch(`${API_BASE}${path}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Session-ID': SESSION_ID
            },
            body: JSON.stringify(body)
        });

        const data = await res.json().catch(() => ({}));

        if (!res.ok) throw new Error(data.message || 'Request failed');

        // sync session
        if (res.headers.get('X-Session-ID')) {
            SESSION_ID = res.headers.get('X-Session-ID');
            localStorage.setItem('CREDGEN_SESSION_ID', SESSION_ID);
        }

        return data;
    }

    async function uploadFilesWithMessage(text, files) {
        const formData = new FormData();
        formData.append('message', text || '');

        files.forEach(f => formData.append('files', f));

        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'X-Session-ID': SESSION_ID },
            body: formData
        });

        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.message || 'Upload failed');

        if (res.headers.get('X-Session-ID')) {
            SESSION_ID = res.headers.get('X-Session-ID');
            localStorage.setItem('CREDGEN_SESSION_ID', SESSION_ID);
        }

        return data;
    }

    // Handle backend actions
    async function handleAction(action) {
        try {
            if (action === 'call_underwriting' || action === 'call_underwriting_api') {
                appendMessage('Running underwriting check...', 'bot');
                const r = await apiCall('/underwrite');
                if (r.underwriting_result) {
                    const status = r.underwriting_result.approval_status ? 'approved' : 'rejected';
                    appendMessage(`Underwriting: ${status} — risk ${r.underwriting_result.risk_score}`, 'bot');
                }
                if (r.action) await handleAction(r.action);
            }

            if (action === 'call_sales' || action === 'call_sales_api') {
                appendMessage('Fetching offer from sales...', 'bot');
                const r = await apiCall('/sales');
                if (r.action) await handleAction(r.action);
            }

            if (action === 'call_documentation' || action === 'call_documentation_api') {
                appendMessage('Generating document...', 'bot');
                const response = await apiCall('/documentation');
                appendMessage('Document generated successfully!', 'bot');
                if (response.download_url) {
                    const link = document.createElement('a');
                    link.href = response.download_url;
                    link.download = 'Sanction_Letter.pdf'
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    appendMessage('Document has been downloaded.', 'bot');
                }
            }

            if (action === 'call_fraud' || action === 'call_fraud_api') {
                appendMessage('Running fraud checks...', 'bot');
                const r = await apiCall('/fraud');
                if (r.action) await handleAction(r.action);
            }
        } catch (err) {
            appendMessage(`Error: ${err.message}`, 'bot');
        }
    }

    // Send message with optional files
    async function sendMessage(text, files = []) {
        if (!text.trim() && files.length === 0) return;

        // Render outgoing messages
        if (files.length > 0) appendMessage('', 'user', files);
        if (text.trim()) appendMessage(text, 'user');

        setSending(true);
        showTypingIndicator(); // Show typing immediately

        try {
            let data;

            if (files.length > 0) {
                data = await uploadFilesWithMessage(text, files);
            } else {
                data = await apiCall('/chat', { message: text });
            }

            // Remove typing before showing response
            hideTypingIndicator();

            if (data.message) appendMessage(data.message, 'bot');
            if (data.suggestions && window.updateQuickReplies) {
                window.updateQuickReplies(data.suggestions);
            }
            if (data.action) await handleAction(data.action);

        } catch (err) {
            hideTypingIndicator(); // Ensure removal on error
            appendMessage(`Error: ${err.message}`, 'bot');
        } finally {
            setSending(false);
            // Double check removal just in case
            hideTypingIndicator();
        }
    }

    // Init
    function init() {
        const ui = UI();

        if (!ui.form || !ui.input) {
            console.error('CREDGEN: UI not found. widget-ui.js missing?');
            return;
        }

        ui.form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const text = ui.input.value.trim();
            ui.input.value = '';

            const files = window.getAttachedFiles();

            await sendMessage(text, files);

            if (files.length > 0) window.clearAttachments();
        });

        ui.input.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                ui.form.dispatchEvent(new Event('submit'));
            }
        });

        // Initial greeting
        setTimeout(() => {
            appendMessage('Hi, I\'m the CRED_GEN assistant. How can I help with your loan today?', 'bot');
        }, 300);

        // Listen for clear events
        window.addEventListener('credgen:clear-chat', () => {
            ui.messages.innerHTML = '';
            resetSession();
            setTimeout(() => {
                appendMessage('Hi, I\'m the CRED_GEN assistant. How can I help with your loan today?', 'bot');
            }, 300);
        });
    }

    // DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
