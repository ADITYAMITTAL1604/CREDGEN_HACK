// frontend/widget-ui.js
// Dynamically builds the entire widget UI so widget.html can stay empty.

(function() {
    'use strict';

    function buildWidgetUI() {
        const root = document.createElement('div');
        root.id = "credgen-chat-root";

        // Full UI recreated from original widget.html
        root.innerHTML = `
            <button id="credgen-toggle" class="credgen-toggle-btn" aria-label="Open chat">
                <i class="fas fa-comment-dots"></i>
            </button>

            <div id="credgen-chatbox" class="credgen-chatbox" aria-hidden="true">
                <header class="credgen-header">
                    <div class="credgen-header-info">
                        <span class="credgen-title">CREDGEN Assistant</span>
                        <span class="credgen-subtitle">AI-powered loan chatbot</span>
                    </div>
                    <div class="credgen-header-controls">
                        <button id="credgen-fullscreen" class="credgen-control-btn" aria-label="Toggle fullscreen">
                            <i class="fas fa-expand"></i>
                        </button>
                        <button id="credgen-clear" class="credgen-control-btn credgen-clear-btn" aria-label="Clear chat">
                            <i class="fas fa-trash-alt"></i>
                        </button>
                        <button id="credgen-close" class="credgen-close-btn" aria-label="Close chat">×</button>
                    </div>
                </header>

                <div id="credgen-messages" class="credgen-messages" role="log" aria-live="polite"></div>

                <div id="credgen-attachments" class="credgen-attachments"></div>

                <form id="credgen-form" class="credgen-input-row">
                    <button type="button" id="credgen-attach" class="credgen-attach-btn" aria-label="Attach files">
                        <i class="fas fa-paperclip"></i>
                    </button>

                    <input
                        id="credgen-input"
                        type="text"
                        placeholder="Type your message..."
                        autocomplete="off"
                        aria-label="Chat message"
                    />

                    <div class="credgen-quick-msg-container">
                        <button type="button" id="credgen-quick-msg" class="credgen-quick-msg-btn" aria-label="Quick messages">
                            <i class="fas fa-comment-dots"></i>
                        </button>
                        <div class="credgen-quick-msg-dropdown">
                            <div class="quick-msg-item" data-msg="I need a loan">I need a loan</div>
                            <div class="quick-msg-item" data-msg="What documents do I need?">What documents do I need?</div>
                            <div class="quick-msg-item" data-msg="What is the interest rate?">What is the interest rate?</div>
                            <div class="quick-msg-item" data-msg="How much can I borrow?">How much can I borrow?</div>
                            <div class="quick-msg-item" data-msg="What is the processing time?">What is the processing time?</div>
                        </div>
                    </div>

                    <input type="file" id="credgen-file-input" multiple style="display:none;" />
                    <button type="submit" class="credgen-send-btn">Send</button>
                </form>
            </div>
        `;

        document.body.appendChild(root);
        initializeUIEvents();
    }

    function initializeUIEvents() {
        const toggleBtn = document.getElementById('credgen-toggle');
        const closeBtn = document.getElementById('credgen-close');
        const fullscreenBtn = document.getElementById('credgen-fullscreen');
        const clearBtn = document.getElementById('credgen-clear');
        const chatbox = document.getElementById('credgen-chatbox');
        const chatRoot = document.getElementById('credgen-chat-root');
        const form = document.getElementById('credgen-form');
        const input = document.getElementById('credgen-input');
        const messagesContainer = document.getElementById('credgen-messages');
        const attachBtn = document.getElementById('credgen-attach');
        const fileInput = document.getElementById('credgen-file-input');
        const attachmentsContainer = document.getElementById('credgen-attachments');
        const quickMsgBtn = document.getElementById('credgen-quick-msg');
        const quickMsgContainer = document.querySelector('.credgen-quick-msg-container');
        const quickMsgDropdown = document.querySelector('.credgen-quick-msg-dropdown');

        window.credgenAttachedFiles = window.credgenAttachedFiles || [];
        const attachedFiles = window.credgenAttachedFiles;

        const MAX_FILES = 5;
        const MAX_TOTAL_SIZE = 10 * 1024 * 1024;

        // Toggle visibility
        toggleBtn.addEventListener('click', () => {
            const isHidden = chatbox.getAttribute('aria-hidden') === 'true';
            chatbox.setAttribute('aria-hidden', !isHidden);
            if (!isHidden) input.focus();
        });

        closeBtn.addEventListener('click', e => {
            e.stopPropagation();
            chatbox.setAttribute('aria-hidden', 'true');
        });

        document.addEventListener('click', e => {
            if (chatbox.getAttribute('aria-hidden') === 'true') return;
            if (!chatRoot.contains(e.target) && !toggleBtn.contains(e.target)) {
                chatbox.setAttribute('aria-hidden', 'true');
            }
        });

        chatbox.addEventListener('click', e => e.stopPropagation());

        fullscreenBtn.addEventListener('click', e => {
            e.stopPropagation();
            const isFullscreen = chatbox.classList.contains('fullscreen');
            chatbox.classList.toggle('fullscreen');
            fullscreenBtn.innerHTML = isFullscreen
                ? '<i class="fas fa-expand"></i>'
                : '<i class="fas fa-compress"></i>';
        });

        clearBtn.addEventListener('click', e => {
            e.stopPropagation();
            if (!confirm('Clear all messages?')) return;

            messagesContainer.innerHTML = '';
            clearAttachments();

            localStorage.removeItem('CREDGEN_SESSION_ID');
            if (window.resetChatSession) window.resetChatSession();

            window.dispatchEvent(new CustomEvent('credgen:clear-chat'));
        });

        // Expose function to update quick replies dynamicallly
        window.updateQuickReplies = function(suggestions) {
            const dropdown = document.querySelector('.credgen-quick-msg-dropdown');
            if (!dropdown) return;
            
            // Clear existing
            dropdown.innerHTML = '';
            
            if (!suggestions || suggestions.length === 0) {
                 // Default fallback if empty
                 suggestions = [
                    "I need a loan",
                    "What documents do I need?",
                    "What is the interest rate?"
                 ];
            }
            
            suggestions.forEach(msg => {
                const item = document.createElement('div');
                item.className = 'quick-msg-item';
                item.dataset.msg = msg;
                item.textContent = msg;
                
                item.addEventListener('click', e => {
                    e.stopPropagation();
                    const input = document.getElementById('credgen-input');
                    if (input) {
                        input.value = item.dataset.msg;
                        input.focus();
                        dropdown.style.display = 'none';
                    }
                });
                
                dropdown.appendChild(item);
            });
        };

        // Quick messages
        quickMsgBtn.addEventListener('click', e => {
            e.stopPropagation();
            quickMsgDropdown.style.display =
                quickMsgDropdown.style.display === 'block' ? 'none' : 'block';
        });

        document.addEventListener('click', e => {
            if (!quickMsgContainer.contains(e.target)) {
                quickMsgDropdown.style.display = 'none';
            }
        });

        // Initialize with defaults
        window.updateQuickReplies(null);

        // File attachments
        attachBtn.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', e => {
            const files = Array.from(e.target.files);

            if (attachedFiles.length + files.length > MAX_FILES) {
                alert(`You can only attach up to ${MAX_FILES} files.`);
                fileInput.value = '';
                return;
            }

            const totalSize =
                attachedFiles.reduce((s, f) => s + f.size, 0) +
                files.reduce((s, f) => s + f.size, 0);

            if (totalSize > MAX_TOTAL_SIZE) {
                alert('Total file size exceeds 10MB');
                fileInput.value = '';
                return;
            }

            files.forEach(file => {
                window.credgenAttachedFiles.push(file);
                addFilePreview(file);
            });

            fileInput.value = '';
            updateAttachmentButton();
            input.focus();
        });

        // Attachment helpers
        function addFilePreview(file) {
            const el = document.createElement('div');
            el.className = 'attachment-preview';
            el.dataset.id = file.name + file.size;

            el.innerHTML = `
                <div class="file-info">
                    <div class="file-icon">${getFileIcon(getFileType(file.type))}</div>
                    <div class="file-details">
                        <div class="file-name">${truncate(file.name)}</div>
                        <div class="file-size">${formatBytes(file.size)}</div>
                    </div>
                </div>
                <button type="button" class="remove-file">
                    <i class="fas fa-times"></i>
                </button>
            `;

            attachmentsContainer.appendChild(el);
            attachmentsContainer.style.display = 'block';

            el.querySelector('.remove-file').addEventListener('click', () => removeFile(file));
        }

        function removeFile(file) {
            const id = file.name + file.size;
            window.credgenAttachedFiles =
                window.credgenAttachedFiles.filter(f => f.name + f.size !== id);

            const el = attachmentsContainer.querySelector(`[data-id="${id}"]`);
            if (el) el.remove();

            if (window.credgenAttachedFiles.length === 0) {
                attachmentsContainer.style.display = 'none';
            }

            updateAttachmentButton();
        }

        function clearAttachments() {
            window.credgenAttachedFiles = [];
            attachmentsContainer.innerHTML = '';
            attachmentsContainer.style.display = 'none';
            updateAttachmentButton();
        }

        function updateAttachmentButton() {
            const count = window.credgenAttachedFiles.length;
            if (count > 0) {
                attachBtn.classList.add('has-files');
                attachBtn.dataset.count = count;
            } else {
                attachBtn.classList.remove('has-files');
                attachBtn.removeAttribute('data-count');
            }
        }

        // Utility helpers
        function getFileType(type) {
            if (type.startsWith('image/')) return 'image';
            if (type === 'application/pdf') return 'pdf';
            if (type.includes('word') || type.includes('text')) return 'document';
            if (type.includes('spreadsheet') || type.includes('excel')) return 'spreadsheet';
            return 'file';
        }

        function getFileIcon(type) {
            const icons = {
                pdf: '<i class="fas fa-file-pdf"></i>',
                image: '<i class="fas fa-file-image"></i>',
                document: '<i class="fas fa-file-word"></i>',
                spreadsheet: '<i class="fas fa-file-excel"></i>',
                file: '<i class="fas fa-file"></i>'
            };
            return icons[type] || icons.file;
        }

        function formatBytes(bytes) {
            if (bytes === 0) return '0B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return (bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i];
        }

        function truncate(name, len = 20) {
            return name.length <= len ? name : name.substring(0, len - 3) + '...';
        }

        // Expose functions for chat.js
        window.getAttachedFiles = () => [...window.credgenAttachedFiles];
        window.clearAttachments = clearAttachments;
        window.getFileType = getFileType;
        window.getFileIcon = getFileIcon;
        window.formatBytes = formatBytes;
    }

    // Build UI immediately
    buildWidgetUI();
})();

