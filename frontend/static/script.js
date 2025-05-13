// API端点
const API_URL = window.location.origin;

// 调试工具
const DEBUG = true;
function debug(...args) {
    if (DEBUG) {
        console.log(`[DEBUG ${new Date().toLocaleTimeString()}]`, ...args);
    }
}

// 给window添加一个全局错误处理器
window.addEventListener('error', function(event) {
    debug('全局错误:', event.message, 'at', event.filename, ':', event.lineno);
});

// DOM元素
const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const uploadStatus = document.getElementById('uploadStatus');
const searchInput = document.getElementById('searchInput');
const searchButton = document.getElementById('searchButton');
const searchResults = document.getElementById('searchResults');
const documentList = document.getElementById('documentList');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendMessageBtn = document.getElementById('sendMessageBtn');
const clearChatBtn = document.getElementById('clearChatBtn');

// 页面加载时获取文档列表
document.addEventListener('DOMContentLoaded', () => {
    debug('页面加载完成，开始初始化');
    
    // 初始化工具提示
    try {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
        debug('工具提示初始化成功');
    } catch (error) {
        debug('工具提示初始化失败:', error);
    }
    
    // 确保Bootstrap已加载
    if (typeof bootstrap === 'undefined') {
        debug('警告: Bootstrap未找到，可能会影响部分功能');
    } else {
        debug('Bootstrap已加载');
    }
    
    // 检查API路径
    debug('当前API路径:', API_URL);
    
    // 获取文档列表
    fetchDocumentList();
    
    // 添加全局事件委托处理
    document.body.addEventListener('click', function(event) {
        // 查找最近的查看详情按钮
        const viewButton = event.target.closest('.view-document-btn');
        if (viewButton) {
            event.preventDefault();
            const docId = viewButton.getAttribute('data-doc-id');
            debug(`通过委托处理点击事件，文档ID: ${docId}`);
            showDocumentDetails(docId);
        }
    });
    
    // 初始化聊天功能
    initChat();
    
    debug('初始化完成');
});

// 初始化聊天功能
function initChat() {
    // 发送按钮点击事件
    sendMessageBtn.addEventListener('click', sendChatMessage);
    
    // 按Enter键发送消息，Shift+Enter换行
    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        } else if (e.key === 'Enter' && e.shiftKey) {
            // 换行时自动调整高度
            setTimeout(() => {
                adjustTextareaHeight(this);
            }, 0);
        }
    });
    
    // 监听输入事件，自动调整高度
    chatInput.addEventListener('input', function() {
        adjustTextareaHeight(this);
    });
    
    // 初始化高度
    adjustTextareaHeight(chatInput);
    
    // 清空聊天按钮
    clearChatBtn.addEventListener('click', function() {
        chatMessages.innerHTML = `
            <div class="chat-message system-message">
                <div class="message-content">
                    <p>您好！我是基于RAG（检索增强生成）技术的助手。您可以向我提问关于已上传文档的内容。</p>
                </div>
            </div>
        `;
        chatInput.value = '';
        chatInput.style.height = 'auto';
        chatInput.focus();
    });
}

// 文本框高度自适应
function adjustTextareaHeight(textarea) {
    // 先将高度重置为auto，使得scrollHeight能够正确计算
    textarea.style.height = 'auto';
    
    // 设置新高度，使用新的基准高度 46px
    const newHeight = Math.min(Math.max(textarea.scrollHeight, 46), 120);
    textarea.style.height = newHeight + 'px';
    
    // 不再需要滚动到底部，因为我们已经隐藏了滚动条
    // textarea.scrollTop = textarea.scrollHeight;
}

// 发送聊天消息
async function sendChatMessage() {
    const message = chatInput.value.trim();
    if (!message) return;
    
    // 添加用户消息到聊天框
    addChatMessage(message, 'user');
    
    // 清空输入框
    chatInput.value = '';
    chatInput.style.height = 'auto'; // 重置高度
    
    // 显示思考中状态
    const thinkingMessageId = addThinkingMessage();
    
    try {
        // 发送消息到后端
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                top_k: 5 // 默认获取5个相关块
            })
        });
        
        // 移除思考中状态
        removeThinkingMessage(thinkingMessageId);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || '请求失败');
        }
        
        const data = await response.json();
        
        // 添加助手回复到聊天框
        addChatMessage(data.answer, 'assistant', data.retrieved_context);
        
        // 滚动到底部
        scrollChatToBottom();
        
    } catch (error) {
        // 移除思考中状态（如果还存在）
        removeThinkingMessage(thinkingMessageId);
        
        // 添加错误消息
        addChatMessage(`很抱歉，处理您的请求时出现错误: ${error.message}`, 'assistant');
        
        // 记录错误
        debug('聊天请求错误:', error);
    }
}

// 添加聊天消息
function addChatMessage(message, type, context = null) {
    const messageElement = document.createElement('div');
    messageElement.className = `chat-message ${type}-message`;
    
    let contentHtml = `<div class="message-content">${formatMessageText(message)}</div>`;
    
    // 如果有上下文信息，添加引用
    if (context && Array.isArray(context) && context.length > 0) {
        // 过滤参考资料，只保留相关性分数较高的
        const filteredReferences = context.filter(ref => {
            // 严格的相关性分数阈值，确保只显示真正相关的文档
            const MIN_SCORE_THRESHOLD = 0.65; 
            return ref.score >= MIN_SCORE_THRESHOLD;
        });
        
        // 如果过滤后没有结果，至少显示得分最高的一个
        let referencesToShow = filteredReferences.length > 0 ? 
                             filteredReferences : 
                             context.length > 0 ? [context.reduce((max, current) => 
                                max.score > current.score ? max : current, context[0])] : [];
        
        // 最多显示3个高质量引用
        referencesToShow = referencesToShow.slice(0, 3);
        
        if (referencesToShow.length > 0) {
            let referencesHtml = '<div class="message-reference">';
            
            // 添加引用信息
            referencesHtml += `<strong>参考资料:</strong><br>`;
            referencesToShow.forEach((ref, index) => {
                // 获取文件名
                let fileName = '未知文件';
                if (ref.metadata && ref.metadata.filename) {
                    fileName = ref.metadata.filename;
                }
                
                // 确保文件名格式正确显示
                if (!fileName || fileName === 'undefined') {
                    fileName = '未知文件';
                }
                
                // 计算相关性百分比
                const relevancePercent = Math.round(ref.score * 100);
                
                // 获取文本块位置信息
                let chunkInfo = '';
                if (ref.metadata && ref.metadata.chunk_index !== undefined) {
                    chunkInfo = `文本块 #${ref.metadata.chunk_index}`;
                }
                
                // 构建完整的参考资料条目
                referencesHtml += `
                    <div class="reference-item">
                        <span class="reference-number">${index + 1}.</span> 
                        <span class="reference-source">${escapeHtml(fileName)}</span>
                        ${chunkInfo ? `<span class="reference-chunk">(${chunkInfo})</span>` : ''}
                        <span class="reference-relevance badge bg-${relevancePercent > 80 ? 'success' : relevancePercent > 60 ? 'info' : 'warning'}">
                            相关度: ${relevancePercent}%
                        </span>
                    </div>
                `;
            });
            
            referencesHtml += '</div>';
            contentHtml += referencesHtml;
        }
    }
    
    messageElement.innerHTML = contentHtml;
    chatMessages.appendChild(messageElement);
    scrollChatToBottom();
}

// 添加"思考中"消息
function addThinkingMessage() {
    const id = 'thinking-' + Date.now();
    const thinkingElement = document.createElement('div');
    thinkingElement.className = 'chat-message assistant-message';
    thinkingElement.id = id;
    thinkingElement.innerHTML = `
        <div class="thinking">
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
        </div>
    `;
    chatMessages.appendChild(thinkingElement);
    scrollChatToBottom();
    return id;
}

// 移除"思考中"消息
function removeThinkingMessage(id) {
    const thinkingElement = document.getElementById(id);
    if (thinkingElement) {
        thinkingElement.remove();
    }
}

// 滚动聊天框到底部
function scrollChatToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 格式化消息文本（换行符转为<br>，链接转为可点击等）
function formatMessageText(text) {
    if (!text) return '';
    
    // 替换URL为可点击的链接
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    text = text.replace(urlRegex, url => `<a href="${url}" target="_blank">${url}</a>`);
    
    // 保留换行符
    text = text.replace(/\n/g, '<br>');
    
    return text;
}

// HTML转义函数，防止XSS攻击
function escapeHtml(text) {
    if (!text) return '';
    
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    
    return text.replace(/[&<>"']/g, m => map[m]);
}

// 截断文本
function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// 上传表单提交
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const file = fileInput.files[0];
    if (!file) {
        showUploadStatus('请选择要上传的文件', 'error');
        return;
    }
    
    // 显示上传进度条
    uploadProgress.classList.remove('d-none');
    uploadProgress.querySelector('.progress-bar').style.width = '0%';
    showUploadStatus('正在上传文件...', 'processing');
    
    // 创建FormData对象
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // 上传文件
        uploadProgress.querySelector('.progress-bar').style.width = '30%';
        showUploadStatus('正在上传文件...', 'processing');
        
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`上传失败: ${response.statusText}`);
        }
        
        // 更新进度
        uploadProgress.querySelector('.progress-bar').style.width = '100%';
        
        // 获取响应数据
        const data = await response.json();
        
        showUploadStatus('文件上传并处理成功！', 'success');
        
        // 刷新文档列表
        fetchDocumentList();
        
        // 重置表单
        uploadForm.reset();
        
        // 添加系统消息到聊天
        addChatMessage(`文档 "${data.original_filename}" 已成功上传并处理。您现在可以询问关于该文档的问题了。`, 'system');
        
    } catch (error) {
        console.error('上传错误:', error);
        showUploadStatus(`上传失败: ${error.message}`, 'error');
    }
    
    // 隐藏进度条
    setTimeout(() => {
        uploadProgress.classList.add('d-none');
    }, 1000);
});

// 搜索按钮点击
searchButton.addEventListener('click', () => {
    performSearch();
});

// 回车键触发搜索
searchInput.addEventListener('keyup', (e) => {
    if (e.key === 'Enter') {
        performSearch();
    }
});

// 执行搜索
async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) {
        searchResults.innerHTML = '<div class="alert alert-warning">请输入搜索关键词</div>';
        return;
    }
    
    searchResults.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p>正在搜索...</p></div>';
    
    try {
        // 实际项目中应替换为真实API端点
        const response = await fetch(`${API_URL}/search?query=${encodeURIComponent(query)}`);
        
        if (!response.ok) {
            throw new Error(`搜索失败: ${response.statusText}`);
        }
        
        const results = await response.json();
        // 检查是否有错误
        if (results.error) {
            throw new Error(`搜索失败: ${results.detail || '未知错误'}`);
        }
        
        displaySearchResults(results, query);
        
    } catch (error) {
        console.error('搜索错误:', error);
        searchResults.innerHTML = `<div class="alert alert-danger">搜索失败: ${error.message}</div>`;
    }
}

// 获取文档列表
async function fetchDocumentList() {
    try {
        // 实际项目中应替换为真实API端点
        const response = await fetch(`${API_URL}/documents`);
        
        if (!response.ok) {
            throw new Error(`获取文档列表失败: ${response.statusText}`);
        }
        
        const documents = await response.json();
        displayDocumentList(documents);
        
    } catch (error) {
        console.error('获取文档列表错误:', error);
        documentList.innerHTML = `<div class="alert alert-danger">获取文档列表失败: ${error.message}</div>`;
        
        // 如果API尚未实现，显示模拟数据
        displayMockDocumentList();
    }
}

// 显示搜索结果
function displaySearchResults(results, query) {
    // 检查是否有检索到的上下文
    const contextResults = results.retrieved_context;
    
    if (!contextResults || contextResults.length === 0) {
        // 如果没有上下文结果但有生成的答案，显示答案
        if (results.answer) {
            searchResults.innerHTML = `
                <div class="alert alert-info">
                    <h6>来自AI的回答:</h6>
                    <div class="llm-answer">${formatMessageText(results.answer)}</div>
                </div>
                <div class="alert alert-warning">没有找到匹配的文档块</div>
            `;
        } else {
            searchResults.innerHTML = '<div class="alert alert-info">没有找到匹配的结果</div>';
        }
        return;
    }
    
    // 按文档分组结果
    const resultsByDocument = {};
    contextResults.forEach(result => {
        // 获取文件名从metadata
        const filename = result.metadata?.filename || '未知文档';
        if (!resultsByDocument[filename]) {
            resultsByDocument[filename] = [];
        }
        resultsByDocument[filename].push(result);
    });
    
    let html = `<h6>搜索结果 - 共找到 ${contextResults.length} 个匹配项</h6>`;
    
    // 如果有LLM生成的答案，显示在最上方
    if (results.answer) {
        html += `
            <div class="alert alert-info mb-4">
                <h6>来自AI的回答:</h6>
                <div class="llm-answer">${formatMessageText(results.answer)}</div>
            </div>
        `;
    }
    
    // 创建文档选项卡
    html += `
        <ul class="nav nav-tabs mb-3" id="searchResultTabs" role="tablist">
    `;
    
    // 添加每个文档的选项卡
    Object.keys(resultsByDocument).forEach((filename, index) => {
        const isActive = index === 0;
        const docId = `doc-${filename.replace(/\s+/g, '-').replace(/[^\w-]/g, '')}`;
        
        html += `
            <li class="nav-item" role="presentation">
                <button class="nav-link ${isActive ? 'active' : ''}" 
                        id="${docId}-tab" 
                        data-bs-toggle="tab" 
                        data-bs-target="#${docId}" 
                        type="button" 
                        role="tab" 
                        aria-controls="${docId}" 
                        aria-selected="${isActive ? 'true' : 'false'}">
                    ${filename} (${resultsByDocument[filename].length})
                </button>
            </li>
        `;
    });
    
    html += `</ul>`;
    
    // 创建选项卡内容
    html += `<div class="tab-content" id="searchResultTabContent">`;
    
    // 添加每个文档的结果
    Object.keys(resultsByDocument).forEach((filename, index) => {
        const isActive = index === 0;
        const docId = `doc-${filename.replace(/\s+/g, '-').replace(/[^\w-]/g, '')}`;
        const docResults = resultsByDocument[filename];
        
        html += `
            <div class="tab-pane fade ${isActive ? 'show active' : ''}" 
                 id="${docId}" 
                 role="tabpanel" 
                 aria-labelledby="${docId}-tab">
        `;
        
        // 添加每个结果
        docResults.forEach((result, i) => {
            // 高亮显示匹配的文本
            const highlightedText = highlightText(result.text, query);
            // 使用result.score或默认为0
            const score = result.score || 0;
            
            html += `
                <div class="search-result">
                    <div class="search-result-header d-flex justify-content-between align-items-center">
                        <span class="result-number">结果 #${i + 1}</span>
                        <span class="result-score">相关度: ${(score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="search-result-content">
                        ${highlightedText}
                    </div>
                    <button class="btn btn-sm btn-outline-secondary mt-2 view-context" 
                            data-id="${filename}" 
                            data-chunk-index="${result.metadata?.chunk_index || 0}">
                        查看上下文
                    </button>
                </div>
            `;
        });
        
        html += `</div>`;
    });
    
    html += `</div>`;
    
    searchResults.innerHTML = html;
    
    // 添加"查看上下文"按钮事件
    document.querySelectorAll('.view-context').forEach(button => {
        button.addEventListener('click', () => {
            const docId = button.getAttribute('data-id');
            const chunkIndex = parseInt(button.getAttribute('data-chunk-index'));
            showDocumentDetails(docId, chunkIndex);
        });
    });
}

// 高亮显示文本中的查询词
function highlightText(text, query) {
    if (!query) return text;
    
    const regex = new RegExp(query, 'gi');
    return text.replace(regex, match => `<span class="search-highlight">${match}</span>`);
}

// 显示文档列表
function displayDocumentList(documents) {
    // 更新文档计数器
    const documentCounter = document.querySelector('.document-counter');
    if (documentCounter) {
        documentCounter.textContent = `${documents.length} 个文档`;
    }
    
    // 控制欢迎消息的显示
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        if (documents && documents.length > 0) {
            welcomeMessage.style.display = 'none';
        } else {
            welcomeMessage.style.display = 'block';
        }
    }
    
    if (!documents || documents.length === 0) {
        documentList.innerHTML = '<li class="list-group-item">没有处理过的文档</li>';
        return;
    }
    
    debug(`显示文档列表: ${documents.length} 个文档`);
    
    let html = '';
    
    documents.forEach(doc => {
        // 处理日期显示
        let dateString = "未知日期";
        if (doc.processedAt && doc.processedAt !== "N/A") {
            try {
                // 尝试解析ISO格式的日期字符串
                const date = new Date(doc.processedAt);
                if (!isNaN(date)) {
                    dateString = date.toLocaleString();
                }
            } catch (e) {
                debug(`日期解析错误: ${e.message}`);
            }
        }
        
        html += `
            <li class="list-group-item document-item">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h6><i class="bi bi-file-earmark-text"></i> ${doc.filename}</h6>
                        <div class="document-metadata">
                            <span><i class="bi bi-clock"></i> ${dateString}</span>
                            <span class="ms-3"><i class="bi bi-card-text"></i> ${doc.chunkCount} 个文本块</span>
                        </div>
                    </div>
                    <div>
                        <button class="btn btn-sm btn-outline-primary view-document-btn" data-doc-id="${doc.id}">
                            <i class="bi bi-eye"></i> 查看详情
                        </button>
                    </div>
                </div>
            </li>
        `;
    });
    
    documentList.innerHTML = html;
    
    // 使用事件委托添加查看详情按钮的事件监听
    document.querySelectorAll('.view-document-btn').forEach(button => {
        button.onclick = function(event) {
            event.preventDefault();
            const docId = this.getAttribute('data-doc-id');
            debug(`点击查看详情按钮，文档ID: ${docId}`);
            showDocumentDetails(docId);
        };
    });
    
    debug('文档列表显示完成，已添加事件监听');
}

// 显示文档详情
async function showDocumentDetails(docId, chunkIndex = null) {
    try {
        debug(`正在加载文档ID: ${docId} 的详情`);
        
        // 创建模态框
        const modalHtml = `
            <div class="modal fade" id="documentDetailsModal" tabindex="-1" aria-labelledby="documentDetailsModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="documentDetailsModalLabel">文档详情</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="关闭"></button>
                        </div>
                        <div class="modal-body">
                            <div class="text-center">
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">加载中...</span>
                                </div>
                                <p>正在加载文档详情...</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        const existingModal = document.getElementById('documentDetailsModal');
        if (existingModal) {
            debug('找到现有模态框，正在移除');
            const oldModal = bootstrap.Modal.getInstance(existingModal);
            if (oldModal) {
                oldModal.dispose();
            }
            existingModal.remove();
        }
        
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        debug('已添加模态框到DOM');
        
        const modalElement = document.getElementById('documentDetailsModal');
        
        if (typeof bootstrap === 'undefined') {
            debug('Bootstrap未加载!');
            alert('无法加载模态框，请确保网络连接正常并刷新页面');
            return;
        }
        
        // 添加模态框事件监听器，确保关闭时恢复页面滚动
        modalElement.addEventListener('hidden.bs.modal', function () {
            document.body.classList.remove('modal-open');
            document.body.style.overflow = 'auto';
            document.body.style.paddingRight = '';
            document.documentElement.style.overflow = 'auto';
            
            // 移除模态框背景
            const modalBackdrops = document.querySelectorAll('.modal-backdrop');
            modalBackdrops.forEach(backdrop => {
                backdrop.remove();
            });
            
            debug('模态框关闭，已恢复页面滚动');
        });
        
        debug('正在初始化Bootstrap模态框');
        const modal = new bootstrap.Modal(modalElement);
        modal.show();
        
        const url = `${API_URL}/document/${encodeURIComponent(docId)}`;
        debug(`正在从API获取数据: ${url}`);
        const response = await fetch(url);
        
        if (!response.ok) {
            const errorText = await response.text();
            debug(`API错误 (${response.status}): ${errorText}`);
            throw new Error(`获取文档详情失败: ${response.statusText}`);
        }
        
        const docData = await response.json();
        debug('获取到文档详情:', docData);
        
        const modalBody = modalElement.querySelector('.modal-body');
        const modalTitle = modalElement.querySelector('.modal-title');
        modalTitle.textContent = `文档详情: ${docData.filename}`;
        
        // 处理日期显示
        let dateString = "未知日期";
        if (docData.processedAt && docData.processedAt !== "N/A") {
            try {
                // 尝试解析ISO格式的日期字符串
                const date = new Date(docData.processedAt);
                if (!isNaN(date)) {
                    dateString = date.toLocaleString();
                }
            } catch (e) {
                debug(`日期解析错误: ${e.message}`);
            }
        }
        
        let detailsHtml = `
            <div class="document-details">
                <div class="mb-3">
                    <h6>文档信息</h6>
                    <div class="card">
                        <div class="card-body">
                            <p><strong>文件名:</strong> ${docData.filename}</p>
                            <p><strong>处理时间:</strong> ${dateString}</p>
                            <p><strong>文本块数:</strong> ${docData.chunks.length}</p>
                        </div>
                    </div>
                </div>
                
                <div>
                    <h6>文本内容预览</h6>
                    <div class="accordion" id="documentChunks">
        `;
        
        docData.chunks.forEach((chunk, index) => {
            const isTargetChunk = (chunkIndex !== null && index === chunkIndex);
            detailsHtml += `
                <div class="accordion-item${isTargetChunk ? ' target-chunk' : ''}">
                    <h2 class="accordion-header">
                        <button class="accordion-button ${isTargetChunk ? '' : 'collapsed'}" 
                                type="button" 
                                data-bs-toggle="collapse" 
                                data-bs-target="#chunk${index}" 
                                aria-expanded="${isTargetChunk ? 'true' : 'false'}" 
                                aria-controls="chunk${index}">
                            文本块 #${index + 1}${isTargetChunk ? ' (匹配内容)' : ''}
                        </button>
                    </h2>
                    <div id="chunk${index}" 
                         class="accordion-collapse collapse ${isTargetChunk ? 'show' : ''}" 
                         data-bs-parent="#documentChunks">
                        <div class="accordion-body">
                            <pre class="chunk-text">${chunk.text}</pre>
                        </div>
                    </div>
                </div>
            `;
        });
        
        detailsHtml += `
                    </div>
                </div>
            </div>
        `;
        
        modalBody.innerHTML = detailsHtml;
        debug('已更新模态框内容');
        
        if (chunkIndex !== null) {
            const targetChunk = modalElement.querySelector('.target-chunk');
            if (targetChunk) {
                debug('滚动到目标文本块');
                setTimeout(() => {
                    targetChunk.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 300);
            }
        }
        
    } catch (error) {
        debug('获取文档详情错误:', error);
        const modalBody = document.querySelector('#documentDetailsModal .modal-body');
        if (modalBody) {
            modalBody.innerHTML = `<div class="alert alert-danger">获取文档详情失败: ${error.message}</div>`;
        } else {
            alert(`获取文档详情失败: ${error.message}`);
        }
    }
}

// 显示上传状态
function showUploadStatus(message, type) {
    let className = '';
    switch (type) {
        case 'success': className = 'status-success'; break;
        case 'error': className = 'status-error'; break;
        case 'processing': className = 'status-processing'; break;
    }
    
    uploadStatus.innerHTML = `<p class="${className}">${message}</p>`;
}

// 显示模拟文档列表（当API未实现时）
function displayMockDocumentList() {
    const mockDocuments = [
        { id: '1', filename: '示例文档.pdf', processedAt: new Date().toISOString(), chunkCount: 15 },
        { id: '2', filename: '使用手册.docx', processedAt: new Date().toISOString(), chunkCount: 8 }
    ];
    
    displayDocumentList(mockDocuments);
} 