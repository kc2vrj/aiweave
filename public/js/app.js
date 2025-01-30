// public/js/app.js
let currentFile = null;
let isRawView = false;

/**
 * Load and display the list of available files
 */
async function loadFiles() {
    try {
        const response = await fetch('/api/files');
        const files = await response.json();
        
        const fileList = document.getElementById('fileList');
        fileList.innerHTML = '';
        
        files.sort((a, b) => new Date(b.created) - new Date(a.created));
        
        files.forEach(file => {
            const fileDiv = document.createElement('div');
            fileDiv.className = 'p-2 hover:bg-gray-100 rounded cursor-pointer';
            fileDiv.onclick = () => loadFile(file.name);
            fileDiv.innerHTML = `
                <div class="font-medium">${file.name}</div>
                <div class="text-xs text-gray-500">
                    ${formatSize(file.size)} • ${new Date(file.created).toLocaleString()}
                </div>
            `;
            fileList.appendChild(fileDiv);
        });
    } catch (error) {
        console.error('Error loading files:', error);
    }
}

/**
 * Load and display a specific file
 */
async function loadFile(filename) {
    try {
        const response = await fetch(`/api/files/${encodeURIComponent(filename)}`);
        if (!response.ok) {
            throw new Error(`File not found: ${filename}`);
        }
        const data = await response.json();
        
        currentFile = filename;
        document.getElementById('viewerControls').classList.remove('hidden');
        
        displayContent(data.content);
    } catch (error) {
        console.error('Error loading file:', error);
        alert(error.message);
    }
}

/**
 * Display content in the viewer, either as raw text or rendered markdown
 */
function displayContent(content) {
    const viewer = document.getElementById('viewer');
    if (isRawView) {
        viewer.className = 'font-mono text-sm whitespace-pre-wrap bg-gray-50 p-4 rounded-lg h-[600px] overflow-auto';
        viewer.textContent = content;
    } else {
        viewer.className = 'prose max-w-none bg-gray-50 p-4 rounded-lg h-[600px] overflow-auto';
        const rendered = marked.parse(content);
        viewer.innerHTML = rendered;
        // Re-run Prism highlighting on the new content
        Prism.highlightAllUnder(viewer);
    }
}

/**
 * Toggle between raw and rendered views
 */
function toggleRawView() {
    isRawView = !isRawView;
    if (currentFile) {
        loadFile(currentFile);
    }
}

/**
 * Download the current file
 */
function downloadCurrent() {
    if (currentFile) {
        window.location.href = `/api/download/${currentFile}`;
    }
}

/**
 * Start the crawler for a given URL
 */
async function startCrawl() {
    const urlInput = document.getElementById('urlInput').value;
    const githubOnly = document.getElementById('githubOnlyCheckbox').checked;
    const statusDiv = document.getElementById('crawlStatus');

    if (!urlInput) {
        alert('Please enter a URL to crawl.');
        return;
    }

    try {
        // Clear previous status and show starting message
        statusDiv.textContent = 'Starting crawler...\n';

        const response = await fetch('/api/crawl', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                url: urlInput,
                githubOnly: githubOnly
            })
        });

        if (!response.ok) {
            throw new Error('Failed to start crawl.');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        // Process the streaming response
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            console.log('Received chunk:', text);  
            const lines = text.split('\n');

            lines.forEach(line => {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.lines) {
                            statusDiv.textContent = data.lines.join('\n');
                            statusDiv.scrollTop = statusDiv.scrollHeight;
                        }
                    } catch (e) {
                        console.error('Error parsing line:', e);
                    }
                }
            });
        }

        console.log('Crawl started successfully.');

        // Reload file list after crawl completes
        await loadFiles();

    } catch (error) {
        console.error('Error starting crawl:', error);
        statusDiv.textContent += `Error: ${error.message}\n`;
    }
}

/**
 * Format a file size in bytes to a human-readable string
 */
function formatSize(bytes) {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
}

// Event listeners for keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + R to toggle raw view
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        toggleRawView();
    }
    // Ctrl/Cmd + D to download current file
    if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        downloadCurrent();
    }
});

// Initialize the app
function init() {
    // Load initial file list
    loadFiles();

    // Set up auto-refresh of file list
    setInterval(loadFiles, 10000);

    // Configure marked.js options
    marked.setOptions({
        gfm: true,             // GitHub Flavored Markdown
        breaks: true,          // Add <br> on single line breaks
        pedantic: false,       // Don't be too strict with markdown spec
        sanitize: false,       // Allow HTML in markdown
        smartLists: true,      // Use smarter list behavior
        smartypants: true,     // Use smart punctuation
        xhtml: false,          // Don't use XHTML style tags
        highlight: function(code, lang) {
            if (Prism.languages[lang]) {
                return Prism.highlight(code, Prism.languages[lang], lang);
            }
            return code;
        }
    });
}

// Start the app
init();
