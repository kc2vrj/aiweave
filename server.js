// server.js
const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const url = require('url');
const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Get list of markdown files
app.get('/api/files', async (req, res) => {
    try {
        const files = await fs.readdir('./crawled_docs');
        const markdownFiles = files.filter(file => file.endsWith('.md'));
        const fileDetails = await Promise.all(markdownFiles.map(async (file) => {
            const stats = await fs.stat(path.join('./crawled_docs', file));
            return {
                name: file,
                size: stats.size,
                created: stats.birthtime
            };
        }));
        res.json(fileDetails);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get content of a specific markdown file
app.get('/api/files/:filename', async (req, res) => {
    try {
        const filePath = path.join('./crawled_docs', req.params.filename);
        const content = await fs.readFile(filePath, 'utf8');
        res.json({ content });
    } catch (error) {
        res.status(404).json({ error: 'File not found' });
    }
});

// Download a markdown file
app.get('/api/download/:filename', async (req, res) => {
    try {
        const filePath = path.join('./crawled_docs', req.params.filename);
        res.download(filePath);
    } catch (error) {
        res.status(404).json({ error: 'File not found' });
    }
});

// Trigger a new crawl
app.post('/api/crawl', (req, res) => {
    const { url } = req.body;
    if (!url) {
        return res.status(400).json({ error: 'URL is required' });
    }

    const outputBuffer = [];
    const MAX_LINES = 10;

    const crawler = spawn('python3', ['ai', url]);
    
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    function sendLines() {
        res.write(`data: ${JSON.stringify({ lines: outputBuffer })}\n\n`);
    }

    crawler.stdout.on('data', (data) => {
        const lines = data.toString().split('\n');
        lines.forEach(line => {
            if (line.trim()) {
                outputBuffer.push(line.trim());
                if (outputBuffer.length > MAX_LINES) {
                    outputBuffer.shift();
                }
                sendLines();
            }
        });
    });

    crawler.stderr.on('data', (data) => {
        const line = `ERROR: ${data}`;
        outputBuffer.push(line.trim());
        if (outputBuffer.length > MAX_LINES) {
            outputBuffer.shift();
        }
        sendLines();
    });

    crawler.on('close', (code) => {
        outputBuffer.push(`Crawler finished with code ${code}`);
        if (outputBuffer.length > MAX_LINES) {
            outputBuffer.shift();
        }
        sendLines();
        res.end();
    });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
