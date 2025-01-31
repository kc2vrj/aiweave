/**
 * Web Viewer Server
 * Express.js server implementation for web content viewing and processing
 * @module server
 */

const express = require('express');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const archiver = require('archiver');
const app = express();
const port = 3000;

// Track active crawl processes and their outputs
const activeCrawls = new Map();
const crawlHistory = new Map();

// Configure middleware for parsing requests and serving static files
app.use(express.json());
app.use(express.static('public'));

// Create necessary directories
const outputDir = path.join(__dirname, 'crawled_docs');
const logsDir = path.join(__dirname, 'logs');
[outputDir, logsDir].forEach(dir => {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
});

/**
 * Validate URL
 * @param {string} url - URL to validate
 * @returns {boolean} True if the URL is valid, false otherwise
 */
function isValidUrl(url) {
    try {
        new URL(url);
        return true;
    } catch (e) {
        return false;
    }
}

// Get crawl history
app.get('/api/crawl/history', (req, res) => {
    const history = Array.from(crawlHistory.entries()).map(([id, data]) => ({
        id,
        url: data.url,
        status: data.status,
        startTime: data.startTime,
        endTime: data.endTime
    }));
    res.json(history);
});

// Get crawl output
app.get('/api/crawl/output/:sessionId', (req, res) => {
    const sessionId = req.params.sessionId;
    const logFile = path.join(logsDir, `${sessionId}.log`);
    
    try {
        if (fs.existsSync(logFile)) {
            const output = fs.readFileSync(logFile, 'utf8');
            res.json({ output });
        } else {
            res.json({ output: '' });
        }
    } catch (error) {
        res.status(500).json({ error: 'Failed to read log file' });
    }
});

// Get crawl status
app.get('/api/crawl/status/:sessionId', (req, res) => {
    const sessionId = req.params.sessionId;
    const crawl = activeCrawls.get(sessionId);
    const history = crawlHistory.get(sessionId);
    
    if (!crawl && !history) {
        return res.json({ status: 'not_found' });
    }
    
    // If crawl is active, read from memory
    if (crawl) {
        res.json({
            status: crawl.status,
            output: crawl.output,
            error: crawl.error
        });
        return;
    }
    
    // If crawl is in history, read from file
    try {
        const logFile = path.join(logsDir, `${sessionId}.log`);
        const output = fs.existsSync(logFile) ? fs.readFileSync(logFile, 'utf8') : '';
        res.json({
            status: history.status,
            output: output,
            error: history.error
        });
    } catch (error) {
        res.status(500).json({ error: 'Failed to read log file' });
    }
});

// Stop crawl
app.post('/api/crawl/stop/:sessionId', (req, res) => {
    const sessionId = req.params.sessionId;
    const crawl = activeCrawls.get(sessionId);
    
    if (!crawl || !crawl.process) {
        return res.json({ success: false, message: 'No active crawl found' });
    }
    
    try {
        // Kill the process and its children
        process.kill(-crawl.process.pid);
        crawl.status = 'stopped';
        
        // Update history
        const history = crawlHistory.get(sessionId);
        if (history) {
            history.status = 'stopped';
            history.endTime = Date.now();
        }
        
        // Save final output to file
        const logFile = path.join(logsDir, `${sessionId}.log`);
        fs.writeFileSync(logFile, crawl.output);
        
        res.json({ success: true });
    } catch (error) {
        res.json({ success: false, message: error.message });
    }
});

// Trigger a new web crawl
app.post('/api/crawl', async (req, res) => {
    const { url } = req.body;

    if (!url) {
        return res.status(400).json({ error: 'URL is required' });
    }

    if (!isValidUrl(url)) {
        return res.status(400).json({ error: 'Invalid URL format' });
    }

    // Generate session ID
    const sessionId = Date.now().toString();
    
    // Start python process in its own process group
    const pythonProcess = spawn('python3', ['ai.py', url], {
        detached: true
    });
    
    // Initialize crawl state
    const crawl = {
        process: pythonProcess,
        status: 'running',
        output: '',
        error: '',
        startTime: Date.now()
    };
    
    // Add to active crawls
    activeCrawls.set(sessionId, crawl);
    
    // Add to history
    crawlHistory.set(sessionId, {
        url,
        status: 'running',
        startTime: Date.now(),
        endTime: null
    });
    
    // Create log file
    const logFile = path.join(logsDir, `${sessionId}.log`);

    pythonProcess.stdout.on('data', (data) => {
        const text = data.toString();
        crawl.output += text;
        fs.appendFileSync(logFile, text);
        console.log(text);
    });

    pythonProcess.stderr.on('data', (data) => {
        const text = data.toString();
        crawl.error += text;
        fs.appendFileSync(logFile, `Error: ${text}`);
        console.error(text);
    });

    pythonProcess.on('close', (code) => {
        crawl.status = 'completed';
        crawl.exitCode = code;
        
        // Update history
        const history = crawlHistory.get(sessionId);
        if (history) {
            history.status = 'completed';
            history.endTime = Date.now();
        }
        
        // Save final output to file
        fs.writeFileSync(logFile, crawl.output);
        
        // Keep the result for 1 hour
        setTimeout(() => {
            activeCrawls.delete(sessionId);
        }, 60 * 60 * 1000);
    });
    
    // Return session ID immediately
    res.json({ 
        sessionId,
        message: 'Crawl started'
    });
});

// Helper function to get a clean display name for a file
function getDisplayName(filename) {
    // Remove domain directory if present
    filename = filename.split('/').pop();
    
    // Remove the hash part if it exists
    filename = filename.replace(/blob-[a-f0-9]+-/, '');
    
    // Remove file extension
    filename = filename.replace(/\.(md|txt)$/, '');
    
    // Replace dashes with spaces
    filename = filename.replace(/-/g, ' ');
    
    // Capitalize words
    filename = filename.split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
    
    return filename;
}

// Get list of all files (recursively)
app.get('/api/files', async (req, res) => {
    try {
        const outputDir = path.join(__dirname, 'crawled_docs');
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
            return res.json([]); // Return empty array if directory was just created
        }

        // Helper function to get files recursively
        async function getFiles(dir) {
            const entries = await fs.promises.readdir(dir, { withFileTypes: true });
            const files = [];

            for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);
                if (!fullPath.startsWith(outputDir)) {
                    continue; // Skip files outside output directory
                }

                const relativePath = path.relative(outputDir, fullPath).replace(/\\/g, '/');

                if (entry.isDirectory()) {
                    const subFiles = await getFiles(fullPath);
                    files.push(...subFiles);
                } else {
                    // Only include .md files (not .txt files or .zip files)
                    if (entry.name.endsWith('.md') && !entry.name.startsWith('.')) {
                        // Get domain from the path
                        const pathParts = relativePath.split('/');
                        const domain = pathParts.length > 1 ? pathParts[0] : null;
                        
                        // Skip files that aren't in a domain directory
                        if (!domain) continue;
                        
                        // Get clean display name
                        const displayName = getDisplayName(entry.name);
                        
                        // Skip files with empty display names
                        if (!displayName.trim()) continue;
                        
                        files.push({
                            path: relativePath,
                            displayName: displayName,
                            domain: domain
                        });
                    }
                }
            }

            return files;
        }

        // Helper function to get a clean display name
        function getDisplayName(filename) {
            // Remove file extension
            filename = filename.replace(/\.md$/, '');
            
            // Remove the hash part if it exists
            filename = filename.replace(/blob-[a-f0-9]+-/, '');
            
            // Replace dashes with spaces
            filename = filename.replace(/-/g, ' ');
            
            // Capitalize words
            filename = filename.split(' ')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
            
            return filename;
        }

        const files = await getFiles(outputDir);
        res.json(files);
    } catch (error) {
        console.error('Error listing files:', error);
        res.status(500).json({ error: 'Failed to list files' });
    }
});

// View file content
app.get('/api/view/:filename(*)', async (req, res) => {
    try {
        const { filename } = req.params;
        const filePath = path.join(__dirname, 'crawled_docs', filename);
        
        // Ensure the file is within the output directory
        if (!filePath.startsWith(path.join(__dirname, 'crawled_docs'))) {
            return res.status(400).json({ error: 'Invalid file path' });
        }
        
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ error: 'File not found' });
        }
        
        const content = await fs.promises.readFile(filePath, 'utf8');
        res.json({ content });
    } catch (error) {
        console.error('Error viewing file:', error);
        res.status(500).json({ error: 'Failed to read file' });
    }
});

// Download a file
app.get('/api/download/:filename(*)', async (req, res) => {
    try {
        const { filename } = req.params;
        const filePath = path.join(__dirname, 'crawled_docs', filename);
        
        // Ensure the file is within the output directory
        if (!filePath.startsWith(path.join(__dirname, 'crawled_docs'))) {
            return res.status(400).json({ error: 'Invalid file path' });
        }
        
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ error: 'File not found' });
        }
        
        res.download(filePath);
    } catch (error) {
        console.error('Error downloading file:', error);
        res.status(500).json({ error: 'Failed to download file' });
    }
});

// Delete files
app.delete('/api/files', async (req, res) => {
    const { files } = req.body;
    console.log('Received delete request for files:', files);
    
    if (!Array.isArray(files) || files.length === 0) {
        return res.status(400).json({ error: 'No files specified for deletion' });
    }
    
    const results = {
        success: [],
        failed: []
    };
    
    for (const filename of files) {
        try {
            const filePath = path.join(__dirname, 'crawled_docs', filename);
            console.log('Attempting to delete:', filePath);
            
            // Ensure the file is within the output directory
            if (!filePath.startsWith(path.join(__dirname, 'crawled_docs'))) {
                results.failed.push({ file: filename, error: 'Invalid file path' });
                continue;
            }
            
            if (fs.existsSync(filePath)) {
                await fs.promises.unlink(filePath);
                results.success.push(filename);
            } else {
                results.failed.push({ file: filename, error: 'File not found' });
            }
        } catch (error) {
            console.error('Error deleting file:', filename, error);
            results.failed.push({ file: filename, error: error.message });
        }
    }
    
    console.log('Delete operation results:', results);
    res.json(results);
});

// Create a zip archive of selected files
app.post('/api/create-zip', async (req, res) => {
    try {
        const { files } = req.body;
        if (!Array.isArray(files) || files.length === 0) {
            return res.status(400).json({ error: 'No files specified' });
        }

        // Get the domain from the first file
        const firstFile = files[0];
        const domainDir = firstFile.split('/')[0];
        
        // Create timestamp for zip filename
        const timestamp = new Date().toISOString().replace(/[-:]/g, '').split('.')[0];
        const zipFilename = `${domainDir}_${timestamp}.zip`;
        const zipPath = path.join(__dirname, 'crawled_docs', domainDir, zipFilename);

        // Create write stream
        const output = fs.createWriteStream(zipPath);
        const archive = archiver('zip', {
            zlib: { level: 9 } // Maximum compression
        });

        // Listen for archive events
        output.on('close', () => {
            console.log(`Archive created: ${zipPath}`);
            console.log(`Total bytes: ${archive.pointer()}`);
            res.json({ 
                success: true, 
                filename: path.join(domainDir, zipFilename)
            });
        });

        archive.on('error', (err) => {
            throw err;
        });

        // Pipe archive data to the file
        archive.pipe(output);

        // Add each file to the archive
        for (const file of files) {
            const filePath = path.join(__dirname, 'crawled_docs', file);
            // Use the file name without the domain prefix in the zip
            const zipPath = file.split('/').slice(1).join('/');
            archive.file(filePath, { name: zipPath });
        }

        // Finalize the archive
        await archive.finalize();

    } catch (error) {
        console.error('Error creating zip:', error);
        res.status(500).json({ error: 'Failed to create zip file' });
    }
});

// Download a zip file
app.get('/api/download-zip/:filename(*)', async (req, res) => {
    try {
        const { filename } = req.params;
        const filePath = path.join(__dirname, 'crawled_docs', filename);
        
        // Ensure the file is within the crawled_docs directory
        if (!filePath.startsWith(path.join(__dirname, 'crawled_docs'))) {
            return res.status(400).json({ error: 'Invalid file path' });
        }
        
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ error: 'Zip file not found' });
        }
        
        res.download(filePath);
    } catch (error) {
        console.error('Error downloading zip:', error);
        res.status(500).json({ error: 'Failed to download zip file' });
    }
});

// Start server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});

module.exports = app;
