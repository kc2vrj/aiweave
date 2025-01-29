# AIWeave 🕸️

AIWeave is a powerful, multi-provider AI web crawler that transforms websites into well-structured markdown documentation. By leveraging various AI providers (Groq, OpenAI, Anthropic, etc.), it intelligently processes and organizes web content while maintaining the original structure and context.

## 🌟 Features

- **Multi-AI Provider Support**: Seamlessly integrates with multiple AI services:
  - Groq
  - OpenAI
  - Anthropic/Claude
  - Google Gemini
  - HuggingFace
  - Deepseek
  - Mistral
  - And more...

- **Intelligent Processing**:
  - Automatic content summarization
  - Smart chunking of large documents
  - Preservation of code blocks and formatting
  - Fallback between AI providers

- **Flexible Crawling**:
  - Single page or entire site crawling
  - Respects site structure
  - Concurrent processing
  - Domain-bound crawling

- **Clean Output**:
  - Organized markdown files
  - Preserved metadata
  - Structured content hierarchy
  - Easy to integrate with documentation systems

## 🚀 Quick Start

1. **Installation**
```bash
# Clone the repository
git clone https://github.com/yourusername/aiweave.git
cd aiweave

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp .env.template .env
cp api-keys-template.json api-keys.json
```

2. **Configure API Keys**
```bash
# Edit api-keys.json with your API keys
nano api-keys.json
```

3. **Basic Usage**
```bash
# Crawl a single page
python3 ai https://example.com/page

# Crawl an entire site
python3 ai https://example.com --max-concurrent 3

# Use a specific AI provider
python3 ai https://example.com --ai-provider groq
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file with your configuration:
```env
MAX_CONCURRENT_CRAWLS=5
DEFAULT_OUTPUT_DIR="crawled_docs"
CRAWL_DELAY=1
MAX_DEPTH=10
```

### API Keys
Configure your AI provider keys in `api-keys.json`:
```json
{
  "Groq_API_KEY": "your-key-here",
  "OpenAI_API_KEY": "your-key-here",
  ...
}
```

## 📚 Command Line Options

```bash
python3 ai [URL] [OPTIONS]

Options:
  --output-dir DIR      Output directory (default: crawled_docs)
  --max-concurrent N    Maximum concurrent crawls (default: 5)
  --ai-provider NAME    Preferred AI provider
  --config FILE         API keys configuration file
```

## 🗂️ Output Structure

```
crawled_docs/
├── index.md
├── about.md
├── docs/
│   ├── getting-started.md
│   └── advanced-usage.md
└── ...
```

Each markdown file includes:
```markdown
---
url: https://original.url/path
---

# Title
[Processed content...]
```

## 🛠️ Advanced Usage

### Custom AI Provider Selection
```bash
# Use specific provider with fallback
python3 ai https://example.com --ai-provider groq

# Specify output directory
python3 ai https://example.com --output-dir ./my-docs
```

### Processing Options
```bash
# Adjust concurrent crawls
python3 ai https://example.com --max-concurrent 3

# Use custom config file
python3 ai https://example.com --config my-config.json
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all the AI providers for their amazing APIs
- Built with [crawl4ai](https://github.com/coleam00/crawl4ai)
- Inspired by the need for better documentation tools

## ⚠️ Disclaimer

This tool is intended for ethical web crawling. Please respect websites' robots.txt files and terms of service. Always ensure you have permission to crawl and process content.

## 📧 Contact

- Project Link: [https://github.com/yourusername/aiweave](https://github.com/yourusername/aiweave)
- Report Issues: [https://github.com/yourusername/aiweave/issues](https://github.com/yourusername/aiweave/issues)

## 📦 Dependencies

See `requirements.txt` for a complete list of dependencies:
- groq
- openai
- anthropic
- google-generativeai
- huggingface-hub
- deepseek-ai
- mistralai
- beautifulsoup4
- crawl4ai
- python-dotenv
- requests
