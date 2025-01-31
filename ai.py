#!/usr/bin/env python3
import os
import re
import gc
import json
import time
import logging
import asyncio
import zipfile
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from groq import AsyncGroq
from openai import AsyncOpenAI
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

load_dotenv()

class RateLimitError(Exception):
    def __init__(self, wait_time, message="Rate limit exceeded"):
        self.wait_time = wait_time
        self.message = message
        super().__init__(self.message)

class APIManager:
    def __init__(self):
        self.groq_rate_limit = 500000  # Default rate limit
        self.groq_used = 0
        self.last_reset_time = time.time()
        self.backoff_time = 1  # Initial backoff time in seconds
        
    async def call_api(self, api_func, *args, **kwargs):
        while True:
            try:
                return await api_func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                if "rate_limit" in error_msg.lower():
                    # Extract wait time from error message
                    wait_time = self._extract_wait_time(error_msg)
                    if wait_time:
                        print(f"Rate limit hit, waiting {wait_time} seconds")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # If we can't extract wait time, use exponential backoff
                    print(f"Using exponential backoff: {self.backoff_time} seconds")
                    await asyncio.sleep(self.backoff_time)
                    self.backoff_time *= 2  # Exponential backoff
                    continue
                
                # For non-rate-limit errors, raise them
                raise

    def _extract_wait_time(self, error_msg):
        try:
            # Look for time patterns like "1m40.3838s" or similar
            match = re.search(r"try again in (\d+)m([\d.]+)s", error_msg)
            if match:
                minutes, seconds = match.groups()
                return float(minutes) * 60 + float(seconds)
            return None
        except:
            return None

    def reset_backoff(self):
        self.backoff_time = 1

class MultiAIProvider:
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.clients = {}
        self.rate_limited_until = {}
        self.request_counts = {}  # Track requests per provider
        self.provider_errors = {}  # Track consecutive errors per provider
        self.max_retries = 3  # Maximum number of retries per provider
        self.initialize_clients()
        self.min_delay = 0.1
        self.api_manager = APIManager()
        # Adjust priority order to spread load
        self.provider_priority = ["openai", "groq", "huggingface"]  # Priority order for fallback
        # Add rate limit windows
        self.rate_limit_windows = {
            "openai": {"requests": 0, "window_start": time.time(), "max_per_minute": 60},
            "groq": {"requests": 0, "window_start": time.time(), "max_per_minute": 50},
            "huggingface": {"requests": 0, "window_start": time.time(), "max_per_minute": 40}
        }

    def initialize_clients(self):
        if self.api_keys.get("Groq_API_KEY"):
            self.clients["groq"] = AsyncGroq(api_key=self.api_keys["Groq_API_KEY"])
            self.request_counts["groq"] = 0
            print("✓ Initialized Groq client")
            
        if self.api_keys.get("HuggingFace_API_KEY"):
            self.clients["huggingface"] = InferenceClient(token=self.api_keys["HuggingFace_API_KEY"])
            self.request_counts["huggingface"] = 0
            print("✓ Initialized Hugging Face client")

        if self.api_keys.get("OpenAI_API_KEY"):
            self.clients["openai"] = AsyncOpenAI(api_key=self.api_keys["OpenAI_API_KEY"])
            self.request_counts["openai"] = 0
            print("✓ Initialized OpenAI client")

    async def get_completion(self, prompt: str) -> Optional[str]:
        """Get completion from the first available provider."""
        while True:
            # Try each provider in order of priority
            for provider in self.provider_priority:
                if provider not in self.clients:
                    continue
                    
                if provider in self.rate_limited_until:
                    wait_time = self.rate_limited_until[provider] - time.time()
                    if wait_time > 0:
                        print(f"{provider} rate limited for {wait_time:.1f}s")
                        continue
                    else:
                        del self.rate_limited_until[provider]
                
                try:
                    if provider == "groq":
                        completion = await self.clients[provider].chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="mixtral-8x7b-32768",
                            temperature=0.1,
                            max_tokens=2000
                        )
                        self.request_counts[provider] += 1
                        return completion.choices[0].message.content
                        
                    elif provider == "openai":
                        completion = await self.clients[provider].chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="gpt-3.5-turbo",
                            temperature=0.1,
                            max_tokens=2000
                        )
                        self.request_counts[provider] += 1
                        return completion.choices[0].message.content
                        
                    elif provider == "huggingface":
                        # Create async function for HuggingFace request
                        async def hf_request():
                            loop = asyncio.get_event_loop()
                            return await loop.run_in_executor(
                                None,
                                lambda: self.clients[provider].text_generation(
                                    prompt,
                                    max_new_tokens=2000,
                                    temperature=0.1,
                                    repetition_penalty=1.1
                                )
                            )
                        
                        completion = await hf_request()
                        self.request_counts[provider] += 1
                        return completion[0]["generated_text"][len(prompt):]
                
                except Exception as e:
                    error_str = str(e).lower()
                    if "rate limit" in error_str or "429" in error_str:
                        # Calculate exponential backoff
                        retries = self.provider_errors.get(provider, 0)
                        wait_time = min(60 * (2 ** retries), 300)  # Max 5 minutes
                        self.provider_errors[provider] = retries + 1
                        self.rate_limited_until[provider] = time.time() + wait_time
                        print(f"{provider} rate limited, waiting {wait_time}s")
                        continue
                    else:
                        print(f"Error from {provider}: {e}")
                        self.provider_errors[provider] = self.provider_errors.get(provider, 0) + 1
                        continue
            
            # If we get here, all providers failed
            await asyncio.sleep(1)  # Wait before retrying all providers

    async def _check_rate_limit(self, provider: str) -> bool:
        """Check if we're within rate limits for the provider"""
        now = time.time()
        window = self.rate_limit_windows[provider]
        
        # Reset window if it's been more than a minute
        if now - window["window_start"] >= 60:
            window["requests"] = 0
            window["window_start"] = now
            
        # Check if we're within limits
        if window["requests"] >= window["max_per_minute"]:
            wait_time = 60 - (now - window["window_start"])
            if wait_time > 0:
                print(f"{provider} rate limit reached, waiting {wait_time:.1f} seconds")
                self.rate_limited_until[provider] = now + wait_time
                return False
                
        window["requests"] += 1
        return True

    async def _get_completion_from_provider(self, provider: str, prompt: str) -> str:
        """Get completion from a specific provider with retries and rate limit handling."""
        task = asyncio.current_task()
        if task:
            task.set_name(provider)
            
        if not await self._check_rate_limit(provider):
            raise Exception(f"{provider} rate limited")
            
        try:
            if provider == "groq":
                response = await self.clients[provider].chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="mixtral-8x7b-32768"
                )
                return response.choices[0].message.content
                
            elif provider == "openai":
                response = await self.clients[provider].chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-3.5-turbo"
                )
                return response.choices[0].message.content
                
            elif provider == "huggingface":
                response = await self.clients[provider].text_generation(
                    prompt,
                    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    max_new_tokens=500,
                    temperature=0.7
                )
                return response[0]["generated_text"]
                
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                # Calculate exponential backoff
                retries = self.provider_errors.get(provider, 0)
                wait_time = min(60 * (2 ** retries), 300)  # Max 5 minutes
                self.rate_limited_until[provider] = time.time() + wait_time
                print(f"{provider} rate limited, backing off for {wait_time:.1f} seconds")
            self.provider_errors[provider] = self.provider_errors.get(provider, 0) + 1
            raise

    def print_stats(self):
        """Print usage statistics for each provider."""
        print("\nAI Provider Usage Statistics:")
        for provider, count in self.request_counts.items():
            print(f"- {provider.capitalize()}: {count} requests")

class FastSiteCrawler:
    def __init__(self, output_dir, ai_provider: MultiAIProvider, max_concurrent=3):  
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent
        self.ai_provider = ai_provider
        self.crawled_urls = set()
        self.failed_urls = set()
        self.to_crawl = set()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_delay = 1.0  
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Optimized browser configuration
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-extensions",
            ],
        )

    def extract_links(self, base_url: str, html_content: str) -> set:
        """Extract links from HTML content."""
        try:
            # Parse base URL components
            parsed_base = urlparse(base_url)
            base_domain = parsed_base.netloc
            
            # For GitHub URLs, get the repo path
            repo_path = None
            if base_domain == 'github.com':
                path_parts = parsed_base.path.strip('/').split('/')
                if len(path_parts) >= 2:
                    # Get owner/repo for GitHub URLs
                    repo_path = '/'.join(path_parts[:2])  # owner/repo
            
            soup = BeautifulSoup(html_content, 'html.parser')
            links = set()
            
            for a in soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)
                
                # Basic URL validation
                if (parsed.netloc == base_domain and 
                    parsed.scheme in ('http', 'https')):
                    
                    # For GitHub URLs, only include links within the same repository
                    if base_domain == 'github.com' and repo_path:
                        url_path = parsed.path.strip('/')
                        if not url_path.startswith(repo_path):
                            continue
                    
                    # Remove fragments and add to set
                    links.add(full_url.split('#')[0])
            
            del soup  # Help garbage collection
            import gc
            gc.collect()  # Force garbage collection
            return links
        except Exception as e:
            print(f"Error extracting links: {e}")
            return set()

    async def process_page(self, url: str, crawler: AsyncWebCrawler) -> set:
        """Process a single page and return new links."""
        try:
            async with self.semaphore:
                # Add delay between requests
                await asyncio.sleep(self.request_delay)
                
                print(f"Processing: {url}")
                result = await crawler.arun(
                    url=url,
                    config=CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS
                    )
                )

                if result and result.success:
                    try:
                        # Extract links before processing content
                        new_links = set()
                        if result.html:
                            new_links = self.extract_links(url, result.html)
                            # Clear HTML content immediately
                            result.html = None
                            gc.collect()

                        # Process markdown content if available
                        if result.markdown_v2 and result.markdown_v2.raw_markdown:
                            markdown_content = result.markdown_v2.raw_markdown
                            # Clear markdown object
                            result.markdown_v2 = None
                            gc.collect()

                            # Process the content
                            processed_content = await self.process_content(markdown_content, url)
                            # Clear original content
                            markdown_content = None
                            gc.collect()

                            # Save processed content
                            if processed_content:
                                self.save_markdown(url, processed_content)
                                processed_content = None
                                gc.collect()

                        # Process human-readable content
                        human_readable_content = await self.process_human_readable_content(result.html, url)
                        if human_readable_content:
                            self.save_human_readable_content(url, human_readable_content)
                            human_readable_content = None
                            gc.collect()

                        # Mark as crawled
                        self.crawled_urls.add(url)
                        print(f"✓ Processed: {url}")
                        
                        return new_links

                    finally:
                        # Clear result object
                        result = None
                        gc.collect()

                return set()

        except Exception as e:
            print(f"Error processing page {url}: {e}")
            self.failed_urls.add(url)
            return set()

    async def process_content(self, content: str, url: str) -> str:
        """Process content using AI providers."""
        if not content:
            return f"# {url}\n\nNo content available"

        # Add a small delay to avoid overwhelming the AI providers
        await asyncio.sleep(0.5)

        # Truncate content more aggressively for longer pages
        content_length = len(content)
        if content_length > 8000:
            content_preview = content[:800] + f"\n\n... {content_length - 1600} characters truncated ...\n\n" + content[-800:]
        else:
            content_preview = content[:2000]

        prompt = f"""Analyze and summarize the following webpage content in markdown format:

URL: {url}

Create a concise document with:
1. A clear, descriptive title (use # for h1)
2. A brief 1-2 sentence overview
3. Key points or features (use bullet points)
4. Any relevant technical details or specifications
5. Additional sections if needed (use ## for h2)

Content to analyze:
{content_preview}"""

        max_retries = 3
        retry_delay = 1
        last_error = None

        for attempt in range(max_retries):
            try:
                result = await self.ai_provider.get_completion(prompt)
                if result:
                    return result
            except Exception as e:
                last_error = e
                error_msg = str(e)
                print(f"Error processing content (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limit hit, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                    continue
                    
                # For non-rate-limit errors, try the next attempt immediately
                continue

        # If all retries failed, create a basic markdown document
        print(f"All processing attempts failed: {last_error}")
        return f"""# {url}

Failed to process with AI due to errors. Showing raw content preview:

{content_preview}"""

    async def process_human_readable_content(self, html_content: str, url: str) -> str:
        """Process human-readable content."""
        if not html_content:
            return f"# {url}\n\nNo content available"

        # Remove script and style elements
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it up
        text = soup.get_text(separator='\n', strip=True)
        
        # Remove extra whitespace and empty lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n\n'.join(lines)
        
        return text

    def save_markdown(self, url: str, content: str) -> None:
        """Save the markdown content to a file with proper subdirectory structure."""
        try:
            # Parse the URL
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.strip('/').split('/')
            
            # Create directory structure
            if parsed_url.netloc == 'github.com':
                # For GitHub URLs, create subdirs for owner/repo
                if len(path_parts) >= 2:
                    # Create directory structure: domain/owner/repo/
                    subdir = os.path.join(self.output_dir, parsed_url.netloc, *path_parts[:2])
                    
                    # Create filename from remaining path parts
                    if len(path_parts) > 2:
                        filename = '-'.join(path_parts[2:])
                    else:
                        filename = 'index'
                else:
                    # Fallback for short paths
                    subdir = os.path.join(self.output_dir, parsed_url.netloc)
                    filename = path_parts[0] if path_parts else 'index'
            else:
                # For other URLs, use domain as subdir
                subdir = os.path.join(self.output_dir, parsed_url.netloc)
                if path_parts:
                    filename = '-'.join(path_parts)
                else:
                    filename = 'index'
            
            # Clean the filename
            filename = re.sub(r'[^\w\-\.]', '-', filename)
            filename = re.sub(r'-+', '-', filename).strip('-')
            
            # Ensure the filename ends with .md
            if not filename.endswith('.md'):
                filename += '.md'
            
            # Create the full file path
            filepath = os.path.join(subdir, filename)
            
            # Create all necessary directories
            os.makedirs(subdir, exist_ok=True)
            
            print(f"Saving file to: {filepath}")
            
            # Write the content with YAML frontmatter
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"---\nurl: {url}\n---\n\n")
                f.write(content)
            
            print(f"✓ Saved: {filename}")
            
        except Exception as e:
            print(f"Error saving file: {e}")
            raise

    def save_human_readable_content(self, url: str, content: str) -> None:
        """Save the human-readable content to a file with proper subdirectory structure."""
        try:
            # Parse the URL
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.strip('/').split('/')
            
            # Create directory structure
            if parsed_url.netloc == 'github.com':
                # For GitHub URLs, create subdirs for owner/repo
                if len(path_parts) >= 2:
                    # Create directory structure: domain/owner/repo/
                    subdir = os.path.join(self.output_dir, parsed_url.netloc, *path_parts[:2])
                    
                    # Create filename from remaining path parts
                    if len(path_parts) > 2:
                        filename = '-'.join(path_parts[2:])
                    else:
                        filename = 'index'
                else:
                    # Fallback for short paths
                    subdir = os.path.join(self.output_dir, parsed_url.netloc)
                    filename = path_parts[0] if path_parts else 'index'
            else:
                # For other URLs, use domain as subdir
                subdir = os.path.join(self.output_dir, parsed_url.netloc)
                if path_parts:
                    filename = '-'.join(path_parts)
                else:
                    filename = 'index'
            
            # Clean the filename
            filename = re.sub(r'[^\w\-\.]', '-', filename)
            filename = re.sub(r'-+', '-', filename).strip('-')
            
            # Ensure the filename ends with .txt
            if not filename.endswith('.txt'):
                filename += '.txt'
            
            # Create the full file path
            filepath = os.path.join(subdir, filename)
            
            # Create all necessary directories
            os.makedirs(subdir, exist_ok=True)
            
            print(f"Saving file to: {filepath}")
            
            # Write the content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ Saved: {filename}")
            
        except Exception as e:
            print(f"Error saving file: {e}")
            raise

    def create_download_zip(self) -> str:
        """Create a zip file containing all markdown files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = os.path.join(self.output_dir, f"markdown_export_{timestamp}.zip")
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the output directory
            for root, _, files in os.walk(self.output_dir):
                for file in files:
                    if file.endswith('.md') or file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        # Calculate path relative to output_dir for the archive
                        arcname = os.path.relpath(file_path, self.output_dir)
                        zipf.write(file_path, arcname)
        
        print(f"\nCreated zip archive: {zip_filename}")
        print(f"Contains {sum(1 for _ in zipfile.ZipFile(zip_filename).namelist())} files")
        return zip_filename

    async def crawl_site(self, start_url: str, github_only: bool = False):
        """Crawl a website starting from the given URL."""
        start_time = time.time()
        
        # Enable garbage collection
        gc.enable()
        gc.set_threshold(100, 5, 5)  # More aggressive GC

        try:
            # Validate URL
            try:
                result = urlparse(start_url)
                if not all([result.scheme, result.netloc]):
                    raise ValueError(f"Invalid URL format: {start_url}")
                if not result.scheme in ['http', 'https']:
                    raise ValueError(f"Invalid URL scheme: {result.scheme}")
                    
                if github_only and result.netloc != 'github.com':
                    raise ValueError("URL must be a GitHub repository when using --github-only flag")
                    
                # Handle GitHub repository setup
                if github_only:
                    path_parts = result.path.strip('/').split('/')
                    if len(path_parts) >= 2:
                        owner, repo = path_parts[0], path_parts[1]
                        project_name = f"{owner}/{repo}"
                        self.output_dir = os.path.join(self.output_dir, project_name)
                        os.makedirs(self.output_dir, exist_ok=True)
                        print(f"GitHub project: {project_name}")
                        print(f"Output directory: {self.output_dir}")
                    else:
                        raise ValueError("Invalid GitHub repository URL format")
                        
            except Exception as e:
                print(f"URL validation error: {e}")
                return

            print(f"Starting crawler...")
            
            # Initialize the crawler
            crawler = AsyncWebCrawler()
            await crawler.start()
            
            try:
                # Process URLs in smaller batches
                self.to_crawl.add(start_url)
                batch_size = 2  # Reduce batch size
                
                while self.to_crawl:
                    print(f"Progress: {len(self.crawled_urls)} crawled, {len(self.to_crawl)} pending, {len(self.failed_urls)} failed")
                    provider_status = []
                    for provider, count in self.ai_provider.request_counts.items():
                        status = " (active)" if provider in self.ai_provider.clients else ""
                        if provider in self.ai_provider.rate_limited_until:
                            wait_time = self.ai_provider.rate_limited_until[provider] - time.time()
                            if wait_time > 0:
                                status = f" (rate limited for {wait_time:.1f}s)"
                        provider_status.append(f"{provider}: {count} reqs{status}")
                    print("Providers:", " | ".join(provider_status))

                    try:
                        # Process current batch
                        current_batch = set()
                        for _ in range(min(batch_size, len(self.to_crawl))):
                            if self.to_crawl:
                                current_batch.add(self.to_crawl.pop())

                        # Create tasks for the batch
                        tasks = [self.process_page(url, crawler) for url in current_batch]
                        
                        # Wait for all tasks to complete
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Process results and collect new links
                        for url, result in zip(current_batch, results):
                            if isinstance(result, Exception):
                                print(f"Failed to process {url}: {str(result)}")
                                self.failed_urls.add(url)
                            elif result:  # If we got new links
                                # Filter links based on github_only flag
                                if github_only:
                                    base_path = f"/{path_parts[0]}/{path_parts[1]}"
                                    result = {link for link in result 
                                            if urlparse(link).netloc == 'github.com' 
                                            and urlparse(link).path.startswith(base_path)}
                                
                                # Add new links that haven't been processed
                                new_links = result - self.crawled_urls - self.to_crawl - self.failed_urls
                                self.to_crawl.update(new_links)
                                
                                # Clear result set
                                result = None
                        
                        # Force garbage collection after batch
                        gc.collect()
                        
                        # Add a small delay between batches
                        await asyncio.sleep(0.5)
                    
                    except Exception as e:
                        print(f"Batch processing error: {e}")
                        continue

            finally:
                await crawler.close()
                gc.collect()

            # Final report
            print("\nCrawl completed!")
            print(f"Successfully crawled: {len(self.crawled_urls)} pages")
            print(f"Failed to crawl: {len(self.failed_urls)} pages")
            
            duration = time.time() - start_time
            print(f"Total time: {duration:.2f} seconds")
            if self.crawled_urls:
                print(f"Average time per page: {duration/len(self.crawled_urls):.2f} seconds")

            if self.failed_urls:
                print("\nFailed URLs:")
                for url in self.failed_urls:
                    print(f"- {url}")

            print("\nAI Provider Usage Statistics:")
            self.ai_provider.print_stats()
            
            # Create download zip
            zip_file = self.create_download_zip()
            print(f"\nDownload your files: {zip_file}")

        except Exception as e:
            print(f"Critical error during crawl: {e}")
        
        finally:
            # Final cleanup
            gc.collect()

import argparse
import itertools

def clean_filename(text):
    # Remove non-word characters (everything except numbers, letters and '-_')
    text = re.sub(r'[^\w\s-]', '', text.lower())
    # Replace all runs of whitespace with a single dash
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')

def get_human_readable_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text and clean it up
    text = soup.get_text(separator='\n', strip=True)
    
    # Remove extra whitespace and empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = '\n\n'.join(lines)
    
    return text

def get_file_hash(content):
    """Generate a hash of the file content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def save_content(url, content, base_dir='crawled_docs'):
    # Create directory for the domain
    parsed_url = urlparse(url)
    domain_dir = os.path.join(base_dir, parsed_url.netloc)
    os.makedirs(domain_dir, exist_ok=True)
    
    # Create a clean filename from the URL path
    path_part = parsed_url.path.strip('/')
    if not path_part:
        path_part = 'index'
    
    # Clean the filename
    clean_name = clean_filename(path_part)
    if not clean_name:
        clean_name = 'index'
    
    # Get human readable content
    human_readable = get_human_readable_content(content)
    
    # Prepare the full content with URL header
    full_content = f"# {url}\n\n{human_readable}"
    
    # Generate hash of the new content
    new_hash = get_file_hash(full_content)
    
    # Check both .md and .txt extensions
    md_filepath = os.path.join(domain_dir, f"{clean_name}.md")
    txt_filepath = os.path.join(domain_dir, f"{clean_name}.txt")
    
    # Check if file exists and compare content
    if os.path.exists(md_filepath):
        with open(md_filepath, 'r', encoding='utf-8') as f:
            existing_content = f.read()
            existing_hash = get_file_hash(existing_content)
            
            if existing_hash == new_hash:
                print(f"Skipping (unchanged): {md_filepath}")
                return md_filepath
    
    # Save the markdown content
    with open(md_filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    # Save the plain text content
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write(human_readable)
    
    print(f"{'Updated' if os.path.exists(md_filepath) else 'Saved'}: {md_filepath}")
    return md_filepath

def crawl(url, visited=None, max_pages=10):
    if visited is None:
        visited = set()
    
    if len(visited) >= max_pages:
        return
    
    if url in visited:
        return
    
    try:
        print(f"Crawling: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        visited.add(url)
        
        # Save the page content
        save_content(url, response.text)
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links on the page
        links = soup.find_all('a', href=True)
        
        # Process each link
        for link in links:
            href = link['href']
            full_url = urljoin(url, href)
            
            # Skip if not same domain
            if urlparse(full_url).netloc != urlparse(url).netloc:
                continue
            
            # Skip fragments and queries
            full_url = full_url.split('#')[0].split('?')[0]
            
            # Skip non-HTML files
            if any(full_url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif']):
                continue
            
            # Recursively crawl the linked page
            crawl(full_url, visited, max_pages)
    
    except Exception as e:
        print(f"Error crawling {url}: {str(e)}")

async def main():
    parser = argparse.ArgumentParser(description='Memory-efficient multi-AI provider web crawler')
    parser.add_argument('url', help='Starting URL to crawl')
    parser.add_argument('--output-dir', default='crawled_docs', help='Output directory')
    parser.add_argument('--max-concurrent', type=int, default=5, help='Maximum concurrent crawls')
    parser.add_argument('--config', default='api-keys.json', help='API keys configuration file')
    parser.add_argument('--github-only', action='store_true', help='Crawl only a GitHub repository')
    args = parser.parse_args()

    if args.github_only and 'github.com' not in urlparse(args.url).netloc:
        print("Error: The URL must be a GitHub repository when using --github-only flag.")
        return

    try:
        with open(args.config) as f:
            api_keys = json.load(f)
    except FileNotFoundError:
        print(f"Config file {args.config} not found")
        return

    ai_provider = MultiAIProvider(api_keys)
    crawler = FastSiteCrawler(args.output_dir, ai_provider, args.max_concurrent)
    await crawler.crawl_site(args.url, args.github_only)

if __name__ == "__main__":
    asyncio.run(main())
