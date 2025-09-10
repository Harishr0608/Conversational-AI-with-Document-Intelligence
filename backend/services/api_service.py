import httpx
import asyncio
from bs4 import BeautifulSoup
try:
    from ddgs import DDGS  # Updated import
except ImportError:
    from duckduckgo_search import DDGS  # Fallback for older versions
import os
import logging
from typing import List, Dict, Any, Optional
import json
from urllib.parse import urljoin, urlparse
from newspaper import Article
import requests

logger = logging.getLogger(__name__)

class APIService:
    def __init__(self):
        """Initialize API service for external data retrieval."""
        self.session = httpx.AsyncClient(timeout=30.0)
        self.ddg_enabled = os.getenv('DUCKDUCKGO_ENABLED', 'true').lower() == 'true'
        self.serp_api_key = os.getenv('SERPAPI_KEY')
        logger.info("Initialized API service")

    async def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web using DuckDuckGo."""
        if not self.ddg_enabled:
            return []

        try:
            def _search():
                ddgs = DDGS()
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': r.get('title', ''),
                        'url': r.get('href', ''),
                        'snippet': r.get('body', ''),
                        'source': 'duckduckgo'
                    })
                return results

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, _search)
            logger.info(f"Found {len(results)} search results for: {query}")
            return results

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

    
    async def get_news(self, topic: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Get news articles about a specific topic."""
        try:
            def _search_news():
                ddgs = DDGS()
                results = []
                for r in ddgs.news(topic, max_results=max_results):
                    results.append({
                        'title': r.get('title', ''),
                        'url': r.get('url', ''),
                        'snippet': r.get('body', ''),
                        'date': r.get('date', ''),
                        'source': r.get('source', ''),
                        'type': 'news'
                    })
                return results
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, _search_news)
            
            logger.info(f"Found {len(results)} news articles for: {topic}")
            return results
            
        except Exception as e:
            logger.error(f"News search error: {e}")
            return []
    
    async def scrape_webpage(self, url: str, max_chars: int = 5000) -> Dict[str, Any]:
        """Scrape content from a webpage."""
        try:
            response = await self.session.get(url)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Extract main content (remove scripts, styles, etc.)
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text_content = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length
            if len(text_content) > max_chars:
                text_content = text_content[:max_chars] + "..."
            
            return {
                'url': url,
                'title': title,
                'content': text_content,
                'success': True,
                'source': 'web_scraping'
            }
            
        except Exception as e:
            logger.error(f"Web scraping error for {url}: {e}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'success': False,
                'error': str(e),
                'source': 'web_scraping'
            }
    
    async def extract_article_content(self, url: str) -> Dict[str, Any]:
        """Extract article content using newspaper3k."""
        try:
            def _extract():
                article = Article(url)
                article.download()
                article.parse()
                
                return {
                    'url': url,
                    'title': article.title or '',
                    'content': article.text or '',
                    'authors': article.authors or [],
                    'publish_date': str(article.publish_date) if article.publish_date else '',
                    'top_image': article.top_image or '',
                    'summary': article.summary or '',
                    'success': True,
                    'source': 'newspaper3k'
                }
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _extract)
            
            logger.info(f"Extracted article content from: {url}")
            return result
            
        except Exception as e:
            logger.error(f"Article extraction error for {url}: {e}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'success': False,
                'error': str(e),
                'source': 'newspaper3k'
            }
    
    async def get_weather(self, location: str) -> Dict[str, Any]:
        """Get weather information (using a free weather API)."""
        try:
            # Using OpenWeatherMap's free tier (requires API key)
            # For demo purposes, using a public weather API
            api_url = f"http://api.openweathermap.org/data/2.5/weather"
            
            # If no API key, use wttr.in as fallback
            if not os.getenv('OPENWEATHER_API_KEY'):
                return await self._get_weather_wttr(location)
            
            params = {
                'q': location,
                'appid': os.getenv('OPENWEATHER_API_KEY'),
                'units': 'metric'
            }
            
            response = await self.session.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                'location': data['name'],
                'temperature': data['main']['temp'],
                'description': data['weather'][0]['description'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'success': True,
                'source': 'openweathermap'
            }
            
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return await self._get_weather_wttr(location)
    
    async def _get_weather_wttr(self, location: str) -> Dict[str, Any]:
        """Fallback weather using wttr.in."""
        try:
            url = f"http://wttr.in/{location}?format=j1"
            response = await self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            current = data['current_condition'][0]
            
            return {
                'location': location,
                'temperature': current['temp_C'],
                'description': current['weatherDesc'][0]['value'],
                'humidity': current['humidity'],
                'pressure': current['pressure'],
                'wind_speed': current['windspeedKmph'],
                'success': True,
                'source': 'wttr.in'
            }
            
        except Exception as e:
            logger.error(f"Weather fallback error: {e}")
            return {
                'location': location,
                'success': False,
                'error': str(e)
            }
    
    async def search_and_extract(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search web and extract content from top results."""
        # First, search for relevant URLs
        search_results = await self.search_web(query, max_results)
        
        enhanced_results = []
        for result in search_results:
            url = result.get('url', '')
            if url:
                # Extract content from each URL
                content = await self.extract_article_content(url)
                
                # Combine search result with extracted content
                enhanced_result = {
                    **result,
                    'extracted_content': content.get('content', ''),
                    'full_title': content.get('title', result.get('title', '')),
                    'extraction_success': content.get('success', False)
                }
                enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    async def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic stock information (using free API)."""
        try:
            # Using Alpha Vantage free tier or Yahoo Finance alternative
            # For demo, using a simple approach
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            
            response = await self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                meta = result['meta']
                
                return {
                    'symbol': symbol.upper(),
                    'price': meta.get('regularMarketPrice'),
                    'change': meta.get('regularMarketDayHigh', 0) - meta.get('regularMarketDayLow', 0),
                    'currency': meta.get('currency'),
                    'market_state': meta.get('marketState'),
                    'success': True,
                    'source': 'yahoo_finance'
                }
            
        except Exception as e:
            logger.error(f"Stock info error: {e}")
            
        return {
            'symbol': symbol.upper(),
            'success': False,
            'error': 'Unable to fetch stock information'
        }
    
    async def search_academic(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search for academic papers (using free sources)."""
        try:
            # Search arXiv for academic papers
            def _search_arxiv():
                import urllib.parse
                import xml.etree.ElementTree as ET
                
                query_encoded = urllib.parse.quote(query)
                url = f"http://export.arxiv.org/api/query?search_query=all:{query_encoded}&start=0&max_results={max_results}"
                
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                namespace = {'atom': 'http://www.w3.org/2005/Atom'}
                
                papers = []
                for entry in root.findall('atom:entry', namespace):
                    title = entry.find('atom:title', namespace)
                    summary = entry.find('atom:summary', namespace)
                    authors = entry.findall('atom:author', namespace)
                    link = entry.find('atom:id', namespace)
                    
                    author_names = []
                    for author in authors:
                        name = author.find('atom:name', namespace)
                        if name is not None:
                            author_names.append(name.text)
                    
                    papers.append({
                        'title': title.text if title is not None else '',
                        'summary': summary.text if summary is not None else '',
                        'authors': author_names,
                        'url': link.text if link is not None else '',
                        'source': 'arxiv'
                    })
                
                return papers
            
            loop = asyncio.get_event_loop()
            papers = await loop.run_in_executor(None, _search_arxiv)
            
            logger.info(f"Found {len(papers)} academic papers for: {query}")
            return papers
            
        except Exception as e:
            logger.error(f"Academic search error: {e}")
            return []
    
    async def close(self):
        """Close the HTTP session."""
        await self.session.aclose()
    
    def __del__(self):
        """Cleanup when service is destroyed."""
        try:
            asyncio.create_task(self.close())
        except:
            pass