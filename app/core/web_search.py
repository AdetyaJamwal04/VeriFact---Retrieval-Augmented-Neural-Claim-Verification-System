"""
Web Search Module - Dual API Support (Tavily + DuckDuckGo) with Optimizations

Uses Tavily as primary (faster, better quality) when API key is available,
falls back to DuckDuckGo (free, no API key required) otherwise.

Optimizations applied:
- Cached Tavily/Brave API clients as singletons (Rank 7)
"""

from typing import List, Dict
import os
import time
import logging

logger = logging.getLogger(__name__)

# Social media domains - included but with lower credibility weight
SOCIAL_MEDIA_DOMAINS = ["twitter.com", "x.com", "facebook.com", "reddit.com", "instagram.com"]

# Optimization #7: Cached API clients
_tavily_client = None
_brave_session = None


def _get_tavily_client():
    """Get or create cached Tavily client."""
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient
        api_key = os.getenv('TAVILY_API_KEY')
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set")
        _tavily_client = TavilyClient(api_key=api_key)
        logger.info("✓ Tavily client initialized")
    return _tavily_client


def _get_brave_session():
    """Get or create cached Brave session."""
    global _brave_session
    if _brave_session is None:
        import requests
        _brave_session = requests.Session()
        _brave_session.headers.update({
            'Accept': 'application/json',
            'X-Subscription-Token': os.getenv('BRAVE_API_KEY', '')
        })
        logger.info("✓ Brave session initialized")
    return _brave_session


def _tavily_search(queries: List[str], max_results: int = 6) -> List[Dict]:
    """
    Search using Tavily API (faster, better quality).
    Requires TAVILY_API_KEY environment variable.
    """
    all_results = []
    seen_urls = set()
    
    tavily = _get_tavily_client()
    
    for query in queries:
        try:
            response = tavily.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_domains=[],
                exclude_domains=[]
            )
            
            for result in response.get('results', []):
                url = result.get('url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        'href': url,
                        'title': result.get('title', ''),
                        'body': result.get('content', '')
                    })
            
            time.sleep(0.3)  # Rate limiting
            
        except Exception as e:
            logger.warning(f"Tavily search failed for '{query}': {e}")
            continue
    
    return all_results


def _brave_search(queries: List[str], max_results: int = 6) -> List[Dict]:
    """
    Search using Brave Search API (reliable, 2000 free requests/month).
    Requires BRAVE_API_KEY environment variable.
    """
    api_key = os.getenv('BRAVE_API_KEY')
    if not api_key:
        raise ValueError("BRAVE_API_KEY not set")
    
    all_results = []
    seen_urls = set()
    
    session = _get_brave_session()
    
    for query in queries:
        try:
            response = session.get(
                'https://api.search.brave.com/res/v1/web/search',
                params={'q': query, 'count': max_results},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            for result in data.get('web', {}).get('results', []):
                url = result.get('url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        'href': url,
                        'title': result.get('title', ''),
                        'body': result.get('description', '')
                    })
            
            time.sleep(1.1)  # Brave free tier: 1 request/second
            
        except Exception as e:
            logger.warning(f"Brave search failed for '{query[:50]}...': {e}")
            continue
    
    logger.info(f"Brave total: {len(all_results)} unique results")
    return all_results


def _ddg_search(queries: List[str], max_results: int = 4, retries: int = 3) -> List[Dict]:
    """
    Search using DuckDuckGo (free, no API key required).
    Uses 'lite' backend for better reliability against rate limiting.
    """
    try:
        from duckduckgo_search import DDGS
        from duckduckgo_search.exceptions import RatelimitException
    except ImportError:
        try:
            from ddgs import DDGS
            RatelimitException = Exception
        except ImportError:
            logger.error("DuckDuckGo search package not found. Please install 'ddgs'.")
            return []
    
    all_results = []
    seen_urls = set()
    
    for attempt in range(retries + 1):
        try:
            with DDGS() as ddgs:
                for query in queries:
                    try:
                        results = list(ddgs.text(
                            query,
                            max_results=max_results,
                            region='wt-wt',
                            safesearch='moderate',
                            backend='lite'
                        ))
                        
                        logger.info(f"DDG lite search '{query[:50]}...': {len(results)} results")
                        
                        if not results:
                            logger.info(f"Trying DDG news search for: {query[:50]}...")
                            results = list(ddgs.news(
                                query,
                                max_results=max_results,
                                region='wt-wt'
                            ))
                            logger.info(f"DDG news search: {len(results)} results")
                        
                        for result in results:
                            url = result.get('href') or result.get('link') or result.get('url')
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                all_results.append({
                                    'href': url,
                                    'title': result.get('title', ''),
                                    'body': result.get('body', result.get('snippet', result.get('excerpt', '')))
                                })
                        
                        time.sleep(1.5)
                        
                    except RatelimitException as e:
                        logger.warning(f"DuckDuckGo rate limited for '{query[:50]}...': {e}")
                        time.sleep(5)
                        continue
                    except Exception as e:
                        logger.warning(f"DuckDuckGo search failed for '{query[:50]}...': {e}")
                        continue
            
            if all_results:
                logger.info(f"DDG total: {len(all_results)} unique results")
                return all_results
                
        except RatelimitException as e:
            logger.warning(f"DuckDuckGo rate limited on attempt {attempt + 1}: {e}")
            if attempt < retries:
                wait_time = (attempt + 1) * 5
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
        except Exception as e:
            logger.warning(f"DuckDuckGo attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(3)
                continue
    
    logger.warning("All DuckDuckGo search attempts failed")
    return all_results


def web_search(queries: List[str], max_results: int = 6) -> List[Dict]:
    """
    Perform web search with automatic API selection and fallback chain.
    
    Priority order:
    1. Tavily (fastest, best quality) - requires TAVILY_API_KEY
    2. Brave Search (reliable, 2000 free/month) - requires BRAVE_API_KEY
    3. DuckDuckGo (free, may hit rate limits) - no key required
    """
    # Try Tavily first (fastest, best quality)
    if os.getenv('TAVILY_API_KEY'):
        try:
            logger.info("Using Tavily API for search")
            results = _tavily_search(queries, max_results)
            if results:
                return results
            logger.warning("Tavily returned no results, trying fallback...")
        except Exception as e:
            logger.warning(f"Tavily failed: {e}")
    
    # Try Brave Search second (reliable, free tier available)
    if os.getenv('BRAVE_API_KEY'):
        try:
            logger.info("Using Brave Search API")
            results = _brave_search(queries, max_results)
            if results:
                return results
            logger.warning("Brave returned no results, trying DuckDuckGo...")
        except Exception as e:
            logger.warning(f"Brave failed: {e}")
    
    # Fallback to DuckDuckGo (free, but may hit rate limits)
    logger.info("Using DuckDuckGo for search (lite backend)")
    return _ddg_search(queries, max_results)


def is_social_media(url: str) -> bool:
    """Check if URL is from a social media platform."""
    return any(domain in url.lower() for domain in SOCIAL_MEDIA_DOMAINS)
