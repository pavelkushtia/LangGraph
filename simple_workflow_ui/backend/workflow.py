#!/usr/bin/env python3
"""
LangGraph Research Workflow for the cluster
"""

import asyncio
import requests
import json
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from datetime import datetime

# Simple imports for basic functionality
import markdown
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

class WorkflowState(TypedDict):
    query: str
    plan: str
    search_results: Dict[str, Any]
    scraped_content: str  # Store raw scraped content for LLM processing
    analysis: str
    final_result: str
    final_result_html: str
    step: str
    error: str

class ResearchWorkflow:
    def __init__(self):
        self.workflow = self._create_workflow()
        self.md = markdown.Markdown(extensions=['fenced_code', 'tables'])
        self.console = Console(record=True, width=80)
    
    def _prepare_search_summary(self, search_results: Dict[str, Any]) -> str:
        """Helper to format search results for analysis"""
        search_summary = ""
        if "results" in search_results:
            for i, result in enumerate(search_results["results"][:3], 1):
                search_summary += f"{i}. **{result.get('title', 'No title')}**\n"
                search_summary += f"   {result.get('snippet', 'No snippet')}\n\n"
        return search_summary
    
    def _call_llm_and_update_state(self, state: WorkflowState, prompt: str) -> WorkflowState:
        """Helper to call LLM and update state"""
        response = requests.post(
            "http://192.168.1.177:11434/api/generate",
            json={
                "model": "llama3.2:3b", 
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            state["analysis"] = result.get("response", "No response received")
        else:
            state["analysis"] = f"Error: HTTP {response.status_code}"
        
        return state
    
    def _extract_relevant_sites(self, search_results: Dict[str, Any], query: str) -> List[str]:
        """Extract relevant URLs from search results based on query type"""
        relevant_sites = []
        if "results" in search_results:
            # Take top results that are most likely to have real-time data
            for result in search_results["results"][:5]:
                url = result.get("url", "")
                title = result.get("title", "").lower()
                snippet = result.get("snippet", "").lower()
                
                # Prioritize sites known for real-time data
                high_priority_domains = [
                    "weather.com", "accuweather.com", "wunderground.com",  # Weather
                    "yahoo.com", "google.com", "finance.yahoo.com",  # Finance  
                    "cnn.com", "bbc.com", "reuters.com", "ap.org",  # News
                    "nasdaq.com", "bloomberg.com", "marketwatch.com"  # Markets
                ]
                
                # Check if URL contains relevant terms or is from a trusted real-time source
                if (any(domain in url.lower() for domain in high_priority_domains) or
                    any(term in title + snippet for term in ["current", "live", "today", "now", "latest"])):
                    relevant_sites.append(url)
                elif url and len(relevant_sites) < 3:  # Include other results as fallback
                    relevant_sites.append(url)
                    
        return relevant_sites[:3]  # Limit to top 3 for efficiency
    
    def _scrape_sites_to_state(self, sites: List[str]) -> str:
        """Simple method to scrape sites and return combined content for LangGraph state"""
        all_content = []
        
        for site_url in sites:
            try:
                print(f"🌐 Scraping: {site_url}")
                
                scrape_response = requests.post(
                    "http://192.168.1.190:8082/web_scrape",
                    json={
                        "url": site_url,
                        "method": "requests",
                        "extract_text": True,
                        "extract_links": False,
                        "wait_time": 3
                    },
                    timeout=30
                )
                
                if scrape_response.status_code == 200:
                    scrape_data = scrape_response.json()
                    text_content = scrape_data.get("text", "")
                    if text_content:
                        all_content.append(f"=== Content from {site_url} ===\n{text_content[:1500]}\n")
                        print(f"✅ Scraped {len(text_content)} chars from {site_url}")
                    else:
                        print(f"❌ No text content from {site_url}")
                else:
                    print(f"❌ Failed to scrape {site_url}: {scrape_response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error scraping {site_url}: {str(e)}")
                continue
        
        return "\n\n".join(all_content) if all_content else ""
    
    def _extract_smart_data(self, relevant_sites: List[str], query: str, strategy: str) -> Dict[str, Any]:
        """Use LLM-guided strategy to extract relevant data from any type of website"""
        extracted_content = []
        scraped_sources = []
        
        print(f"🕷️ Starting smart scraping of {len(relevant_sites)} sites...")
        
        try:
            for site_url in relevant_sites:
                try:
                    print(f"🌐 Scraping: {site_url}")
                    
                    # Use tools server web scraping endpoint
                    scrape_response = requests.post(
                        "http://192.168.1.190:8082/web_scrape",
                        json={
                            "url": site_url,
                            "method": "requests",  # Fast method first
                            "extract_text": True,
                            "extract_links": False,
                            "wait_time": 3
                        },
                        timeout=30
                    )
                    
                    print(f"📡 Scrape response status: {scrape_response.status_code}")
                    
                    if scrape_response.status_code == 200:
                        scrape_data = scrape_response.json()
                        
                        # Extract relevant text content
                        text_content = scrape_data.get("text", "")
                        print(f"📄 Scraped {len(text_content)} characters from {site_url}")
                        
                        # Use LLM strategy to extract relevant data
                        relevant_data = self._extract_data_using_strategy(text_content, site_url, query, strategy)
                        if relevant_data:
                            print(f"✅ Relevant data extracted: {relevant_data[:200]}...")
                            extracted_content.append(relevant_data)
                            scraped_sources.append(site_url)
                        else:
                            print(f"❌ No relevant data found in {site_url}")
                            
                except Exception as scrape_error:
                    print(f"💥 Failed to scrape {site_url}: {scrape_error}")
                    continue
            
            if extracted_content:
                return {
                    "success": True,
                    "content": "\n\n".join(extracted_content),
                    "sources": ", ".join(scraped_sources)
                }
            else:
                return {
                    "success": False,
                    "error": "No relevant data found in scraped content",
                    "content": "",
                    "sources": ""
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Smart data extraction failed: {str(e)}",
                "content": "",
                "sources": ""
            }
    
    def _extract_data_using_strategy(self, text_content: str, source_url: str, query: str, strategy: str) -> str:
        """Use LLM with extraction strategy to find relevant data in scraped content"""
        try:
            # Limit text size for LLM processing
            text_sample = text_content[:2000] if len(text_content) > 2000 else text_content
            
            extraction_prompt = f"""User Query: "{query}"
Extraction Strategy: {strategy}

Scraped Website Content:
{text_sample}

Based on the extraction strategy, find and extract the specific data that answers the user's query.

Instructions:
1. Look for the patterns and keywords specified in the strategy
2. Extract relevant data points (numbers, dates, conditions, etc.)
3. Include enough context to understand each data point
4. If no relevant data found, return "NO_DATA_FOUND"

Format your response as:
SOURCE: {source_url}
EXTRACTED_DATA: [the specific data that answers the query]
CONTEXT: [brief context around each data point]"""

            response = requests.post(
                "http://192.168.1.177:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": extraction_prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                extracted_info = response.json()["response"]
                
                # Check if LLM found relevant data
                if "NO_DATA_FOUND" not in extracted_info.upper():
                    return extracted_info
                else:
                    return ""
            else:
                print(f"❌ LLM extraction failed: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"❌ Error in LLM data extraction: {str(e)}")
            return ""

    def _contains_weather_data(self, text: str, query: str) -> bool:
        """Check if scraped text contains relevant weather data"""
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Look for temperature indicators
        temp_indicators = ["°f", "°c", "fahrenheit", "celsius", "temperature", "temp", "degrees"]
        weather_indicators = ["weather", "forecast", "humidity", "wind", "precipitation", "cloudy", "sunny"]
        
        # Check if text contains weather-related content
        has_temp = any(indicator in text_lower for indicator in temp_indicators)
        has_weather = any(indicator in text_lower for indicator in weather_indicators)
        
        # For location-specific queries, check if location is mentioned
        location_mentioned = True  # Default to true, improve this logic as needed
        if "mountain view" in query_lower:
            location_mentioned = "mountain view" in text_lower or "mountain_view" in text_lower
        
        return (has_temp or has_weather) and location_mentioned and len(text.strip()) > 100
    
    def _extract_temperature_data(self, text: str, source_url: str) -> str:
        """Extract clean temperature data from scraped content"""
        import re
        
        # Look for temperature patterns in the text
        temp_patterns = [
            r'(\d+)°\s*[FC]',  # 62°F or 62°C 
            r'(\d+)\s*degrees?\s*[FC]',  # 62 degrees F
            r'(\d+)°',  # Just 62°
        ]
        
        found_temps = []
        text_lines = text.split('\n')
        
        for i, line in enumerate(text_lines):
            line_clean = line.strip()
            
            # Skip empty lines and obvious navigation/ads
            if (len(line_clean) < 3 or 
                'advertisement' in line_clean.lower() or
                'cookie' in line_clean.lower() or
                len(line_clean) > 200):
                continue
                
            # Look for temperature in this line
            for pattern in temp_patterns:
                matches = re.findall(pattern, line_clean, re.IGNORECASE)
                if matches:
                    # Get some context around the temperature
                    context_start = max(0, i-2)
                    context_end = min(len(text_lines), i+3)
                    context_lines = text_lines[context_start:context_end]
                    
                    # Clean up context
                    clean_context = []
                    for ctx_line in context_lines:
                        clean_line = ctx_line.strip()
                        if (clean_line and 
                            len(clean_line) < 100 and 
                            'advertisement' not in clean_line.lower()):
                            clean_context.append(clean_line)
                    
                    temp_info = f"Source: {source_url}\nTemperature found: {line_clean}\nContext: {' | '.join(clean_context[:5])}"
                    found_temps.append(temp_info)
                    break
        
        if found_temps:
            return "\n\n".join(found_temps[:2])  # Return top 2 temperature findings
        
        return ""
    
    def _format_with_rich(self, query: str, analysis: str, plan: str, sources_count: int = 0) -> str:
        """Format response using Rich for better display"""
        # Clear console
        self.console = Console(record=True, width=80)
        
        # Title
        title_panel = Panel(
            f"🔍 [bold blue]{query}[/bold blue]",
            style="blue",
            padding=(0, 1)
        )
        self.console.print(title_panel)
        self.console.print()
        
        # Answer
        answer_content = Markdown(f"**Answer:**\n\n{analysis}")
        answer_panel = Panel(
            answer_content,
            title="🎯 Direct Answer",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(answer_panel)
        self.console.print()
        
        # Summary
        summary_content = f"**Research Plan:** {plan}\n\n**Sources Found:** {sources_count} web sources"
        summary_panel = Panel(
            Markdown(summary_content),
            title="📝 Research Summary",
            border_style="dim",
            padding=(1, 2)
        )
        self.console.print(summary_panel)
        
        # Footer
        self.console.print("\n[dim italic]⚡ Powered by LangGraph distributed AI cluster[/dim italic]")
        
        # Export as HTML
        return self.console.export_html(inline_styles=True)

    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph research workflow with adaptive routing"""
        
        def analyze_query_type(state: WorkflowState) -> str:
            """Analyze query using semantic embeddings to determine optimal processing path"""
            from query_classifier import get_query_classifier
            
            try:
                classifier = get_query_classifier()
                category, confidence = classifier.classify_query(state["query"])
                
                # Log classification details for debugging
                details = classifier.get_classification_details(state["query"])
                print(f"🔍 Query Classification: {category} (confidence: {confidence:.3f})")
                print(f"📊 All scores: {details}")
                
                return category
                
            except Exception as e:
                print(f"❌ Semantic classification failed: {e}")
                # Fallback to general research if embeddings server is down
                return "general_research"
        
        def planning_node(state: WorkflowState) -> WorkflowState:
            """Planning step using Jetson (fast model)"""
            try:
                state["step"] = "Planning research strategy..."
                
                prompt = f"""Create a brief research plan for: "{state['query']}"
                
Just provide:
1. 2-3 search terms to use
2. What to look for

Keep it short - maximum 3 sentences."""

                response = requests.post(
                    "http://192.168.1.177:11434/api/generate",
                    json={
                        "model": "llama3.2:3b",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    state["plan"] = result.get("response", "No response received")
                else:
                    state["plan"] = f"Error: HTTP {response.status_code}"
                    
            except Exception as e:
                state["plan"] = f"Planning failed: {str(e)}"
                state["error"] = str(e)
            
            return state
        
        def search_node(state: WorkflowState) -> WorkflowState:
            """Search step using Tools server"""
            try:
                state["step"] = "Searching for information..."
                
                # Extract search terms from plan or use original query
                search_query = state["query"]
                
                response = requests.post(
                    "http://192.168.1.190:8082/web_search",
                    json={
                        "query": search_query,
                        "max_results": 5
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    state["search_results"] = response.json()
                else:
                    # Mock search results if service unavailable
                    state["search_results"] = {
                        "status": "mock",
                        "results": [
                            {
                                "title": f"Sample result for {search_query}",
                                "url": "https://example.com",
                                "snippet": f"This is a mock search result for the query: {search_query}"
                            }
                        ]
                    }
                    
            except Exception as e:
                state["search_results"] = {
                    "error": str(e),
                    "status": "failed"
                }
                state["error"] = str(e)
            
            return state
        
        def realtime_analysis_node(state: WorkflowState) -> WorkflowState:
            """Simple, powerful real-time analysis using LangGraph state - let LLM do the extraction"""
            try:
                state["step"] = "Scraping websites and extracting data..."
                
                # Step 1: Get relevant sites and scrape content
                relevant_sites = self._extract_relevant_sites(state["search_results"], state["query"])
                print(f"🌐 Found {len(relevant_sites)} relevant sites for: {state['query']}")
                
                # Step 2: Scrape raw content and store in state
                scraped_content = self._scrape_sites_to_state(relevant_sites)
                state["scraped_content"] = scraped_content  # Store in LangGraph state
                print(f"📄 Scraped {len(scraped_content)} characters total")
                
                # Step 3: Let LLM extract what's needed using state content
                if scraped_content:
                    state["step"] = "Analyzing scraped content with LLM..."
                    
                    # Ultra-simple prompt with less content
                    limited_content = scraped_content[:1000]  # Much smaller content
                    
                    extraction_prompt = f"""Question: {state['query']}

Website content:
{limited_content}

Answer the question using this content. Be direct and specific."""

                    return self._call_llm_and_update_state(state, extraction_prompt)
                
                else:
                    # No content scraped
                    fallback_prompt = f"""User Query: "{state['query']}"

I searched for relevant websites but couldn't scrape useful content. Based on this type of query, here's where you can find reliable real-time information:

- For weather: Weather.com, AccuWeather.com, or National Weather Service
- For financial data: Yahoo Finance, Bloomberg, or MarketWatch  
- For news: Major news sites like CNN, BBC, or Reuters

Please check these sources directly for the most current information."""

                    return self._call_llm_and_update_state(state, fallback_prompt)
                
            except Exception as e:
                state["analysis"] = f"Real-time analysis failed: {str(e)}"
                state["error"] = str(e)
            return state
        
        def factual_analysis_node(state: WorkflowState) -> WorkflowState:
            """Analysis optimized for factual/historical queries"""
            try:
                state["step"] = "Analyzing factual information..."
                search_summary = self._prepare_search_summary(state["search_results"])
                
                prompt = f"""FACTUAL QUERY: {state['query']}

**Search Results:**
{search_summary}

For factual questions, synthesize authoritative information to provide:
1. Clear, factual answer based on reliable sources
2. Key facts from the search results  
3. Context from authoritative sources

**Direct Answer:** [Clear factual answer based on search results]
**Key Facts:** [3-4 important facts from sources]
**Confidence:** [high for authoritative sources, medium otherwise]
**Brief Summary:** [One sentence factual summary]"""

                return self._call_llm_and_update_state(state, prompt)
                
            except Exception as e:
                state["analysis"] = f"Factual analysis failed: {str(e)}"
                state["error"] = str(e)
            return state
        
        def complex_analysis_node(state: WorkflowState) -> WorkflowState:
            """Analysis optimized for complex research queries"""
            try:
                state["step"] = "Performing complex analysis..."
                search_summary = self._prepare_search_summary(state["search_results"])
                
                prompt = f"""COMPLEX ANALYSIS QUERY: {state['query']}

**Search Results:**
{search_summary}

For complex analysis, provide:
1. Comprehensive answer synthesizing multiple sources
2. Analytical insights beyond basic facts
3. Connections between different pieces of information

**Direct Answer:** [Comprehensive analysis-based answer]
**Key Insights:** [3-4 analytical insights from the research]
**Confidence:** [based on depth and quality of sources]
**Brief Summary:** [One sentence analytical summary]"""

                return self._call_llm_and_update_state(state, prompt)
                
            except Exception as e:
                state["analysis"] = f"Complex analysis failed: {str(e)}"
                state["error"] = str(e)
            return state
        
        def general_analysis_node(state: WorkflowState) -> WorkflowState:
            """General analysis for standard queries"""
            try:
                state["step"] = "Analyzing search results..."
                search_summary = self._prepare_search_summary(state["search_results"])
                
                prompt = f"""RESEARCH QUERY: {state['query']}

**Search Results:**
{search_summary}

Provide a helpful answer based on the search results:

**Direct Answer:** [Clear answer based on search results]
**Key Findings:** [2-4 key findings from the results]
**Confidence:** [high/medium/low based on source quality]
**Brief Summary:** [One sentence summary]"""

                return self._call_llm_and_update_state(state, prompt)
                
            except Exception as e:
                state["analysis"] = f"General analysis failed: {str(e)}"
                state["error"] = str(e)
            return state
        
        def finalize_node(state: WorkflowState) -> WorkflowState:
            """Finalize results with Rich formatting"""
            state["step"] = "Completed"
            
            try:
                # Get data
                query = state.get('query', 'Unknown query')
                analysis = state.get('analysis', 'No analysis available')
                plan = state.get('plan', 'No plan available')
                sources_count = len(state.get("search_results", {}).get("results", []))
                
                # Create Rich formatted HTML
                state["final_result_html"] = self._format_with_rich(query, analysis, plan, sources_count)
                
                # Keep simple markdown version for compatibility
                final_output = f"""# {query}

## 🎯 **Answer**
{analysis}

## 📝 **Research Summary**
**Plan**: {plan}

**Sources**: Found {sources_count} sources from web search.

*Powered by LangGraph distributed AI cluster*"""
                
                state["final_result"] = final_output
                
            except Exception as e:
                # Fallback formatting
                fallback = f"# {state.get('query', 'Query')}\n\n**Answer:** {state.get('analysis', 'Error in analysis')}\n\n**Error:** {str(e)}"
                state["final_result"] = fallback
                state["final_result_html"] = f"<pre>{fallback}</pre>"
            
            return state
        
        # Build the StateGraph with conditional routing
        workflow = StateGraph(WorkflowState)
        
        # Add all nodes
        workflow.add_node("planning", planning_node)
        workflow.add_node("search", search_node)
        workflow.add_node("realtime_analysis", realtime_analysis_node)
        workflow.add_node("factual_analysis", factual_analysis_node)
        workflow.add_node("complex_analysis", complex_analysis_node)
        workflow.add_node("general_analysis", general_analysis_node)
        workflow.add_node("finalize", finalize_node)
        
        # Set conditional routing after search
        def route_to_analysis(state: WorkflowState) -> str:
            """Route to appropriate analysis node based on query type"""
            return analyze_query_type(state)
        
        # Linear flow: planning → search → [conditional analysis] → finalize
        workflow.set_entry_point("planning")
        workflow.add_edge("planning", "search")
        
        # Conditional routing after search
        workflow.add_conditional_edges(
            "search",
            route_to_analysis,
            {
                "realtime_research": "realtime_analysis",
                "factual_research": "factual_analysis", 
                "complex_research": "complex_analysis",
                "general_research": "general_analysis"
            }
        )
        
        # All analysis nodes lead to finalize
        workflow.add_edge("realtime_analysis", "finalize")
        workflow.add_edge("factual_analysis", "finalize")
        workflow.add_edge("complex_analysis", "finalize")
        workflow.add_edge("general_analysis", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def execute(self, query: str) -> Dict[str, Any]:
        """Execute the research workflow with structured outputs"""
        initial_state = {
            "query": query,
            "plan": "",
            "search_results": {},
            "analysis": "",
            "final_result": "",
            "final_result_html": "",
            "step": "Starting...",
            "error": ""
        }
        
        try:
            # Execute the workflow
            result = await self.workflow.ainvoke(initial_state)
            
            # Use Rich-formatted HTML if available, otherwise fallback
            formatted_html = result.get("final_result_html", "")
            if not formatted_html and result.get("final_result"):
                # Simple fallback formatting
                formatted_html = f"<pre>{result['final_result']}</pre>"
            
            return {
                "success": True,
                "result": result,
                "formatted_html": formatted_html
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": initial_state
            }
    
    def format_response(self, text: str) -> str:
        """Format LLM response using Rich"""
        return self._format_with_rich("Response", text, "Direct formatting", 0)
