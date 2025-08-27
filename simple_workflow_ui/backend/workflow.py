#!/usr/bin/env python3
"""
LangGraph Research Workflow for the cluster
"""

import asyncio
import requests
import json
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
import markdown

class WorkflowState(TypedDict):
    query: str
    plan: str
    search_results: Dict[str, Any]
    analysis: str
    final_result: str
    step: str
    error: str

class ResearchWorkflow:
    def __init__(self):
        self.workflow = self._create_workflow()
        self.md = markdown.Markdown(extensions=['fenced_code', 'tables', 'toc'])
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph research workflow"""
        
        def planning_node(state: WorkflowState) -> WorkflowState:
            """Planning step using Jetson (fast model)"""
            try:
                state["step"] = "Planning research strategy..."
                
                prompt = f"""Create a focused research plan for: "{state['query']}"
                
Please provide:
1. 3-5 specific search terms/queries to investigate
2. Key aspects to focus on
3. Expected deliverables

Keep it concise and actionable."""

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
                        "num_results": 5
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
        
        def analysis_node(state: WorkflowState) -> WorkflowState:
            """Analysis step using CPU (powerful model)"""
            try:
                state["step"] = "Analyzing findings..."
                
                # Prepare context for analysis
                search_summary = ""
                if "results" in state["search_results"]:
                    for i, result in enumerate(state["search_results"]["results"][:3], 1):
                        search_summary += f"{i}. **{result.get('title', 'No title')}**\n"
                        search_summary += f"   {result.get('snippet', 'No snippet')}\n\n"
                
                prompt = f"""Based on the research plan and search results, provide a comprehensive analysis:

**Original Query:** {state['query']}

**Research Plan:**
{state['plan']}

**Search Results:**
{search_summary}

Please provide:
1. **Key Findings**: Main insights discovered
2. **Analysis**: Critical evaluation of the information
3. **Conclusions**: Direct answers to the original query
4. **Recommendations**: Next steps or further areas to explore

Format your response clearly with headings and bullet points."""

                response = requests.post(
                    "http://192.168.1.81:11435/api/generate",
                    json={
                        "model": "mistral:7b", 
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=90
                )
                
                if response.status_code == 200:
                    result = response.json()
                    state["analysis"] = result.get("response", "No analysis received")
                else:
                    state["analysis"] = f"Analysis failed: HTTP {response.status_code}"
                    
            except Exception as e:
                state["analysis"] = f"Analysis failed: {str(e)}"
                state["error"] = str(e)
            
            return state
        
        def finalize_node(state: WorkflowState) -> WorkflowState:
            """Finalize the results"""
            state["step"] = "Completed"
            
            # Combine all results into final formatted output
            final_output = f"""# Research Results: {state['query']}

## 📋 Research Plan
{state['plan']}

## 🔍 Search Results Summary
"""
            
            if "results" in state["search_results"]:
                for i, result in enumerate(state["search_results"]["results"][:3], 1):
                    final_output += f"{i}. [{result.get('title', 'No title')}]({result.get('url', '#')})\n"
                    final_output += f"   *{result.get('snippet', 'No snippet')}*\n\n"
            
            final_output += f"""
## 📊 Analysis & Insights
{state['analysis']}

---
*Research completed using LangGraph distributed AI cluster*
"""
            
            state["final_result"] = final_output
            return state
        
        # Build the StateGraph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("planning", planning_node)
        workflow.add_node("search", search_node)
        workflow.add_node("analysis", analysis_node)
        workflow.add_node("finalize", finalize_node)
        
        # Add edges
        workflow.set_entry_point("planning")
        workflow.add_edge("planning", "search")
        workflow.add_edge("search", "analysis")
        workflow.add_edge("analysis", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def execute(self, query: str) -> Dict[str, Any]:
        """Execute the research workflow"""
        initial_state = {
            "query": query,
            "plan": "",
            "search_results": {},
            "analysis": "",
            "final_result": "",
            "step": "Starting...",
            "error": ""
        }
        
        try:
            # Execute the workflow
            result = await self.workflow.ainvoke(initial_state)
            
            # Convert markdown to HTML for better display
            if result["final_result"]:
                result["final_result_html"] = self.md.convert(result["final_result"])
            
            return {
                "success": True,
                "result": result,
                "formatted_html": result.get("final_result_html", "")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": initial_state
            }
    
    def format_response(self, text: str) -> str:
        """Format LLM response as HTML"""
        return self.md.convert(text)
