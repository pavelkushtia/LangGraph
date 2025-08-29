#!/usr/bin/env python3
"""
Example LangGraph Workflows for Local Model Setup
Demonstrates various AI workflows using your distributed infrastructure
"""

import asyncio
from langchain.schema import HumanMessage, AIMessage
from langgraph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict
import operator
import requests
import json

# Import your local components (assuming they're in the same directory)
try:
    from local_models import JetsonOllamaLLM, CPULlamaLLM, LocalEmbeddings
    from local_tools import WebSearchTool, WebScrapeTool, CommandExecuteTool
except ImportError:
    print(
        "⚠️  Local model imports not found. Make sure to create the files from the integration guide."
    )


class WorkflowState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    context: str
    workflow_type: str
    results: Dict


class LocalAIWorkflows:
    def __init__(self):
        """Initialize all local AI components"""
        # Models
        self.jetson_fast = JetsonOllamaLLM(model_name="tinyllama:1.1b")
        self.jetson_standard = JetsonOllamaLLM(model_name="llama3.2:3b")
        self.cpu_heavy = CPULlamaLLM()

        # Tools
        self.web_search = WebSearchTool()
        self.web_scrape = WebScrapeTool()
        self.command_exec = CommandExecuteTool()

        # Embeddings
        self.embeddings = LocalEmbeddings()

    def create_research_workflow(self):
        """Research assistant workflow - great for learning"""

        def research_planner(state: WorkflowState) -> WorkflowState:
            """Plan research strategy"""
            query = state["messages"][-1]["content"]

            # Use fast model to plan research
            plan_prompt = f"""
            Research Query: {query}
            
            Create a research plan with 3-4 specific search queries to gather comprehensive information.
            Format as a JSON list: ["query1", "query2", "query3"]
            """

            plan = self.jetson_fast.invoke(plan_prompt)

            return {
                "messages": state["messages"]
                + [{"role": "assistant", "content": f"Research plan: {plan}"}],
                "context": plan,
                "workflow_type": "research",
                "results": {"plan": plan},
            }

        def execute_searches(state: WorkflowState) -> WorkflowState:
            """Execute planned searches"""
            try:
                # Extract queries from plan (simplified parsing)
                plan = state["context"]
                # In real implementation, properly parse JSON
                search_queries = [
                    "AI trends 2024",
                    "machine learning applications",
                    "local AI setup",
                ]

                search_results = []
                for query in search_queries[:2]:  # Limit to avoid overload
                    try:
                        result = self.web_search.run(query)
                        search_results.append(f"Query: {query}\nResults: {result}\n")
                    except:
                        search_results.append(
                            f"Query: {query}\nResults: Search failed\n"
                        )

                combined_results = "\n".join(search_results)

                return {
                    "messages": state["messages"]
                    + [{"role": "assistant", "content": "Search completed"}],
                    "context": combined_results,
                    "workflow_type": state["workflow_type"],
                    "results": {**state["results"], "searches": combined_results},
                }
            except Exception as e:
                return {
                    "messages": state["messages"]
                    + [{"role": "assistant", "content": f"Search failed: {e}"}],
                    "context": state["context"],
                    "workflow_type": state["workflow_type"],
                    "results": state["results"],
                }

        def synthesize_research(state: WorkflowState) -> WorkflowState:
            """Synthesize findings using heavy model"""
            search_data = state["context"]
            original_query = state["messages"][0]["content"]

            synthesis_prompt = f"""
            Original Question: {original_query}
            
            Research Data:
            {search_data}
            
            Please provide a comprehensive, well-structured answer based on the research data.
            Include key findings, different perspectives, and practical insights.
            """

            final_answer = self.cpu_heavy.invoke(synthesis_prompt)

            return {
                "messages": state["messages"]
                + [{"role": "assistant", "content": final_answer}],
                "context": state["context"],
                "workflow_type": state["workflow_type"],
                "results": {**state["results"], "final_answer": final_answer},
            }

        # Build workflow graph
        workflow = StateGraph(WorkflowState)
        workflow.add_node("plan", research_planner)
        workflow.add_node("search", execute_searches)
        workflow.add_node("synthesize", synthesize_research)

        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "search")
        workflow.add_edge("search", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def create_coding_assistant_workflow(self):
        """Coding assistant with code analysis and generation"""

        def analyze_request(state: WorkflowState) -> WorkflowState:
            """Analyze coding request"""
            request = state["messages"][-1]["content"]

            analysis_prompt = f"""
            Coding Request: {request}
            
            Analyze this request and determine:
            1. Type: [explanation/code_generation/debugging/review]
            2. Complexity: [simple/medium/complex]
            3. Language/Framework needed
            
            Respond in JSON format.
            """

            analysis = self.jetson_standard.invoke(analysis_prompt)

            return {
                "messages": state["messages"]
                + [{"role": "assistant", "content": f"Analysis: {analysis}"}],
                "context": analysis,
                "workflow_type": "coding",
                "results": {"analysis": analysis},
            }

        def route_coding_task(state: WorkflowState) -> str:
            """Route to appropriate coding handler"""
            analysis = state["context"].lower()

            if "simple" in analysis:
                return "quick_response"
            elif "complex" in analysis or "generation" in analysis:
                return "detailed_coding"
            else:
                return "explanation"

        def quick_coding_response(state: WorkflowState) -> WorkflowState:
            """Handle simple coding questions with Jetson"""
            request = state["messages"][0]["content"]

            response = self.jetson_standard.invoke(
                f"Coding question: {request}\nProvide a clear, concise answer with code examples if needed."
            )

            return {
                "messages": state["messages"]
                + [{"role": "assistant", "content": response}],
                "context": state["context"],
                "workflow_type": state["workflow_type"],
                "results": {**state["results"], "response": response},
            }

        def detailed_coding_response(state: WorkflowState) -> WorkflowState:
            """Handle complex coding with CPU model"""
            request = state["messages"][0]["content"]

            detailed_prompt = f"""
            Complex Coding Request: {request}
            
            Provide a comprehensive solution including:
            1. Detailed explanation
            2. Complete code implementation
            3. Best practices and considerations
            4. Alternative approaches if applicable
            5. Testing suggestions
            """

            response = self.cpu_heavy.invoke(detailed_prompt)

            return {
                "messages": state["messages"]
                + [{"role": "assistant", "content": response}],
                "context": state["context"],
                "workflow_type": state["workflow_type"],
                "results": {**state["results"], "detailed_response": response},
            }

        def coding_explanation(state: WorkflowState) -> WorkflowState:
            """Provide coding explanations"""
            request = state["messages"][0]["content"]

            explanation = self.jetson_standard.invoke(
                f"Explain this programming concept clearly: {request}"
            )

            return {
                "messages": state["messages"]
                + [{"role": "assistant", "content": explanation}],
                "context": state["context"],
                "workflow_type": state["workflow_type"],
                "results": {**state["results"], "explanation": explanation},
            }

        # Build coding workflow
        workflow = StateGraph(WorkflowState)
        workflow.add_node("analyze", analyze_request)
        workflow.add_node("quick_response", quick_coding_response)
        workflow.add_node("detailed_coding", detailed_coding_response)
        workflow.add_node("explanation", coding_explanation)

        workflow.set_entry_point("analyze")
        workflow.add_conditional_edges(
            "analyze",
            route_coding_task,
            {
                "quick_response": "quick_response",
                "detailed_coding": "detailed_coding",
                "explanation": "explanation",
            },
        )

        workflow.add_edge("quick_response", END)
        workflow.add_edge("detailed_coding", END)
        workflow.add_edge("explanation", END)

        return workflow.compile()

    def create_data_analysis_workflow(self):
        """Data analysis workflow using local tools"""

        def data_processor(state: WorkflowState) -> WorkflowState:
            """Process data analysis request"""
            request = state["messages"][-1]["content"]

            # Check if data URL is provided
            if "http" in request:
                # Extract URL and scrape data
                url = [word for word in request.split() if word.startswith("http")][0]
                try:
                    scraped_data = self.web_scrape.run(url)
                    context = f"Scraped data from {url}:\n{scraped_data}"
                except:
                    context = "Failed to scrape data"
            else:
                context = "No data source provided"

            return {
                "messages": state["messages"]
                + [{"role": "assistant", "content": "Data collected"}],
                "context": context,
                "workflow_type": "data_analysis",
                "results": {"data": context},
            }

        def analyze_data(state: WorkflowState) -> WorkflowState:
            """Analyze the collected data"""
            data = state["context"]
            original_request = state["messages"][0]["content"]

            analysis_prompt = f"""
            Analysis Request: {original_request}
            
            Data Available:
            {data}
            
            Perform a thorough analysis including:
            1. Data summary and key insights
            2. Patterns and trends identified
            3. Statistical observations
            4. Recommendations based on findings
            """

            analysis = self.cpu_heavy.invoke(analysis_prompt)

            return {
                "messages": state["messages"]
                + [{"role": "assistant", "content": analysis}],
                "context": state["context"],
                "workflow_type": state["workflow_type"],
                "results": {**state["results"], "analysis": analysis},
            }

        # Build data workflow
        workflow = StateGraph(WorkflowState)
        workflow.add_node("collect_data", data_processor)
        workflow.add_node("analyze", analyze_data)

        workflow.set_entry_point("collect_data")
        workflow.add_edge("collect_data", "analyze")
        workflow.add_edge("analyze", END)

        return workflow.compile()


# Example usage and testing
class WorkflowDemo:
    def __init__(self):
        self.ai_workflows = LocalAIWorkflows()

    async def demo_research_workflow(self):
        """Demo the research workflow"""
        print("🔬 Running Research Workflow Demo...")

        research_flow = self.ai_workflows.create_research_workflow()

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": "What are the latest trends in local AI and edge computing?",
                }
            ],
            "context": "",
            "workflow_type": "research",
            "results": {},
        }

        try:
            result = await research_flow.ainvoke(initial_state)
            print("✅ Research completed!")
            print(
                f"Final Answer: {result['results'].get('final_answer', 'No final answer')}"
            )
            return result
        except Exception as e:
            print(f"❌ Research failed: {e}")
            return None

    async def demo_coding_workflow(self):
        """Demo the coding assistant workflow"""
        print("💻 Running Coding Assistant Demo...")

        coding_flow = self.ai_workflows.create_coding_assistant_workflow()

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": "How do I implement a simple REST API with FastAPI and include error handling?",
                }
            ],
            "context": "",
            "workflow_type": "coding",
            "results": {},
        }

        try:
            result = await coding_flow.ainvoke(initial_state)
            print("✅ Coding assistance completed!")
            print(f"Response type: {result['workflow_type']}")
            return result
        except Exception as e:
            print(f"❌ Coding assistance failed: {e}")
            return None

    async def demo_data_analysis_workflow(self):
        """Demo the data analysis workflow"""
        print("📊 Running Data Analysis Demo...")

        data_flow = self.ai_workflows.create_data_analysis_workflow()

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": "Analyze this webpage for key insights: https://example.com",
                }
            ],
            "context": "",
            "workflow_type": "data_analysis",
            "results": {},
        }

        try:
            result = await data_flow.ainvoke(initial_state)
            print("✅ Data analysis completed!")
            return result
        except Exception as e:
            print(f"❌ Data analysis failed: {e}")
            return None


async def main():
    """Run all workflow demos"""
    print("🚀 Starting Local AI Workflow Demonstrations")
    print("=" * 50)

    demo = WorkflowDemo()

    # Test each workflow
    await demo.demo_research_workflow()
    print("\n" + "=" * 50)

    await demo.demo_coding_workflow()
    print("\n" + "=" * 50)

    await demo.demo_data_analysis_workflow()
    print("\n" + "=" * 50)

    print("🎉 All demos completed!")


if __name__ == "__main__":
    print("Local AI Workflow Examples")
    print("Make sure your cluster is running before executing these examples.")
    print("Use: python cluster_orchestrator.py start")
    print()

    choice = input("Run demos? (y/n): ")
    if choice.lower() == "y":
        asyncio.run(main())
    else:
        print("Examples ready to run when you are!")
