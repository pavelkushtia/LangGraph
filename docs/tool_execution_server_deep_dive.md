# Tool Execution Server Deep Dive 🛠️

## 🎯 What is the Tool Execution Server?

The **Tool Execution Server** is the action hub of your LangGraph cluster that executes external operations and data gathering tasks. It runs on your **worker-node3 (192.168.1.190:8082)** and provides powerful capabilities for web search, web scraping, and safe command execution.

### Why Do We Need a Tool Execution Server?

The tool execution server solves a fundamental need: **AI agents need to interact with the real world**. By providing a secure, isolated environment for tool execution, we enable:

- **Web Search**: Gather real-time information from the internet
- **Web Scraping**: Extract content from specific websites and APIs
- **Command Execution**: Perform system operations and data processing
- **Security Isolation**: Protect your main systems from potentially dangerous operations
- **Scalability**: Offload tool execution to dedicated hardware

## 🏗️ Technical Architecture

### Server Components

```text
worker-node3 (192.168.1.190:8082)
├── FastAPI Web Server
├── Tool Services
│   ├── Web Search Service (DuckDuckGo)
│   ├── Web Scraping Service
│   │   ├── Requests + BeautifulSoup (lightweight)
│   │   └── Selenium + Chrome (dynamic content)
│   └── Command Execution Service (sandboxed)
├── Security Layer
├── Async Processing Engine
└── Health Monitoring
```

### Hardware Optimization

**Why worker-node3 for Tools?**

- **6GB VM**: Sufficient for Chrome/Selenium operations
- **Network Isolation**: Separate network boundary for external operations
- **Resource Dedication**: Won't impact LLM inference performance
- **Security**: Isolated environment for potentially risky operations
- **Scalability**: Easy to add more worker nodes for load distribution

### Tool Capabilities

```python
# Web Search Tool
web_search:
  - Engine: DuckDuckGo (privacy-focused)
  - Max Results: 50 (configurable)
  - Response Time: ~2-5 seconds
  - Use: Real-time information gathering

# Web Scraping Tool  
web_scrape:
  - Methods: Requests (fast) + Selenium (dynamic)
  - Content: Text extraction, link extraction
  - Timeout: Configurable up to 30 seconds
  - Use: Specific website content extraction

# Command Execution Tool
execute_command:
  - Security: Whitelist-based command filtering
  - Timeout: Configurable up to 300 seconds  
  - Working Directory: Customizable sandbox
  - Use: System operations, data processing
```

## 🔌 API Endpoints

### POST /web_search

**Purpose**: Search the web for information

```python
# Request
{
  "query": "artificial intelligence news",
  "max_results": 10,
  "search_engine": "duckduckgo"
}

# Response
{
  "query": "artificial intelligence news",
  "results": [
    {
      "title": "Latest AI Developments",
      "url": "https://example.com/ai-news",
      "snippet": "Recent breakthroughs in AI technology..."
    }
  ],
  "count": 10,
  "search_engine": "duckduckgo"
}
```

### POST /web_scrape

**Purpose**: Extract content from specific websites

```python
# Request
{
  "url": "https://example.com/article",
  "method": "requests",  # or "selenium"
  "extract_text": true,
  "extract_links": false,
  "wait_time": 5
}

# Response
{
  "url": "https://example.com/article",
  "status_code": 200,
  "title": "Article Title",
  "text": "Full article content...",
  "links": [
    {"text": "Related Link", "href": "/related"}
  ]
}
```

### POST /execute_command

**Purpose**: Execute safe shell commands

```python
# Request
{
  "command": "ls -la /tmp",
  "timeout": 30,
  "working_dir": "/tmp"
}

# Response
{
  "command": "ls -la /tmp",
  "working_dir": "/tmp",
  "returncode": 0,
  "stdout": "total 12\ndrwxrwxrwt...",
  "stderr": "",
  "timeout": 30
}
```

### GET /health

**Purpose**: Check server status

```python
# Response
{
  "status": "healthy",
  "memory_usage_percent": 45.2,
  "cpu_usage_percent": 12.8,
  "tools_available": ["web_search", "web_scrape", "execute_command"]
}
```

### GET /stats

**Purpose**: Get detailed system statistics

```python
# Response
{
  "cpu_percent": 15.3,
  "memory": {
    "total": 6442450944,
    "available": 3221225472,
    "percent": 50.0
  },
  "disk": {
    "total": 21474836480,
    "free": 10737418240,
    "percent": 50.0
  }
}
```

## 🔗 LangGraph Integration

### Tool Classes

```python
class WebSearchTool(BaseTool):
    """Web search via tools server"""
    
    name = "web_search"
    description = "Search the web for current information"
    args_schema: Type[BaseModel] = WebSearchInput
    
    def _run(self, query: str, max_results: int = 10) -> str:
        try:
            response = requests.post(
                "http://192.168.1.190:8082/web_search",
                json={"query": query, "max_results": max_results},
                timeout=30
            )
            result = response.json()
            
            if "results" in result:
                formatted_results = []
                for item in result["results"][:max_results]:
                    formatted_results.append(
                        f"**{item['title']}**\n{item['url']}\n{item['snippet']}\n"
                    )
                return "\n".join(formatted_results)
            else:
                return f"Search failed: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Web search error: {e}"

class WebScrapeTool(BaseTool):
    """Web scraping via tools server"""
    
    name = "web_scrape"
    description = "Extract content from specific web pages"
    args_schema: Type[BaseModel] = WebScrapeInput
    
    def _run(self, url: str, extract_text: bool = True, method: str = "requests") -> str:
        try:
            response = requests.post(
                "http://192.168.1.190:8082/web_scrape",
                json={
                    "url": url, 
                    "extract_text": extract_text,
                    "method": method
                },
                timeout=60
            )
            result = response.json()
            
            if "text" in result:
                return f"**Title:** {result.get('title', 'No title')}\n\n**Content:** {result['text'][:3000]}..."
            else:
                return f"Scraping failed: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Web scraping error: {e}"

class CommandExecuteTool(BaseTool):
    """Safe command execution via tools server"""
    
    name = "execute_command"
    description = "Execute safe shell commands for data processing"
    args_schema: Type[BaseModel] = CommandInput
    
    def _run(self, command: str, timeout: int = 30, working_dir: str = "/tmp") -> str:
        try:
            response = requests.post(
                "http://192.168.1.190:8082/execute_command",
                json={
                    "command": command, 
                    "timeout": timeout,
                    "working_dir": working_dir
                },
                timeout=timeout + 10
            )
            result = response.json()
            
            if "stdout" in result:
                output = f"**Return Code:** {result.get('returncode', 'unknown')}\n"
                if result.get('stdout'):
                    output += f"**Output:**\n{result['stdout']}\n"
                if result.get('stderr'):
                    output += f"**Errors:**\n{result['stderr']}\n"
                return output
            else:
                return f"Command failed: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Command execution error: {e}"
```

### Usage in LangGraph Workflows

```python
# Initialize tools
web_search_tool = WebSearchTool()
web_scrape_tool = WebScrapeTool()
command_tool = CommandExecuteTool()

# In your workflow nodes
def search_and_respond_node(state: AgentState) -> AgentState:
    """Search web and generate response"""
    user_query = state["messages"][-1].content
    
    # Extract search terms from user query
    search_query = extract_search_terms(user_query)
    
    # Search for information
    search_results = web_search_tool._run(search_query, max_results=5)
    
    # Generate enhanced response with search context
    enhanced_prompt = f"""
    User Question: {user_query}
    
    Search Results:
    {search_results}
    
    Please provide a comprehensive answer using the search results above.
    Include relevant URLs for further reading.
    """
    
    response = llm.invoke(enhanced_prompt)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "context": search_results,
        "tools_used": state.get("tools_used", []) + ["web_search"]
    }

def scrape_and_respond_node(state: AgentState) -> AgentState:
    """Scrape specific URL and generate response"""
    user_message = state["messages"][-1].content
    
    # Extract URL from user message
    url = extract_url(user_message)
    
    if url:
        # Scrape the webpage
        scraped_content = web_scrape_tool._run(url, extract_text=True)
        
        # Generate response based on scraped content
        enhanced_prompt = f"""
        User Request: {user_message}
        
        Scraped Content:
        {scraped_content}
        
        Please analyze and summarize the content above.
        """
        
        response = llm.invoke(enhanced_prompt)
        
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response)],
            "context": scraped_content,
            "tools_used": state.get("tools_used", []) + ["web_scrape"]
        }
    else:
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content="Please provide a valid URL to scrape.")]
        }

def command_and_respond_node(state: AgentState) -> AgentState:
    """Execute command and generate response"""
    user_message = state["messages"][-1].content
    
    # Extract command from user message (with safety checks)
    command = extract_safe_command(user_message)
    
    if command:
        # Execute the command
        command_result = command_tool._run(command, timeout=60)
        
        # Generate response based on command output
        enhanced_prompt = f"""
        User Request: {user_message}
        
        Command Executed: {command}
        
        Command Output:
        {command_result}
        
        Please explain the command result and what it means.
        """
        
        response = llm.invoke(enhanced_prompt)
        
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response)],
            "context": command_result,
            "tools_used": state.get("tools_used", []) + ["execute_command"]
        }
    else:
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content="I can only execute safe, approved commands.")]
        }
```

## 🎯 Use Cases & Examples

### 1. Real-Time Information Gathering

**Scenario**: User asks about current events or recent developments

```python
# User Question
user_question = "What are the latest developments in quantum computing?"

# Process
1. Extract search query: "quantum computing latest developments 2024"
2. Call web_search tool → Get recent articles
3. Filter and rank results by relevance
4. Generate comprehensive response with sources

# Example Search Results
[
  {
    "title": "IBM Announces New Quantum Processor",
    "url": "https://research.ibm.com/quantum-news",
    "snippet": "IBM's latest 1000-qubit quantum processor breakthrough..."
  },
  {
    "title": "Google's Quantum Supremacy Update",
    "url": "https://quantumai.google/updates",
    "snippet": "New results showing quantum advantage in optimization..."
  }
]

# Generated Response
"Based on the latest information, there have been several significant quantum computing developments:

1. **IBM's 1000-Qubit Processor**: IBM recently announced their breakthrough quantum processor with over 1000 qubits, representing a major leap in quantum computing capability.

2. **Google's Quantum Advantage**: Google has published new results demonstrating quantum advantage in optimization problems...

Sources:
- IBM Research: https://research.ibm.com/quantum-news
- Google Quantum AI: https://quantumai.google/updates"
```

### 2. Website Content Analysis

**Scenario**: User wants to analyze or summarize content from a specific website

```python
# User Request
user_request = "Please analyze the content of https://example.com/research-paper"

# Process
1. Validate URL and check accessibility
2. Choose scraping method (requests vs selenium based on site complexity)
3. Extract title, text content, and key elements
4. Analyze and summarize the content

# Scraping Result
{
  "title": "Advanced Machine Learning Techniques",
  "text": "This research paper explores novel approaches to deep learning...",
  "url": "https://example.com/research-paper"
}

# Generated Analysis
"**Content Analysis: Advanced Machine Learning Techniques**

**Summary**: This research paper discusses cutting-edge developments in deep learning, focusing on:

1. **Novel Architectures**: The paper introduces new neural network architectures that improve efficiency by 30%
2. **Training Methods**: Innovative training techniques that reduce computational requirements
3. **Applications**: Real-world applications in computer vision and natural language processing

**Key Insights**:
- The proposed methods show significant improvements over existing baselines
- The approach is particularly effective for resource-constrained environments
- Future work suggests potential for even greater efficiency gains

**Relevance**: This research has important implications for deploying AI systems in edge computing scenarios."
```

### 3. Data Processing and Analysis

**Scenario**: User needs to process data or perform system operations

```python
# User Request
user_request = "Analyze the log files in /var/log and find any error patterns"

# Process
1. Validate command safety
2. Execute safe data analysis commands
3. Process results and identify patterns
4. Generate human-readable insights

# Safe Commands Executed
commands = [
    "find /var/log -name '*.log' -type f | head -10",
    "grep -i error /var/log/syslog | tail -20",
    "wc -l /var/log/*.log"
]

# Command Results
{
  "command": "grep -i error /var/log/syslog | tail -20",
  "stdout": "Dec 01 10:15:32 server kernel: USB disconnect, address 1\nDec 01 10:20:15 server NetworkManager: Device eth0 connection failed",
  "returncode": 0
}

# Generated Analysis
"**Log Analysis Results**

**Files Found**: 10 log files in /var/log directory

**Error Pattern Analysis**:
1. **Network Issues**: Multiple instances of network connection failures detected
2. **USB Disconnects**: Several USB device disconnection errors
3. **Timing**: Most errors occurred between 10:00-10:30 AM

**Recommendations**:
- Check network cable and switch connectivity
- Investigate USB device stability
- Monitor system during morning hours for recurring issues

**Command Summary**:
- Searched through system logs for error patterns
- Found 20 recent error entries
- No critical system failures detected"
```

### 4. Competitive Intelligence and Research

**Scenario**: Research competitors or gather market intelligence

```python
# User Request
user_request = "Research the latest pricing and features of competitor AI platforms"

# Process
1. Search for competitor information
2. Scrape pricing pages and feature lists
3. Compile and compare offerings
4. Generate competitive analysis

# Multi-Tool Workflow
1. web_search("AI platform pricing comparison 2024")
2. web_scrape("https://competitor1.com/pricing")
3. web_scrape("https://competitor2.com/features")
4. execute_command("curl -s api.competitor.com/pricing")

# Generated Competitive Analysis
"**AI Platform Competitive Analysis**

**Market Overview**:
Based on recent research, the AI platform market shows these trends:

**Competitor 1**:
- Pricing: $0.02/1K tokens for GPT-4 equivalent
- Features: API access, fine-tuning, enterprise support
- Strengths: Established ecosystem, strong documentation

**Competitor 2**:
- Pricing: $0.015/1K tokens
- Features: Open-source models, custom deployments
- Strengths: Cost-effective, privacy-focused

**Our Advantages**:
- On-premises deployment (privacy + control)
- No per-token costs after setup
- Distributed architecture for reliability
- Customizable for specific use cases

**Recommendations**:
- Emphasize privacy and cost benefits
- Highlight distributed reliability features
- Position as enterprise-grade self-hosted solution"
```

## ⚡ Performance & Optimization

### Tool-Specific Optimizations

```python
class OptimizedToolsService:
    def __init__(self):
        # Connection pooling for web requests
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=30),
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # Selenium Chrome options for performance
        self.chrome_options = self._setup_optimized_chrome()
        
        # Command execution sandbox
        self.safe_commands = self._load_safe_commands()
    
    def _setup_optimized_chrome(self):
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-images')  # Faster loading
        options.add_argument('--disable-javascript')  # When not needed
        options.add_argument('--memory-pressure-off')
        return options
    
    async def optimized_web_search(self, query: str, max_results: int):
        # Implement caching for frequent searches
        cache_key = f"search:{hash(query)}:{max_results}"
        cached_result = await self.redis_cache.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        
        # Perform fresh search
        result = await self.web_search(query, max_results)
        
        # Cache for 1 hour
        await self.redis_cache.setex(cache_key, 3600, json.dumps(result))
        
        return result
```

### Performance Expectations

| Tool | Operation | Average Time | Resource Usage |
|------|-----------|--------------|----------------|
| Web Search | 5 results | ~3 seconds | ~100MB RAM |
| Web Search | 20 results | ~8 seconds | ~150MB RAM |
| Web Scrape (requests) | Simple page | ~2 seconds | ~50MB RAM |
| Web Scrape (selenium) | Dynamic page | ~8 seconds | ~300MB RAM |
| Command Execute | Simple command | ~1 second | ~20MB RAM |
| Command Execute | Complex script | ~30 seconds | ~100MB RAM |

### Concurrent Processing

```python
class ConcurrentToolsService:
    async def process_multiple_urls(self, urls: List[str]) -> List[Dict]:
        """Process multiple URLs concurrently"""
        tasks = []
        for url in urls:
            task = self.web_scrape_requests(url, True, False)
            tasks.append(task)
        
        # Process up to 5 URLs concurrently
        semaphore = asyncio.Semaphore(5)
        
        async def bounded_scrape(url_task):
            async with semaphore:
                return await url_task
        
        results = await asyncio.gather(*[bounded_scrape(task) for task in tasks])
        return results
    
    async def parallel_search_and_scrape(self, query: str, target_url: str):
        """Run web search and web scraping in parallel"""
        search_task = self.web_search(query, 10)
        scrape_task = self.web_scrape_requests(target_url, True, False)
        
        search_result, scrape_result = await asyncio.gather(search_task, scrape_task)
        
        return {
            "search": search_result,
            "scrape": scrape_result
        }
```

## 🔄 HAProxy Integration

### Load Balancing Tools

```bash
# HAProxy Configuration for Tools Server
frontend tools_frontend
    bind *:9003
    mode http
    default_backend tools_servers

backend tools_servers
    mode http
    balance roundrobin
    option httpchk GET /health
    
    server tools_primary 192.168.1.190:8082 check inter 30s fall 3 rise 2
    # server tools_secondary 192.168.1.106:8082 check inter 30s fall 3 rise 2  # Future expansion
```

**Access Methods**:

- **Direct**: `http://192.168.1.190:8082/web_search`
- **Load Balanced**: `http://192.168.1.81:9003/web_search`

### Failover Strategy

If worker-node3 goes down:

- HAProxy marks it unhealthy after 3 failed checks (90 seconds)
- LangGraph workflows fall back to simpler responses
- You can add backup tool servers on other worker nodes
- Critical: Tools are enhancement features, system continues without them

## 🛡️ Security & Safety

### Command Execution Security

```python
class SecureCommandExecutor:
    def __init__(self):
        # Whitelist of safe commands
        self.safe_commands = {
            'ls', 'cat', 'head', 'tail', 'wc', 'grep', 'find', 
            'echo', 'date', 'pwd', 'whoami', 'df', 'free',
            'ps', 'top', 'netstat', 'curl', 'wget'
        }
        
        # Blacklist of dangerous patterns
        self.dangerous_patterns = [
            'rm -rf', 'sudo', 'su -', 'chmod 777', 'chown',
            'mkfs', 'dd if=', 'init ', 'shutdown', 'reboot',
            '> /dev/', 'format', 'fdisk', 'mount', 'umount'
        ]
        
        # Safe working directories
        self.safe_dirs = ['/tmp', '/var/tmp', '/home/sanzad/workspace']
    
    def validate_command(self, command: str, working_dir: str) -> bool:
        """Validate command safety"""
        # Check working directory
        if not any(working_dir.startswith(safe_dir) for safe_dir in self.safe_dirs):
            return False
        
        # Check for dangerous patterns
        if any(pattern in command.lower() for pattern in self.dangerous_patterns):
            return False
        
        # Extract command name
        cmd_name = command.split()[0] if command else ""
        
        # Check if command is in whitelist
        return cmd_name in self.safe_commands
```

### Web Scraping Ethics and Limits

```python
class EthicalScrapingService:
    def __init__(self):
        # Rate limiting
        self.rate_limiter = AsyncLimiter(max_rate=10, time_period=60)  # 10 requests per minute
        
        # Robots.txt compliance
        self.robots_cache = {}
        
        # User agent identification
        self.user_agent = "LangGraph-ToolServer/1.0 (Educational Use)"
    
    async def check_robots_txt(self, url: str) -> bool:
        """Check if scraping is allowed by robots.txt"""
        domain = urlparse(url).netloc
        
        if domain not in self.robots_cache:
            try:
                robots_url = f"https://{domain}/robots.txt"
                async with self.session.get(robots_url) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        # Parse robots.txt (simplified)
                        self.robots_cache[domain] = "Disallow: /" not in robots_content
                    else:
                        self.robots_cache[domain] = True  # Allow if no robots.txt
            except:
                self.robots_cache[domain] = True  # Allow on error
        
        return self.robots_cache[domain]
    
    async def ethical_scrape(self, url: str) -> Dict:
        """Scrape with ethical considerations"""
        # Check rate limiting
        async with self.rate_limiter:
            # Check robots.txt compliance
            if not await self.check_robots_txt(url):
                return {"error": "Scraping not allowed by robots.txt"}
            
            # Add delay to be respectful
            await asyncio.sleep(1)
            
            # Proceed with scraping
            return await self.web_scrape_requests(url, True, False)
```

## 🛠️ Service Management

### Systemd Service

```bash
# Service Configuration
/etc/systemd/system/tools-server.service:

[Unit]
Description=LangGraph Tools Execution Server
After=network.target

[Service]
Type=simple
User=sanzad
WorkingDirectory=/home/sanzad/tools-server
Environment=PATH=/home/sanzad/tools-env/bin
Environment=DISPLAY=:99  # For headless Chrome
ExecStart=/home/sanzad/tools-env/bin/python tools_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/tmp /var/tmp
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

### Management Commands

```bash
# Service Control
sudo systemctl start tools-server
sudo systemctl stop tools-server
sudo systemctl restart tools-server
sudo systemctl status tools-server

# Logs
sudo journalctl -u tools-server -f
sudo journalctl -u tools-server --since "1 hour ago"

# Health Checks
curl http://192.168.1.190:8082/health
curl http://192.168.1.190:8082/stats

# Test Tools
# Web Search Test
curl -X POST http://192.168.1.190:8082/web_search \
  -H "Content-Type: application/json" \
  -d '{"query": "test search", "max_results": 3}'

# Web Scrape Test
curl -X POST http://192.168.1.190:8082/web_scrape \
  -H "Content-Type: application/json" \
  -d '{"url": "https://httpbin.org/html", "extract_text": true}'

# Command Test
curl -X POST http://192.168.1.190:8082/execute_command \
  -H "Content-Type: application/json" \
  -d '{"command": "echo Hello from tools server", "timeout": 10}'
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Chrome/Selenium Issues

```bash
# Check Chrome installation
chromium-browser --version

# Test headless Chrome
chromium-browser --headless --no-sandbox --dump-dom https://example.com

# Common fixes:
sudo apt update && sudo apt install -y chromium-browser chromium-chromedriver
sudo apt install -y xvfb  # Virtual display for headless mode

# Set display for service
export DISPLAY=:99
```

#### 2. Network/Firewall Issues

```bash
# Check port accessibility
netstat -ln | grep 8082
telnet 192.168.1.190 8082

# Configure firewall
sudo ufw allow 8082/tcp
sudo ufw status

# Test from coordinator
curl -v http://192.168.1.190:8082/health
```

#### 3. Performance Issues

```bash
# Monitor resource usage
htop
free -h
df -h

# Check for memory leaks
ps aux | grep tools_server
cat /proc/$(pgrep -f tools_server)/status

# Restart if needed
sudo systemctl restart tools-server
```

#### 4. Tool-Specific Errors

```bash
# Web Search Issues
# - Check internet connectivity
ping 8.8.8.8
nslookup duckduckgo.com

# - Test direct search
curl "https://duckduckgo.com/html/?q=test"

# Web Scraping Issues
# - Check target site accessibility
curl -I https://target-site.com

# - Test with different user agent
curl -H "User-Agent: Mozilla/5.0..." https://target-site.com

# Command Execution Issues
# - Check command whitelist
# - Verify working directory permissions
ls -la /tmp
```

## 🚀 Advanced Features

### Custom Tool Extensions

Add your own tools to the server:

```python
class CustomDataTool:
    async def process_csv_data(self, file_path: str, operation: str) -> Dict:
        """Process CSV data with pandas"""
        try:
            df = pd.read_csv(file_path)
            
            if operation == "summary":
                return {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "summary": df.describe().to_dict()
                }
            elif operation == "head":
                return {"data": df.head().to_dict()}
            
        except Exception as e:
            return {"error": str(e)}

# Add to FastAPI app
@app.post("/process_csv")
async def process_csv(request: CSVProcessRequest):
    return await custom_tool.process_csv_data(request.file_path, request.operation)
```

### AI-Powered Tool Selection

Implement intelligent tool routing:

```python
class IntelligentToolRouter:
    def __init__(self, embeddings_service):
        self.embeddings = embeddings_service
        self.tool_descriptions = {
            "web_search": "Find current information, news, facts, recent events",
            "web_scrape": "Extract content from specific websites, analyze web pages",
            "execute_command": "Process data, run calculations, system operations"
        }
        
        # Pre-compute tool embeddings
        self.tool_embeddings = {}
        for tool, desc in self.tool_descriptions.items():
            self.tool_embeddings[tool] = self.embeddings.embed_query(desc)
    
    def select_best_tool(self, user_query: str) -> str:
        """Select most appropriate tool for user query"""
        query_embedding = self.embeddings.embed_query(user_query)
        
        best_tool = None
        best_score = -1
        
        for tool, tool_embedding in self.tool_embeddings.items():
            similarity = cosine_similarity(query_embedding, tool_embedding)
            if similarity > best_score:
                best_score = similarity
                best_tool = tool
        
        return best_tool
```

### Multi-Tool Workflows

Create complex multi-step tool workflows:

```python
class MultiToolWorkflow:
    async def research_workflow(self, topic: str) -> Dict:
        """Complete research workflow using multiple tools"""
        results = {}
        
        # Step 1: Search for general information
        search_results = await self.web_search(topic, 10)
        results["search"] = search_results
        
        # Step 2: Scrape top results for detailed content
        if "results" in search_results:
            top_urls = [r["url"] for r in search_results["results"][:3]]
            scrape_results = await self.process_multiple_urls(top_urls)
            results["scraped_content"] = scrape_results
        
        # Step 3: Process and analyze data
        analysis_command = f"echo 'Analyzing {len(results)} sources for topic: {topic}'"
        command_result = await self.execute_command(analysis_command, 30, "/tmp")
        results["analysis"] = command_result
        
        return results
    
    async def competitive_analysis_workflow(self, company: str) -> Dict:
        """Automated competitive analysis"""
        workflow_results = {}
        
        # Search for company information
        company_search = await self.web_search(f"{company} company information", 5)
        workflow_results["company_info"] = company_search
        
        # Search for competitors
        competitor_search = await self.web_search(f"{company} competitors", 5)
        workflow_results["competitors"] = competitor_search
        
        # Search for pricing information
        pricing_search = await self.web_search(f"{company} pricing plans", 5)
        workflow_results["pricing"] = pricing_search
        
        return workflow_results
```

### Caching and Performance

Add Redis caching for frequently used results:

```python
import redis
import json
import hashlib

class CachedToolsService:
    def __init__(self):
        self.redis_client = redis.Redis(
            host='192.168.1.81', 
            port=6379, 
            password='langgraph_redis_pass',
            decode_responses=True
        )
        self.cache_ttl = 3600  # 1 hour default
    
    def get_cache_key(self, tool: str, params: Dict) -> str:
        """Generate cache key for tool and parameters"""
        key_data = f"{tool}:{json.dumps(params, sort_keys=True)}"
        return f"tools:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    async def cached_web_search(self, query: str, max_results: int) -> Dict:
        """Web search with caching"""
        cache_key = self.get_cache_key("web_search", {"query": query, "max_results": max_results})
        
        # Check cache
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Perform search
        result = await self.web_search(query, max_results)
        
        # Cache result
        self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(result))
        
        return result
    
    async def cached_web_scrape(self, url: str, extract_text: bool) -> Dict:
        """Web scraping with caching"""
        cache_key = self.get_cache_key("web_scrape", {"url": url, "extract_text": extract_text})
        
        # Check cache (longer TTL for scraping since content changes less frequently)
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Perform scraping
        result = await self.web_scrape_requests(url, extract_text, False)
        
        # Cache result for 24 hours
        self.redis_client.setex(cache_key, 86400, json.dumps(result))
        
        return result
```

## 🎯 Why This Architecture Works

### Benefits of Dedicated Tool Execution

1. **Security Isolation**: Tools run in isolated environment, protecting main systems
2. **Resource Management**: Tool operations don't impact LLM inference performance  
3. **Scalability**: Easy to add more worker nodes for increased tool capacity
4. **Reliability**: If tools fail, core LangGraph continues functioning
5. **Flexibility**: Can add new tools without touching core system
6. **Compliance**: Easier to audit and control external operations

### Integration with LangGraph Workflows

The tool execution server enables sophisticated AI capabilities:

- **Real-Time Information**: Access to current web information and data
- **Content Analysis**: Deep analysis of websites and documents
- **Data Processing**: Automated data manipulation and analysis
- **Research Automation**: Multi-step research workflows
- **Competitive Intelligence**: Automated market and competitor analysis
- **System Operations**: Safe execution of system commands and scripts

### Workflow Enhancement Patterns

1. **Information Augmentation**: LLM responses enhanced with real-time data
2. **Content Verification**: Cross-reference LLM outputs with web sources
3. **Automated Research**: Multi-step information gathering workflows
4. **Dynamic Decision Making**: Tool selection based on query analysis
5. **Context Enrichment**: Enhanced responses with external data sources

## 🏆 Conclusion

The Tool Execution Server is not just "another microservice" - it's the **action arm** of your LangGraph cluster. It transforms your AI system from a static knowledge base to a dynamic, real-time information processing and action execution platform.

**Key Takeaways**:

- Runs on isolated worker hardware (worker-node3) for security and performance
- Provides three core capabilities: web search, web scraping, and command execution  
- Integrates seamlessly with LangGraph through custom tool classes
- Enables advanced AI patterns like research automation and competitive intelligence
- Maintains security through command whitelisting and sandboxed execution
- Scales with your needs through load balancing and worker node expansion

Without the tool execution server, your LangGraph would be limited to static knowledge and pre-trained responses. With it, your AI becomes a dynamic, capable agent that can research, analyze, and act on real-world information - just like a skilled human assistant! 🛠️✨
