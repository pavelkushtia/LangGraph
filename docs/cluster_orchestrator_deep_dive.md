# Cluster Orchestrator Deep Dive 🎛️

## 🎯 What is the Cluster Orchestrator?

The **Cluster Orchestrator** is the mission control center of your LangGraph cluster that manages the entire distributed AI infrastructure lifecycle. It runs on your **cpu-node (192.168.1.81)** and serves as the centralized command and control system for starting, stopping, monitoring, and testing all services across your 5-machine cluster.

### Why Do We Need a Cluster Orchestrator?

The cluster orchestrator solves a critical operational challenge: **coordinating complex distributed systems**. By providing centralized lifecycle management, we enable:

- **Ordered Service Startup**: Services start in correct dependency order to prevent failures
- **Graceful Shutdown**: Proper service termination to prevent data corruption
- **Centralized Management**: Single interface to control entire cluster infrastructure
- **Health Monitoring**: Comprehensive status checking across all nodes and services
- **Automated Testing**: Built-in test suite to verify cluster functionality
- **Operational Simplicity**: Complex distributed operations made simple

## 🏗️ Technical Architecture

### Orchestrator Components

```text
cpu-node (192.168.1.81) - Cluster Orchestrator
├── Cluster Management Engine
│   ├── Service Startup Sequencer
│   ├── Graceful Shutdown Controller
│   └── Dependency Order Manager
├── Remote Execution Framework
│   ├── SSH Command Dispatcher
│   ├── Local Command Runner
│   └── Error Handling & Retry Logic
├── Health Monitoring System
│   ├── Service Status Checker
│   ├── HTTP Health Probes
│   └── Redis Connectivity Tests
├── Testing Framework
│   ├── Individual Service Tests
│   ├── Integration Tests
│   └── Load Balancer Tests
└── Convenience Scripts Generator
    ├── Start/Stop Scripts
    ├── Status Scripts
    └── Restart Scripts
```

### Hardware Placement

**Why cpu-node for Orchestration?**

- **Central Location**: 32GB Intel machine with reliable connectivity to all nodes
- **Resource Availability**: Sufficient resources to handle orchestration without impacting core services
- **Network Position**: Can reach all cluster nodes via SSH and HTTP
- **Reliability**: Most stable machine in cluster for critical control operations
- **Co-location**: Same machine as HAProxy and Redis for reduced latency

### Managed Infrastructure

```python
# Complete Cluster Topology
cluster_machines = {
    "jetson": "192.168.1.177",        # Jetson Orin Nano 8GB
    "cpu_coordinator": "192.168.1.81", # Intel i5-6500T 32GB  
    "rp_embeddings": "192.168.1.178",  # ARM Cortex-A76 8GB
    "worker_tools": "192.168.1.190",   # Intel i5 VM 6GB
    "worker_monitor": "192.168.1.191"  # Intel i5 VM 6GB
}

# Managed Services by Machine
services_topology = {
    "jetson": ["ollama"],
    "cpu_coordinator": ["redis-server", "llama-server", "haproxy"],
    "rp_embeddings": ["embeddings-server"],
    "worker_tools": ["tools-server"], 
    "worker_monitor": ["monitoring-server"]
}

# Service Dependencies & Startup Order
startup_sequence = [
    # Phase 1: Core Infrastructure
    {"service": "redis-server", "machine": "cpu_coordinator", "wait": 3},
    
    # Phase 2: Model Servers  
    {"service": "ollama", "machine": "jetson", "wait": 10},
    {"service": "llama-server", "machine": "cpu_coordinator", "wait": 10},
    
    # Phase 3: Application Services
    {"service": "embeddings-server", "machine": "rp_embeddings", "wait": 5},
    {"service": "tools-server", "machine": "worker_tools", "wait": 5},
    
    # Phase 4: Load Balancing & Monitoring
    {"service": "haproxy", "machine": "cpu_coordinator", "wait": 5},
    {"service": "monitoring-server", "machine": "worker_monitor", "wait": 5}
]
```

## 🔌 Command Interface

### python3 cluster_orchestrator.py start

**Purpose**: Start entire cluster in correct dependency order

```bash
# Command Execution
python3 cluster_orchestrator.py start

# Output Example
🚀 Starting LangGraph cluster services...
Starting Redis cache...
  [cpu_coordinator] redis-server: ✅ Started
Starting model servers...
  [jetson] ollama: ✅ Started  
  [cpu_coordinator] llama-server: ✅ Started
Starting application services...
  [rp_embeddings] embeddings-server: ✅ Started
  [worker_tools] tools-server: ✅ Started
Starting load balancer and monitoring...
  [cpu_coordinator] haproxy: ✅ Started
  [worker_monitor] monitoring-server: ✅ Started
✅ Cluster startup completed!

🌐 Services available at:
  - LLM Load Balancer: http://192.168.1.81:9000
  - Tools Load Balancer: http://192.168.1.81:9001
  - Embeddings Load Balancer: http://192.168.1.81:9002
  - Cluster Health: http://192.168.1.191:8083/cluster_health
  - HAProxy Stats: http://192.168.1.81:9000/haproxy_stats
```

### python3 cluster_orchestrator.py stop

**Purpose**: Gracefully stop all cluster services in reverse order

```bash
# Command Execution
python3 cluster_orchestrator.py stop

# Output Example
🛑 Stopping cluster services...
Stopping monitoring-server on worker_monitor...
  [worker_monitor] monitoring-server: ✅ Stopped
Stopping haproxy on cpu_coordinator...
  [cpu_coordinator] haproxy: ✅ Stopped
Stopping tools-server on worker_tools...
  [worker_tools] tools-server: ✅ Stopped
Stopping embeddings-server on rp_embeddings...
  [rp_embeddings] embeddings-server: ✅ Stopped
Stopping llama-server on cpu_coordinator...
  [cpu_coordinator] llama-server: ✅ Stopped
Stopping ollama on jetson...
  [jetson] ollama: ✅ Stopped
Stopping redis-server on cpu_coordinator...
  [cpu_coordinator] redis-server: ✅ Stopped
✅ Cluster shutdown completed!
```

### python3 cluster_orchestrator.py status

**Purpose**: Check status of all services across cluster

```bash
# Command Execution
python3 cluster_orchestrator.py status

# Output Example
📊 LangGraph Cluster Status:
============================================================

jetson (192.168.1.177):
  ollama: 🟢 Active

cpu_coordinator (192.168.1.81):
  llama-server: 🟢 Active
  haproxy: 🟢 Active
  redis-server: 🟢 Active

rp_embeddings (192.168.1.178):
  embeddings-server: 🟢 Active

worker_tools (192.168.1.190):
  tools-server: 🔴 Failed

worker_monitor (192.168.1.191):
  monitoring-server: 🟢 Active

📡 Cluster Health Check:
  Overall Status: DEGRADED
  Alerts: 1
```

### python3 cluster_orchestrator.py test

**Purpose**: Run comprehensive functionality tests across all services

```bash
# Command Execution  
python3 cluster_orchestrator.py test

# Output Example
🧪 Testing cluster functionality...
  ✅ Jetson Ollama: Healthy
  ✅ CPU Llama.cpp: Healthy
  ✅ Embeddings Server: Healthy
  ❌ Tools Server: Connection refused
  ✅ Monitoring Server: Healthy
  ✅ Load Balancer: Healthy
  ✅ Redis: Healthy
```

### python3 cluster_orchestrator.py restart

**Purpose**: Perform complete cluster restart with proper sequencing

```bash
# Command Execution
python3 cluster_orchestrator.py restart

# Equivalent to:
# 1. python3 cluster_orchestrator.py stop
# 2. sleep 10
# 3. python3 cluster_orchestrator.py start
```

## 🔗 LangGraph Integration

### Orchestrator Integration Class

```python
class LangGraphOrchestrator:
    """Integration between LangGraph workflows and cluster orchestrator"""
    
    def __init__(self, orchestrator_path: str = "~/ai-infrastructure/langgraph-config"):
        self.orchestrator_path = orchestrator_path
        self.orchestrator_script = f"{orchestrator_path}/cluster_orchestrator.py"
    
    async def ensure_cluster_ready(self) -> bool:
        """Ensure cluster is running before starting workflows"""
        try:
            # Check cluster status
            result = subprocess.run([
                "python3", self.orchestrator_script, "status"
            ], capture_output=True, text=True, cwd=self.orchestrator_path)
            
            if result.returncode == 0:
                # Parse status output to check if all critical services are running
                output = result.stdout
                critical_services = ["ollama", "embeddings-server", "tools-server", "redis-server"]
                
                for service in critical_services:
                    if f"{service}: 🔴" in output:
                        logger.warning(f"Critical service {service} is down")
                        return False
                
                return True
            else:
                logger.error(f"Cluster status check failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to check cluster status: {e}")
            return False
    
    async def auto_start_cluster(self) -> bool:
        """Automatically start cluster if not running"""
        try:
            logger.info("Starting cluster automatically...")
            
            result = subprocess.run([
                "python3", self.orchestrator_script, "start"
            ], capture_output=True, text=True, cwd=self.orchestrator_path)
            
            if result.returncode == 0:
                logger.info("Cluster started successfully")
                # Wait for services to be fully ready
                await asyncio.sleep(30)
                return await self.ensure_cluster_ready()
            else:
                logger.error(f"Failed to start cluster: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Auto-start cluster failed: {e}")
            return False
    
    async def run_cluster_tests(self) -> Dict[str, Any]:
        """Run cluster functionality tests"""
        try:
            result = subprocess.run([
                "python3", self.orchestrator_script, "test"
            ], capture_output=True, text=True, cwd=self.orchestrator_path)
            
            # Parse test results
            test_results = {}
            lines = result.stdout.split('\n')
            
            for line in lines:
                if '✅' in line:
                    service = line.split('✅')[1].split(':')[0].strip()
                    test_results[service] = "healthy"
                elif '❌' in line:
                    service = line.split('❌')[1].split(':')[0].strip()
                    test_results[service] = "failed"
            
            return {
                "overall_success": result.returncode == 0,
                "service_results": test_results,
                "output": result.stdout
            }
            
        except Exception as e:
            return {
                "overall_success": False,
                "error": str(e)
            }
```

### Pre-Workflow Cluster Validation

```python
# Integration with LangGraph workflows
async def cluster_aware_workflow(state: AgentState) -> AgentState:
    """LangGraph workflow with cluster health validation"""
    
    orchestrator = LangGraphOrchestrator()
    
    # Ensure cluster is ready before processing
    cluster_ready = await orchestrator.ensure_cluster_ready()
    
    if not cluster_ready:
        logger.warning("Cluster not ready, attempting auto-start...")
        
        # Try to auto-start cluster
        start_success = await orchestrator.auto_start_cluster()
        
        if not start_success:
            return {
                **state,
                "messages": state["messages"] + [AIMessage(
                    content="System is currently unavailable. Please try again in a few minutes."
                )],
                "error": "cluster_unavailable"
            }
    
    # Proceed with normal workflow
    return await standard_workflow_processing(state)

# Workflow health monitoring
async def health_monitoring_node(state: AgentState) -> AgentState:
    """Monitor cluster health during workflow execution"""
    
    orchestrator = LangGraphOrchestrator()
    
    # Run quick health test
    test_results = await orchestrator.run_cluster_tests()
    
    # Add health information to state
    state["cluster_health"] = {
        "timestamp": datetime.now().isoformat(),
        "overall_healthy": test_results.get("overall_success", False),
        "service_status": test_results.get("service_results", {}),
        "alerts": []
    }
    
    # Generate alerts for failed services
    for service, status in test_results.get("service_results", {}).items():
        if status == "failed":
            state["cluster_health"]["alerts"].append(f"Service {service} is unhealthy")
    
    return state

# Automated cluster management workflow
async def cluster_management_workflow(action: str) -> Dict[str, Any]:
    """LangGraph workflow for cluster management operations"""
    
    orchestrator = LangGraphOrchestrator()
    
    try:
        if action == "start":
            success = await orchestrator.auto_start_cluster()
            return {
                "action": "start",
                "success": success,
                "message": "Cluster started successfully" if success else "Failed to start cluster"
            }
        
        elif action == "test":
            results = await orchestrator.run_cluster_tests()
            return {
                "action": "test",
                "success": results.get("overall_success", False),
                "details": results.get("service_results", {}),
                "message": "All services healthy" if results.get("overall_success") else "Some services failed"
            }
        
        elif action == "status":
            healthy = await orchestrator.ensure_cluster_ready()
            return {
                "action": "status", 
                "success": healthy,
                "message": "Cluster healthy" if healthy else "Cluster has issues"
            }
            
    except Exception as e:
        return {
            "action": action,
            "success": False,
            "error": str(e),
            "message": f"Cluster management failed: {e}"
        }
```

## 🎯 Use Cases & Examples

### 1. Daily Operations Management

**Scenario**: Starting and managing the cluster for daily AI operations

```python
# Morning startup routine
async def morning_cluster_startup():
    """Complete morning cluster startup sequence"""
    
    print("🌅 Starting daily LangGraph cluster operations...")
    
    # 1. Start cluster
    start_result = subprocess.run([
        "python3", "cluster_orchestrator.py", "start"
    ], capture_output=True, text=True)
    
    if start_result.returncode != 0:
        print(f"❌ Cluster startup failed: {start_result.stderr}")
        return False
    
    print("✅ Cluster started successfully")
    
    # 2. Wait for services to stabilize
    print("⏳ Waiting for services to stabilize...")
    await asyncio.sleep(30)
    
    # 3. Run comprehensive tests
    test_result = subprocess.run([
        "python3", "cluster_orchestrator.py", "test"
    ], capture_output=True, text=True)
    
    print("🧪 Test Results:")
    print(test_result.stdout)
    
    # 4. Check health monitoring
    try:
        health_response = requests.get("http://192.168.1.191:8083/cluster_health")
        health_data = health_response.json()
        
        print(f"📊 Cluster Health: {health_data['overall_status'].upper()}")
        
        if health_data['alerts']:
            print("🚨 Active Alerts:")
            for alert in health_data['alerts']:
                print(f"  - {alert}")
        
        return health_data['overall_status'] == 'healthy'
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

# Evening shutdown routine  
async def evening_cluster_shutdown():
    """Complete evening cluster shutdown sequence"""
    
    print("🌙 Starting evening cluster shutdown...")
    
    # 1. Check for active workflows (optional)
    # Implementation depends on your workflow tracking
    
    # 2. Graceful shutdown
    stop_result = subprocess.run([
        "python3", "cluster_orchestrator.py", "stop"
    ], capture_output=True, text=True)
    
    if stop_result.returncode == 0:
        print("✅ Cluster shutdown completed successfully")
        return True
    else:
        print(f"❌ Cluster shutdown failed: {stop_result.stderr}")
        return False
```

### 2. Automated Recovery and Restart

**Scenario**: Automatic cluster recovery when issues are detected

```python
class ClusterRecoveryManager:
    def __init__(self):
        self.max_restart_attempts = 3
        self.restart_delay = 60  # seconds
        self.last_restart_time = 0
        
    async def check_and_recover(self) -> bool:
        """Check cluster health and recover if needed"""
        
        # Get current health status
        try:
            health_response = requests.get(
                "http://192.168.1.191:8083/cluster_health", 
                timeout=10
            )
            health_data = health_response.json()
            
            if health_data['overall_status'] == 'unhealthy':
                logger.warning("Cluster unhealthy, attempting recovery...")
                return await self.perform_recovery()
            elif health_data['overall_status'] == 'degraded':
                logger.warning("Cluster degraded, monitoring...")
                return await self.monitor_degraded_cluster()
            else:
                logger.info("Cluster healthy")
                return True
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return await self.perform_recovery()
    
    async def perform_recovery(self) -> bool:
        """Perform cluster recovery sequence"""
        
        current_time = time.time()
        
        # Rate limiting - don't restart too frequently
        if current_time - self.last_restart_time < self.restart_delay:
            logger.warning("Recovery rate limited, waiting...")
            return False
        
        self.last_restart_time = current_time
        
        try:
            logger.info("Starting cluster recovery...")
            
            # 1. Stop cluster
            stop_result = subprocess.run([
                "python3", "cluster_orchestrator.py", "stop"
            ], capture_output=True, text=True, timeout=60)
            
            if stop_result.returncode != 0:
                logger.error(f"Failed to stop cluster: {stop_result.stderr}")
            
            # 2. Wait for clean shutdown
            await asyncio.sleep(15)
            
            # 3. Start cluster
            start_result = subprocess.run([
                "python3", "cluster_orchestrator.py", "start"  
            ], capture_output=True, text=True, timeout=120)
            
            if start_result.returncode != 0:
                logger.error(f"Failed to start cluster: {start_result.stderr}")
                return False
            
            # 4. Wait for services to initialize
            logger.info("Waiting for services to initialize...")
            await asyncio.sleep(45)
            
            # 5. Verify recovery
            test_result = subprocess.run([
                "python3", "cluster_orchestrator.py", "test"
            ], capture_output=True, text=True)
            
            recovery_success = test_result.returncode == 0
            
            if recovery_success:
                logger.info("✅ Cluster recovery successful")
            else:
                logger.error("❌ Cluster recovery failed")
            
            return recovery_success
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    async def monitor_degraded_cluster(self) -> bool:
        """Monitor degraded cluster and decide if recovery is needed"""
        
        # Wait and check again
        await asyncio.sleep(30)
        
        try:
            health_response = requests.get(
                "http://192.168.1.191:8083/cluster_health",
                timeout=10
            )
            health_data = health_response.json()
            
            if health_data['overall_status'] == 'unhealthy':
                logger.warning("Cluster degraded to unhealthy, starting recovery")
                return await self.perform_recovery()
            elif health_data['overall_status'] == 'healthy':
                logger.info("Cluster recovered to healthy state")
                return True
            else:
                logger.info("Cluster still degraded, continuing to monitor")
                return True
                
        except Exception as e:
            logger.error(f"Degraded cluster monitoring failed: {e}")
            return await self.perform_recovery()
```

### 3. Cluster Performance Optimization

**Scenario**: Optimize cluster performance based on usage patterns

```python
class ClusterPerformanceOptimizer:
    def __init__(self):
        self.performance_history = []
        self.optimization_thresholds = {
            "high_load": 0.8,
            "low_load": 0.2, 
            "memory_pressure": 0.85
        }
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """Analyze cluster performance and suggest optimizations"""
        
        performance_data = {}
        
        # Get cluster health data
        try:
            health_response = requests.get("http://192.168.1.191:8083/cluster_health")
            health_data = health_response.json()
            
            performance_data["cluster_status"] = health_data["overall_status"]
            performance_data["service_performance"] = {}
            
            # Analyze individual service performance
            for service_name, service_data in health_data["services"].items():
                health_info = service_data.get("health", {})
                
                performance_data["service_performance"][service_name] = {
                    "status": health_info.get("status", "unknown"),
                    "response_time": health_info.get("response_time", 0),
                    "healthy": health_info.get("status") == "healthy"
                }
            
            # Generate optimization recommendations
            recommendations = self.generate_optimization_recommendations(performance_data)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "performance_data": performance_data,
                "recommendations": recommendations,
                "optimization_actions": self.get_optimization_actions(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {"error": str(e)}
    
    def generate_optimization_recommendations(self, perf_data: Dict) -> List[str]:
        """Generate performance optimization recommendations"""
        
        recommendations = []
        
        # Check service response times
        for service, metrics in perf_data.get("service_performance", {}).items():
            response_time = metrics.get("response_time", 0)
            
            if service == "jetson_ollama" and response_time > 1.0:
                recommendations.append(f"Jetson Ollama slow ({response_time}s) - consider smaller model")
            elif service == "embeddings" and response_time > 0.5:
                recommendations.append(f"Embeddings slow ({response_time}s) - check ARM optimization")
            elif service == "tools" and response_time > 5.0:
                recommendations.append(f"Tools server slow ({response_time}s) - investigate network/timeouts")
        
        # Check overall cluster health
        if perf_data.get("cluster_status") == "degraded":
            recommendations.append("Cluster degraded - investigate failing services")
        
        # Resource utilization recommendations
        recommendations.extend([
            "Monitor Jetson GPU usage with tegrastats",
            "Check cpu-node memory usage for large model operations", 
            "Verify network bandwidth between nodes",
            "Review HAProxy load distribution"
        ])
        
        return recommendations
    
    def get_optimization_actions(self, recommendations: List[str]) -> List[Dict]:
        """Convert recommendations into actionable commands"""
        
        actions = []
        
        for rec in recommendations:
            if "smaller model" in rec:
                actions.append({
                    "type": "model_optimization",
                    "command": "ollama pull llama3.2:1b",
                    "description": "Switch to smaller model on Jetson"
                })
            elif "network" in rec:
                actions.append({
                    "type": "network_check", 
                    "command": "python3 cluster_orchestrator.py test",
                    "description": "Run network connectivity tests"
                })
            elif "investigate failing" in rec:
                actions.append({
                    "type": "service_restart",
                    "command": "python3 cluster_orchestrator.py restart",
                    "description": "Restart cluster to resolve service issues"
                })
        
        return actions

# Usage example
async def optimize_cluster_performance():
    """Run performance optimization analysis"""
    
    optimizer = ClusterPerformanceOptimizer()
    analysis = await optimizer.analyze_performance()
    
    print("📊 Cluster Performance Analysis:")
    print(f"Overall Status: {analysis.get('performance_data', {}).get('cluster_status', 'unknown')}")
    
    print("\n🔍 Recommendations:")
    for rec in analysis.get('recommendations', []):
        print(f"  - {rec}")
    
    print("\n⚡ Optimization Actions:")
    for action in analysis.get('optimization_actions', []):
        print(f"  - {action['description']}")
        print(f"    Command: {action['command']}")
```

### 4. Cluster Development and Testing

**Scenario**: Development workflow with cluster management

```python
class ClusterDevelopmentManager:
    def __init__(self):
        self.development_mode = True
        self.auto_restart = True
        
    async def development_cycle(self, test_workflow: callable):
        """Complete development and testing cycle"""
        
        try:
            # 1. Ensure cluster is running
            print("🔧 Development Cycle Starting...")
            
            cluster_ready = await self.ensure_development_cluster()
            if not cluster_ready:
                print("❌ Failed to prepare development cluster")
                return False
            
            # 2. Run the test workflow
            print("🧪 Running test workflow...")
            test_result = await test_workflow()
            
            # 3. Analyze results
            if test_result.get("success", False):
                print("✅ Test workflow successful")
            else:
                print(f"❌ Test workflow failed: {test_result.get('error', 'Unknown error')}")
            
            # 4. Optional: restart cluster for clean state
            if self.auto_restart:
                print("🔄 Restarting cluster for clean state...")
                await self.restart_for_development()
            
            return test_result.get("success", False)
            
        except Exception as e:
            print(f"❌ Development cycle failed: {e}")
            return False
    
    async def ensure_development_cluster(self) -> bool:
        """Ensure cluster is ready for development"""
        
        # Quick health check
        test_result = subprocess.run([
            "python3", "cluster_orchestrator.py", "test"
        ], capture_output=True, text=True)
        
        if test_result.returncode == 0:
            print("✅ Cluster already healthy")
            return True
        
        # Try to start cluster
        print("🚀 Starting cluster for development...")
        start_result = subprocess.run([
            "python3", "cluster_orchestrator.py", "start"
        ], capture_output=True, text=True)
        
        if start_result.returncode != 0:
            print(f"❌ Failed to start cluster: {start_result.stderr}")
            return False
        
        # Wait and verify
        await asyncio.sleep(20)
        
        verify_result = subprocess.run([
            "python3", "cluster_orchestrator.py", "test"
        ], capture_output=True, text=True)
        
        return verify_result.returncode == 0
    
    async def restart_for_development(self) -> bool:
        """Quick restart for development testing"""
        
        restart_result = subprocess.run([
            "python3", "cluster_orchestrator.py", "restart"
        ], capture_output=True, text=True)
        
        success = restart_result.returncode == 0
        
        if success:
            print("✅ Development restart completed")
        else:
            print(f"❌ Development restart failed: {restart_result.stderr}")
        
        return success

# Example test workflow
async def test_langgraph_workflow():
    """Example LangGraph workflow test"""
    
    try:
        # Test simple LLM interaction
        llm_response = requests.post(
            "http://192.168.1.81:9000/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": "Test prompt for development",
                "stream": False
            },
            timeout=30
        )
        
        if llm_response.status_code != 200:
            return {"success": False, "error": f"LLM test failed: {llm_response.status_code}"}
        
        # Test embeddings
        embed_response = requests.post(
            "http://192.168.1.81:9002/embeddings",
            json={
                "texts": ["development test"],
                "model": "default"
            },
            timeout=10
        )
        
        if embed_response.status_code != 200:
            return {"success": False, "error": f"Embeddings test failed: {embed_response.status_code}"}
        
        # Test tools
        tools_response = requests.post(
            "http://192.168.1.81:9001/web_search",
            json={
                "query": "test search",
                "max_results": 1
            },
            timeout=15
        )
        
        if tools_response.status_code != 200:
            return {"success": False, "error": f"Tools test failed: {tools_response.status_code}"}
        
        return {"success": True, "message": "All services tested successfully"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# Development workflow usage
async def run_development_session():
    """Complete development session"""
    
    dev_manager = ClusterDevelopmentManager()
    
    # Run multiple test cycles
    for i in range(3):
        print(f"\n🔄 Development Cycle {i+1}/3")
        success = await dev_manager.development_cycle(test_langgraph_workflow)
        
        if not success:
            print(f"❌ Cycle {i+1} failed, stopping development session")
            break
        
        print(f"✅ Cycle {i+1} completed successfully")
```

## ⚡ Performance & Optimization

### Orchestration Performance Tuning

```python
class OptimizedClusterOrchestrator:
    def __init__(self):
        # Optimized timeouts for different operations
        self.timeouts = {
            "service_start": 30,
            "service_stop": 15,
            "health_check": 10,
            "ssh_command": 20
        }
        
        # Concurrent operations where safe
        self.concurrent_operations = {
            "independent_services": ["embeddings-server", "tools-server"],
            "model_servers": ["ollama", "llama-server"]
        }
        
        # Retry logic
        self.retry_config = {
            "max_attempts": 3,
            "retry_delay": 5,
            "exponential_backoff": True
        }
    
    async def optimized_cluster_start(self):
        """Optimized cluster startup with concurrent operations"""
        
        try:
            # Phase 1: Core infrastructure (sequential - Redis must be first)
            await self.start_service_with_retry("cpu_coordinator", "redis-server", 3)
            
            # Phase 2: Model servers (concurrent - independent)
            model_tasks = [
                self.start_service_with_retry("jetson", "ollama", 15),
                self.start_service_with_retry("cpu_coordinator", "llama-server", 15)
            ]
            await asyncio.gather(*model_tasks)
            
            # Phase 3: Application services (concurrent - independent)
            app_tasks = [
                self.start_service_with_retry("rp_embeddings", "embeddings-server", 5),
                self.start_service_with_retry("worker_tools", "tools-server", 5)
            ]
            await asyncio.gather(*app_tasks)
            
            # Phase 4: Infrastructure services (sequential - HAProxy needs apps)
            await self.start_service_with_retry("cpu_coordinator", "haproxy", 5)
            await self.start_service_with_retry("worker_monitor", "monitoring-server", 5)
            
            return True
            
        except Exception as e:
            logger.error(f"Optimized cluster start failed: {e}")
            return False
    
    async def start_service_with_retry(self, machine: str, service: str, wait_time: int):
        """Start service with retry logic and timeout"""
        
        for attempt in range(self.retry_config["max_attempts"]):
            try:
                # Start service
                result = await self.run_remote_command_async(
                    machine, 
                    f"sudo systemctl start {service}",
                    timeout=self.timeouts["service_start"]
                )
                
                if result["success"]:
                    logger.info(f"✅ {service} started on {machine}")
                    
                    # Wait for service to stabilize
                    await asyncio.sleep(wait_time)
                    
                    # Verify service is running
                    status_result = await self.run_remote_command_async(
                        machine,
                        f"systemctl is-active {service}",
                        timeout=5
                    )
                    
                    if status_result.get("output", "").strip() == "active":
                        return True
                    else:
                        logger.warning(f"Service {service} not active after start")
                
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed for {service}: {e}")
            
            # Exponential backoff retry delay
            if attempt < self.retry_config["max_attempts"] - 1:
                delay = self.retry_config["retry_delay"]
                if self.retry_config["exponential_backoff"]:
                    delay *= (2 ** attempt)
                
                logger.info(f"Retrying {service} in {delay} seconds...")
                await asyncio.sleep(delay)
        
        raise Exception(f"Failed to start {service} after {self.retry_config['max_attempts']} attempts")
    
    async def run_remote_command_async(self, machine: str, command: str, timeout: int = 20) -> Dict:
        """Async wrapper for remote command execution"""
        
        try:
            if machine == 'cpu_coordinator':
                # Local execution
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                return {
                    "success": process.returncode == 0,
                    "output": stdout.decode(),
                    "error": stderr.decode()
                }
            else:
                # Remote execution
                ip = self.machines[machine]
                ssh_cmd = f"ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no sanzad@{ip} '{command}'"
                
                process = await asyncio.create_subprocess_shell(
                    ssh_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                return {
                    "success": process.returncode == 0,
                    "output": stdout.decode(),
                    "error": stderr.decode()
                }
                
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### Performance Expectations

| Operation | Time | Concurrency | Success Rate |
|-----------|------|-------------|--------------|
| Full Cluster Start | ~45 seconds | 2-3 parallel ops | >95% |
| Full Cluster Stop | ~25 seconds | Sequential | >98% |  
| Health Check All | ~10 seconds | Parallel | >99% |
| Service Restart | ~20 seconds | Individual | >95% |
| Complete Test Suite | ~30 seconds | Parallel tests | >90% |

### Concurrent Operations

```python
class ConcurrentClusterManager:
    async def parallel_health_checks(self) -> Dict[str, Any]:
        """Run health checks on all services concurrently"""
        
        health_tasks = []
        
        # Define all health check endpoints
        health_endpoints = [
            ("jetson_ollama", "http://192.168.1.177:11434/api/tags"),
            ("embeddings", "http://192.168.1.178:8081/health"),
            ("tools", "http://192.168.1.190:8082/health"),
            ("monitoring", "http://192.168.1.191:8083/health"),
            ("haproxy", "http://192.168.1.81:8888/health")
        ]
        
        # Create concurrent health check tasks
        for service_name, url in health_endpoints:
            task = self.check_service_health_async(service_name, url)
            health_tasks.append(task)
        
        # Add Redis health check (different protocol)
        redis_task = self.check_redis_health_async()
        health_tasks.append(redis_task)
        
        # Execute all health checks concurrently
        results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        # Process results
        health_summary = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                service_name = health_endpoints[i][0] if i < len(health_endpoints) else "redis"
                health_summary[service_name] = {
                    "status": "error",
                    "error": str(result)
                }
            else:
                health_summary.update(result)
        
        return health_summary
    
    async def check_service_health_async(self, service_name: str, url: str) -> Dict[str, Any]:
        """Async health check for HTTP services"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        return {
                            service_name: {
                                "status": "healthy",
                                "response_time": response.headers.get("X-Response-Time", "unknown"),
                                "status_code": response.status
                            }
                        }
                    else:
                        return {
                            service_name: {
                                "status": "unhealthy",
                                "status_code": response.status
                            }
                        }
        except Exception as e:
            return {
                service_name: {
                    "status": "error",
                    "error": str(e)
                }
            }
    
    async def check_redis_health_async(self) -> Dict[str, Any]:
        """Async health check for Redis"""
        
        try:
            import redis.asyncio as redis
            
            r = redis.Redis(
                host='192.168.1.81', 
                port=6379, 
                password='langgraph_redis_pass',
                socket_timeout=5
            )
            
            await r.ping()
            info = await r.info()
            
            return {
                "redis": {
                    "status": "healthy",
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory_human", "unknown")
                }
            }
        except Exception as e:
            return {
                "redis": {
                    "status": "error",
                    "error": str(e)
                }
            }
```

## 🛠️ Service Management

### Convenient Startup Scripts

The orchestrator automatically creates convenient shell scripts:

```bash
# ~/start_cluster.sh
#!/bin/bash
cd ~/ai-infrastructure/langgraph-config
source ~/langgraph-env/bin/activate
python3 cluster_orchestrator.py start

# ~/stop_cluster.sh  
#!/bin/bash
cd ~/ai-infrastructure/langgraph-config
source ~/langgraph-env/bin/activate
python3 cluster_orchestrator.py stop

# ~/cluster_status.sh
#!/bin/bash
cd ~/ai-infrastructure/langgraph-config
source ~/langgraph-env/bin/activate
python3 cluster_orchestrator.py status
```

### Management Commands

```bash
# Cluster Lifecycle
~/start_cluster.sh              # Start entire cluster
~/stop_cluster.sh               # Stop entire cluster  
~/cluster_status.sh             # Check cluster status

# Advanced Operations
cd ~/ai-infrastructure/langgraph-config
source ~/langgraph-env/bin/activate

# Full restart with wait
python3 cluster_orchestrator.py restart

# Comprehensive testing
python3 cluster_orchestrator.py test

# Individual service management (via orchestrator)
python3 cluster_orchestrator.py start --service ollama --machine jetson
python3 cluster_orchestrator.py stop --service tools-server --machine worker_tools

# Status with details
python3 cluster_orchestrator.py status --verbose

# Health monitoring
python3 cluster_orchestrator.py monitor --interval 30
```

### Integration with System Services

```bash
# Create systemd service for orchestrator auto-start (optional)
sudo tee /etc/systemd/system/langgraph-cluster.service << EOF
[Unit]
Description=LangGraph Cluster Orchestrator Auto-Start
After=network.target

[Service]
Type=oneshot
User=sanzad
WorkingDirectory=/home/sanzad/ai-infrastructure/langgraph-config
Environment=PATH=/home/sanzad/langgraph-env/bin
ExecStart=/home/sanzad/langgraph-env/bin/python cluster_orchestrator.py start
ExecStop=/home/sanzad/langgraph-env/bin/python cluster_orchestrator.py stop
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# Enable auto-start (optional)
sudo systemctl enable langgraph-cluster

# Cron job for health monitoring
echo "*/10 * * * * cd ~/ai-infrastructure/langgraph-config && source ~/langgraph-env/bin/activate && python3 cluster_orchestrator.py test >> ~/cluster_health.log 2>&1" | crontab -e
```

## 🔧 Troubleshooting

### Common Issues

#### 1. SSH Connection Problems

```bash
# Check SSH connectivity to all nodes
for ip in 192.168.1.177 192.168.1.178 192.168.1.190 192.168.1.191; do
    echo "Testing SSH to $ip..."
    ssh -o ConnectTimeout=5 sanzad@$ip "echo 'SSH OK'" || echo "SSH FAILED to $ip"
done

# Fix SSH key issues
ssh-copy-id sanzad@192.168.1.177  # Jetson
ssh-copy-id sanzad@192.168.1.178  # rp-node
ssh-copy-id sanzad@192.168.1.190  # worker-node3
ssh-copy-id sanzad@192.168.1.191  # worker-node4

# Test SSH without password prompt
ssh -o BatchMode=yes sanzad@192.168.1.177 "echo 'Passwordless SSH working'"
```

#### 2. Service Startup Failures

```bash
# Debug service startup issues
python3 cluster_orchestrator.py start --debug

# Check individual service logs
ssh sanzad@192.168.1.177 "sudo journalctl -u ollama -f"
ssh sanzad@192.168.1.178 "sudo journalctl -u embeddings-server -f"
ssh sanzad@192.168.1.190 "sudo journalctl -u tools-server -f"
ssh sanzad@192.168.1.191 "sudo journalctl -u monitoring-server -f"

# Local services
sudo journalctl -u redis-server -f
sudo journalctl -u haproxy -f

# Common fixes:
# - Ensure Python virtual environments are activated
# - Check service dependencies are installed
# - Verify network ports are available
# - Check disk space and memory
```

#### 3. Partial Cluster Failures

```bash
# Identify failed services
python3 cluster_orchestrator.py status | grep "🔴"

# Restart specific service
ssh sanzad@192.168.1.190 "sudo systemctl restart tools-server"

# Check service health
curl -v http://192.168.1.190:8082/health

# Force restart problematic services
python3 cluster_orchestrator.py stop
sleep 5
python3 cluster_orchestrator.py start

# Emergency cluster recovery
./emergency_cluster_recovery.sh
```

#### 4. Performance Issues

```bash
# Check system resources on all nodes
for ip in 192.168.1.177 192.168.1.81 192.168.1.178 192.168.1.190 192.168.1.191; do
    echo "=== Resources on $ip ==="
    ssh sanzad@$ip "free -h && df -h / && uptime"
done

# Monitor cluster startup time
time python3 cluster_orchestrator.py restart

# Check network latency between nodes
for ip in 192.168.1.177 192.168.1.178 192.168.1.190 192.168.1.191; do
    echo "Ping to $ip:"
    ping -c 3 $ip
done

# Optimize startup sequence
# Edit cluster_orchestrator.py to adjust wait times and dependencies
```

#### 5. Test Failures

```bash
# Debug test failures
python3 cluster_orchestrator.py test --verbose

# Test individual endpoints
curl -v http://192.168.1.177:11434/api/tags
curl -v http://192.168.1.178:8081/health
curl -v http://192.168.1.190:8082/health
curl -v http://192.168.1.191:8083/health
curl -v http://192.168.1.81:8888/health

# Test Redis separately
redis-cli -h 192.168.1.81 -p 6379 -a langgraph_redis_pass ping

# Check load balancer endpoints
curl -v http://192.168.1.81:9000/health  # LLM load balancer
curl -v http://192.168.1.81:9001/health  # Tools load balancer
curl -v http://192.168.1.81:9002/health  # Embeddings load balancer
```

## 🚀 Advanced Features

### Custom Orchestration Profiles

Create different orchestration profiles for different use cases:

```python
class ProfiledClusterOrchestrator:
    def __init__(self):
        self.profiles = {
            "development": {
                "services": ["redis-server", "ollama", "embeddings-server"],
                "skip_monitoring": True,
                "fast_startup": True
            },
            "production": {
                "services": "all",
                "enable_monitoring": True,
                "health_checks": True,
                "backup_services": True
            },
            "testing": {
                "services": ["redis-server", "ollama", "tools-server"],
                "mock_services": ["embeddings-server"],
                "reset_between_tests": True
            }
        }
    
    async def start_profile(self, profile_name: str):
        """Start cluster with specific profile"""
        
        profile = self.profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Unknown profile: {profile_name}")
        
        logger.info(f"Starting cluster with profile: {profile_name}")
        
        if profile.get("fast_startup"):
            await self.fast_startup_sequence(profile["services"])
        else:
            await self.standard_startup_sequence(profile["services"])
        
        if profile.get("enable_monitoring"):
            await self.enable_enhanced_monitoring()
        
        if profile.get("health_checks"):
            await self.run_post_startup_health_checks()
```

### Cluster State Management

Advanced state management and recovery:

```python
class ClusterStateManager:
    def __init__(self):
        self.state_file = "cluster_state.json"
        self.backup_interval = 300  # 5 minutes
        
    async def save_cluster_state(self):
        """Save current cluster state"""
        
        state = {
            "timestamp": datetime.now().isoformat(),
            "services": await self.get_all_service_states(),
            "health": await self.get_cluster_health(),
            "performance": await self.get_performance_metrics()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    async def restore_cluster_state(self, target_state: str = None):
        """Restore cluster to previous state"""
        
        if target_state and os.path.exists(target_state):
            state_file = target_state
        else:
            state_file = self.state_file
        
        if not os.path.exists(state_file):
            logger.error("No cluster state file found")
            return False
        
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        logger.info(f"Restoring cluster state from {state['timestamp']}")
        
        # Restore services to previous state
        for service_name, service_state in state["services"].items():
            if service_state["active"]:
                await self.ensure_service_running(service_name)
            else:
                await self.ensure_service_stopped(service_name)
        
        return True
```

### Automated Cluster Scaling

Automatic scaling based on load:

```python
class AutoScalingOrchestrator:
    def __init__(self):
        self.scaling_thresholds = {
            "cpu_high": 80,
            "memory_high": 85,
            "response_time_high": 5.0,
            "queue_length_high": 10
        }
        
        self.scaling_actions = {
            "add_worker_node": self.add_worker_node,
            "enable_cpu_fallback": self.enable_cpu_fallback,
            "restart_overloaded_service": self.restart_overloaded_service
        }
    
    async def monitor_and_scale(self):
        """Monitor cluster and auto-scale if needed"""
        
        while True:
            try:
                # Get current cluster metrics
                metrics = await self.get_cluster_metrics()
                
                # Evaluate scaling needs
                scaling_needed = self.evaluate_scaling_needs(metrics)
                
                if scaling_needed:
                    await self.execute_scaling_action(scaling_needed)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def add_worker_node(self, node_type: str):
        """Dynamically add worker node to cluster"""
        
        # This would integrate with cloud providers or container orchestration
        # For local setup, it could start additional VM instances
        
        logger.info(f"Scaling up: Adding {node_type} worker node")
        
        # Implementation depends on your infrastructure setup
        # Could use Docker, VMs, or cloud instances
```

## 🎯 Why This Architecture Works

### Benefits of Centralized Orchestration

1. **Simplified Operations**: Single command to manage entire distributed cluster
2. **Dependency Management**: Ensures services start in correct order
3. **Error Handling**: Centralized error handling and recovery procedures
4. **Operational Consistency**: Same management interface across all environments
5. **Monitoring Integration**: Built-in health checking and status reporting
6. **Development Productivity**: Fast iteration cycles with automated cluster management

### Integration with LangGraph Ecosystem

The cluster orchestrator enables sophisticated operational patterns:

- **Workflow-Aware Management**: Coordinate cluster state with LangGraph workflows
- **Health-Driven Operations**: Auto-recovery based on service health
- **Performance Optimization**: Dynamic resource allocation based on load
- **Development Acceleration**: Rapid cluster provisioning for testing
- **Production Reliability**: Robust startup/shutdown procedures
- **Disaster Recovery**: Automated backup and restore capabilities

### Orchestration Enhancement Patterns

1. **Lifecycle Management**: Complete control over distributed service lifecycle
2. **Dependency Coordination**: Proper service startup and shutdown ordering
3. **Health Orchestration**: Integrated health monitoring and auto-recovery
4. **Performance Management**: Load-based scaling and optimization
5. **Development Integration**: Seamless development and testing workflows
6. **Operational Excellence**: Production-grade cluster management

## 🏆 Conclusion

The Cluster Orchestrator is not just "another management script" - it's the **mission control center** for your distributed LangGraph infrastructure. It transforms complex multi-machine operations into simple, reliable, and repeatable procedures.

**Key Takeaways**:

- Runs on the central coordinator node (cpu-node) for maximum reliability
- Manages 6 services across 5 machines with proper dependency ordering
- Provides comprehensive health monitoring and testing capabilities
- Integrates seamlessly with LangGraph workflows for intelligent operations
- Enables development productivity through automated cluster management
- Scales from simple development setups to production-grade operations

Without the cluster orchestrator, managing your distributed LangGraph cluster would require complex manual procedures across multiple machines. With it, you have simple, reliable, one-command control over your entire AI infrastructure - just like having a skilled DevOps engineer managing your cluster 24/7! 🎛️✨
