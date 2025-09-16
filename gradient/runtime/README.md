# Gradient Runtime System

The Gradient Runtime System provides automatic instrumentation and tracking for AI agent execution across multiple frameworks like LangGraph, LangChain, and others.

## Overview

The runtime system automatically:

1. **Tracks Requests**: Creates a context for each entrypoint call
2. **Instruments Frameworks**: Automatically instruments framework code to track node executions
3. **Provides Observability**: Logs and tracks timing, inputs, outputs, and errors
4. **Extensible Architecture**: Clean interfaces for adding new framework instrumentors

## Architecture

```
gradient/runtime/
├── __init__.py                 # Public API exports
├── interfaces.py               # Abstract interfaces
├── context.py                  # Request context management
├── manager.py                  # Runtime coordination
├── tracker.py                  # Default execution tracker
└── langgraph_instrumentor.py   # LangGraph-specific instrumentation
```

### Core Components

#### 1. Request Context (`context.py`)
- Manages request lifecycle and context variables
- Associates all framework executions with a specific entrypoint call
- Tracks request timing, inputs, outputs, and errors

#### 2. Execution Tracker (`tracker.py`)
- Tracks individual node/step executions within frameworks
- Records timing, inputs, outputs, and error information
- Provides summary reporting

#### 3. Framework Instrumentors (`langgraph_instrumentor.py`)
- Framework-specific monkey-patching logic
- Automatically wraps framework methods to add tracking
- Clean install/uninstall for testing and debugging

#### 4. Runtime Manager (`manager.py`)
- Coordinates all runtime components
- Manages instrumentor lifecycle
- Provides global runtime control

## Usage

### Automatic Usage (Recommended)

Just use the `@entrypoint` decorator - runtime tracking is automatic:

```python
from gradient.sdk import entrypoint

@entrypoint
def my_agent(prompt: str) -> str:
    # Your LangGraph code here
    graph = create_my_langgraph()
    result = graph.invoke({"input": prompt})
    return result["output"]
```

### Manual Usage (Advanced)

For direct control over the runtime system:

```python
from gradient.runtime import get_runtime_manager, RequestContext

# Get runtime manager
runtime = get_runtime_manager()

# Start request tracking
runtime.start_request("my_function", inputs={"prompt": "test"})

# Your code with framework calls here
# (LangGraph nodes will be automatically tracked)

# End request tracking
runtime.end_request(outputs={"result": "success"})
```

## Extensibility

### Adding New Framework Instrumentors

1. **Implement the FrameworkInstrumentor interface**:

```python
from gradient.runtime.interfaces import FrameworkInstrumentor, ExecutionTracker

class MyFrameworkInstrumentor(FrameworkInstrumentor):
    @property
    def framework_name(self) -> str:
        return "myframework"
    
    def install(self, tracker: ExecutionTracker) -> None:
        # Monkey-patch your framework here
        pass
    
    def uninstall(self) -> None:
        # Restore original functions
        pass
    
    def is_installed(self) -> bool:
        return self._installed
```

2. **Register with the RuntimeManager**:

```python
# In manager.py _register_default_instrumentors()
self._instrumentors.append(MyFrameworkInstrumentor())
```

### Custom Execution Trackers

Implement the `ExecutionTracker` interface to customize how executions are tracked:

```python
from gradient.runtime.interfaces import ExecutionTracker, NodeExecution

class MyCustomTracker(ExecutionTracker):
    def start_node_execution(self, node_id, node_name, framework, inputs=None, metadata=None):
        # Custom tracking logic
        execution = NodeExecution(...)
        # Store in database, send to monitoring system, etc.
        return execution
    
    def end_node_execution(self, execution, outputs=None, error=None):
        # Custom completion logic
        pass
```

## Output Example

When running an agent with LangGraph nodes, you'll see output like:

```
[RUNTIME] Started request a1b2c3d4 for entrypoint: my_agent
[RUNTIME] Request a1b2c3d4 | Started langgraph node: data_loader (id: node_data_loader)
[RUNTIME] Request a1b2c3d4 | COMPLETED langgraph node: data_loader (45.2ms)
[RUNTIME] Request a1b2c3d4 | Started langgraph node: processor (id: node_processor)
[RUNTIME] Request a1b2c3d4 | COMPLETED langgraph node: processor (123.7ms)
[RUNTIME] Request a1b2c3d4 | Started langgraph node: generator (id: node_generator)
[RUNTIME] Request a1b2c3d4 | COMPLETED langgraph node: generator (89.1ms)
[RUNTIME] COMPLETED request a1b2c3d4 (258.3ms)

[RUNTIME] === Request Summary ===
[RUNTIME] Request ID: a1b2c3d4-5e6f-7890-abcd-ef1234567890
[RUNTIME] Entrypoint: my_agent
[RUNTIME] Status: completed
[RUNTIME] Duration: 258.3ms
[RUNTIME] Node Executions: 3
[RUNTIME] === Node Details ===
[RUNTIME]  1. [langgraph] data_loader - COMPLETED (45.2ms)
[RUNTIME]  2. [langgraph] processor - COMPLETED (123.7ms)
[RUNTIME]  3. [langgraph] generator - COMPLETED (89.1ms)
[RUNTIME] === End Summary ===
```

## Supported Frameworks

### LangGraph ✅
- Automatically instruments `CompiledGraph.invoke()` and `CompiledGraph.stream()`
- Tracks individual node executions within graphs
- Captures timing and basic input/output information

### Future Framework Support 🚧
- **LangChain**: Chain and tool execution tracking
- **AutoGen**: Multi-agent conversation tracking
- **Custom Frameworks**: Easy integration via instrumentor interface

## Configuration

The runtime system is designed to work automatically with minimal configuration. Advanced users can:

- Disable specific instrumentors
- Customize tracking granularity
- Integrate with external monitoring systems
- Control logging verbosity

## Testing

The runtime system includes comprehensive testing infrastructure:

```python
# Test with mock framework
from gradient.runtime.tracker import DefaultExecutionTracker

tracker = DefaultExecutionTracker()
execution = tracker.start_node_execution("test", "test_node", "test_framework")
tracker.end_node_execution(execution, outputs={"result": "success"})

print(f"Tracked {len(tracker.get_executions())} executions")
```