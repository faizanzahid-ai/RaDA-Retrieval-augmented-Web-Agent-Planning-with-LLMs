# RaDA: Retrieval-augmented Web Agent Planning with LLMs

Implementation of the RaDA paper from ACL 2024 Findings.

## Overview

RaDA (Retrieval-augmented web Driver Agent) is a novel planning method for web agents that:
- Does not require manual exemplars
- Efficiently leverages LLM context
- Enhances generalization through retrieval-augmented planning

## Architecture

RaDA disentangles planning into two stages:

1. **RaD (Retrieval-augmented Task Decomposition)**: Decomposes tasks into high-level subtasks
2. **RaA (Retrieval-augmented Action Generation)**: Traverses the trajectory to iteratively synthesize actions based on dynamically retrieved exemplars

## Novel Contributions

This implementation introduces three major upgrades over the original RaDA framework:

### 1. Graph-Based Task Decomposition

**Problem:** RaDA uses linear subtasks, limiting planning flexibility.

**Novelty:** Represent tasks as a directed graph where:
- **Nodes** = subtasks
- **Edges** = dependencies between subtasks

**Benefits:**
- Parallel execution of independent subtasks
- Better planning through explicit dependency modeling
- More robust handling of complex, multi-step workflows

### 2. Reinforcement Learning + RaDA

**Problem:** RaDA has no learning loop; policies remain static.

**Novelty:** Integrate Reinforcement Learning into the planning pipeline:
- **Reward function:** Task success + efficiency + fewer steps
- **Trainable policies:**
  - Retrieval policy (what to retrieve and when)
  - Decomposition policy (how to break down tasks)

**Result:** The agent becomes an optimal planner rather than a heuristic-driven system.

### 3. Lifelong Retrieval-Augmented Planning (Adaptive Memory)

**Problem:** RaDA uses fixed trajectory memory with no learning or improvement over time.

**Novelty:** Build an Adaptive Memory system that evolves after each task:
- **After successful trajectories:** Compress → Store → Rank → Re-weight
- **After failures:** Penalize bad exemplars

**Memory Scoring Formula:**
```
Memory Score = Success Rate × Recency × Diversity
```

**Why this is unique:**
- Current RaDA = static memory
- This model = continually learning agent

**Outcome:** A true "Lifelong Retrieval-Augmented Planning" agent that improves with every task.


## Installation

```bash
pip install -r requirements.txt
playwright install
```

## Usage

```python
from rada.agent import RadaAgent

# Initialize the agent
agent = RadaAgent(
    llm_model="gpt-4",
    embedding_model="all-MiniLM-L6-v2"
)

# Run a task
result = await agent.execute_task(
    task="Book a flight from New York to Los Angeles on December 15th",
    url="https://example-travel-site.com"
)
```

## Project Structure

```
rada/
├── core/
│   ├── __init__.py
│   ├── rad.py          # Task Decomposition
│   ├── raa.py          # Action Generation
│   └── agent.py        # Main agent orchestrator
├── retrieval/
│   ├── __init__.py
│   ├── store.py        # Vector store
│   └── embedder.py     # Embedding model
├── environment/
│   ├── __init__.py
│   ├── browser.py      # Browser automation
│   └── dom_parser.py   # DOM processing
├── exemplars/          # Stored exemplars
├── evaluation/
│   ├── __init__.py
│   └── metrics.py      # Evaluation metrics
└── utils/
    ├── __init__.py
    └── prompts.py      # LLM prompts
```

## Benchmarks

The implementation supports:
- CompWoB (Complex World of Bits)
- Mind2Web

# Setup Guide for RaDA

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
- (Optional) GPU for faster embedding computation

## Installation Steps

### 1. Create Virtual Environment

```bash
cd rada
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Playwright Browsers

```bash
playwright install
```

### 4. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-api-key-here
```

### 5. Verify Installation

```bash
python -c "from rada.core.agent import RadaAgent; print('Installation successful!')"
```

## Quick Start

### Basic Usage

```python
import asyncio
from rada.core.agent import RadaAgent

async def main():
    # Initialize the agent
    agent = RadaAgent(
        llm_model="gpt-4",
        api_key="your-api-key",
        headless=True
    )
    
    # Execute a task
    result = await agent.execute_task(
        task="Search for Python tutorials",
        start_url="https://www.google.com"
    )
    
    print(result)

asyncio.run(main())
```

### Running Examples

```bash
python examples/basic_usage.py
```

## Project Structure

```
rada/
├── core/                      # Core RaDA components
│   ├── rad.py                # Retrieval-augmented Task Decomposition
│   ├── raa.py                # Retrieval-augmented Action Generation
│   └── agent.py              # Main agent orchestrator
├── retrieval/                 # Exemplar retrieval system
│   ├── embedder.py           # Embedding model wrapper
│   └── store.py              # Vector store for exemplars
├── environment/               # Web interaction environment
│   ├── browser.py            # Playwright browser automation
│   └── dom_parser.py         # HTML/DOM parsing utilities
├── evaluation/                # Evaluation framework
│   └── metrics.py            # Performance metrics
├── utils/                     # Utilities
│   └── prompts.py            # LLM prompt templates
├── exemplars/                 # Stored exemplars (auto-created)
└── examples/                  # Usage examples
    └── basic_usage.py
```

## Configuration

All configuration is managed through environment variables in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LLM_MODEL`: LLM model to use (default: gpt-4)
- `EMBEDDING_MODEL`: Embedding model (default: all-MiniLM-L6-v2)
- `HEADLESS`: Browser headless mode (default: false)
- `MAX_ACTIONS_PER_SUBTASK`: Max actions per subtask (default: 10)
- `TOP_K_EXEMPLARS`: Number of exemplars to retrieve (default: 3)

## How RaDA Works

### Two-Stage Planning Process

1. **RaD (Retrieval-augmented Task Decomposition)**
   - Takes a complex task as input
   - Uses LLM to decompose it into high-level subtasks
   - Can use retrieved similar tasks for better decomposition

2. **RaA (Retrieval-augmented Action Generation)**
   - Takes each subtask and current page state
   - Retrieves similar exemplars from the exemplar store
   - Generates specific actions (CLICK, TYPE, SELECT, etc.)
   - Verifies subtask completion after each action sequence
   - Saves successful trajectories as new exemplars

### Exemplar Learning

- Successfully completed subtasks are stored as exemplars
- Exemplars are embedded and stored in a vector database
- Similar exemplars are retrieved for new tasks
- This enables continuous learning and improvement

## Supported Actions

- `CLICK`: Click on an element
- `TYPE`: Type text into an input field
- `SELECT`: Select a dropdown option
- `SCROLL`: Scroll the page
- `NAVIGATE`: Navigate to a URL
- `WAIT`: Wait for page loading
- `PRESS`: Press a keyboard key

## Evaluation Metrics

- **Task Success Rate**: Percentage of tasks completed successfully
- **Action Efficiency**: Average number of actions per task
- **Subtask Completion Rate**: Percentage of subtasks completed

## Troubleshooting

### Common Issues

1. **OpenAI API Error**
   - Ensure your API key is correctly set in `.env`
   - Check your API quota and billing

2. **Playwright Browser Error**
   - Run `playwright install` to install browsers
   - Check firewall/antivirus settings

3. **Import Errors**
   - Ensure you're in the correct directory
   - Verify virtual environment is activated
   - Run `pip install -r requirements.txt` again

4. **Slow Performance**
   - Use `HEADLESS=true` for faster execution
   - Use GPU for embeddings if available
   - Consider using `gpt-3.5-turbo` instead of `gpt-4`

## Advanced Usage

### Adding Custom Exemplars

```python
await agent.add_exemplar_manually(
    task="Search for information",
    subtask="Navigate to search engine",
    actions=[...],
    outcome="Success",
    context="Search task"
)
```

### Using Different LLM Models

```python
agent = RadaAgent(
    llm_model="gpt-3.5-turbo",  # Faster and cheaper
    # or
    llm_model="gpt-4",          # More capable
)
```

### Custom Embedding Models

```python
agent = RadaAgent(
    embedding_model="all-mpnet-base-v2",  # More accurate
    # or
    embedding_model="all-MiniLM-L6-v2",   # Faster
)
```

## 🎯 Key Features

### 1. Real Browser Automation
- Full Playwright integration
- Headless/visible mode
- Real DOM interaction
- Screenshot support

### 2. Advanced DOM Parsing
- XPath generation for every element
- Unique element IDs
- CSS selector generation
- Accessibility tree extraction
- Action type inference

### 3. Action Grounding
- CLICK: Precise element targeting
- TYPE: Text input with element ID
- SELECT: Dropdown selection
- PRESS: Keyboard actions
- SCROLL: Page navigation

### 4. Retrieval-Based Planning
- Exemplar storage & retrieval
- Vector similarity search
- Continuous learning from successes
- Context-aware action generation

### 5. Paper-Level Evaluation
- All metrics from RaDA paper
- Mind2Web benchmark (10 tasks, 10 domains)
- CompWoB benchmark (3 complex tasks)
- Detailed metric reports

## 🎯 What's Implemented

This implementation of the RaDA paper (ACL 2024) now includes:

✅ **Groq API Integration** - 10x faster than OpenAI  
✅ **Mind2Web Dataset** - 10 tasks across multiple domains  
✅ **CompWoB Dataset** - 3 complex web tasks  
✅ **Comprehensive Metrics** - Detailed evaluation reports  
✅ **Beautiful Console Output** - Rich formatted tables  

## 📊 Metrics Printed

The evaluation script prints the following metrics:

### 1. **Overall Performance Metrics**
- Total Tasks Evaluated
- Successful Tasks Count
- **Success Rate** (primary metric from paper)
- Average Execution Time
- Average Actions per Task
- Average Subtask Match Rate

### 2. **Performance by Difficulty**
- Easy Tasks: Success rate, time, actions
- Medium Tasks: Success rate, time, actions
- Hard Tasks: Success rate, time, actions

### 3. **Performance by Domain**
- E-commerce
- Travel
- Food Delivery
- Job Search
- Entertainment
- Education
- News
- Finance
- Real Estate
- Social Media

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd rada
pip install -r requirements.txt
playwright install
```

### Step 2: Get Groq API Key (FREE)

1. Visit https://console.groq.com
2. Sign up (no credit card needed)
3. Create API key
4. You get **10,000 requests/day FREE**

### Step 3: Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Groq API key
nano .env  # or use any text editor
```

Your `.env` should look like:
```env
GROQ_API_KEY=gsk_your_actual_key_here
LLM_MODEL=llama-3.3-70b-versatile
LLM_PROVIDER=groq
```

### Step 4: Test Setup

```bash
python test_setup.py
```

This will verify:
- ✅ Groq API connection
- ✅ RaD (Task Decomposition) working
- ✅ RaA (Action Generation) working
- ✅ Datasets loaded correctly

### Step 5: Run Evaluation

```bash
python evaluate_with_dataset.py
```

## 📈 Sample Output

When you run the evaluation, you'll see:

```
================================================================================
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃              MIND2WEB EVALUATION REPORT                             ┃
┃ Model: llama-3.3-70b-versatile                                     ┃
┃ Timestamp: 2024-04-24T10:30:00                                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

                 Overall Performance Metrics                 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric                    ┃ Value       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Total Tasks               │ 10          │
│ Successful Tasks          │ 8           │
│ Success Rate              │ 80.00%      │
│ Avg Execution Time        │ 2.45s       │
│ Avg Actions per Task      │ 8.50        │
│ Avg Subtask Match Rate    │ 75.00%      │
└───────────────────────────┴─────────────┘

              Performance by Difficulty              
┏━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Difficulty ┃ Total ┃ Successful ┃ Success Rate ┃ Avg Time(s) ┃ Avg Actions ┃
┡━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ EASY       │ 4     │ 4          │ 100.00%      │ 1.80         │ 6.00        │
│ MEDIUM     │ 4     │ 3          │ 75.00%       │ 2.50         │ 9.00        │
│ HARD       │ 2     │ 1          │ 50.00%       │ 3.50         │ 12.00       │
└────────────┴───────┴────────────┴──────────────┴──────────────┴─────────────┘

              Performance by Domain              
┏━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Domain          ┃ Total ┃ Successful ┃ Success Rate ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ E-commerce      │ 1     │ 1          │ 100.00%      │
│ Travel          │ 1     │ 1          │ 100.00%      │
│ Food Delivery   │ 1     │ 0          │ 0.00%        │
│ Job Search      │ 1     │ 1          │ 100.00%      │
│ Entertainment   │ 1     │ 1          │ 100.00%      │
│ Education       │ 1     │ 1          │ 100.00%      │
│ News            │ 1     │ 1          │ 100.00%      │
│ Finance         │ 1     │ 1          │ 100.00%      │
│ Real Estate     │ 1     │ 1          │ 100.00%      │
│ Social Media    │ 1     │ 0          │ 0.00%        │
└─────────────────┴───────┴────────────┴──────────────┘
================================================================================
```
## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{kim-etal-2024-rada,
    title = "{R}a{DA}: Retrieval-augmented Web Agent Planning with {LLM}s",
    author = "Kim, Minsoo and Bursztyn, Victor and Koh, Eunyee and Guo, Shunan and Hwang, Seung-won",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    year = "2024",
    pages = "13511--13525"
}
```
