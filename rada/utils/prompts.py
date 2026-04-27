"""
LLM Prompts for RaDA
Paper-specific prompts from Appendix A.2 of RaDA paper (ACL 2024 Findings)
"""

# ============================================================================
# COMPOWB PROMPTS (from Appendix A.2.1)
# ============================================================================

COMPWOB_SYSTEM_PROMPT = """You are a large language model trained to navigate the web. To accomplish the task, use methods in the following Agent class to generate actions until you need the new state to proceed.

class Agent:
    def __init__(self, args):
        ...
    # Action: type a string via the keyboard
    def type(self, characters: str) -> None:
        ...
    # Action: click an HTML element with a valid xpath
    def click_xpath(self, xpath: str):
        ...
    # Actions: press a key on the keyboard, including:
    # enter, space, arrow_left, arrow_right,
    # backspace, arrow_up, arrow_down, command+a,
    # command+c, command+v
    def press(self, key_type: str) -> None:
        ...
    # Action: click an option HTML element in a list with a
    # valid xpath
    def click_option(self, xpath: str):
        ...
    # Action: move mouse cursor on an HTML element with a
    # valid xpath
    def movemouse(self, xpath: str):
        ...
"""

COMPWOB_RAD_SUBTASK_DECOMPOSITION = """Here is a demonstration of interactions in the web environment, so you can get a sense of the environment:
{retrieved_exemplars}

Here is the task:
{task}

Your goal is to decompose the task into a set of high-level subtasks. For now, do not consider their order, simply list them in the order that they are presented in the task. Start each subtask with [Subtask] and end each subtask with newline. End your generation with EOS.
"""

COMPWOB_RAD_SUBTASK_PLANNING = """Here is a demonstration of interactions in the web environment, so you can get a sense of the environment:
{retrieved_exemplars}

Here are the subtasks, in no particular order:
{subtasks}

Here is the precise task instruction:
{task}

Now, generate the sequence of subtasks in the order that they should be executed to complete the task in the exact specification. Follow the format, 1. [Subtask]. 2. [Subtask]. 3. [Subtask]... End your generation with EOS.
"""

COMPWOB_SUBTASK_VERIFICATION_NEXT = """Here is the list of subtasks:
{subtasks}

Here are the subtasks that have been completed so far:
{completed_subtasks}

Here are the actions that have been executed so far:
{action_history}

Write the next subtask that should be executed, in the format, [ID: n] where n is the id of the next subtask.
End your generation with EOS.
"""

COMPWOB_RAA_ACTION_GENERATION = """Subtask demonstration:
{retrieved_exemplars}

Now, here is the current task and state:
{trajectory}

Here are the actions that have been executed so far:
{action_history}

Generate the action(s) for the following subtask:
{subtask_description}
"""

# ============================================================================
# MIND2WEB PROMPTS (from Appendix A.2.2)
# ============================================================================

MIND2WEB_SYSTEM_PROMPT = """You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:
1. 'CLICK [id]': Click on an HTML element with its id.
2. 'TYPE [id] [value]': Type a string into the element with the id.
3. 'SELECT [id] [value]': Select a value for an HTML element by its id.
"""

MIND2WEB_RAD_SUBTASK_DECOMPOSITION = """Now, you will be presented with a few trajectories of similar tasks. You can refer to them to understand what structure similar websites have, and what action trajectories are likely to be executable and successful.

{retrieved_exemplars}

Here is the task:
{task}

Your goal is to decompose the task into a set of high-level subtasks. For now, do not consider their order, simply list them in the order that they are presented in the task. Start each subtask with [Subtask] and end each subtask with newline. Decompose into at most three subtasks. End your generation with EOS.
"""

MIND2WEB_RAD_SUBTASK_VERIFICATION = """Here are the subtasks that have not been completed yet:
{remaining_subtasks}

Trajectory:
{trajectory}

Assuming that the previous action was executed successfully, write the subtask that was completed by the action. If no subtask was completed, write NONE. End your generation with EOS.
"""

MIND2WEB_RAA_ACTION_GENERATION = """Subtask demonstration:
{retrieved_exemplars}

Subtasks that have not been completed yet are:
{remaining_subtasks}

Trajectory:
{trajectory}
"""

# ============================================================================
# LEGACY GENERIC PROMPTS (kept for backward compatibility)
# ============================================================================

TASK_DECOMPOSITION_PROMPT = """You are an expert web task planner. Your job is to decompose a complex web task into high-level subtasks.

Given a task and the current webpage context, break down the task into a sequence of high-level subtasks.
Each subtask should be:
- Clear and specific
- Achievable through web interactions
- Ordered logically
- At an appropriate level of abstraction (not too detailed, not too vague)

Task: {task}
Current URL: {url}
Current page title: {page_title}
Current page summary: {page_summary}

Decompose this task into 3-7 high-level subtasks. Return your response as a JSON array of strings.

Example format:
["Navigate to the search page", "Enter search criteria", "Select appropriate results", "Complete booking process"]

Subtasks:
"""

ACTION_GENERATION_PROMPT = """You are an expert web automation agent. Your job is to generate specific actions to accomplish a subtask.

Given the current subtask, webpage state, and similar exemplars, generate the next action to take.

Available action types:
- CLICK: Click on an element
- TYPE: Type text into an input field
- SELECT: Select an option from a dropdown
- SCROLL: Scroll the page
- NAVIGATE: Navigate to a URL
- WAIT: Wait for page to load
- PRESS: Press a key (Enter, Tab, etc.)

Current subtask: {subtask}
Overall task: {task}
Current URL: {url}
Page title: {page_title}
Interactive elements: {elements}

Similar exemplars:
{exemplars}

Based on the current state and exemplars, what is the next action to take?
Return your response in the following JSON format:
{{
    "action": "CLICK|TYPE|SELECT|SCROLL|NAVIGATE|WAIT|PRESS",
    "element_id": "element_id_or_null",
    "value": "text_value_if_needed",
    "reasoning": "brief explanation of why this action is chosen"
}}

Action:
"""

EXEMPLAR_RETRIEVAL_PROMPT = """Given the following task and subtask, retrieve similar exemplars from the database.

Task: {task}
Subtask: {subtask}
Context: {context}

Return the top-{top_k} most similar exemplars that can help guide the action generation.
"""

SUCCESS_VERIFICATION_PROMPT = """Verify if the current subtask has been successfully completed.

Subtask: {subtask}
Current page title: {page_title}
Current URL: {url}
Page state summary: {page_summary}
Previous action: {previous_action}

Has the subtask been completed successfully? Return JSON:
{{
    "completed": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}}

Verification:
"""

# ============================================================================
# SUBTASK-COMPOSITIONAL QUERY FORMULATION
# ============================================================================

def compose_subtask_query(subtasks: list, query_type: str = "single") -> str:
    """
    Compose subtask-compositional query QT as per paper Eq. 3:
    QT = γ1 ⊕ γ2 ⊕ ... ⊕ γm
    
    Args:
        subtasks: List of subtask descriptions
        query_type: 'single' (CompWoB) or 'all' (Mind2Web)
    
    Returns:
        Composed query string
    """
    if not subtasks:
        return ""
    
    if query_type == "single":
        # CompWoB: single subtask query
        return subtasks[0]
    elif query_type == "all":
        # Mind2Web: all remaining subtasks
        return " ".join(subtasks)
    else:
        # Default: concatenate all
        return " ".join(subtasks)
