"""
agent.py — Agentic RAG with ReAct Loop
========================================
DLM Focus: Agentic reasoning goes beyond standard RAG by allowing the model to:
  1. Decide WHEN to retrieve (not always necessary)
  2. Decide WHAT to retrieve (adaptive query formulation)
  3. Use TOOLS (calculator, fact-checker, web search)
  4. SELF-CORRECT by checking its own reasoning
  5. Ask FOLLOW-UP retrieval questions mid-generation

ReAct Framework: Reason + Act
  Thought → Action → Observation → Thought → Action → ... → Final Answer

This transforms RAG from a pipeline into an intelligent reasoning system.
"""

from __future__ import annotations

import json
import re
import asyncio
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from retriever import HybridRetriever, Document
from generator import BaseLLM


class AgentAction(Enum):
    """Available agent actions."""
    RETRIEVE = "retrieve"
    CALCULATE = "calculate"
    SUMMARIZE = "summarize"
    FACT_CHECK = "fact_check"
    WEB_SEARCH = "web_search"
    FINAL_ANSWER = "final_answer"


@dataclass
class AgentStep:
    """One step in the agent's reasoning trace."""
    thought: str
    action: str
    action_input: str
    observation: str
    step_num: int
    
    def to_dict(self) -> Dict:
        return {
            "step": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "input": self.action_input,
            "observation": self.observation[:200] + "..." if len(self.observation) > 200 else self.observation,
        }


@dataclass
class AgentResult:
    """Complete agent execution result."""
    final_answer: str
    steps: List[AgentStep]
    total_steps: int
    retrieved_docs: List[Document]
    tools_used: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "answer": self.final_answer,
            "reasoning_steps": [s.to_dict() for s in self.steps],
            "total_steps": self.total_steps,
            "tools_used": list(set(self.tools_used)),
        }


# ─── Tool Definitions ─────────────────────────────────────────────────────────

class ToolRegistry:
    """Registry of tools available to the agent."""
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self._tools: Dict[str, Callable] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register built-in tools."""
        
        async def retrieve_tool(query: str) -> str:
            """Search the knowledge base for relevant information."""
            result = await self.retriever.retrieve(query, top_k=3)
            if not result.documents:
                return "No relevant documents found."
            
            context = "\n\n".join([
                f"[Source: {doc.source}]\n{doc.content[:500]}"
                for doc in result.documents
            ])
            return context
        
        async def calculate_tool(expression: str) -> str:
            """Evaluate a mathematical expression."""
            try:
                # Safe evaluation
                allowed_names = {
                    k: v for k, v in __builtins__.items() 
                    if k in ['abs', 'round', 'min', 'max', 'sum', 'len']
                } if isinstance(__builtins__, dict) else {}
                
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return f"Result: {result}"
            except Exception as e:
                return f"Calculation error: {e}"
        
        async def summarize_tool(text: str) -> str:
            """Summarize a long piece of text."""
            if len(text) < 500:
                return text
            # Simple extractive summary (first + last sentences)
            sentences = text.split('. ')
            if len(sentences) > 4:
                summary = '. '.join(sentences[:2] + ['...'] + sentences[-2:])
            else:
                summary = text[:500] + "..."
            return summary
        
        async def fact_check_tool(claim: str) -> str:
            """Check if a claim is supported by the knowledge base."""
            result = await self.retriever.retrieve(
                f"Evidence for or against: {claim}", top_k=3
            )
            if not result.documents:
                return f"No evidence found for claim: '{claim}'"
            
            evidence = result.documents[0].content[:400]
            return f"Evidence found: {evidence}"
        
        self._tools = {
            "retrieve": retrieve_tool,
            "calculate": calculate_tool,
            "summarize": summarize_tool,
            "fact_check": fact_check_tool,
        }
    
    def get_tool_descriptions(self) -> str:
        """Format tool descriptions for the LLM prompt."""
        descriptions = {
            "retrieve": "Search the knowledge base. Input: search query string. Use when you need specific information.",
            "calculate": "Evaluate mathematical expressions. Input: valid Python math expression like '2 + 2' or '(100 * 0.15)'.",
            "summarize": "Summarize text. Input: text to summarize.",
            "fact_check": "Verify a specific claim against the knowledge base. Input: specific claim to verify.",
            "final_answer": "Provide your final answer to the user. Input: your complete, well-formatted answer.",
        }
        
        lines = []
        for name, desc in descriptions.items():
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)
    
    async def execute(self, action: str, action_input: str) -> str:
        """Execute a tool and return the observation."""
        tool = self._tools.get(action)
        if tool is None:
            return f"Unknown tool: {action}. Available tools: {list(self._tools.keys())}"
        
        try:
            result = await tool(action_input)
            return str(result)
        except Exception as e:
            logger.error(f"Tool {action} failed: {e}")
            return f"Tool execution failed: {e}"


# ─── ReAct Agent ──────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are an intelligent research assistant using the ReAct framework (Reasoning + Acting).

You have access to these tools:
{tool_descriptions}

CRITICAL: You MUST follow this EXACT format for each step:
Thought: [Your reasoning about what to do next]
Action: [tool name - must be one of: retrieve, calculate, summarize, fact_check, final_answer]
Action Input: [the input to the tool]

After each action, you will receive an Observation. Use it to inform your next Thought.

When you have enough information, use Action: final_answer with your complete answer.

Rules:
1. Always think before acting
2. Use retrieve when you need specific information
3. Use fact_check to verify important claims
4. Use calculate for any numerical computations
5. Maximum {max_steps} steps total
6. Your final_answer must be comprehensive and cite sources
"""

REACT_HUMAN_PROMPT = """Question: {question}

History:
{history}

Continue reasoning:"""


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agentic RAG.
    
    Loop:
    1. LLM generates Thought + Action + Action Input
    2. Execute the action (tool call)
    3. Get Observation
    4. Feed back to LLM with full history
    5. Repeat until "final_answer" action or max steps
    
    Key advantage over vanilla RAG:
    - Multi-hop reasoning (retrieve → reason → retrieve again)
    - Dynamic tool selection based on question type
    - Can handle complex questions requiring calculation + retrieval
    - Self-correction via fact_check tool
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
    ):
        self.llm = llm
        self.tools = tool_registry
        self.max_steps = max_steps
    
    def _parse_llm_output(self, text: str) -> Dict[str, str]:
        """Parse Thought/Action/Action Input from LLM output."""
        result = {"thought": "", "action": "", "action_input": ""}
        
        # Extract Thought
        thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Extract Action
        action_match = re.search(r'Action:\s*(.*?)(?=Action Input:|$)', text, re.DOTALL | re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip().lower()
        
        # Extract Action Input
        input_match = re.search(r'Action Input:\s*(.*?)(?=Thought:|Observation:|$)', text, re.DOTALL | re.IGNORECASE)
        if input_match:
            result["action_input"] = input_match.group(1).strip()
        
        return result
    
    async def run(
        self,
        question: str,
        callback: Optional[Callable] = None,
    ) -> AgentResult:
        """
        Run the ReAct agent loop.
        
        Args:
            question: User's question
            callback: Optional async callback for streaming step updates
        """
        steps: List[AgentStep] = []
        history_parts: List[str] = []
        all_retrieved_docs: List[Document] = []
        tools_used: List[str] = []
        final_answer = ""
        
        # System prompt
        system = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=self.tools.get_tool_descriptions(),
            max_steps=self.max_steps,
        )
        
        for step_num in range(self.max_steps):
            # Build history string
            history = "\n".join(history_parts) if history_parts else "No steps taken yet."
            
            # Create prompt
            prompt = REACT_HUMAN_PROMPT.format(
                question=question,
                history=history,
            )
            
            # Get LLM response
            try:
                llm_output = await self.llm.generate(prompt, system=system)
            except Exception as e:
                logger.error(f"LLM generation failed at step {step_num}: {e}")
                break
            
            # Parse output
            parsed = self._parse_llm_output(llm_output)
            thought = parsed["thought"]
            action = parsed["action"]
            action_input = parsed["action_input"]
            
            # Default action if parsing fails
            if not action:
                logger.warning(f"Could not parse action at step {step_num}, defaulting to retrieve")
                action = "retrieve"
                action_input = question
            
            logger.debug(f"Step {step_num + 1}: {action}({action_input[:50]}...)")
            
            # Final answer
            if action == "final_answer":
                final_answer = action_input
                steps.append(AgentStep(
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation="[Final answer provided]",
                    step_num=step_num + 1,
                ))
                break
            
            # Execute tool
            tools_used.append(action)
            observation = await self.tools.execute(action, action_input)
            
            # Create step record
            step = AgentStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                step_num=step_num + 1,
            )
            steps.append(step)
            
            # Callback for streaming updates
            if callback:
                await callback(step)
            
            # Update history
            history_parts.append(
                f"Thought: {thought}\n"
                f"Action: {action}\n"
                f"Action Input: {action_input}\n"
                f"Observation: {observation[:500]}"
            )
        
        # If no final answer was produced, generate one
        if not final_answer:
            logger.warning("Max steps reached, generating final answer from history")
            summary_prompt = (
                f"Based on the following research, answer the question: {question}\n\n"
                f"Research:\n" + "\n".join(history_parts[-3:])
            )
            final_answer = await self.llm.generate(summary_prompt)
        
        return AgentResult(
            final_answer=final_answer,
            steps=steps,
            total_steps=len(steps),
            retrieved_docs=all_retrieved_docs,
            tools_used=tools_used,
        )
    
    async def run_stream(
        self, question: str
    ) -> AsyncGenerator[Dict, None]:
        """Stream agent steps as they happen."""
        
        async def step_callback(step: AgentStep):
            pass  # Will be handled by the generator
        
        steps = []
        
        async def collecting_callback(step: AgentStep):
            steps.append(step)
        
        # Start agent in background
        task = asyncio.create_task(self.run(question, callback=collecting_callback))
        
        # Stream steps as they're added
        last_sent = 0
        while not task.done():
            while last_sent < len(steps):
                yield {"type": "step", "data": steps[last_sent].to_dict()}
                last_sent += 1
            await asyncio.sleep(0.1)
        
        # Send remaining steps and final answer
        result = await task
        while last_sent < len(steps):
            yield {"type": "step", "data": steps[last_sent].to_dict()}
            last_sent += 1
        
        yield {"type": "final", "data": result.to_dict()}


# ─── Reflexion Agent (Advanced) ───────────────────────────────────────────────

class ReflexionAgent:
    """
    Reflexion: Agent learns from its mistakes through verbal reinforcement.
    
    Process:
    1. Attempt task with ReAct
    2. Evaluate the answer quality
    3. Generate reflection on what went wrong
    4. Retry with reflection in context
    
    This simulates how humans learn from mistakes — a true DLM concept.
    """
    
    def __init__(
        self,
        react_agent: ReActAgent,
        llm: BaseLLM,
        max_attempts: int = 3,
    ):
        self.agent = react_agent
        self.llm = llm
        self.max_attempts = max_attempts
    
    async def evaluate_answer(self, question: str, answer: str, context: str) -> Dict:
        """Score the quality of an answer."""
        eval_prompt = f"""Evaluate this answer for the given question.

Question: {question}
Answer: {answer}

Rate on these dimensions (0-10):
- Completeness: Does it fully answer the question?
- Accuracy: Is it factually accurate based on the context?
- Clarity: Is it clearly written?

Output JSON only:
{{"completeness": 8, "accuracy": 9, "clarity": 7, "overall": 8, "issues": "..."}}"""
        
        response = await self.llm.generate(eval_prompt)
        
        try:
            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        
        return {"overall": 7, "issues": "Could not evaluate"}
    
    async def generate_reflection(
        self, question: str, answer: str, evaluation: Dict
    ) -> str:
        """Generate a reflection on how to improve."""
        reflection_prompt = f"""You attempted to answer this question: {question}

Your answer: {answer}

Evaluation feedback: {evaluation.get('issues', 'No specific feedback')}
Score: {evaluation.get('overall', 'N/A')}/10

What specific strategies would improve your next attempt?
Be concrete and actionable. Output 2-3 bullet points."""
        
        return await self.llm.generate(reflection_prompt)
    
    async def run(self, question: str) -> AgentResult:
        """Run agent with reflexion (retry + learn from mistakes)."""
        reflections = []
        best_result = None
        best_score = 0
        
        for attempt in range(self.max_attempts):
            logger.info(f"Reflexion attempt {attempt + 1}/{self.max_attempts}")
            
            # Run agent with reflections in context
            if reflections:
                augmented_q = (
                    f"{question}\n\n"
                    f"[Previous attempt insights]:\n" + 
                    "\n".join(reflections[-2:])  # Last 2 reflections
                )
            else:
                augmented_q = question
            
            result = await self.agent.run(augmented_q)
            
            # Evaluate
            evaluation = await self.evaluate_answer(
                question, result.final_answer, ""
            )
            score = evaluation.get("overall", 0)
            
            if score > best_score:
                best_score = score
                best_result = result
            
            if score >= 8:  # Good enough
                break
            
            # Generate reflection for next attempt
            reflection = await self.generate_reflection(
                question, result.final_answer, evaluation
            )
            reflections.append(reflection)
            logger.info(f"Reflection: {reflection[:100]}...")
        
        return best_result or result


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_agent(
    config: dict,
    llm: BaseLLM,
    retriever: HybridRetriever,
) -> ReActAgent:
    """Create agent from config."""
    agent_config = config.get("agent", {})
    
    tool_registry = ToolRegistry(retriever)
    
    return ReActAgent(
        llm=llm,
        tool_registry=tool_registry,
        max_steps=agent_config.get("max_steps", 10),
    )
