import ast
import operator
from google.adk.models.google_llm import Gemini
from google.genai import types
from datetime import datetime

from google.adk.agents.llm_agent import Agent
from google.adk.tools import ToolContext

from .rag import BASE_DIR, retrieve

NOTES_FILE = BASE_DIR / "notes.txt"   # save_note isi file me likhega
MAX_DISTANCE = 1.2                    # isse zyada distance = "notes me match nahi mila"



# Tool 1: calculate
                             
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval_node(node):
    """Expression tree ke ek node ko evaluate karta hai. Sirf numbers aur + - * / allowed."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("unsupported")


def calculate(expression: str) -> dict:
    """Evaluates a basic arithmetic expression and returns the numeric result.

    Use this for ANY math calculation instead of computing it yourself.
    Supported: numbers, + - * / and parentheses. Example: "12 * 4 + 7".

    Args:
        expression: The arithmetic expression to evaluate.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        return {"status": "success", "expression": expression, "result": result}
    except ZeroDivisionError:
        return {"status": "error", "error_message": "Division by zero is not possible."}
    except (SyntaxError, ValueError):
        return {
            "status": "error",
            "error_message": "Only numbers and + - * / ( ) are allowed in the expression.",
        }



# Tool 2: save_note

def save_note(topic: str, note: str) -> dict:
    """Saves a short note about a topic to the student's local notes file.

    Use this when the student asks you to remember or save something.

    Args:
        topic: Short topic name, e.g. "biology".
        note: The note text to save.
    """
    if not topic.strip() or not note.strip():
        return {"status": "error", "error_message": "Both topic and note are required."}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {topic.strip()}: {note.strip()}\n")
    return {"status": "success", "message": f"Note saved under topic '{topic.strip()}'."}



# Tool 3: track_quiz_score

def track_quiz_score(was_correct: bool, tool_context: ToolContext) -> dict:
    """Updates the running quiz score for this session and returns the current score.

    Call this exactly once after the student answers each quiz question.

    Args:
        was_correct: True if the student's last answer was correct, otherwise False.
    """
    correct = tool_context.state.get("quiz_correct", 0)
    total = tool_context.state.get("quiz_total", 0)
    total += 1
    if was_correct:
        correct += 1
    tool_context.state["quiz_correct"] = correct
    tool_context.state["quiz_total"] = total
    return {
        "status": "success",
        "correct": correct,
        "total": total,
        "score": f"{correct}/{total}",
    }



# Tool 4: search_study_material  (RAG ka retrieve stage)

def search_study_material(question: str) -> dict:
    """Searches the student's own study notes and returns the most relevant passages.

    Use this BEFORE answering any question that might be covered by the student's notes.

    Args:
        question: The student's question or topic to look up.
    """
    try:
        hits = retrieve(question, k=3)
    except Exception:
        return {
            "status": "error",
            "error_message": "Notes database not found. Run: python study_agent/ingest.py",
        }
    relevant = [h for h in hits if h["distance"] <= MAX_DISTANCE]
    if not relevant:
        return {
            "status": "no_match",
            "message": "No relevant passages were found in the student's notes.",
        }
    return {
        "status": "success",
        "passages": [{"source": h["source"], "text": h["text"]} for h in relevant],
    }



MODEL = Gemini(
    model="gemini-3.1-flash-lite",
    retry_options=types.HttpRetryOptions(
        attempts=6,              # maximum 6 koshish
        initial_delay=2.0,       # pehli baar 2 second ruko
        max_delay=30.0,          # ruk-ruk ke maximum 30 second tak
        exp_base=2.0,            # har baar wait double (2, 4, 8, 16, 30...)
        http_status_codes=[429, 500, 503, 504],   # sirf in errors par dobara try karo
    ),
)




# Agent: instructions + tools jodna (Part 4)

INSTRUCTION = """
You are StudyBuddy, a friendly and patient study assistant for a student.

Follow these rules:

1. MATH: For any calculation, always use the `calculate` tool. Never compute numbers yourself.

2. NOTES: Before answering any question that might be covered by the student's own study
   notes, first call `search_study_material`.
   - If it returns passages, base your answer on them and mention the source file,
     for example "According to your notes (water_cycle.txt) ...".
   - If it returns "no_match", say clearly that this was not found in the student's notes.
     You may then give a short general answer, but label it as general knowledge.
   - If it returns an error, tell the student the notes database is not ready and that
     they need to run the ingestion script.

3. SAVING: When the student asks you to remember or save something, call `save_note`
   with a short topic and the note text, then confirm that it was saved.

4. QUIZ: When the student asks to be quizzed:
   - Build questions from the student's notes (use `search_study_material`).
   - Ask exactly ONE question at a time and wait for the answer.
   - After the answer, say whether it was correct with a one-line explanation, then call
     `track_quiz_score` exactly once (was_correct=true or false) and share the running score.
   - Then offer the next question.

Style: keep answers short and simple. If the student writes in Hindi or Hinglish,
reply in the same language.
"""



def log_tool_call(tool, args, tool_context):
    """Har tool call se pehle terminal me print karta hai (debugging + report ke saboot ke liye)."""
    print(f"[TOOL CALL] {tool.name}({args})", flush=True)
    return None   # None = tool ko normal tarah chalne do


root_agent = Agent(
    model=MODEL,
    name="study_buddy",
    description=(
        "A study assistant that answers from the student's own notes, "
        "does math, saves notes and runs quizzes."
    ),
    instruction=INSTRUCTION,
    tools=[calculate, save_note, track_quiz_score, search_study_material],
    before_tool_callback=log_tool_call,
)