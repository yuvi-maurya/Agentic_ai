from google.adk.agents.llm_agent import Agent

root_agent = Agent(
    model="gemini-flash-latest",
    name="study_buddy",
    description="A study assistant.",
    instruction="You are StudyBuddy, a friendly study assistant. Keep answers short.",
)
