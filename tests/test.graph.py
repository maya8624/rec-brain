import asyncio
from app.agents.graph import build_graph
from app.agents.state import initial_state
from langchain_core.messages import HumanMessage


async def test():
    agent = build_graph()
    state = initial_state()
    state["messages"] = [HumanMessage(
        content="Find me 3 bedroom houses in Castle Hill")]

    result = await agent.ainvoke(
        state,
        config={"configurable": {"thread_id": "test-123"}}
    )

    for msg in result["messages"]:
        print(type(msg).__name__, ":", msg.content)

asyncio.run(test())
