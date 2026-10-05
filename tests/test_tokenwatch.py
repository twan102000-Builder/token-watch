from tokenwatch.decomposer import PromptSlotDecomposer
from tokenwatch.advisor import TokenAdvisorEngine
from tokenwatch.models import TelemetryEvent, PromptBreakdown

def test_decomposer():
    decomposer = PromptSlotDecomposer()
    payload = {
        "messages": [
            {"role": "system", "content": "You are a smart home assistant."},
            {"role": "user", "content": "Sensor temperature reading: 72F in living room."},
            {"role": "user", "content": "Turn down the thermostat to 68F."}
        ],
        "tools": [
            {"type": "function", "function": {"name": "set_climate", "parameters": {}}}
        ]
    }
    breakdown = decomposer.decompose(payload)
    assert breakdown.system_instructions > 0
    assert breakdown.tool_schemas > 0
    assert breakdown.iot_telemetry > 0
    assert breakdown.task_payload > 0
    assert breakdown.total_prompt_tokens > 0

def test_advisor_tool_pruning():
    advisor = TokenAdvisorEngine()
    event = TelemetryEvent(
        trace_id="t1",
        node_id="n1",
        agent_id="hvac",
        model="gpt-4o",
        input_tokens=2500,
        output_tokens=50,
        prompt_breakdown=PromptBreakdown(
            system_instructions=200,
            tool_schemas=1800,
            conversation_history=100,
            iot_telemetry=300,
            task_payload=100,
            total_prompt_tokens=2500
        ),
        tools_declared=["tool1", "tool2", "tool3", "tool4", "tool5", "tool6"],
        tools_invoked=["tool1"]
    )
    advisor.record_event(event)
    advisories = advisor.generate_advisories("hvac")
    assert any(a.advisory_type == "TOOL_PRUNING" for a in advisories)
