from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import CLARIFY_DECISION_TOOL_NAME

_CLARIFY_DECISION_DESCRIPTION = """Decides whether the main agent must seek clarification from the user.

Use this tool when:
- You have reviewed the latest context and can advise if clarification is needed

Required field:
- `needs_clarification`: boolean flag indicating if the parent should ask the user for clarification

Optional fields:
- `message`: short summary or recommendation to send back
- `reasons`: brief justification for the decision
"""

ClarifyDecisionTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=CLARIFY_DECISION_TOOL_NAME,
        description=_CLARIFY_DECISION_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['needs_clarification'],
            'properties': {
                'needs_clarification': {
                    'type': 'boolean',
                    'description': 'Whether the parent must ask the user for clarification',
                },
                'message': {
                    'type': 'string',
                    'description': 'Optional short recommendation or summary for the parent agent',
                },
                'reasons': {
                    'type': 'string',
                    'description': 'Optional brief justification for the decision',
                },
            },
            'additionalProperties': False,
        },
    ),
)
