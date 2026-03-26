from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import FINISH_TOOL_NAME

# _FINISH_DESCRIPTION = """Signals the completion of the current task or conversation.

# Use this tool when:
# - You have successfully completed the user's requested task
# - You cannot proceed further due to technical limitations or missing information

# Always include:
# - `message`: human-readable summary for the parent agent
# - `needs_clarification`: whether the parent must ask the user for clarification
# Optional:
# - `reasons`: short justification for the verdict
# """

# FinishTool = ChatCompletionToolParam(
#     type='function',
#     function=ChatCompletionToolParamFunctionChunk(
#         name=FINISH_TOOL_NAME,
#         description=_FINISH_DESCRIPTION,
#         parameters={
#             'type': 'object',
#             'required': ['message', 'needs_clarification'],
#             'properties': {
#                 'message': {
#                     'type': 'string',
#                     'description': 'Final message to send to the user',
#                 },
#                 'needs_clarification': {
#                     'type': 'boolean',
#                     'description': 'Whether the parent must ask the user for clarification',
#                 },
#                 'reasons': {
#                     'type': 'string',
#                     'description': 'Short justification for the verdict',
#                 },
#             },
#         'additionalProperties': False,
#         },
#     ),
# )

_FINISH_DESCRIPTION = """Signals the completion of the current task or conversation.

Use this tool when:
- You have successfully completed the user's requested task

Always include:
- `needs_clarification`: whether the parent must ask the user for clarification
Optional:
- `reasons`: short justification for the verdict
"""

FinishTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=FINISH_TOOL_NAME,
        description=_FINISH_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['needs_clarification'],
            'properties': {
                'message': {
                    'type': 'string',
                    'description': 'Final message to send to the user',
                },
                'needs_clarification': {
                    'type': 'boolean',
                    'description': 'Whether the parent must ask the user for clarification',
                },
                'reasons': {
                    'type': 'string',
                    'description': 'Short justification for the verdict',
                },
            },
        'additionalProperties': False,
        },
    ),
)
