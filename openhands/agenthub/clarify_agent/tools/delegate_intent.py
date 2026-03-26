from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_DELEGATE_DESCRIPTION = """
Hand off the current user request to the specialized IntentAgent when you suspect missing, ambiguous, or risky requirements.

Provide any relevant context you have already gathered so the IntentAgent can craft targeted clarification questions.
"""


IntentAgentDelegateTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='delegate_to_intent_agent',
        description=_DELEGATE_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'context': {
                    'type': 'string',
                    'description': 'Optional notes or hypotheses to share with the IntentAgent.',
                },
            },
            'additionalProperties': False,
        },
    ),
)

