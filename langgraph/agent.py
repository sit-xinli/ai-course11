import os

from collections.abc import AsyncIterable
from typing import Any, Literal

import httpx

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


memory = MemorySaver()


@tool
def get_exchange_rate(
    currency_from: str = 'USD',
    currency_to: str = 'EUR',
    currency_date: str = 'latest',
):
    """現在の為替レートを取得するために使用します。

    引数
        currency_from: 変換元の通貨 (例 "USD")。
        currency_to： 変換先の通貨 (例 "EUR")。
        currency_date: 為替レートの日付あるいは「最新」。 デフォルトは
            "最新"。

    戻り値：
        為替レートデータを含む辞書。
        を返します。
    """
    try:
        response = httpx.get(
            f'https://api.frankfurter.app/{currency_date}',
            params={'from': currency_from, 'to': currency_to},
        )
        response.raise_for_status()

        data = response.json()
        if 'rates' not in data:
            return {'error': '無効なAPI応答フォーマット.'}
        return data
    except httpx.HTTPError as e:
        return {'error': f'APIリクエストの失敗: {e}'}
    except ValueError:
        return {'error': 'API からの JSON 応答が無効です。'}


class ResponseFormat(BaseModel):
    """このフォーマットでユーザーに返答する。"""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class CurrencyAgent:
    """CurrencyAgent - 通貨取引に特化したアシスタントツール."""

    SYSTEM_INSTRUCTION = (
        'あなたは通貨変換の専門アシスタントですね。 '
        "あなたの唯一の目的は、為替レートに関する質問に答えるために'get_exchange_rate'ツールを使用することです。 "
        'ユーザーが通貨換算や為替レート以外のことを尋ねてきたら、'
        'そのトピックについては手助けできず、通貨関連の問い合わせにのみ手助けできることを丁寧に述べてください。 '
        '関係のない質問に答えようとしたり、他の目的のためにツールを使用したりしないでください。'
    )

    FORMAT_INSTRUCTION = (
        'リクエストを完了するためにユーザーがさらに情報を提供する必要がある場合は、レスポンスステータスを input_required に設定する'
        'リクエストの処理中にエラーが発生した場合は、レスポンスステータスを error に設定する'
        'リクエストが完了した場合は、応答ステータスを completed にする。'
    )

    def __init__(self):
        model_source = os.getenv('model_source', 'google')
        if model_source == 'google':
            self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        else:
            self.model = ChatOpenAI(
                model=os.getenv('TOOL_LLM_NAME'),
                openai_api_key=os.getenv('API_KEY', 'EMPTY'),
                openai_api_base=os.getenv('TOOL_LLM_URL'),
                temperature=0,
            )
        self.tools = [get_exchange_rate]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': '為替レートを調べてみる...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': '為替レートの処理中...',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                '現在、あなたのリクエストを処理することができません。 '
                'もう一度お試しください.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
