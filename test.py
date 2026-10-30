#!/usr/bin/env python3
"""Test an OpenAI-compatible local LLM endpoint (llama.cpp, vLLM, Ollama).

With --prompt, sends one message and exits. Without it, asks for messages
interactively and keeps the conversation history between turns.
"""
import argparse
import json
import sys
import time

import requests


def api_request(session, method, url, **kwargs):
    response = session.request(method, url, timeout=kwargs.pop('timeout', 60), **kwargs)
    response.raise_for_status()
    return response


def reasoning_text(message):
    """Reasoning text from a message or stream delta. Servers name the field differently."""
    for key in ('reasoning_content', 'reasoning'):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def chat(session, base, model, messages, args):
    """Send one chat request, print the reasoning, reply and timings.

    Returns the reply text without the reasoning, or None on failure.
    """
    payload = {
        'model': model,
        'messages': messages,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'stream': args.stream,
    }
    if args.stream:
        payload['stream_options'] = {'include_usage': True}

    start = time.perf_counter()
    first_token_time = None  # First generated text of either kind.
    first_content_time = None  # First answer text, after any reasoning.
    usage = None
    text_parts = []
    section = None
    dim, reset = ('\033[2m', '\033[0m') if sys.stdout.isatty() else ('', '')

    def show(kind, text):
        """Print a piece of reasoning or answer text under its section header."""
        nonlocal section, first_token_time, first_content_time
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now
        if kind == 'content' and first_content_time is None:
            first_content_time = now
        if kind != section:
            if section is not None:
                print(reset)
            print('\n--- Reasoning ---' if kind == 'reasoning' else '\n--- Model response ---')
            section = kind
        print(f'{dim}{text}' if kind == 'reasoning' else text, end='', flush=True)

    try:
        with session.post(f'{base}/v1/chat/completions', json=payload, timeout=(10, args.timeout), stream=args.stream) as response:
            response.raise_for_status()
            if args.stream:
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line or not raw_line.startswith('data:'):
                        continue
                    data = raw_line[5:].strip()
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if chunk.get('usage'):
                        usage = chunk['usage']
                    for choice in chunk.get('choices', []):
                        delta = choice.get('delta') or {}
                        reasoning = reasoning_text(delta)
                        if reasoning:
                            show('reasoning', reasoning)
                        content = delta.get('content')
                        if isinstance(content, str) and content:
                            text_parts.append(content)
                            show('content', content)
            else:
                result = response.json()
                usage = result.get('usage')
                choices = result.get('choices', [])
                message = (choices[0].get('message') if choices else None) or {}
                reasoning = reasoning_text(message)
                if reasoning:
                    show('reasoning', reasoning)
                content = message.get('content') or ''
                text_parts.append(content)
                show('content', content)
            print(reset if section == 'reasoning' else '')
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f'\nRequest failed: {exc}', file=sys.stderr)
        return None
    end = time.perf_counter()

    elapsed = end - start
    print('\n--- Performance ---')
    print(f'Total request time: {elapsed:.3f} s')
    if first_token_time is not None and args.stream:
        print(f'Time to first token: {first_token_time - start:.3f} s (client-observed)')
        if first_content_time is not None and first_content_time != first_token_time:
            print(f'Time to first answer text: {first_content_time - start:.3f} s')
    if usage:
        prompt_tokens = usage.get('prompt_tokens')
        completion_tokens = usage.get('completion_tokens')
        if prompt_tokens is not None:
            print(f'Prompt tokens: {prompt_tokens}')
        if completion_tokens is not None:
            print(f'Completion tokens: {completion_tokens}')
            reasoning_tokens = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
            if reasoning_tokens:
                print(f'  of which reasoning: {reasoning_tokens}')
            generation_window = (end - first_token_time) if args.stream and first_token_time is not None else elapsed
            if generation_window > 0:
                print(f'Approx. throughput: {completion_tokens / generation_window:.2f} tokens/s')
    else:
        print('Token usage unavailable from server; throughput not calculated.')
    return ''.join(text_parts)


def read_message():
    """Read one user message from the terminal. Returns None at end of input."""
    try:
        return input('\nYou> ').strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', default='http://localhost:8080', help='Server base URL (default: %(default)s)')
    parser.add_argument('--model', help='Override automatically detected model ID')
    parser.add_argument('--prompt', help='Send this message and exit instead of asking interactively')
    parser.add_argument('--system', default='You are a helpful AI assistant.', help='System prompt')
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--stream', action='store_true', help='Stream output token chunks')
    parser.add_argument('--api-key', default=None, help='Optional bearer token for authenticated servers')
    parser.add_argument('--timeout', type=float, default=120, help='Read timeout in seconds')
    args = parser.parse_args()

    base = args.url.rstrip('/')
    session = requests.Session()
    session.headers.update({'Content-Type': 'application/json'})
    if args.api_key:
        session.headers['Authorization'] = f'Bearer {args.api_key}'

    print(f'Server: {base}')
    try:
        health = api_request(session, 'GET', f'{base}/health', timeout=5)
        print(f'Health: HTTP {health.status_code}')
    except requests.RequestException as exc:
        print(f'Health endpoint unavailable (may not be supported): {exc}')

    model = args.model
    try:
        response = api_request(session, 'GET', f'{base}/v1/models', timeout=10)
        models = response.json().get('data', [])
        model_ids = [item['id'] for item in models if isinstance(item, dict) and 'id' in item]
        print('Available models:', ', '.join(model_ids) if model_ids else '(none listed)')
        if not model and model_ids:
            model = model_ids[0]
    except (requests.RequestException, ValueError) as exc:
        print(f'Model discovery unavailable: {exc}')

    if not model:
        print('Error: No model detected. Supply --model MODEL_ID.', file=sys.stderr)
        return 2

    print(f'Using model: {model}')
    print(f'Mode: {"streaming" if args.stream else "standard"}')
    system = [{'role': 'system', 'content': args.system}] if args.system else []

    if args.prompt is not None:
        reply = chat(session, base, model, system + [{'role': 'user', 'content': args.prompt}], args)
        return 0 if reply is not None else 1

    print('\nType a message and press Enter. /reset clears the conversation; /exit or Ctrl-D quits.')
    history = []
    while True:
        message = read_message()
        if message is None or message in ('/exit', '/quit'):
            return 0
        if not message:
            continue
        if message == '/reset':
            history = []
            print('Conversation cleared.')
            continue
        history.append({'role': 'user', 'content': message})
        reply = chat(session, base, model, system + history, args)
        if reply is None:
            history.pop()  # Let the user retry without a dangling unanswered message.
        else:
            history.append({'role': 'assistant', 'content': reply})


if __name__ == '__main__':
    sys.exit(main())
