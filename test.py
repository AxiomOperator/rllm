#!/usr/bin/env python3
"""Test an OpenAI-compatible local LLM endpoint (llama.cpp, vLLM, Ollama)."""
import argparse
import json
import sys
import time

import requests


def api_request(session, method, url, **kwargs):
    response = session.request(method, url, timeout=kwargs.pop('timeout', 60), **kwargs)
    response.raise_for_status()
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://localhost:8080', help='Server base URL (default: %(default)s)')
    parser.add_argument('--model', help='Override automatically detected model ID')
    parser.add_argument('--prompt', default='Explain how a GPU processes LLM inference.', help='User prompt')
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
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': args.system},
            {'role': 'user', 'content': args.prompt},
        ],
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'stream': args.stream,
    }
    if args.stream:
        payload['stream_options'] = {'include_usage': True}

    start = time.perf_counter()
    first_token_time = None
    usage = None
    text_parts = []
    print('\n--- Model response ---')
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
                        content = delta.get('content')
                        if isinstance(content, str) and content:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            text_parts.append(content)
                            print(content, end='', flush=True)
                print()
            else:
                result = response.json()
                usage = result.get('usage')
                choices = result.get('choices', [])
                content = (choices[0].get('message', {}).get('content') if choices else '') or ''
                print(content)
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f'\nRequest failed: {exc}', file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - start
    print('\n--- Performance ---')
    print(f'Total request time: {elapsed:.3f} s')
    if first_token_time is not None:
        ttft = first_token_time - start
        print(f'Time to first content: {ttft:.3f} s (client-observed)')
    if usage:
        prompt_tokens = usage.get('prompt_tokens')
        completion_tokens = usage.get('completion_tokens')
        if prompt_tokens is not None:
            print(f'Prompt tokens: {prompt_tokens}')
        if completion_tokens is not None:
            print(f'Completion tokens: {completion_tokens}')
            generation_window = (time.perf_counter() - first_token_time) if first_token_time is not None else elapsed
            if generation_window > 0:
                print(f'Approx. throughput: {completion_tokens / generation_window:.2f} tokens/s')
    else:
        print('Token usage unavailable from server; throughput not calculated.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
