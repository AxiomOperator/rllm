#!/usr/bin/env python3
"""Check every endpoint of a running rllm server and report pass/fail.

Sends a small request to each endpoint and checks the response's status and
shape: models, chat completions (plain, streaming, n, logprobs, JSON mode,
JSON schema, tools, stored completions), legacy completions, Responses
(plain, streaming, chained, background and cancel, stored items), embeddings,
/tokenize, /detokenize, /apply-template, /metrics, /health and the error
responses. Needs only `requests`.

On a terminal it opens a menu: run all checks, a group or chosen checks,
rerun the failed ones, look at each check's requests and responses, send
your own requests (with templates, and live output when streaming), and
change the settings. With --batch or --only (or when not on a terminal) it
runs the checks and exits.

    python3 tools/test_endpoints.py                    # asks for the server
    python3 tools/test_endpoints.py --url http://host:8080
    python3 tools/test_endpoints.py --batch
    python3 tools/test_endpoints.py --only chat,responses -v
    python3 tools/test_endpoints.py --api-key "$RLLM_API_KEY" --list

For a decision model (Clef), generation endpoints are expected to answer 400
and /v1/systemone is checked instead. The exit status is the number of
failed checks (0 when all pass).
"""
import argparse
import base64
import json
import sys
import time
import traceback

import requests

GROUPS = ('basic', 'chat', 'store', 'completions', 'responses', 'embeddings', 'extras', 'errors', 'systemone')


class CheckFailed(Exception):
    pass


def expect(cond, message):
    if not cond:
        raise CheckFailed(message)


class Client:
    def __init__(self, base, api_key, timeout, verbose):
        self.base = base.rstrip('/')
        self.timeout = timeout
        self.verbose = verbose
        self.s = requests.Session()
        if api_key:
            self.s.headers['Authorization'] = f'Bearer {api_key}'
        self.model = None
        self.decision = False
        self.effort = None
        self.log = []  # (method, path, status, request body, response text) of each request, for viewing later

    def with_effort(self, path, body):
        """Add the reasoning effort to generation requests that don't set one."""
        if not self.effort or not isinstance(body, dict):
            return body
        if path == '/v1/chat/completions' and 'messages' in body and 'reasoning_effort' not in body:
            return {**body, 'reasoning_effort': self.effort}
        if path == '/v1/responses' and 'input' in body and 'reasoning' not in body:
            return {**body, 'reasoning': {'effort': self.effort}}
        return body

    def request(self, method, path, status=200, **kw):
        """Send a request and check its status; returns the response."""
        kw.setdefault('timeout', self.timeout)
        r = self.s.request(method, self.base + path, **kw)
        r.encoding = 'utf-8'
        if not kw.get('stream'):
            self.log.append((method, path, r.status_code, kw.get('json', kw.get('data')), r.text))
        if self.verbose and not kw.get('stream'):
            body = r.text if len(r.text) < 600 else r.text[:600] + ' ...'
            print(f'      {method} {path} -> {r.status_code}: {body.strip()}')
        expect(r.status_code == status, f'{method} {path}: HTTP {r.status_code}, expected {status}: {r.text[:300]}')
        return r

    def json(self, method, path, body=None, status=200, **kw):
        r = self.request(method, path, status=status, json=self.with_effort(path, body), **kw)
        try:
            return r.json()
        except ValueError:
            raise CheckFailed(f'{method} {path}: response is not JSON: {r.text[:200]}')

    def post(self, path, body, status=200):
        return self.json('POST', path, body, status=status)

    def get(self, path, status=200):
        return self.json('GET', path, status=status)

    def stream(self, path, body):
        """POST a streaming request; returns the list of parsed SSE events and whether [DONE] came."""
        r = self.request('POST', path, json=self.with_effort(path, body), stream=True)
        expect('text/event-stream' in r.headers.get('content-type', ''), f'content-type is {r.headers.get("content-type")}')
        events, done = [], False
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            events.append(json.loads(data))
        r.close()
        self.log.append(('POST', path, r.status_code, self.with_effort(path, body),
                         '\n'.join(json.dumps(e) for e in events) + ('\n[DONE]' if done else '')))
        if self.verbose:
            print(f'      {len(events)} events, [DONE]={done}')
        return events, done


def error_shape(body, param=None):
    err = body.get('error')
    expect(isinstance(err, dict), f'no error object: {body}')
    for k in ('message', 'type', 'param', 'code'):
        expect(k in err, f'error lacks {k!r}: {err}')
    if param is not None:
        expect(err['param'] == param, f'error param is {err["param"]!r}, expected {param!r}')
    return err


def msgs(text, system=None):
    m = [{'role': 'system', 'content': system}] if system else []
    return m + [{'role': 'user', 'content': text}]


WEATHER_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get the current weather for a city.',
        'parameters': {
            'type': 'object',
            'properties': {'city': {'type': 'string'}, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}},
            'required': ['city'],
        },
    },
}

PERSON_SCHEMA = {
    'type': 'object',
    'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}, 'tags': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 3}},
    'required': ['name', 'age', 'tags'],
    'additionalProperties': False,
}


# --- checks ---------------------------------------------------------------
# Each check is (group, name, function(client, args)). A check returns an
# optional short detail string, or raises CheckFailed.

CHECKS = []


def check(group, name):
    def register(fn):
        CHECKS.append((group, name, fn))
        return fn
    return register


@check('basic', 'GET /health')
def _(c, a):
    r = c.request('GET', '/health')
    expect(r.text.strip() == 'ok', f'body is {r.text!r}')


@check('basic', 'GET /v1/models')
def _(c, a):
    b = c.get('/v1/models')
    expect(b.get('object') == 'list' and b.get('data'), f'unexpected body {b}')
    m = b['data'][0]
    expect(m.get('object') == 'model' and m.get('id'), f'bad model object {m}')
    return m['id']


@check('basic', 'GET /v1/models/{model}')
def _(c, a):
    b = c.get(f'/v1/models/{c.model}')
    expect(b.get('id') == c.model, f'id is {b.get("id")}')


@check('basic', 'GET /v1/models/{unknown} is 404')
def _(c, a):
    error_shape(c.get('/v1/models/no-such-model-xyz', status=404))


@check('basic', 'DELETE /v1/models/{model} is 403')
def _(c, a):
    error_shape(c.json('DELETE', f'/v1/models/{c.model}', status=403))


@check('basic', 'response headers')
def _(c, a):
    r = c.request('GET', '/v1/models', headers={'X-Client-Request-Id': 'endpoint-test-1'})
    for h in ('x-request-id', 'openai-processing-ms', 'openai-version'):
        expect(h in r.headers, f'missing header {h}')
    expect(r.headers.get('x-client-request-id') == 'endpoint-test-1', 'X-Client-Request-Id not echoed')


# Chat completions

@check('chat', 'chat: basic')
def _(c, a):
    b = c.post('/v1/chat/completions', {'model': c.model, 'messages': msgs('Say hello in five words or fewer.'), 'max_tokens': a.max_tokens, 'temperature': 0})
    expect(b.get('object') == 'chat.completion', f'object is {b.get("object")}')
    ch = b['choices'][0]
    expect(ch['message']['role'] == 'assistant', 'role is not assistant')
    expect(isinstance(ch['message'].get('content'), str), 'content is not a string')
    expect(ch['finish_reason'] in ('stop', 'length'), f'finish_reason {ch["finish_reason"]}')
    u = b['usage']
    expect(u['prompt_tokens'] > 0 and u['completion_tokens'] > 0, f'usage {u}')
    expect(u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens'], 'total_tokens mismatch')
    return repr(ch['message']['content'][:60])


@check('chat', 'chat: streaming with usage')
def _(c, a):
    t0 = time.perf_counter()
    ev, done = c.stream('/v1/chat/completions', {'model': c.model, 'messages': msgs('Count from 1 to 10.'), 'max_tokens': a.max_tokens,
                                                 'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}})
    dt = time.perf_counter() - t0
    expect(done, 'no [DONE]')
    expect(all(e.get('object') == 'chat.completion.chunk' for e in ev), 'chunk with wrong object')
    deltas = [ch.get('delta') or {} for e in ev for ch in e.get('choices', [])]
    text = ''.join((d.get('content') or '') + (d.get('reasoning_content') or '') for d in deltas)
    expect(text, 'no content or reasoning streamed')
    finishes = [ch['finish_reason'] for e in ev for ch in e.get('choices', []) if ch.get('finish_reason')]
    expect(finishes, 'no finish_reason')
    usage = [e['usage'] for e in ev if e.get('usage')]
    expect(usage and not ev[-1]['choices'], 'last chunk is not the usage chunk')
    n = usage[-1]['completion_tokens']
    return f'{n} tokens, {n / dt:.1f} tok/s'


@check('chat', 'chat: n=2, logprobs, seed')
def _(c, a):
    body = {'model': c.model, 'messages': msgs('Name a color.'), 'max_tokens': 8, 'n': 2, 'temperature': 0.8, 'seed': 42,
            'logprobs': True, 'top_logprobs': 3, 'reasoning_effort': 'none'}  # logprobs cover answer text only
    b1 = c.post('/v1/chat/completions', body)
    expect(len(b1['choices']) == 2, f'{len(b1["choices"])} choices')
    expect(sorted(ch['index'] for ch in b1['choices']) == [0, 1], 'bad choice indexes')
    lp = b1['choices'][0].get('logprobs') or {}
    content = lp.get('content') or []
    expect(content, 'no logprobs.content')
    t = content[0]
    expect('token' in t and 'logprob' in t and 'bytes' in t and len(t['top_logprobs']) == 3, f'bad logprob entry {t}')
    expect(t['logprob'] <= 0, 'logprob > 0')
    b2 = c.post('/v1/chat/completions', body)
    same = [x['message']['content'] for x in b1['choices']] == [x['message']['content'] for x in b2['choices']]
    expect(same, 'same seed gave different output')


@check('chat', 'chat: stop sequence and length limit')
def _(c, a):
    b = c.post('/v1/chat/completions', {'model': c.model, 'messages': msgs('Write the numbers 1 to 20 separated by commas.'),
                                        'max_tokens': 64, 'temperature': 0, 'stop': ['5']})
    expect('5' not in (b['choices'][0]['message']['content'] or ''), 'stop sequence included in output')
    b = c.post('/v1/chat/completions', {'model': c.model, 'messages': msgs('Write a long story about a dragon.'),
                                        'max_completion_tokens': 5, 'temperature': 0, 'reasoning_effort': 'none'})
    expect(b['choices'][0]['finish_reason'] == 'length', f'finish_reason {b["choices"][0]["finish_reason"]}')
    expect(b['usage']['completion_tokens'] == 5, f'{b["usage"]["completion_tokens"]} tokens')


@check('chat', 'chat: json_object')
def _(c, a):
    b = c.post('/v1/chat/completions', {'model': c.model, 'messages': msgs('Give a JSON object describing a cat, with keys name and color.'),
                                        'max_tokens': 128, 'temperature': 0, 'response_format': {'type': 'json_object'}})
    out = json.loads(b['choices'][0]['message']['content'])
    expect(isinstance(out, dict), f'not an object: {out}')
    return json.dumps(out)[:60]


@check('chat', 'chat: json_schema')
def _(c, a):
    b = c.post('/v1/chat/completions', {'model': c.model, 'messages': msgs('Make up a person.'), 'max_tokens': 128, 'temperature': 0,
                                        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'person', 'schema': PERSON_SCHEMA, 'strict': True}}})
    out = json.loads(b['choices'][0]['message']['content'])
    expect(set(out) == {'name', 'age', 'tags'}, f'keys {set(out)}')
    expect(isinstance(out['age'], int) and isinstance(out['name'], str), f'types {out}')
    expect(isinstance(out['tags'], list) and len(out['tags']) <= 3, f'tags {out["tags"]}')
    return json.dumps(out)[:60]


@check('chat', 'chat: tool call (required) and result round trip')
def _(c, a):
    messages = msgs('What is the weather in Paris?')
    b = c.post('/v1/chat/completions', {'model': c.model, 'messages': messages, 'tools': [WEATHER_TOOL], 'tool_choice': 'required',
                                        'max_tokens': 128, 'temperature': 0})
    ch = b['choices'][0]
    calls = ch['message'].get('tool_calls') or []
    expect(calls, f'no tool_calls: {ch["message"]}')
    expect(ch['finish_reason'] == 'tool_calls', f'finish_reason {ch["finish_reason"]}')
    call = calls[0]
    expect(call['function']['name'] == 'get_weather', f'called {call["function"]["name"]}')
    args = json.loads(call['function']['arguments'])
    expect('city' in args, f'arguments {args}')
    messages += [ch['message'], {'role': 'tool', 'tool_call_id': call['id'], 'content': '{"temperature": 18, "unit": "celsius"}'}]
    b = c.post('/v1/chat/completions', {'model': c.model, 'messages': messages, 'tools': [WEATHER_TOOL], 'max_tokens': 96, 'temperature': 0})
    m = b['choices'][0]['message']
    expect(m.get('content') or m.get('reasoning_content') or m.get('tool_calls'), 'empty reply after tool result')
    return f'get_weather({json.dumps(args)})'


@check('chat', 'chat: streamed tool call')
def _(c, a):
    ev, done = c.stream('/v1/chat/completions', {'model': c.model, 'messages': msgs('What is the weather in Tokyo?'), 'tools': [WEATHER_TOOL],
                                                 'tool_choice': {'type': 'function', 'function': {'name': 'get_weather'}},
                                                 'max_tokens': 128, 'temperature': 0, 'stream': True})
    expect(done, 'no [DONE]')
    parts = [tc for e in ev for ch in e.get('choices', []) for tc in (ch.get('delta') or {}).get('tool_calls') or []]
    expect(parts, 'no tool_calls deltas')
    name = ''.join(p.get('function', {}).get('name') or '' for p in parts)
    args = json.loads(''.join(p.get('function', {}).get('arguments') or '' for p in parts))
    expect(name == 'get_weather' and 'city' in args, f'{name} {args}')


# Stored chat completions

@check('store', 'stored chat completion: create, list, retrieve, update, messages, delete')
def _(c, a):
    tag = f'endpoint-test-{int(time.time() * 1000)}'
    b = c.post('/v1/chat/completions', {'model': c.model, 'messages': msgs('Say OK.'), 'max_tokens': 8, 'temperature': 0,
                                        'store': True, 'metadata': {'suite': tag}})
    cid = b['id']
    lst = c.get(f'/v1/chat/completions?metadata[suite]={tag}')
    expect([x['id'] for x in lst['data']] == [cid], f'list gave {[x["id"] for x in lst["data"]]}')
    got = c.get(f'/v1/chat/completions/{cid}')
    expect(got['id'] == cid and got['choices'][0]['message']['content'] == b['choices'][0]['message']['content'], 'retrieved differs')
    upd = c.post(f'/v1/chat/completions/{cid}', {'metadata': {'suite': tag, 'updated': 'yes'}})
    expect(upd['metadata'].get('updated') == 'yes', f'metadata {upd.get("metadata")}')
    m = c.get(f'/v1/chat/completions/{cid}/messages')
    expect(m['data'] and m['data'][0]['content'] == 'Say OK.', f'messages {m["data"]}')
    d = c.json('DELETE', f'/v1/chat/completions/{cid}')
    expect(d.get('deleted') is True, f'delete gave {d}')
    error_shape(c.get(f'/v1/chat/completions/{cid}', status=404))


# Legacy completions

@check('completions', 'completions: basic')
def _(c, a):
    b = c.post('/v1/completions', {'model': c.model, 'prompt': 'The capital of France is', 'max_tokens': 8, 'temperature': 0})
    expect(b.get('object') == 'text_completion', f'object is {b.get("object")}')
    expect(isinstance(b['choices'][0]['text'], str), 'text not a string')
    return repr(b['choices'][0]['text'][:60])


@check('completions', 'completions: echo, logprobs, batch prompts, token prompt')
def _(c, a):
    b = c.post('/v1/completions', {'model': c.model, 'prompt': 'One two three', 'max_tokens': 4, 'temperature': 0, 'echo': True, 'logprobs': 2})
    ch = b['choices'][0]
    expect(ch['text'].startswith('One two three'), f'echo missing: {ch["text"]!r}')
    expect(ch['logprobs'] and ch['logprobs']['tokens'] and ch['logprobs']['token_logprobs'], 'no logprobs')
    b = c.post('/v1/completions', {'model': c.model, 'prompt': ['Red is a', 'Blue is a'], 'max_tokens': 4, 'temperature': 0})
    expect(sorted(x['index'] for x in b['choices']) == [0, 1], 'batch prompts not both answered')
    toks = c.post('/tokenize', {'content': 'Hello there'})['tokens']
    b = c.post('/v1/completions', {'model': c.model, 'prompt': toks, 'max_tokens': 4, 'temperature': 0})
    expect(b['usage']['prompt_tokens'] == len(toks), f'prompt_tokens {b["usage"]["prompt_tokens"]} vs {len(toks)}')


@check('completions', 'completions: streaming, best_of')
def _(c, a):
    ev, done = c.stream('/v1/completions', {'model': c.model, 'prompt': 'Once upon a time', 'max_tokens': 16, 'temperature': 0, 'stream': True})
    expect(done and ''.join(ch['text'] for e in ev for ch in e.get('choices', [])), 'nothing streamed')
    b = c.post('/v1/completions', {'model': c.model, 'prompt': 'A good name for a dog is', 'max_tokens': 6, 'best_of': 3, 'n': 1, 'temperature': 0.8})
    expect(len(b['choices']) == 1, f'{len(b["choices"])} choices')


@check('completions', 'completions: suffix rejected')
def _(c, a):
    error_shape(c.post('/v1/completions', {'model': c.model, 'prompt': 'def f(', 'suffix': ')', 'max_tokens': 4}, status=400))


# Responses

@check('responses', 'responses: basic, retrieve, input_items')
def _(c, a):
    b = c.post('/v1/responses', {'model': c.model, 'input': 'Say hello.', 'max_output_tokens': a.max_tokens, 'temperature': 0})
    expect(b.get('object') == 'response' and b['status'] in ('completed', 'incomplete'), f'status {b.get("status")}')
    msg = [o for o in b['output'] if o['type'] == 'message']
    expect(msg and msg[0]['content'][0]['type'] == 'output_text', f'output {b["output"]}')
    expect(b['usage']['input_tokens'] > 0, 'no usage')
    got = c.get(f'/v1/responses/{b["id"]}')
    expect(got['id'] == b['id'] and got['output'] == b['output'], 'retrieved response differs')
    items = c.get(f'/v1/responses/{b["id"]}/input_items')
    expect(items['data'] and items['data'][0]['type'] == 'message', f'input items {items["data"]}')
    return repr(msg[0]['content'][0]['text'][:60])


@check('responses', 'responses: streaming events')
def _(c, a):
    ev, _done = c.stream('/v1/responses', {'model': c.model, 'input': 'Count from 1 to 5.', 'max_output_tokens': a.max_tokens,
                                           'temperature': 0, 'stream': True})
    types = [e['type'] for e in ev]
    expect(types[:2] == ['response.created', 'response.in_progress'], f'starts with {types[:2]}')
    expect(types[-1] in ('response.completed', 'response.incomplete'), f'ends with {types[-1]}')
    expect('response.output_text.delta' in types and 'response.output_text.done' in types, 'no output_text events')
    seq = [e['sequence_number'] for e in ev]
    expect(seq == sorted(seq) and len(set(seq)) == len(seq), 'sequence numbers not increasing')
    text = ''.join(e['delta'] for e in ev if e['type'] == 'response.output_text.delta')
    done = [e['text'] for e in ev if e['type'] == 'response.output_text.done']
    expect(done and done[0] == text, 'output_text.done text differs from the deltas')
    return f'{len(ev)} events'


@check('responses', 'responses: previous_response_id')
def _(c, a):
    r1 = c.post('/v1/responses', {'model': c.model, 'input': 'My favourite number is 47. Reply with just OK.', 'max_output_tokens': 16,
                                  'temperature': 0, 'reasoning': {'effort': 'none'}})
    r2 = c.post('/v1/responses', {'model': c.model, 'input': 'What is my favourite number? Answer with the number only.', 'max_output_tokens': 16,
                                  'temperature': 0, 'previous_response_id': r1['id'], 'reasoning': {'effort': 'none'}})
    expect(r2['previous_response_id'] == r1['id'], 'previous_response_id not echoed')
    expect(r2['usage']['input_tokens'] > r1['usage']['input_tokens'], 'second turn prompt not longer than the first')
    text = ''.join(p['text'] for o in r2['output'] if o['type'] == 'message' for p in o['content'])
    return ('remembered' if '47' in text else 'did not answer 47 (model quality, not a failure)') + f': {text[:40]!r}'


@check('responses', 'responses: function call round trip')
def _(c, a):
    tool = {'type': 'function', **WEATHER_TOOL['function']}
    r1 = c.post('/v1/responses', {'model': c.model, 'input': 'What is the weather in Rome?', 'tools': [tool], 'tool_choice': 'required',
                                  'max_output_tokens': 128, 'temperature': 0})
    calls = [o for o in r1['output'] if o['type'] == 'function_call']
    expect(calls and calls[0]['name'] == 'get_weather', f'output {r1["output"]}')
    json.loads(calls[0]['arguments'])
    r2 = c.post('/v1/responses', {'model': c.model, 'previous_response_id': r1['id'], 'tools': [tool], 'max_output_tokens': 96, 'temperature': 0,
                                  'input': [{'type': 'function_call_output', 'call_id': calls[0]['call_id'], 'output': '{"temperature": 22}'}]})
    expect(r2['status'] in ('completed', 'incomplete') and r2['output'], f'status {r2["status"]}')


@check('responses', 'responses: json_schema text format')
def _(c, a):
    b = c.post('/v1/responses', {'model': c.model, 'input': 'Make up a person.', 'max_output_tokens': 128, 'temperature': 0,
                                 'text': {'format': {'type': 'json_schema', 'name': 'person', 'schema': PERSON_SCHEMA, 'strict': True}}})
    text = ''.join(p['text'] for o in b['output'] if o['type'] == 'message' for p in o['content'])
    expect(set(json.loads(text)) == {'name', 'age', 'tags'}, f'output {text[:100]}')


@check('responses', 'responses: input_tokens')
def _(c, a):
    b = c.post('/v1/responses/input_tokens', {'model': c.model, 'input': 'How many tokens is this?'})
    expect(isinstance(b.get('input_tokens'), int) and b['input_tokens'] > 0, f'body {b}')
    return f'{b["input_tokens"]} tokens'


@check('responses', 'responses: incomplete status')
def _(c, a):
    b = c.post('/v1/responses', {'model': c.model, 'input': 'Write a long essay about the sea.', 'max_output_tokens': 4, 'temperature': 0})
    expect(b['status'] == 'incomplete', f'status {b["status"]}')
    expect(b['incomplete_details']['reason'] == 'max_output_tokens', f'details {b["incomplete_details"]}')


@check('responses', 'responses: background, cancel, delete')
def _(c, a):
    b = c.post('/v1/responses', {'model': c.model, 'input': 'Write a very long story.', 'max_output_tokens': 2048, 'background': True})
    expect(b['status'] in ('queued', 'in_progress'), f'status {b["status"]}')
    x = c.post(f'/v1/responses/{b["id"]}/cancel', {})
    expect(x['status'] in ('cancelled', 'completed', 'incomplete'), f'status after cancel {x["status"]}')
    status = x['status']
    for _ in range(50):
        status = c.get(f'/v1/responses/{b["id"]}')['status']
        if status not in ('queued', 'in_progress'):
            break
        time.sleep(0.1)
    d = c.json('DELETE', f'/v1/responses/{b["id"]}')
    expect(d.get('deleted') is True, f'delete gave {d}')
    error_shape(c.get(f'/v1/responses/{b["id"]}', status=404))
    # A finished, non-background response cannot be cancelled.
    r = c.post('/v1/responses', {'model': c.model, 'input': 'Hi', 'max_output_tokens': 4})
    error_shape(c.post(f'/v1/responses/{r["id"]}/cancel', {}, status=400))
    c.json('DELETE', f'/v1/responses/{r["id"]}')
    return f'final status {status}'


# Embeddings

@check('embeddings', 'embeddings: float, batch, dimensions, base64')
def _(c, a):
    b = c.post('/v1/embeddings', {'model': c.model, 'input': ['The cat sat on the mat.', 'A dog ran in the park.']})
    expect(b.get('object') == 'list' and len(b['data']) == 2, f'body keys {list(b)}')
    v = b['data'][0]['embedding']
    expect(isinstance(v, list) and v and all(isinstance(x, float) for x in v[:8]), 'embedding is not a float list')
    dim = len(v)
    d = min(64, dim)
    b = c.post('/v1/embeddings', {'model': c.model, 'input': 'Hello', 'dimensions': d})
    vd = b['data'][0]['embedding']
    expect(len(vd) == d, f'{len(vd)} dimensions, asked {d}')
    norm = sum(x * x for x in vd) ** 0.5
    expect(abs(norm - 1) < 1e-3, f'truncated vector norm {norm}')
    b = c.post('/v1/embeddings', {'model': c.model, 'input': 'Hello', 'encoding_format': 'base64'})
    raw = base64.b64decode(b['data'][0]['embedding'])
    expect(len(raw) == 4 * dim, f'base64 decodes to {len(raw)} bytes, expected {4 * dim}')
    error_shape(c.post('/v1/embeddings', {'model': c.model, 'input': 'Hello', 'dimensions': dim + 1}, status=400), 'dimensions')
    return f'{dim} dimensions'


# llama.cpp-style extras

@check('extras', '/tokenize and /detokenize round trip')
def _(c, a):
    text = 'Hello, world! Ünïcödé 🙂'
    t = c.post('/tokenize', {'content': text})['tokens']
    expect(t and all(isinstance(x, int) for x in t), f'tokens {t}')
    back = c.post('/detokenize', {'tokens': t})['content']
    expect(back == text, f'round trip gave {back!r}')
    p = c.post('/tokenize', {'content': 'Hello', 'with_pieces': True})['tokens']
    expect(p and 'id' in p[0] and 'piece' in p[0], f'pieces {p}')
    error_shape(c.post('/detokenize', {'tokens': [10 ** 9]}, status=400), 'tokens')
    return f'{len(t)} tokens'


@check('extras', '/apply-template')
def _(c, a):
    b = c.post('/apply-template', {'messages': msgs('Hi there', system='Be brief.')})
    expect('Hi there' in b['prompt'] and 'Be brief.' in b['prompt'], f'prompt {b["prompt"][:200]!r}')
    b2 = c.post('/apply-template', {'messages': msgs('Weather?'), 'tools': [WEATHER_TOOL]})
    expect('get_weather' in b2['prompt'], 'tool not in templated prompt')


@check('extras', 'GET /metrics')
def _(c, a):
    r = c.request('GET', '/metrics')
    lines = [l for l in r.text.splitlines() if l and not l.startswith('#')]
    names = {l.split('{')[0].split(' ')[0] for l in lines}
    expect(any(n.startswith('rllm_') for n in names), 'no rllm_ metrics')
    return f'{len(names)} series'


# Errors

@check('errors', 'unknown URL is 404 Invalid URL')
def _(c, a):
    e = error_shape(c.get('/v1/no-such-endpoint', status=404))
    expect('Invalid URL' in e['message'], f'message {e["message"]}')
    error_shape(c.json('PUT', '/v1/chat/completions', {}, status=404))


@check('errors', 'bad parameters are 400 naming the param')
def _(c, a):
    error_shape(c.post('/v1/chat/completions', {'model': c.model}, status=400), 'messages')
    error_shape(c.post('/v1/chat/completions', {'model': c.model, 'messages': msgs('x'), 'temperature': 5}, status=400), 'temperature')
    error_shape(c.post('/v1/chat/completions', {'model': c.model, 'messages': msgs('x'), 'n': 0}, status=400), 'n')
    error_shape(c.post('/v1/chat/completions', {'model': c.model, 'messages': msgs('x'), 'top_logprobs': 50, 'logprobs': True}, status=400), 'top_logprobs')
    bad_tool = {'type': 'function', 'function': {'name': 'bad name!', 'parameters': {'type': 'object'}}}
    error_shape(c.post('/v1/chat/completions', {'model': c.model, 'messages': msgs('x'), 'tools': [bad_tool]}, status=400))


@check('errors', 'non-JSON body is 415, malformed JSON is 400')
def _(c, a):
    c.request('POST', '/v1/chat/completions', status=415, data='model=x', headers={'Content-Type': 'application/x-www-form-urlencoded'})
    c.request('POST', '/v1/chat/completions', status=400, data='{not json', headers={'Content-Type': 'application/json'})


@check('errors', 'unsupported content is 400')
def _(c, a):
    image = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]}]
    error_shape(c.post('/v1/chat/completions', {'model': c.model, 'messages': image}, status=400))
    error_shape(c.post('/v1/responses', {'model': c.model, 'input': 'x', 'tools': [{'type': 'web_search'}]}, status=400))


@check('errors', 'context length exceeded is 400')
def _(c, a):
    long = 'word ' * 400_000
    e = error_shape(c.post('/v1/chat/completions', {'model': c.model, 'messages': msgs(long), 'max_tokens': 4}, status=400))
    return e.get('code') or e['message'][:60]


@check('errors', 'missing or wrong API key is 401')
def _(c, a):
    if not a.api_key:
        return 'SKIP: no --api-key given'
    r = requests.get(c.base + '/v1/models', timeout=c.timeout)
    expect(r.status_code == 401, f'no key: HTTP {r.status_code}')
    r = requests.get(c.base + '/v1/models', headers={'Authorization': 'Bearer wrong-key'}, timeout=c.timeout)
    expect(r.status_code == 401 and r.json()['error']['code'] == 'invalid_api_key', f'wrong key: HTTP {r.status_code}')


# Decision models

@check('systemone', 'POST /v1/systemone')
def _(c, a):
    body = {'state': 'I was charged twice for my plan this month.', 'questions': {
        'route': {'type': 'choice', 'instructions': 'Which team handles this?', 'criteria': {'billing': 'Payments, refunds', 'tech': 'Bugs, outages', 'sales': None}},
        'urgent': {'type': 'noul', 'instructions': 'Does the customer need an answer today?'},
        'anger': {'type': 'score', 'instructions': 'How upset is the customer?', 'criteria': ['calm', 'annoyed', 'angry']}}}
    if not c.decision:
        error_shape(c.post('/v1/systemone', body, status=400))
        return 'not a decision model: 400 as expected'
    b = c.post('/v1/systemone', body)
    ans = b['answers']
    expect(set(ans) == {'route', 'urgent', 'anger'}, f'answers {list(ans)}')
    probs = ans['route']['probabilities']
    expect(abs(sum(probs.values()) - 1) < 1e-3, f'probabilities sum to {sum(probs.values())}')
    expect(0 <= ans['urgent']['noul'] <= 1, f'noul {ans["urgent"]["noul"]}')
    expect(0 <= ans['anger']['score'] <= 2, f'score {ans["anger"]["score"]}')
    return f'route={ans["route"]["choice"]} ({max(probs.values()):.2f})'


def decision_generation_check(c, a):
    """For a decision model: every generation endpoint answers 400."""
    error_shape(c.post('/v1/chat/completions', {'model': c.model, 'messages': msgs('hi')}, status=400))
    error_shape(c.post('/v1/completions', {'model': c.model, 'prompt': 'hi'}, status=400))
    error_shape(c.post('/v1/responses', {'model': c.model, 'input': 'hi'}, status=400))


# --- runner ---------------------------------------------------------------

TTY = sys.stdout.isatty()
GREEN, RED, YELLOW, BOLD, DIM, RESET = ('\033[32m', '\033[31m', '\033[33m', '\033[1m', '\033[2m', '\033[0m') if TTY else ('',) * 6
MARK = {'pass': f'{GREEN}PASS{RESET}', 'fail': f'{RED}FAIL{RESET}', 'skip': f'{YELLOW}SKIP{RESET}'}


def connect(c, model=None):
    """Find the model and whether it is a decision model. Returns an error message, or None."""
    log = c.log
    try:
        models = c.get('/v1/models')
        c.model = model or models['data'][0]['id']
        # A decision model answers /v1/systemone; anything else refuses it.
        probe = c.s.post(c.base + '/v1/systemone', json={'state': 'x', 'questions': {'q': {'type': 'noul', 'instructions': 'x'}}}, timeout=c.timeout)
        c.decision = probe.status_code == 200
        return None
    except (requests.RequestException, CheckFailed, KeyError, IndexError, ValueError) as e:
        c.model, c.decision = None, False
        return f'cannot reach {c.base}/v1/models: {e}'
    finally:
        c.log = log


def plan(c, checks):
    """The checks that apply to the connected model."""
    if not c.decision:
        return list(checks)
    return [x for x in checks if x[0] in ('basic', 'extras', 'systemone')] + \
        [('errors', 'generation endpoints refuse a decision model', decision_generation_check)]


def run_checks(c, a, checks):
    """Run checks in order, printing a line each; returns one result dict per check run."""
    print(f'\nServer {c.base}, model {c.model}{" (decision model)" if c.decision else ""}, {len(checks)} checks\n')
    results = []
    t_all = time.perf_counter()
    try:
        for group, name, fn in checks:
            c.log = []
            t0 = time.perf_counter()
            res = {'group': group, 'name': name, 'fn': fn, 'detail': '', 'log': c.log}
            try:
                detail = fn(c, a)
                res['dt'] = time.perf_counter() - t0
                if isinstance(detail, str) and detail.startswith('SKIP'):
                    res.update(status='skip', detail=detail[5:].strip())
                    print(f'{MARK["skip"]} {name} {DIM}{res["detail"]}{RESET}')
                else:
                    res.update(status='pass', detail=detail or '')
                    print(f'{MARK["pass"]} {name} {DIM}({res["dt"]:.2f} s){" " + detail if detail else ""}{RESET}')
            except Exception as e:
                res['dt'] = time.perf_counter() - t0
                msg = str(e) if isinstance(e, (CheckFailed, requests.RequestException)) else f'{type(e).__name__}: {e}'
                res.update(status='fail', detail=msg)
                print(f'{MARK["fail"]} {name} {DIM}({res["dt"]:.2f} s){RESET}\n     {msg}')
                if a.verbose and not isinstance(e, CheckFailed):
                    traceback.print_exc()
            results.append(res)
            if a.exitfirst and res['status'] == 'fail':
                break
    except KeyboardInterrupt:
        print(f'\n{YELLOW}interrupted{RESET}')
    count = {k: sum(r['status'] == k for r in results) for k in ('pass', 'fail', 'skip')}
    print(f'\n{count["pass"]} passed, {count["fail"]} failed, {count["skip"]} skipped in {time.perf_counter() - t_all:.1f} s')
    for r in results:
        if r['status'] == 'fail':
            print(f'  {RED}failed:{RESET} {r["name"]}')
    return results


# --- interactive mode -----------------------------------------------------

def ask(prompt, default=None):
    """Read a line; returns the default for an empty line, None at Ctrl-D / Ctrl-C."""
    try:
        line = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return line if line else default


def pretty(text, limit):
    """Pretty-print JSON (or JSON lines), cut to `limit` characters."""
    if text is None:
        return ''
    if not isinstance(text, str):
        text = json.dumps(text, indent=2, ensure_ascii=False)
    else:
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except ValueError:
            pass
    if limit and len(text) > limit:
        text = text[:limit] + f'\n{DIM}... {len(text) - limit} more characters (raise the view limit in settings){RESET}'
    return text


def parse_selection(text, items):
    """Indexes chosen by '3', '2,5', '4-7', 'all', or a name substring."""
    if text in ('all', '*'):
        return list(range(len(items)))
    chosen = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if part.replace('-', '').isdigit() and part.count('-') <= 1:
            lo, _, hi = part.partition('-')
            lo, hi = int(lo), int(hi or lo)
            chosen += [i - 1 for i in range(lo, hi + 1) if 1 <= i <= len(items)]
        else:
            chosen += [i for i, it in enumerate(items) if part.lower() in it.lower()]
    return sorted(set(chosen), key=chosen.index)


def show_exchanges(res, limit):
    print(f'\n{BOLD}{res["name"]}{RESET}  [{MARK[res["status"]]}] {res["detail"]}')
    if not res['log']:
        print(f'{DIM}(no requests recorded){RESET}')
    for method, path, status, body, text in res['log']:
        color = GREEN if 200 <= status < 300 else YELLOW if status < 500 else RED
        print(f'\n{BOLD}> {method} {path}{RESET}')
        if body is not None:
            print(pretty(body, limit))
        print(f'{BOLD}< {color}HTTP {status}{RESET}')
        print(pretty(text, limit))


def view_results(results, a):
    while True:
        if not results:
            print('No results yet: run some checks first.')
            return
        print()
        for i, r in enumerate(results, 1):
            print(f'{i:3}  {MARK[r["status"]]} {r["name"]} {DIM}{r["detail"][:70]}{RESET}')
        sel = ask('\nShow requests and responses for (number, "f" for all failed, Enter to go back): ')
        if sel is None:
            return
        idx = [i for i, r in enumerate(results) if r['status'] == 'fail'] if sel == 'f' else parse_selection(sel, [r['name'] for r in results])
        for i in idx:
            show_exchanges(results[i], a.view_chars)
        if idx:
            ask(f'\n{DIM}Enter to continue{RESET}')


# Bodies for the custom request mode. A plain-text answer at the body prompt
# goes where PROMPT is.
TEMPLATES = {
    '/v1/chat/completions': lambda c, a, t: {'model': c.model, 'messages': msgs(t), 'max_tokens': 256},
    '/v1/completions': lambda c, a, t: {'model': c.model, 'prompt': t, 'max_tokens': 64},
    '/v1/responses': lambda c, a, t: {'model': c.model, 'input': t, 'max_output_tokens': 256},
    '/v1/responses/input_tokens': lambda c, a, t: {'model': c.model, 'input': t},
    '/v1/embeddings': lambda c, a, t: {'model': c.model, 'input': t},
    '/tokenize': lambda c, a, t: {'content': t, 'with_pieces': True},
    '/detokenize': lambda c, a, t: {'tokens': json.loads(t) if t.startswith('[') else [int(x) for x in t.replace(',', ' ').split()]},
    '/apply-template': lambda c, a, t: {'messages': msgs(t)},
}
GET_PATHS = ['/health', '/v1/models', '/metrics', '/v1/chat/completions?limit=5']


def read_body(c, a, path):
    """Ask for a request body: JSON typed over several lines, or text for the path's template."""
    template = TEMPLATES.get(path.split('?')[0])
    hint = 'text for the template, ' if template else ''
    print(f'Body: {hint}JSON (finish with an empty line), or Enter for none.')
    first = ask('  ')
    if first is None:
        return False, None
    if not first:
        return True, None
    if first[0] in '{[' and not path.startswith('/detokenize'):
        lines = [first]
        while True:
            try:
                json.loads('\n'.join(lines))
                break
            except ValueError:
                pass
            line = ask('  ')
            if line is None:
                return False, None
            if line == '':
                break
            lines.append(line)
        try:
            return True, json.loads('\n'.join(lines))
        except ValueError as e:
            print(f'{RED}Not valid JSON: {e}{RESET}')
            return False, None
    if not template:
        print(f'{RED}No template for {path}: type JSON.{RESET}')
        return False, None
    try:
        return True, template(c, a, first)
    except ValueError as e:
        print(f'{RED}{e}{RESET}')
        return False, None


def custom_requests(c, a):
    print(f'\n{BOLD}Custom request{RESET}. Paths with templates: {", ".join(TEMPLATES)}.')
    print(f'GET paths: {", ".join(GET_PATHS)}. Add "stream" after a POST path to stream it. Enter on an empty path goes back.')
    last = '/v1/chat/completions'
    while True:
        line = ask(f'\nMethod and path {DIM}(last: {last}; Enter to go back){RESET}: ')
        if not line:
            return
        words = line.split()
        method = words[0].upper() if words[0].upper() in ('GET', 'POST', 'DELETE', 'PUT', 'PATCH') else None
        if method:
            words = words[1:]
        stream = 'stream' in words
        words = [w for w in words if w != 'stream']
        path = words[0] if words else last
        if not path.startswith('/'):
            path = '/' + path
        method = method or ('GET' if path in GET_PATHS else 'POST')
        last = path
        body = None
        if method in ('POST', 'PUT', 'PATCH'):
            ok, body = read_body(c, a, path)
            if not ok:
                continue
            body = c.with_effort(path.split('?')[0], body)
            if stream and isinstance(body, dict):
                body['stream'] = True
                if path == '/v1/chat/completions':
                    body['stream_options'] = {'include_usage': True}
            if body is not None:
                print(f'{DIM}{json.dumps(body, ensure_ascii=False)[:a.view_chars]}{RESET}')
        send(c, a, method, path, body)


def send(c, a, method, path, body):
    """Send one request and print the status, timings and body (streamed live for SSE)."""
    t0 = time.perf_counter()
    try:
        r = c.s.request(method, c.base + path, json=body, timeout=c.timeout, stream=True)
        r.encoding = 'utf-8'
        color = GREEN if r.ok else YELLOW if r.status_code < 500 else RED
        print(f'{BOLD}< {color}HTTP {r.status_code}{RESET} {DIM}x-request-id {r.headers.get("x-request-id", "-")}, '
              f'{r.headers.get("content-type", "")}{RESET}')
        if 'text/event-stream' in r.headers.get('content-type', ''):
            stream_events(r, t0)
        else:
            print(pretty(r.text, a.view_chars))
        r.close()
        print(f'{DIM}{time.perf_counter() - t0:.2f} s{RESET}')
    except requests.RequestException as e:
        print(f'{RED}request failed: {e}{RESET}')
    except KeyboardInterrupt:
        print(f'\n{YELLOW}interrupted{RESET}')


def stream_events(r, t0):
    """Print streamed text as it arrives; summarize the other events."""
    first, kinds, usage, events = None, {}, None, 0
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            break
        events += 1
        e = json.loads(data)
        kinds[e.get('type') or e.get('object')] = kinds.get(e.get('type') or e.get('object'), 0) + 1
        usage = e.get('usage') or (e.get('response') or {}).get('usage') or usage
        pieces = []
        for ch in e.get('choices', []):
            d = ch.get('delta') or {}
            if d.get('reasoning_content'):
                pieces.append(DIM + d['reasoning_content'] + RESET)
            pieces.append(d.get('content') or ch.get('text') or '')
            for tc in d.get('tool_calls') or []:
                f = tc.get('function') or {}
                pieces.append(f'{YELLOW}{f.get("name") or ""}{f.get("arguments") or ""}{RESET}')
        if e.get('type') == 'response.output_text.delta':
            pieces.append(e['delta'])
        elif e.get('type') == 'response.reasoning_text.delta':
            pieces.append(DIM + e['delta'] + RESET)
        elif e.get('type') == 'response.function_call_arguments.delta':
            pieces.append(YELLOW + e['delta'] + RESET)
        elif e.get('type') == 'error' or 'error' in e:
            pieces.append(f'\n{RED}{json.dumps(e)}{RESET}')
        text = ''.join(pieces)
        if text:
            if first is None:
                first = time.perf_counter()
            print(text, end='', flush=True)
    print()
    ttft = f', first text after {first - t0:.2f} s' if first else ''
    print(f'{DIM}{events} events{ttft}: {", ".join(f"{k} x{n}" for k, n in kinds.items())}{RESET}')
    if usage:
        print(f'{DIM}usage {json.dumps(usage)}{RESET}')


def settings(c, a, base_checks):
    """Edit the settings; returns the checks for the (possibly new) connection."""
    while True:
        key = (c.s.headers.get('Authorization') or '')[7:]
        rows = [
            ('url', c.base), ('api key', ('*' * 4 + key[-4:]) if key else '(none)'), ('model', c.model or '(not connected)'),
            ('reasoning effort', a.reasoning_effort), ('max tokens', a.max_tokens), ('timeout', f'{c.timeout:g} s'),
            ('verbose', a.verbose), ('stop at first failure', a.exitfirst), ('view limit', f'{a.view_chars} characters'),
        ]
        print()
        for i, (k, v) in enumerate(rows, 1):
            print(f'{i:3}  {k:22} {v}')
        sel = ask('\nChange which (number, Enter to go back): ')
        if sel is None or not sel.isdigit() or not 1 <= int(sel) <= len(rows):
            return plan(c, base_checks)
        name = rows[int(sel) - 1][0]
        if name in ('verbose', 'stop at first failure'):
            attr = 'verbose' if name == 'verbose' else 'exitfirst'
            setattr(a, attr, not getattr(a, attr))
            c.verbose = a.verbose
            continue
        hint = {'reasoning effort': ' (none, minimal, low, medium, high, or default to send nothing)', 'api key': ' ("-" for none)'}.get(name, '')
        value = ask(f'New {name}{hint}: ')
        if value is None:
            continue
        try:
            if name == 'url':
                c.base = normalize_url(value)
            elif name == 'api key':
                c.s.headers.pop('Authorization', None)
                a.api_key = None if value == '-' else value
                if a.api_key:
                    c.s.headers['Authorization'] = f'Bearer {a.api_key}'
            elif name == 'model':
                a.model = value
            elif name == 'reasoning effort':
                a.reasoning_effort = value
                c.effort = None if value == 'default' else value
            elif name == 'max tokens':
                a.max_tokens = int(value)
            elif name == 'timeout':
                c.timeout = a.timeout = float(value)
            elif name == 'view limit':
                a.view_chars = int(value)
        except ValueError:
            print(f'{RED}not a number{RESET}')
            continue
        if name in ('url', 'api key', 'model'):
            err = connect(c, a.model)
            print(f'{RED}{err}{RESET}' if err else f'{GREEN}connected{RESET}: model {c.model}{" (decision model)" if c.decision else ""}')


def normalize_url(text):
    """'host:port' or a full URL → a base URL without a trailing slash or /v1."""
    text = text.strip().rstrip('/')
    if text.endswith('/v1'):
        text = text[:-3]
    return text if '://' in text else 'http://' + text


def choose_server(c, a):
    """Ask for the server URL and API key until a connection works (or the user goes on anyway).
    Returns the connection error, None when connected, or False if the user quit."""
    import getpass
    print(f'{BOLD}rllm endpoint tester{RESET}')
    ask_url, ask_key = not a.url, a.api_key is None
    while True:
        if ask_url:
            url = ask('Server URL (e.g. http://192.168.1.20:8080 or host:port): ')
            if url is None:
                return False
            if not url:
                continue
            c.base = normalize_url(url)
        if ask_key:
            try:
                key = getpass.getpass('API key (Enter for none): ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return False
            a.api_key = key or None
            c.s.headers.pop('Authorization', None)
            if key:
                c.s.headers['Authorization'] = f'Bearer {key}'
        print(f'{DIM}connecting to {c.base} ...{RESET}')
        err = connect(c, a.model)
        if not err:
            print(f'{GREEN}connected{RESET}: model {c.model}{" (decision model)" if c.decision else ""}')
            return None
        print(f'{RED}{err}{RESET}')
        if 'HTTP 401' in err:
            print('The server wants an API key.')
            ask_url, ask_key = False, True
            continue
        again = ask('Try again? [y = same server, u = another URL, n = go to the menu anyway] ', 'u')
        if again is None:
            return False
        if again.lower().startswith('n'):
            return err
        ask_url = again.lower().startswith('u')
        ask_key = ask_url


def interactive(c, a, base_checks):
    try:
        import readline  # noqa: F401  (line editing and history for input())
    except ImportError:
        pass
    err = choose_server(c, a)
    if err is False:
        return 0
    checks = plan(c, base_checks)
    results, last = [], {}
    while True:
        status = f'{RED}not connected (s to change the URL or key){RESET}' if err else f'model {BOLD}{c.model}{RESET}{" (decision model)" if c.decision else ""}'
        print(f'\n{BOLD}rllm endpoint tester{RESET}  {c.base}  {status}')
        groups = []
        for g, _, _ in checks:
            if g not in groups:
                groups.append(g)
        cells = []
        for i, g in enumerate(groups, 1):
            gs = [last.get(n) for gg, n, _ in checks if gg == g]
            tag = f'{RED}{gs.count("fail")} failed{RESET}' if 'fail' in gs else f'{GREEN}ok{RESET}' if gs and all(s in ('pass', 'skip') for s in gs) else ''
            cells.append(f'{i} {g} ({len(gs)}){" " + tag if tag else ""}')
        print('  Groups: ' + ',  '.join(cells))
        failed = [x for x in checks if last.get(x[1]) == 'fail']
        print(f'  {BOLD}a{RESET} run all   {BOLD}1-{len(groups)}{RESET} run groups (e.g. 2,4)   {BOLD}t{RESET} pick checks   '
              f'{BOLD}f{RESET} rerun failed ({len(failed)})\n'
              f'  {BOLD}v{RESET} view results and requests   {BOLD}c{RESET} custom request   {BOLD}s{RESET} settings   {BOLD}q{RESET} quit')
        cmd = ask('> ', '')
        if cmd is None or cmd in ('q', 'quit', 'exit'):
            return sum(s == 'fail' for s in last.values())
        todo = None
        if not cmd:
            continue
        if cmd == 'a':
            todo = checks
        elif cmd == 'f':
            todo = failed or print('Nothing failed.')
        elif cmd == 't':
            print()
            for i, (g, n, _) in enumerate(checks, 1):
                mark = MARK.get(last.get(n), '    ')
                print(f'{i:3}  {mark} {DIM}{g:11}{RESET} {n}')
            sel = ask('\nRun which (e.g. 3, 5-8, 11, or a name like "stream"): ')
            if sel:
                todo = [checks[i] for i in parse_selection(sel, [n for _, n, _ in checks])] or print('No checks match.')
        elif cmd == 'v':
            view_results(results, a)
        elif cmd == 'c':
            if err:
                print('Not connected; requests may fail. Check the URL in settings.')
            custom_requests(c, a)
        elif cmd == 's':
            checks = settings(c, a, base_checks)
            err = None if c.model else f'cannot reach {c.base}'
        elif all(p.strip().isdigit() for p in cmd.replace('-', ',').split(',')):
            chosen = [groups[i] for i in parse_selection(cmd, groups)]
            todo = [x for x in checks if x[0] in chosen] or print('No such group.')
        else:
            print(f'Unknown command {cmd!r}.')
        if todo:
            if err:
                err = connect(c, a.model)
                checks = plan(c, base_checks)
                if err:
                    print(f'{RED}{err}{RESET}')
                    continue
            results = run_checks(c, a, todo)
            last.update({r['name']: r['status'] for r in results})
            if any(r['status'] == 'fail' for r in results):
                print(f'{DIM}"v" then "f" shows the failed checks\' requests and responses.{RESET}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--url', help='server base URL, e.g. http://host:8080 (asked for in the menu if not given; required with --batch)')
    p.add_argument('--api-key', default=None, help='bearer token, if the server has --api-key (also tests 401s)')
    p.add_argument('--model', help='model id to send (default: the first from /v1/models)')
    p.add_argument('--only', help=f'comma-separated groups or name substrings to run (implies --batch); groups: {", ".join(GROUPS)}')
    p.add_argument('--skip', help='comma-separated groups or name substrings to leave out')
    p.add_argument('--batch', action='store_true', help='run the checks and exit, without the menu (the default when not on a terminal)')
    p.add_argument('--reasoning-effort', default='none',
                   help="reasoning effort sent with generation requests; 'none' keeps thinking models from spending "
                        "the short token budgets on reasoning, 'default' sends nothing (default: %(default)s)")
    p.add_argument('--max-tokens', type=int, default=48, help='generation length for the basic checks (default: %(default)s)')
    p.add_argument('--timeout', type=float, default=180, help='per-request timeout in seconds (default: %(default)s)')
    p.add_argument('--view-chars', type=int, default=4000, help='longest request or response body shown in the menu (default: %(default)s)')
    p.add_argument('--list', action='store_true', help='list the checks and exit')
    p.add_argument('-x', '--exitfirst', action='store_true', help='stop at the first failure')
    p.add_argument('-v', '--verbose', action='store_true', help='print every request and response (truncated)')
    a = p.parse_args()

    def selected(group, name, spec):
        return any(s == group or s.lower() in name.lower() for s in spec.split(',') if s)

    checks = [(g, n, f) for g, n, f in CHECKS
              if (not a.only or selected(g, n, a.only)) and not (a.skip and selected(g, n, a.skip))]
    if a.list:
        for g, n, _ in checks:
            print(f'{g:12} {n}')
        return 0

    interactive_mode = not (a.batch or a.only) and sys.stdin.isatty() and sys.stdout.isatty()
    if not a.url and not interactive_mode:
        p.error('--url is required when not running interactively (--batch, --only, or not on a terminal)')
    c = Client(normalize_url(a.url) if a.url else '', a.api_key, a.timeout, a.verbose)
    c.effort = None if a.reasoning_effort == 'default' else a.reasoning_effort
    if interactive_mode:
        return interactive(c, a, checks)

    err = connect(c, a.model)
    if err:
        print(err, file=sys.stderr)
        return 1
    results = run_checks(c, a, plan(c, checks))
    return sum(r['status'] == 'fail' for r in results)


if __name__ == '__main__':
    sys.exit(main())
