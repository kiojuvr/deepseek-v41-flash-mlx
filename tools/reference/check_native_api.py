#!/usr/bin/env python3
"""Offline validation orchestration only; not part of the API runtime."""
import concurrent.futures
import json
import math
import os
import re
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    run = Path(sys.argv[1])
    port = int(os.environ.get('DSV41_SMOKE_PORT', '18082'))
    timeout = float(os.environ.get('DSV41_CHECK_TIMEOUT', '1800'))
    # Refuse an occupied port before launching the model process.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    env = dict(os.environ, DSV41_NATIVE_MODEL='1', DSV41_BIND=f'127.0.0.1:{port}')
    base = f'http://127.0.0.1:{port}'
    model = env.get('DSV41_MODEL', 'DeepSeek-V4.1-Flash')
    thinking = os.environ.get('DSV41_CHECK_THINKING', 'enabled')
    assert thinking in ('enabled', 'disabled', 'omitted')
    reasoning_effort = os.environ.get('DSV41_CHECK_REASONING_EFFORT')
    assert reasoning_effort in (None, 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')
    max_tokens = int(os.environ.get('DSV41_CHECK_MAX_TOKENS', '4'))
    assert 1 <= max_tokens <= 256
    require_eos = os.environ.get('DSV41_CHECK_REQUIRE_EOS', '0') == '1'
    parity_only = os.environ.get('DSV41_CHECK_PARITY_ONLY', '0') == '1'
    temperature = float(os.environ.get('DSV41_CHECK_TEMPERATURE', '0'))
    seed = int(os.environ.get('DSV41_CHECK_SEED', '0'))
    assert math.isfinite(temperature) and temperature >= 0
    assert 0 <= seed < 2**64
    stop = os.environ.get('DSV41_CHECK_STOP')
    stop_json = os.environ.get('DSV41_CHECK_STOP_JSON')
    assert not (stop is not None and stop_json is not None), 'set only one stop environment variable'
    if stop_json is not None:
        stop = json.loads(stop_json)
    if stop == '':
        stop = None
    stop_values = [] if stop is None else ([stop] if isinstance(stop, str) else stop)
    assert isinstance(stop_values, list) and all(isinstance(value, str) and value for value in stop_values)
    assert stop is None or stop_values, 'stop array must not be empty'
    request = {'model': model, 'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': max_tokens, 'temperature': temperature, 'seed': seed}
    if thinking != 'omitted':
        request['thinking'] = {'type': thinking}
    if reasoning_effort is not None:
        request['reasoning_effort'] = reasoning_effort
    if stop is not None:
        request['stop'] = stop
    (run / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path, payload=None, deadline=timeout):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with opener.open(req, timeout=deadline) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def save(name, result):
        (run / f'{name}.json').write_text(json.dumps({'status': result[0], 'body': result[1]}, ensure_ascii=False, indent=2) + '\n')
        return result

    with (run / 'server.log').open('w') as log:
        process = subprocess.Popen(['server/target/debug/dsv41-developer-server'], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited during startup: {process.returncode}')
                try:
                    status, health = call('/health', deadline=2)
                    if status == 200 and health['runtime'] == 'native-model':
                        save('health', (status, health))
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('model startup timed out')
            # A short concurrent health request must not wait for native prefill.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(call, '/v1/chat/completions', request)
                time.sleep(0.25)
                assert save('health-during-generation', call('/health', deadline=5))[0] == 200
                first = save('completion-1', future.result())
            second = save('completion-2', call('/v1/chat/completions', request))
            for status, response in (first, second):
                assert status == 200, response
                assert response['object'] == 'chat.completion' and response['model'] == model
                choice = response['choices'][0]
                assert choice['finish_reason'] in ('length', 'stop')
                assert choice['message']['role'] == 'assistant'
                assert choice['message'].get('content') or choice['message'].get('reasoning_content')
                if require_eos:
                    assert choice['finish_reason'] == 'stop', 'EOS was not reached within the requested budget'
                if stop is not None:
                    assert choice['finish_reason'] == 'stop', 'local stop sequence was not reached within the requested budget'
                    assert all(value not in (choice['message'].get('content') or '') for value in stop_values)
                effective_thinking = thinking == 'enabled' or (thinking == 'omitted' and reasoning_effort != 'none')
                if not effective_thinking:
                    assert choice['message'].get('content') and not choice['message'].get('reasoning_content')
                usage = response['usage']
                assert usage['prompt_tokens'] > 0 and 0 < usage['completion_tokens'] <= max_tokens
                if choice['finish_reason'] == 'length':
                    assert usage['completion_tokens'] == max_tokens
                assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
            assert first[1]['id'] != second[1]['id']
            assert first[1]['choices'] == second[1]['choices']
            assert first[1]['usage'] == second[1]['usage']
            busy_retries = {}
            def stream(payload, name, disconnect=False, include_usage=True):
                req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(dict(payload, stream=True, stream_options={'include_usage': include_usage})).encode(), headers={'Content-Type': 'application/json'})
                events = []
                busy_deadline = time.monotonic() + timeout
                retries = 0
                while True:
                    try:
                        response = opener.open(req, timeout=timeout)
                        break
                    except urllib.error.HTTPError as error:
                        if error.code != 503 or time.monotonic() >= busy_deadline:
                            raise
                        # A local stop can finish the client stream just before
                        # native cancellation releases the admission slot.
                        error.read()
                        retries += 1
                        time.sleep(0.2)
                busy_retries[name] = retries
                with response:
                    assert response.status == 200 and response.headers.get_content_type() == 'text/event-stream'
                    with (run / (name + '.sse')).open('w') as record:
                        for raw in response:
                            line = raw.decode()
                            record.write(line)
                            record.flush()
                            if not line.startswith('data:'):
                                continue
                            data = line[5:].strip()
                            if data == '[DONE]':
                                assert not disconnect, 'generation ended before disconnect test'
                                return events
                            event = json.loads(data)
                            assert 'error' not in event, event
                            events.append(event)
                            if disconnect and any(c['delta'].get('reasoning_content') or c['delta'].get('content') for c in event.get('choices', [])):
                                return events
                raise AssertionError('SSE ended without required terminal/output')

            events = stream(request, 'completion-stream')
            reasoning = ''.join(c['delta'].get('reasoning_content') or '' for e in events for c in e.get('choices', []))
            content = ''.join(c['delta'].get('content') or '' for e in events for c in e.get('choices', []))
            finishes = [c['finish_reason'] for e in events for c in e.get('choices', []) if c.get('finish_reason')]
            message = first[1]['choices'][0]['message']
            assert reasoning == (message.get('reasoning_content') or '') and content == (message.get('content') or '')
            assert finishes == [first[1]['choices'][0]['finish_reason']]
            usages = [e['usage'] for e in events if e.get('usage')]
            assert len(usages) == 1 and usages[0]['completion_tokens'] == first[1]['usage']['completion_tokens'] and usages[0]['total_tokens'] == first[1]['usage']['total_tokens']
            assert events[-1]['choices'] == [] and events[-1].get('usage')
            assert all(e.get('usage') is None for e in events[:-1])
            assert events[-2]['choices'][0]['finish_reason'] == finishes[0]
            if parity_only:
                # A decoded local stop lets SSE finish as soon as the parser has
                # requested cancellation. The native owner can still be completing
                # its current cooperative step, so wait for its terminal record
                # instead of racing it and killing the server in finally.
                settle_deadline = time.monotonic() + timeout
                while True:
                    server_text = (run / 'server.log').read_text()
                    option_records = re.findall(r'native request=\d+ terminal=\w+ committed=\d+ budget=' + str(max_tokens) + r' temperature=([^ ]+) seed=(\d+)', server_text)
                    if len(option_records) >= 3:
                        break
                    if process.poll() is not None:
                        raise RuntimeError(f'server exited before native terminal record: {process.returncode}')
                    if time.monotonic() >= settle_deadline:
                        raise TimeoutError('native terminal record missing after SSE completion')
                    time.sleep(0.2)
                assert all(math.isclose(float(temp), temperature, rel_tol=0, abs_tol=1e-7) and int(recorded_seed) == seed for temp, recorded_seed in option_records[:3]), 'native request options changed before bridge submission'
                local_stops = []
                if stop is not None:
                    local_stops = re.findall(r'native request=\d+ terminal=(\w+) committed=(\d+) budget=' + str(max_tokens), server_text)
                    expected = first[1]['usage']['completion_tokens']
                    assert len(local_stops) >= 3 and all(reason == 'cancelled' and int(count) == expected for reason, count in local_stops[:3]), 'parity-only local-stop cancellations were not recorded at the API usage boundary'
                (run / 'result.json').write_text(json.dumps({'passed': True, 'thinking': thinking, 'reasoning_effort': reasoning_effort, 'max_tokens': max_tokens, 'temperature': temperature, 'seed': seed, 'stop': stop, 'parity_only': True, 'native_option_records': option_records[:3], 'local_stop_records': local_stops[:3], 'busy_retries': busy_retries, 'scope': 'native nonstream repeatability and HTTP/SSE parity for request temperature/seed and optional request-local stop, with pre-bridge option records; not official RNG, disconnect, M2, or 256K qualification'}, indent=2, ensure_ascii=False) + '\n')
                print('PASS: native request repeatability and HTTP/SSE parity', flush=True)
                return
            no_usage = stream(request, 'completion-stream-no-usage', include_usage=False)
            assert all('usage' not in e for e in no_usage)
            no_usage_reasoning = ''.join(c['delta'].get('reasoning_content') or '' for e in no_usage for c in e.get('choices', []))
            no_usage_content = ''.join(c['delta'].get('content') or '' for e in no_usage for c in e.get('choices', []))
            assert no_usage_reasoning == reasoning and no_usage_content == content
            disconnect_request = dict(request, max_tokens=256, thinking={'type': 'enabled'})
            disconnect_request.pop('stop', None)
            stream(disconnect_request, 'disconnect', disconnect=True)
            # Retry only BUSY while the cancelled native step returns and releases admission.
            cancel_deadline = time.monotonic() + timeout
            while True:
                resumed = call('/v1/chat/completions', request)
                if resumed[0] != 503:
                    break
                assert time.monotonic() < cancel_deadline, 'admission not released after disconnect'
                time.sleep(0.2)
            save('after-disconnect', resumed)
            assert resumed[0] == 200 and resumed[1]['choices'] == first[1]['choices'] and resumed[1]['usage'] == first[1]['usage']
            if stop is not None:
                local_stops = re.findall(r'native request=\d+ terminal=(\w+) committed=(\d+) budget=' + str(max_tokens), (run / 'server.log').read_text())
                expected = first[1]['usage']['completion_tokens']
                assert len(local_stops) >= 5 and all(reason == 'cancelled' and int(count) == expected for reason, count in local_stops[:5]), 'incremental nonstream/SSE local-stop cancellations were not recorded at API usage boundary'
            cancellations = re.findall(r'native request=\d+ terminal=cancelled committed=(\d+) budget=256', (run / 'server.log').read_text())
            assert cancellations and all(int(count) < 256 for count in cancellations), 'no native cancellation before budget recorded'
            (run / 'result.json').write_text(json.dumps({'passed': True, 'thinking': thinking, 'reasoning_effort': reasoning_effort, 'max_tokens': max_tokens, 'temperature': temperature, 'seed': seed, 'require_eos': require_eos, 'stop': stop, 'parity_only': False, 'busy_retries': busy_retries, 'scope': 'native incremental nonstream/SSE parity, local stop when configured, disconnect recovery and health; not official RNG, M2, or 256K qualification'}, indent=2) + '\n')
            print('PASS: native HTTP/SSE parity, recipe usage, disconnect recovery and responsive health', flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
