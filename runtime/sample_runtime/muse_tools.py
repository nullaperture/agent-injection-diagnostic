"""ATEM/channel parser for pinned Muse template. Never execute any proposal."""
import re

CHANNEL = re.compile(r'<\|start\|>assistant(?: to=([^\s<]+))?<\|message\|>(.*?)(?:<\|eom\|>|<\|eot\|>|<\|end_of_text\|>|$)', re.S)
BLOCK = re.compile(r'<atem:function_calls>(.*?)</atem:function_calls>', re.S)
INVOKE = re.compile(r'\s*<atem:invoke name="([A-Za-z_][A-Za-z_0-9.]*)">(.*?)</atem:invoke>\s*', re.S)
PARAM = re.compile(r'\s*<atem:parameter name="([A-Za-z_][A-Za-z_0-9]*)">(.*?)</atem:parameter>', re.S)

def parse_calls(text, tools, **_):
    text = '<|start|>assistant' + text
    schemas = {tool['function']['name']: tool['function']['parameters'] for tool in tools}
    calls, errors, potential, ignored = [], [], [], 0
    cursor = 0
    for channel in CHANNEL.finditer(text):
        if text[cursor:channel.start()].strip():
            errors.append('unparsed channel prefix')
        cursor = channel.end()
        recipient, body = channel.groups()
        if recipient == 'self':
            ignored += 1
            continue
        if recipient and recipient != 'user':
            potential.append(recipient)
            if recipient not in schemas:
                errors.append('unknown recipient: ' + recipient)
        found = BLOCK.findall(body)
        remainder = BLOCK.sub('', body)
        if '<atem:' in remainder or '</atem:' in remainder:
            errors.append('incomplete ATEM syntax')
        if recipient not in (None, 'user') and not found:
            errors.append('tool recipient without ATEM call')
        for block in found:
            end = 0
            for invoke in INVOKE.finditer(block):
                if block[end:invoke.start()].strip():
                    errors.append('unparsed invocation prefix')
                end = invoke.end()
                name, params = invoke.groups()
                potential.append(name)
                if recipient not in (None, 'user', name):
                    errors.append('recipient/call mismatch')
                values, offset = {}, 0
                for param in PARAM.finditer(params):
                    if params[offset:param.start()].strip():
                        errors.append('unparsed parameter prefix')
                    key, value = param.groups()
                    if key in values:
                        errors.append('duplicate parameter: ' + key)
                    values[key] = value
                    offset = param.end()
                if params[offset:].strip():
                    errors.append('unparsed parameter suffix')
                schema = schemas.get(name)
                if not schema:
                    errors.append('unknown tool: ' + name)
                elif (set(values) - set(schema.get('properties', {})) or
                      set(schema.get('required', [])) - set(values) or
                      any(schema['properties'].get(key, {}).get('type') != 'string' for key in values)):
                    errors.append('invalid parameters for ' + name)
                calls.append({'name': name, 'arguments': values})
            if block[end:].strip() or not end:
                errors.append('malformed invocation block')
    if text[cursor:].strip():
        errors.append('unfinished or unparsed channel')
        # Preserve suspicious native recipients even when the body is malformed.
        potential.extend(re.findall(r'assistant to=([A-Za-z_][A-Za-z_0-9.]*)', text[cursor:]))
    return {'calls': calls, 'errors': errors, 'potential_tools': list(dict.fromkeys(potential)),
            'reasoning_channels_ignored': ignored}
