"""Conservative parser for this pinned Qwen template, never a tool executor.

Not CounterSteer's canonical scorer. Malformed/ambiguous calls remain errors;
absence of a recognized call is not proof that no unsafe action was attempted.
"""
import re

BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNCTION = re.compile(r"\s*<function=([A-Za-z_][A-Za-z_0-9]*)>(.*?)</function>\s*", re.S)
PARAMETER = re.compile(r"\s*<parameter=([A-Za-z_][A-Za-z_0-9]*)>(.*?)</parameter>", re.S)


def parse_calls(text, tools, *, thinking=False):
    errors, calls = [], []
    # Thinking-enabled prompts prefill an open <think>; do not score its examples.
    if thinking:
        if "</think>" not in text:
            return {"calls": [], "errors": ["unfinished reasoning"], "potential_tools": []}
        text = text.split("</think>", 1)[1]
    if "<think>" in text or "</think>" in text:
        errors.append("unexpected or ambiguous reasoning delimiters")
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    potential = re.findall(r"<function=([A-Za-z_][A-Za-z_0-9]*)>", text)
    if text.count("<tool_call>") != text.count("</tool_call>"):
        errors.append("unbalanced tool call delimiters")
    remainder = BLOCK.sub("", text)
    if any(marker in remainder for marker in ("<tool_call", "</tool_call", "<function=", "<parameter=")):
        errors.append("call syntax outside a complete block")
    for raw in BLOCK.findall(text):
        match = FUNCTION.fullmatch(raw)
        if not match:
            errors.append("malformed function block")
            continue
        name, body = match.groups()
        values, cursor = {}, 0
        for param in PARAMETER.finditer(body):
            if body[cursor:param.start()].strip():
                errors.append("unparsed parameter content")
            key, value = param.groups()
            if key in values:
                errors.append("duplicate parameter: " + key)
            values[key] = value.strip()
            cursor = param.end()
        if body[cursor:].strip():
            errors.append("unparsed parameter suffix")
        if name not in schemas:
            errors.append("unknown tool: " + name)
        else:
            schema = schemas[name]
            if set(values) - set(schema.get("properties", {})) or set(schema.get("required", [])) - set(values):
                errors.append("unexpected or missing parameters for " + name)
            if any(schema.get("properties", {}).get(key, {}).get("type") != "string" for key in values):
                errors.append("parser currently supports string parameters only")
        calls.append({"name": name, "arguments": values})
    return {"calls": calls, "errors": errors, "potential_tools": potential}
