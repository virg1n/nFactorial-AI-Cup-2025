









def clean_json(content):
    if content.startswith("```json"):
        content = content[
            len("```json") :
        ].strip()

    elif content.startswith("```"):
        content = content[
            len("```") :
        ].strip()

    if content.endswith("```"):
        content = content[
            : -len("```")
        ].strip()

    content = "\n".join(line.strip() for line in content.splitlines())

    return content