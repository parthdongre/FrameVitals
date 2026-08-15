# Security Policy

## Supported versions

FrameVitals is currently pre-alpha. Security fixes are applied to the latest development version only.

## Reporting a vulnerability

Please do not open a public GitHub issue for a vulnerability that could put users, data, or systems at risk.

Use a private GitHub security advisory for this repository when that option is available. If private reporting is not available, contact the maintainer through the GitHub profile associated with this repository and clearly mark the message as a security report.

A useful report includes:

- The affected component and version/commit
- Reproduction steps or a minimal proof of concept
- The expected security impact
- Any known prerequisites or limitations
- A suggested mitigation, if you have one

Please avoid accessing data that is not yours, disrupting services, or publishing exploit details before a fix is available.

## Security-sensitive areas

Extra care is expected around:

- Dataset upload and file parsing
- Path handling and generated files
- The safe pandas expression evaluator
- LLM/tool execution boundaries
- External AI providers and credentials
- Deserialization and model artifacts
- Web endpoints that return user-generated analysis data
