"""
code_block_validator.py
自动检测和修复 Markdown 中代码块的语法合法性
"""

import re
import ast
import json
import html
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CodeBlockStatus(Enum):
    VALID = "valid"
    FIXED = "fixed"
    INVALID = "invalid"
    UNKNOWN_LANG = "unknown_lang"


@dataclass
class CodeBlock:
    raw: str
    language: Optional[str]
    code: str
    start_pos: int
    end_pos: int
    indent: str


@dataclass
class ValidationResult:
    block: CodeBlock
    status: CodeBlockStatus
    fixed_code: Optional[str] = None
    error_message: Optional[str] = None
    detected_lang: Optional[str] = None


LANGUAGE_HEURISTICS = [
    (r'^\s*(def\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import|'
     r'if\s+__name__\s*==\s*["\']__main__["\']|class\s+\w+.*:)', 'python'),
    (r'^\s*(const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=|'
     r'function\s+\w+\s*\(|=>|console\.(log|error|warn))', 'javascript'),
    (r'^\s*(interface\s+\w+|type\s+\w+\s*=|:\s*(string|number|boolean)\s*[,;=])', 'typescript'),
    (r'^\s*(public\s+class|private\s+|protected\s+|System\.out\.print)', 'java'),
    (r'^\s*(#include\s*<|int\s+main\s*\(|printf\s*\(|cout\s*<<|'
     r'uint\d*_t\s+|int\d*_t\s+|size_t\s+|uintptr_t\s+|'
     r'struct\s+\w+\s*\{|typedef\s+|enum\s+\w*\s*\{|'
     r'\*\s*\(|\w+\s*\*\s*\w+|\w+\s*\*\)|'
     r'//\s|/\*\s|\*/|NULL|nullptr|sizeof\s*\(|'
     r'malloc\s*\(|free\s*\(|memset\s*\(|memcpy\s*\()', 'cpp'),
    (r'^\s*(package\s+\w+|func\s+\w+\s*\(|import\s*\()', 'go'),
    (r'^\s*(fn\s+\w+\s*\(|let\s+mut\s+|use\s+\w+::|impl\s+)', 'rust'),
    (r'^\s*(#!/bin/(bash|sh|zsh)|echo\s+|export\s+\w+=|if\s+\[)', 'bash'),
    (r'^\s*(SELECT\s+|INSERT\s+INTO|UPDATE\s+|CREATE\s+TABLE|DELETE\s+FROM)', 'sql'),
    (r'^\s*<(!DOCTYPE\s+html|html|div|body|head)', 'html'),
    (r'^\s*([.#]\w+\s*\{|@\w+\s+|margin\s*:|padding\s*:|color\s*:)', 'css'),
    (r'^\s*[\{\[]', 'json'),
    (r'^\s*\w[^:\n]*:\s*\S|^\s*-\s+\S', 'yaml'),
    (r'^\s*#{1,6}\s', 'markdown'),
]


def detect_language(code: str) -> str:
    code_stripped = code.strip()
    if not code_stripped:
        return 'text'

    try:
        json.loads(code_stripped)
        return 'json'
    except (json.JSONDecodeError, ValueError):
        pass

    for pattern, lang in LANGUAGE_HEURISTICS:
        if re.search(pattern, code_stripped, re.MULTILINE | re.IGNORECASE):
            return lang

    c_indicators = [
        r'\b(?:uint8_t|uint16_t|uint32_t|uint64_t|int8_t|int16_t|int32_t|int64_t|size_t|ssize_t|uintptr_t|intptr_t)\b',
        r'\b(?:struct|union|enum|typedef|volatile|register|extern|static|const)\s+\w+',
        r'\*\s*\w+\s*[=;]',
        r'\w+\s*\*\s*\w+',
        r'\b(?:malloc|free|memset|memcpy|strcpy|strlen|sprintf|fprintf)\s*\(',
        r'\b(?:NULL|nullptr|true|false|bool)\b',
        r'//\s.*$',
        r'/\*.*\*/',
    ]
    for pattern in c_indicators:
        if re.search(pattern, code_stripped, re.MULTILINE):
            return 'cpp'

    return 'text'


def _contains_html_special_chars(code: str) -> bool:
    html_pattern = re.compile(r'<[^\s>][^>]*>')
    return bool(html_pattern.search(code))


CODE_FENCE_RE = re.compile(
    r'(?ms)^([ \t]{0,3})```([^\n]*)\n(.*?)(\n[ \t]{0,3}```)[ \t]*\r?\n?'
)

INCOMPLETE_FENCE_RE = re.compile(
    r'(?ms)^([ \t]{0,3})```([^\n]*)\n(.*)$'
)


def extract_code_blocks(markdown: str) -> List[CodeBlock]:
    blocks = []
    for match in CODE_FENCE_RE.finditer(markdown):
        indent, info, code, closing = match.groups()
        lang = info.strip().split()[0] if info.strip() else None
        blocks.append(CodeBlock(
            raw=match.group(0),
            language=lang,
            code=code,
            start_pos=match.start(),
            end_pos=match.end(),
            indent=indent
        ))

    last_end = 0
    if blocks:
        last_end = max(b.end_pos for b in blocks)
    remaining = markdown[last_end:]
    incomplete_match = INCOMPLETE_FENCE_RE.search(remaining)
    if incomplete_match:
        indent, info, code = incomplete_match.groups()
        lang = info.strip().split()[0] if info.strip() else None
        start_pos = last_end + incomplete_match.start()
        end_pos = last_end + incomplete_match.end()
        blocks.append(CodeBlock(
            raw=incomplete_match.group(0),
            language=lang,
            code=code,
            start_pos=start_pos,
            end_pos=end_pos,
            indent=indent
        ))
    return blocks


class PythonSyntaxValidator:
    @staticmethod
    def check(code: str) -> Tuple[bool, Optional[str]]:
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"
        except ValueError as e:
            return False, f"ValueError: {e}"

    @staticmethod
    def fix_common_issues(code: str) -> Tuple[str, bool]:
        original = code
        fixed = code
        fixed = PythonSyntaxValidator._fix_unbalanced_brackets(fixed)
        fixed = PythonSyntaxValidator._fix_unclosed_quotes(fixed)
        fixed = PythonSyntaxValidator._fix_missing_colons(fixed)
        fixed = PythonSyntaxValidator._fix_indentation(fixed)
        is_valid, _ = PythonSyntaxValidator.check(fixed)
        return fixed, is_valid and fixed != original

    @staticmethod
    def _fix_unbalanced_brackets(code: str) -> str:
        pairs = {'(': ')', '[': ']', '{': '}'}
        stack = []
        lines = code.split('\n')
        for i, line in enumerate(lines):
            for char in line:
                if char in pairs:
                    stack.append((char, i))
                elif char in pairs.values():
                    if stack and pairs[stack[-1][0]] == char:
                        stack.pop()
        if stack:
            for open_char, line_idx in reversed(stack):
                closing = pairs[open_char]
                if line_idx < len(lines):
                    lines[line_idx] = lines[line_idx].rstrip() + closing
                else:
                    lines.append(closing)
        return '\n'.join(lines)

    @staticmethod
    def _fix_unclosed_quotes(code: str) -> str:
        lines = code.split('\n')
        in_triple_single = False
        in_triple_double = False
        for i, line in enumerate(lines):
            triple_single = line.count("'''")
            triple_double = line.count('"""')
            if triple_single % 2 == 1:
                in_triple_single = not in_triple_single
            if triple_double % 2 == 1:
                in_triple_double = not in_triple_double
        if in_triple_single:
            lines.append("'''")
        if in_triple_double:
            lines.append('"""')
        return '\n'.join(lines)

    @staticmethod
    def _fix_missing_colons(code: str) -> str:
        patterns = [
            (r'^(def\s+\w+\s*\([^)]*\))$', r'\1:'),
            (r'^(class\s+\w+[^:]*?)$', r'\1:'),
            (r'^(if\s+.+)$', r'\1:'),
            (r'^(elif\s+.+)$', r'\1:'),
            (r'^(else\s*)$', r'\1:'),
            (r'^(for\s+.+)$', r'\1:'),
            (r'^(while\s+.+)$', r'\1:'),
            (r'^(try\s*)$', r'\1:'),
            (r'^(except\s*.*)$', r'\1:'),
            (r'^(finally\s*)$', r'\1:'),
            (r'^(with\s+.+)$', r'\1:'),
        ]
        lines = code.split('\n')
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            for pattern, replacement in patterns:
                if re.match(pattern, stripped) and not stripped.endswith(':'):
                    if i + 1 < len(lines) and lines[i + 1].startswith('    '):
                        continue
                    lines[i] = re.sub(pattern, replacement, stripped)
                    break
        return '\n'.join(lines)

    @staticmethod
    def _fix_indentation(code: str) -> str:
        lines = code.split('\n')
        fixed_lines = []
        prev_indent = 0
        for line in lines:
            stripped = line.lstrip()
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue
            current_indent = len(line) - len(stripped)
            if current_indent > prev_indent + 4:
                expected_indent = prev_indent + 4 if prev_indent > 0 else 0
                fixed_lines.append(' ' * expected_indent + stripped)
                current_indent = expected_indent
            else:
                fixed_lines.append(line)
            prev_indent = current_indent
        return '\n'.join(fixed_lines)


class GenericCodeValidator:
    @staticmethod
    def check(code: str, lang: str) -> Tuple[bool, Optional[str]]:
        brackets_ok, bracket_err = GenericCodeValidator._check_brackets(code)
        if not brackets_ok:
            return False, bracket_err
        quotes_ok, quote_err = GenericCodeValidator._check_quotes(code)
        if not quotes_ok:
            return False, quote_err
        if lang in ('json', 'javascript'):
            try:
                json.loads(code)
            except json.JSONDecodeError:
                pass
        return True, None

    @staticmethod
    def _check_brackets(code: str) -> Tuple[bool, Optional[str]]:
        pairs = {'(': ')', '[': ']', '{': '}'}
        stack = []
        in_string = False
        string_char = None
        escaped = False
        for i, char in enumerate(code):
            if escaped:
                escaped = False
                continue
            if char == '\\':
                escaped = True
                continue
            if char in ('"', "'"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                continue
            if in_string:
                continue
            if char == '/' and i + 1 < len(code) and code[i + 1] == '/':
                break
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False, f"Unmatched closing bracket '{char}' at position {i}"
                if pairs[stack[-1]] != char:
                    return False, f'Mismatched brackets at position {i}'
                stack.pop()
        if stack:
            return False, f'Unclosed brackets: {stack}'
        return True, None

    @staticmethod
    def _check_quotes(code: str) -> Tuple[bool, Optional[str]]:
        single_quotes = code.count("'") - code.count("\\'")
        double_quotes = code.count('"') - code.count('\\"')
        if single_quotes % 2 != 0:
            return False, "Unbalanced single quotes"
        if double_quotes % 2 != 0:
            return False, "Unbalanced double quotes"
        return True, None


def validate_and_fix_code_blocks(markdown: str) -> Tuple[str, List[ValidationResult]]:
    blocks = extract_code_blocks(markdown)
    if not blocks:
        return markdown, []
    results: List[ValidationResult] = []
    replacements: List[Tuple[int, int, str]] = []
    for block in blocks:
        result = _validate_single_block(block)
        results.append(result)
        if result.status in (CodeBlockStatus.FIXED, CodeBlockStatus.INVALID, CodeBlockStatus.UNKNOWN_LANG):
            new_block = _rebuild_code_block(result)
            replacements.append((block.start_pos, block.end_pos, new_block))
    new_markdown = markdown
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        new_markdown = new_markdown[:start] + replacement + new_markdown[end:]
    return new_markdown, results


def _validate_single_block(block: CodeBlock) -> ValidationResult:
    lang = block.language
    code = block.code
    if not lang:
        detected = detect_language(code)
        if detected != 'text':
            block.language = detected
            return ValidationResult(
                block=block,
                status=CodeBlockStatus.UNKNOWN_LANG,
                detected_lang=detected
            )
        lang = 'text'
    if lang == 'text':
        if _contains_html_special_chars(code):
            escaped_code = code.replace('<', '&lt;').replace('>', '&gt;')
            redetected = detect_language(code)
            if redetected != 'text':
                block.language = redetected
                return ValidationResult(
                    block=block,
                    status=CodeBlockStatus.UNKNOWN_LANG,
                    detected_lang=redetected
                )
            block.code = escaped_code
            return ValidationResult(
                block=block,
                status=CodeBlockStatus.FIXED,
                fixed_code=escaped_code,
                error_message="Escaped HTML special characters in text block"
            )
        return ValidationResult(block=block, status=CodeBlockStatus.VALID)
    if lang == 'python':
        is_valid, error = PythonSyntaxValidator.check(code)
        if is_valid:
            return ValidationResult(block=block, status=CodeBlockStatus.VALID)
        fixed_code, fix_success = PythonSyntaxValidator.fix_common_issues(code)
        if fix_success:
            block.code = fixed_code
            return ValidationResult(
                block=block,
                status=CodeBlockStatus.FIXED,
                fixed_code=fixed_code,
                error_message=f"Original: {error}"
            )
        block.language = 'text'
        block.code = code.replace('<', '&lt;').replace('>', '&gt;')
        return ValidationResult(
            block=block,
            status=CodeBlockStatus.INVALID,
            error_message=error
        )
    is_valid, error = GenericCodeValidator.check(code, lang)
    if is_valid:
        return ValidationResult(block=block, status=CodeBlockStatus.VALID)
    fixed = PythonSyntaxValidator._fix_unbalanced_brackets(code)
    fixed_valid, _ = GenericCodeValidator.check(fixed, lang)
    if fixed_valid:
        block.code = fixed
        return ValidationResult(
            block=block,
            status=CodeBlockStatus.FIXED,
            fixed_code=fixed,
            error_message=f"Original: {error}"
        )
    block.language = 'text'
    block.code = code.replace('<', '&lt;').replace('>', '&gt;')
    return ValidationResult(
        block=block,
        status=CodeBlockStatus.INVALID,
        error_message=error
    )


def _rebuild_code_block(result: ValidationResult) -> str:
    lang = result.block.language or 'text'
    code = result.fixed_code or result.block.code
    return f"{result.block.indent}```{lang}\n{code}\n{result.block.indent}```\n"


def sanitize_llm_output(text: str, log_results: bool = True) -> str:
    new_text, results = validate_and_fix_code_blocks(text)
    if log_results and results:
        for r in results:
            if r.status == CodeBlockStatus.FIXED:
                logger.info(f"[CodeBlock] FIXED {r.block.language}: {r.error_message}")
            elif r.status == CodeBlockStatus.INVALID:
                logger.warning(f"[CodeBlock] INVALID -> text: {r.error_message}")
            elif r.status == CodeBlockStatus.UNKNOWN_LANG:
                logger.info(f"[CodeBlock] DETECTED lang={r.detected_lang}")
    return new_text


def escape_html_in_code(text: str) -> str:
    blocks = extract_code_blocks(text)
    if not blocks:
        return text
    replacements = []
    for block in blocks:
        escaped_code = block.code.replace('<', '&lt;').replace('>', '&gt;')
        if escaped_code != block.code:
            new_block = f"{block.indent}```{block.language or ''}\n{escaped_code}\n{block.indent}```\n"
            replacements.append((block.start_pos, block.end_pos, new_block))
    new_text = text
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        new_text = new_text[:start] + replacement + new_text[end:]
    return new_text


"""
在 server.py 中使用：

from code_block_validator import sanitize_llm_output, escape_html_in_code

# 在返回 LLM 响应前调用
cleaned_content = sanitize_llm_output(agent_resp.content)
cleaned_content = escape_html_in_code(cleaned_content)
"""