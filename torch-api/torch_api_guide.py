# torch_api_guide.py —— 本地 torch API 速查 + 真机演示
# 用法:
#   python torch_api_guide.py            # 交互: 输入 API 名 (list/quit)
#   python torch_api_guide.py matmul     # 直接查某个 API
#   python torch_api_guide.py --list     # 列出知识库里所有可查的 API
#   python torch_api_guide.py matmul --no-open   # 只生成文件, 不弹浏览器
#
# 结果会生成 HTML(+同名的 .md) 并自动用浏览器打开。内容来源: 自己整理的知识库
# (torch_api_kb.py) + 本机 torch 的真实运行输出 + 签名/文档兜底 —— 不跳官网。

import io
import os
import sys
import contextlib
import difflib
import inspect
import webbrowser
from html import escape
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch_api_kb import KB

OUT_DIR = Path(__file__).resolve().parent / "results" / "torch_api"

_CSS = """
body{font-family:-apple-system,'PingFang SC',Helvetica,Arial,sans-serif;
     max-width:920px;margin:24px auto;padding:0 20px;line-height:1.7;color:#222}
h1{border-bottom:3px solid #4a6;padding-bottom:8px}
h2{color:#345;border-left:5px solid #69c;padding-left:10px;margin-top:28px}
.tag{display:inline-block;background:#eef;color:#45c;border-radius:4px;
     padding:2px 10px;font-size:13px;margin-left:8px}
table{border-collapse:collapse;width:100%;margin:8px 0}
th,td{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:14px}
th{background:#f4f6fa}
pre{background:#0f1626;color:#d7e0ee;padding:12px;border-radius:8px;
    overflow-x:auto;font-size:13px;line-height:1.5}
pre.out{background:#f4f9f4;color:#1d3;border:1px solid #bec;color:#14532d}
.note{color:#666;font-size:13px}
.src{color:#999;font-size:12px;margin-top:4px}
"""


def _norm(s):
    return "".join(s.lower().split())


def _matches(entry, q):
    if q in entry["title"].lower():
        return True
    return any(q in _norm(a) for a in entry["keys"]) or any(
        _norm(a) in q for a in entry["keys"])


def find(q):
    """返回 (命中条目或候选列表, 是否精确命中, 是否多个候选)。q 已归一化。"""
    qn = _norm(q)
    for e in KB:
        for a in e["keys"]:
            if _norm(a) == qn:
                return e, True, False
    cands = [e for e in KB if _matches(e, qn)]
    return cands, False, len(cands) > 1


def run_example(code):
    buf = io.StringIO()
    err = None
    with contextlib.redirect_stdout(buf):
        try:
            exec(compile(code, "<api_example>", "exec"), {"torch": torch, "nn": nn})
        except Exception as e:  # noqa: BLE001 示例代码兜底，不崩溃
            err = f"{type(e).__name__}: {e}"
    return buf.getvalue(), err


def render(entry, output, err):
    title = entry["title"]
    esc = lambda s: escape(str(s))
    rows = "".join(
        f"<tr><td><code>{esc(n)}</code></td><td>{esc(d)}</td></tr>"
        for n, d in entry["params"])
    notes = "".join(f"<li>{esc(n)}</li>" for n in entry["notes"]) or "<li>—</li>"
    err_html = f"<pre class='out' style='color:#b00'>示例报错: {esc(err)}</pre>" if err else ""

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{esc(title)} · torch 速查</title><style>{_CSS}</style></head><body>
<h1>{esc(title)} <span class="tag">{esc(entry['cat'])}</span></h1>
<p>{esc(entry['desc'])}</p>
<h2>参数</h2>
<table><tr><th style="width:30%">参数</th><th>说明</th></tr>{rows}</table>
<h2>要点 / 坑</h2><ul>{notes}</ul>
<h2>示例（本机 torch 真实运行）</h2>
<pre>{esc(entry['example'])}</pre>
<pre class="out">{esc(output) if output else '(无输出)'}</pre>{err_html}
<p class="src">来源: 自建知识库 + 本机 torch 执行结果（非官网）· 生成于 {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}</p>
</body></html>"""

    md = f"""# {title}（{entry['cat']}）

{entry['desc']}

## 参数
| 参数 | 说明 |
|---|---|
""" + "\n".join(f"| `{n}` | {d} |" for n, d in entry["params"]) + """

## 要点 / 坑
""" + "\n".join(f"- {n}" for n in entry["notes"]) + f"""

## 示例（本机 torch 真实运行）
```python
{entry['example']}
```

```text
{output if output else '(无输出)'}
```
"""
    return html, md


def _obj_from_name(name):
    """安全地把 'torch.matmul'/'nn.Linear'/'F.softmax' 等解析成本地对象（不 eval 任意代码）。"""
    root = {"torch": torch, "nn": nn}
    seg = name.split(".")
    if not seg or seg[0] not in root:
        return None, None
    obj = root[seg[0]]
    try:
        for s in seg[1:]:
            obj = getattr(obj, s)
    except AttributeError:
        return None, None
    if inspect.ismodule(obj):
        return None, None
    return obj, name


def render_fallback(name):
    obj, full = _obj_from_name(name)
    if obj is None:
        return None
    try:
        sig = str(inspect.signature(obj))
    except (ValueError, TypeError):
        sig = "(签名不可用)"
    doc = (inspect.getdoc(obj) or "").strip().splitlines()
    shown = "\n".join(doc[:14])
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{escape(name)} · torch 本地文档</title><style>{_CSS}</style></head><body>
<h1>{escape(name)} <span class="tag">本地未收录·兜底显示</span></h1>
<p>这个 API 还没写进知识库，下面展示 <b>本机安装的 torch</b> 自带的签名与文档开头
（不是跳官网）。想加自己的中文讲解和例子，编辑 <code>torch_api_kb.py</code> 照着加一条即可。</p>
<h2>签名</h2><pre>{escape(sig)}</pre>
<h2>文档（本机 docstring）</h2><pre class="out">{escape(shown) if shown else '(该对象无 docstring)'}</pre>
<p class="src">来源: 本机 torch {torch.__version__}</p>
</body></html>"""
    return html


def pick_interactive(cands, q):
    print(f"「{q}」匹配到多个，选一个：")
    for i, e in enumerate(cands[:8]):
        print(f"  [{i}] {e['title']}  ({e['cat']})")
    idx = input("序号 (默认0): ").strip()
    try:
        return cands[max(0, int(idx))] if idx else cands[0]
    except (ValueError, IndexError):
        return cands[0]


def process(q, open_browser=True):
    res, exact, ambiguous = find(q)
    if exact:
        entry = res
    elif res and not ambiguous:
        entry = res[0]
    elif res and ambiguous:
        entry = pick_interactive(res, q) if sys.stdin.isatty() else res[0]
    else:
        html = render_fallback(q)
        if html is None:
            tips = difflib.get_close_matches(_norm(q), {a for e in KB for a in e["keys"]}, n=4)
            print(f"[!] 知识库里没有「{q}」，也无法解析成本地对象。")
            if tips:
                print("  你是不是想查: " + " / ".join(sorted(set(tips))))
            return False
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"fallback_{_norm(q)[:40]}.html"
        out_path.write_text(html, encoding="utf-8")
        print(f"生成(兜底): {out_path}")
        if open_browser:
            webbrowser.open(out_path.as_uri())
        return True

    # 知识库条目: 真机跑示例
    output, err = run_example(entry["example"])
    html, md = render(entry, output, err)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = _norm(entry["keys"][0]).replace(".", "-")
    html_path = OUT_DIR / f"{slug}.html"
    md_path = OUT_DIR / f"{slug}.md"
    html_path.write_text(html, encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    print(f"[OK] {entry['title']}   ->  {html_path.name} (md 一并生成)")
    print(f"  示例输出片段: {output.strip().splitlines()[0] if output.strip() else '(空)'}")
    if open_browser:
        webbrowser.open(html_path.as_uri())
    return True


def show_list():
    print("知识库已收录（cat / title / 别名）：\n")
    last = None
    for e in KB:
        if e["cat"] != last:
            print(f"\n== {e['cat']} ==")
            last = e["cat"]
        print(f"  {e['title']:34s} 别名: {', '.join(e['keys'][:4])}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    open_browser = "--no-open" not in sys.argv

    if "--list" in sys.argv or args[:1] == ["list"]:
        show_list()
        return

    if args:
        process(args[0], open_browser)
        return

    print("=== 本地 torch API 速查（真机跑示例，不跳官网）===")
    print("输入 API 名回车即查；list=看全部；quit/q=退出\n")
    while True:
        try:
            q = input("torch API? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("quit", "q", "exit"):
            break
        if q == "list":
            show_list()
            continue
        process(q, open_browser)


if __name__ == "__main__":
    main()
