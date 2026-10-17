"""Render the simple_model member-type graph from this checkout (Python + Pillow).

Run from any directory: python MyDoc/generate_simple_model_map.py
This intentionally follows data members, not types in function signatures or void*.
"""

from __future__ import annotations

import html
import math
import re
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
STEM = "simple_model-struct-map"
SOURCES = [
    "examples/simple/simple-backend.cpp", "include/ggml.h",
    "include/ggml-backend.h", "include/ggml-alloc.h",
    "src/ggml-backend-impl.h", "src/ggml-backend.cpp",
    "src/ggml-alloc.c", "src/ggml-impl.h", "src/ggml.c",
]
TEXT = {p: (ROOT / p).read_text(encoding="utf-8") for p in SOURCES}


def uncomment(s):
    return re.sub(r"/\*.*?\*/|//[^\n]*", lambda m: "\n" * m[0].count("\n"), s, flags=re.S)


CLEAN = {p: uncomment(t) for p, t in TEXT.items()}
ALIASES = {}
for path, s in CLEAN.items():
    for m in re.finditer(r"typedef\s+struct\s+(\w+)\s*\*\s*(\w+)\s*;", s):
        ALIASES[m[2]] = m[1]


@dataclass
class Member:
    code: str
    name: str
    target: str | None
    pointer: bool = False
    conditional: bool = False


@dataclass
class Node:
    name: str
    ident: str
    note: str
    category: str
    path: str
    line: int
    source: str
    members: list[Member]
    x: int = 0
    y: int = 0
    w: int = 1800
    h: int = 0


def definition(name, kind="struct"):
    for path, s in CLEAN.items():
        m = re.search(r"\b" + kind + r"\s+" + re.escape(name) + r"\s*\{", s)
        if not m:
            continue
        start = s.index("{", m.start())
        depth, end = 1, start + 1
        while depth:
            depth += (s[end] == "{") - (s[end] == "}")
            end += 1
        # The comment-free string has different offsets; locate original by line.
        line = s.count("\n", 0, m.start()) + 1
        last = s.count("\n", 0, end) + 1
        source = "\n".join(TEXT[path].splitlines()[line - 1:last])
        return path, line, source, s[start + 1:end - 1]
    raise ValueError(f"Definition not found: {kind} {name}")


def parse_members(body):
    body = re.sub(r"^\s*#.*$", "", body, flags=re.M)
    parts, start, depth = [], 0, 0
    for i, char in enumerate(body):
        depth += (char == "{") - (char == "}")
        if char == ";" and depth == 0:
            parts.append(body[start:i + 1])
            start = i + 1
    members = []
    for part in parts:
        code = re.sub(r"\s+", " ", part).strip()
        if not code:
            continue
        fn = re.search(r"\(\s*\*\s*(\w+)\s*\)", code)
        if fn:
            # Callback parameters/return values are not stored subobjects.
            members.append(Member(code, fn[1], None))
            continue
        if "allocated_tensors" in code:
            members.append(Member("struct { ... } allocated_tensors[1024];", "allocated_tensors", "allocated_tensor_debug", conditional=True))
            continue
        decl = re.sub(r"\{\s*\}", "", code).rstrip("; ")
        match = re.search(r"(\w+)\s*((?:\[[^\]]*\]\s*)*)$", decl)
        if not match:
            raise ValueError(f"Cannot parse: {code}")
        target_match = re.search(r"\bstruct\s+(\w+)", code)
        target = target_match[1] if target_match else None
        if target is None:
            target = next((actual for alias, actual in ALIASES.items() if re.search(r"\b" + alias + r"\b", code)), None)
        pointer = "*" in decl or any(re.search(r"\b" + a + r"\b", decl) for a in ALIASES)
        members.append(Member(code, match[1], target, pointer))
    return members


# A stable layout makes it possible to compare regenerated maps across revisions.
GROUPS = [
    ("T", "01  张量与存储", "#087f8c", [
        ("ggml_tensor", "张量描述、运算依赖与视图来源"),
        ("ggml_backend_buffer", "一块后端存储及其访问接口"),
        ("ggml_backend_buffer_type", "分配策略、设备与内存对齐"),
        ("ggml_backend_buffer_i", "函数指针成员：逐项保留完整签名"),
        ("ggml_backend_buffer_type_i", "函数指针成员：逐项保留完整签名"),
    ]),
    ("B", "02  后端、设备与注册", "#7450b6", [
        ("ggml_backend", "backend / cpu_backend 可指向同一对象"),
        ("ggml_backend_i", "执行后端接口；成员均为函数指针"),
        ("ggml_backend_device", "设备对象及所属后端注册信息"),
        ("ggml_backend_device_i", "设备接口；签名中的类型不是内嵌成员"),
        ("ggml_backend_reg", "后端注册入口；iface 按值内嵌"),
        ("ggml_backend_reg_i", "设备枚举与扩展函数查询接口"),
        ("ggml_backend_event", "调度器保存的后端同步事件"),
    ]),
    ("S", "03  调度器、计算图与上下文", "#2266c2", [
        ("ggml_backend_sched", "成员依次保留源码顺序；galloc 展开见 A01"),
        ("ggml_backend_sched_split", "一个切分区间；graph 是内嵌的图视图"),
        ("ggml_cgraph", "图节点、叶子、梯度与访问集合"),
        ("ggml_hash_set", "集合键是 tensor 指针；used 是位图"),
        ("ggml_context", "描述对象所在的内存池与对象链表"),
        ("ggml_object", "对象管理头；next 形成链表"),
    ]),
    ("A", "04  图分配器递归展开", "#b96913", [
        ("ggml_gallocr", "多缓冲区分配、哈希状态与节点分配记录"),
        ("vbuffer", "当前版本：逻辑缓冲区由多块 chunk 组成"),
        ("ggml_dyn_tallocr", "按对齐要求管理多个 chunk 的空闲空间"),
        ("allocated_tensor_debug", "匿名结构体；仅 GGML_ALLOCATOR_DEBUG 启用时存在"),
        ("tallocr_chunk", "单个 chunk 的空闲块表"),
        ("free_block", "全部成员均为基础类型：到此结束"),
        ("hash_node", "张量引用计数与分配状态"),
        ("node_alloc", "运算结果与各输入的分配记录"),
        ("leaf_alloc", "叶子张量的分配记录"),
        ("tensor_alloc", "记录 buffer_id、相对地址与最大尺寸"),
        ("buffer_address", "chunk 索引 + chunk 内偏移：到此结束"),
    ]),
]
COLORS = {prefix: color for prefix, _, color, _ in GROUPS} | {"M": "#24354d"}
NODES = {}


def add(name, ident, note, category):
    if name == "allocated_tensor_debug":
        path, _, _, body = definition("ggml_dyn_tallocr")
        m = re.search(r"struct\s*\{([^{}]+)\}\s*allocated_tensors", body, re.S)
        assert m
        members = parse_members(m[1])
        line = next(i for i, s in enumerate(TEXT[path].splitlines(), 1) if "const struct ggml_tensor * tensor;" in s)
        source = "#ifdef GGML_ALLOCATOR_DEBUG\nstruct {\n" + m[1].strip() + "\n} allocated_tensors[1024];\n#endif"
    else:
        path, line, source, body = definition(name)
        members = parse_members(body)
    NODES[name] = Node(name, ident, note, category, path, line, source, members)


add("simple_model", "M01", "递归起点 · examples/simple/simple-backend.cpp", "M")
for prefix, _, _, specs in GROUPS:
    for i, (name, note) in enumerate(specs, 1):
        add(name, f"{prefix}{i:02}", note, prefix)

# Fail if the graph contains an unexpanded structure or an unreachable extra node.
edges = [(n.name, m.name, m.target, m.pointer, m.conditional) for n in NODES.values() for m in n.members if m.target]
assert all(e[2] in NODES for e in edges), [e for e in edges if e[2] not in NODES]
reached, pending = set(), ["simple_model"]
while pending:
    name = pending.pop()
    if name not in reached:
        reached.add(name)
        pending += [m.target for m in NODES[name].members if m.target]
assert reached == set(NODES), set(NODES) - reached


FONTDIR = Path("C:/Windows/Fonts")
FONT_CACHE = {}


def font(size, mono=False, bold=False):
    key = size, mono, bold
    if key not in FONT_CACHE:
        candidates = ([FONTDIR / ("consolab.ttf" if bold else "consola.ttf")] if mono else [])
        candidates += [FONTDIR / ("msyhbd.ttc" if bold else "msyh.ttc")]
        candidates += [Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")]
        candidate = next((p for p in candidates if p.exists()), None)
        FONT_CACHE[key] = ImageFont.truetype(str(candidate), size) if candidate else ImageFont.load_default(size=size)
    return FONT_CACHE[key]


def wrap_code(code, width=116):
    return textwrap.wrap(code, width, subsequent_indent="    ", break_long_words=False, break_on_hyphens=False)


ROW = 33
HEADER = 150
MEMBER_Y = {}
for node in NODES.values():
    node.h = HEADER + sum(len(wrap_code(m.code)) * ROW + 12 for m in node.members) + 20

root = NODES["simple_model"]
root.x, root.y = 120, 190
START_Y = max(root.y + root.h + 175, 750)
for col, (_, _, _, specs) in enumerate(GROUPS):
    y = START_Y
    for name, _ in specs:
        n = NODES[name]
        n.x, n.y = 120 + col * 2080, y
        y += n.h + 64
    COLORS.setdefault("M", "#24354d")
BOTTOM = max(n.y + n.h for n in NODES.values())
WIDTH, HEIGHT = 8440, BOTTOM + 490


class Canvas:
    def __init__(self):
        self.im = Image.new("RGB", (WIDTH, HEIGHT), "#f5f7fb")
        self.draw = ImageDraw.Draw(self.im)
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
                      '<title id="title">simple_model 成员类型递归展开图</title>',
                      f'<desc id="desc">{len(NODES)} 个结构体节点，{len(edges)} 条结构体成员关系。包含指针、数组、typedef 和条件编译匿名结构体。每个成员保留源码声明。</desc>',
                      '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif}.mono{font-family:Consolas,"DejaVu Sans Mono",monospace}.node:target>rect:first-of-type{stroke:#ee375c;stroke-width:6}a{cursor:pointer}</style>',
                      f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#f5f7fb"/>']

    def rect(self, x, y, w, h, fill, stroke=None, radius=0, sw=1):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')
        self.draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=stroke, width=sw)

    def text(self, x, y, value, size=26, color="#26354a", mono=False, bold=False, link=None):
        if link:
            self.parts.append(f'<a href="#{link}"><title>跳转到 {link}</title>')
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{"700" if bold else "400"}" class="{"mono" if mono else "label"}" dominant-baseline="text-before-edge">{html.escape(value)}</text>')
        self.draw.text((x, y), value, font=font(size, mono, bold), fill=color, anchor="lt")
        if link:
            self.parts.append('</a>')

    def line(self, points, color="#91a2ba", dashed=False, width=3, arrow=False, diamond=False):
        pts = " ".join(f"{x},{y}" for x, y in points)
        self.parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" {"stroke-dasharray=\"10 7\"" if dashed else ""}/>')
        for a, b in zip(points, points[1:]):
            if dashed:
                length = math.dist(a, b)
                for start in range(0, int(length), 17):
                    end = min(start + 10, length)
                    if length:
                        self.draw.line([(a[0] + (b[0] - a[0]) * t / length, a[1] + (b[1] - a[1]) * t / length) for t in [start, end]], fill=color, width=width)
            else:
                self.draw.line([a, b], fill=color, width=width)
        if arrow:
            end, prev = points[-1], points[-2]
            angle = math.atan2(end[1] - prev[1], end[0] - prev[0])
            tri = [end] + [(end[0] - 16 * math.cos(angle + t), end[1] - 16 * math.sin(angle + t)) for t in [-0.42, 0.42]]
            self.polygon(tri, color)
        if diamond:
            x, y = points[0]
            self.polygon([(x, y), (x + 10, y - 7), (x + 20, y), (x + 10, y + 7)], color)

    def polygon(self, points, color):
        self.parts.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in points)}" fill="{color}"/>')
        self.draw.polygon(points, fill=color)


c = Canvas()
c.text(120, 45, "simple_model  /  成员类型递归展开", 54, bold=True)
c.text(120, 115, "按当前工作区源码生成 · 结构关系图（不表示内存偏移或运行时对象数量） · 2026-10-17", 28, "#64748b")

# The adjacent reading guide is part of the same standalone image.
c.rect(2200, 190, 6080, root.h, "#eaf0f8", radius=18)
c.text(2240, 215, "阅读方式", 33, bold=True)
guide = [
    "① 从 M01 出发：a / b → T01；backend / cpu_backend → B01；sched → S01。",
    "② 每行右侧 →编号 指向对应的结构体卡片；SVG 中可点击编号跳转。共享类型只展开一次。",
    "③ 实线 + 菱形：按值内嵌（含结构体数组）；虚线：指针 / 指针数组 / 结构体指针 typedef。",
    "④ 同栏首次展开画连接线；跨栏与重复引用通过编号连接，避免长线遮挡成员。",
    "⑤ 基础类型、enum、void *、函数指针是终点；std::vector<uint8_t> 的元素是 uint8_t。",
    "⑥ iface 也是结构体，已逐成员展开；函数签名中的参数 / 返回类型不作为数据成员递归。",
    f"覆盖：{len(NODES) - 1} 个具名结构体 + 1 个条件编译匿名结构体；{len(edges)} 条结构体成员关系。",
]
for i, line in enumerate(guide):
    c.text(2240, 270 + i * 42, line, 27)

for col, (prefix, title, color, specs) in enumerate(GROUPS):
    x = 120 + col * 2080
    c.rect(x - 26, START_Y - 100, 1950, BOTTOM - START_Y + 136, "#edf1f7", radius=18)
    c.rect(x - 26, START_Y - 100, 1950, 72, color, radius=12)
    c.text(x, START_Y - 85, title, 32, "#ffffff", bold=True)


def draw_node(n):
    color = COLORS[n.category]
    c.parts.append(f'<g id="{n.ident}" class="node">')
    c.rect(n.x, n.y, n.w, n.h, "#ffffff", "#ced8e6", 12, 2)
    c.rect(n.x, n.y, n.w, 60, color, radius=10)
    c.text(n.x + 23, n.y + 10, n.ident, 30, "#ffffff", mono=True, bold=True)
    name = "anonymous struct  (allocated_tensors[])" if n.name == "allocated_tensor_debug" else "struct " + n.name
    c.text(n.x + 122, n.y + 12, name, 29, "#ffffff", mono=True, bold=True)
    c.text(n.x + 23, n.y + 75, f"{n.path}:{n.line}", 24, "#63758d", mono=True)
    c.text(n.x + 23, n.y + 111, n.note, 25, color)
    y = n.y + HEADER
    for m in n.members:
        lines = wrap_code(m.code)
        height = len(lines) * ROW + 12
        if m.target:
            c.rect(n.x + 3, y - 2, n.w - 6, height, "#f0f5fa")
        for i, line in enumerate(lines):
            c.text(n.x + 23, y + i * ROW, line, 23, "#26354a", mono=True)
        if m.target:
            target = NODES[m.target]
            marker = "◇" if not m.pointer else "*"
            label = f"{marker} → {target.ident}"
            if m.conditional:
                label = "if → " + target.ident
            c.text(n.x + n.w - 155, y, label, 23, COLORS[target.category], bold=True, link=target.ident)
        elif re.search(r"\(\s*\*", m.code):
            c.text(n.x + n.w - 100, y, "fn ptr", 22, "#7b8596", mono=True)
        MEMBER_Y[n.name, m.name] = y + 15
        y += height
    c.parts.append('</g>')


for n in NODES.values():
    draw_node(n)

# Draw a spanning forest in each column. All other edges remain explicit row
# references, so that shared nodes/cycles do not create a tangled wiring diagram.
seen, rail_count = {"simple_model", "ggml_tensor", "ggml_backend", "ggml_backend_sched", "ggml_gallocr"}, {}
for source, field, target, pointer, conditional in edges:
    a, b = NODES[source], NODES[target]
    if target in seen or a.category != b.category or a.y >= b.y:
        continue
    seen.add(target)
    key = a.category
    rail_count[key] = rail_count.get(key, 0) + 1
    rail = a.x + a.w + 25 + (rail_count[key] % 6) * 18
    c.line([(a.x + a.w, MEMBER_Y[source, field]), (rail, MEMBER_Y[source, field]), (rail, b.y + 30), (b.x + b.w, b.y + 30)], COLORS[a.category], dashed=pointer or conditional, width=3, arrow=True, diamond=not pointer)

# Root branch lines run through the clear band above the four columns.
for i, (field, target) in enumerate([("a", "ggml_tensor"), ("backend", "ggml_backend"), ("sched", "ggml_backend_sched")]):
    b = NODES[target]
    sx, sy = root.x + root.w + 8, MEMBER_Y[root.name, field]
    rail_x = 1980 + i * 35
    band_y = START_Y - 135 - (2 - i) * 18
    c.line([(sx, sy), (rail_x, sy), (rail_x, band_y), (b.x + b.w // 2, band_y), (b.x + b.w // 2, b.y - 105)], COLORS[b.category], dashed=True, arrow=True)

# The graph allocator is the scheduler's largest subbranch.
a, b = NODES["ggml_backend_sched"], NODES["ggml_gallocr"]
sy = MEMBER_Y[a.name, "galloc"]
c.line([(a.x + a.w, sy), (a.x + a.w + 205, sy), (a.x + a.w + 205, b.y + 30), (b.x, b.y + 30)], COLORS[b.category], dashed=True, arrow=True)

# Explicit, short loops for recursive pointer types.
for name, fields in [("ggml_tensor", ["src", "view_src"]), ("ggml_object", ["next"])]:
    n = NODES[name]
    for i, field in enumerate(fields):
        y = MEMBER_Y[name, field]
        rail = n.x - 30 - i * 22
        c.line([(n.x, y), (rail, y), (rail, n.y + 27 + i * 18), (n.x, n.y + 27 + i * 18)], COLORS[n.category], dashed=True, arrow=True)

footer_y = BOTTOM + 80
c.rect(120, footer_y, 8160, 335, "#e8eef6", radius=16)
c.text(155, footer_y + 20, "typedef 与终点说明", 33, bold=True)
footer = [
    "ggml_backend_t = struct ggml_backend *    ggml_backend_sched_t = struct ggml_backend_sched *    ggml_gallocr_t = struct ggml_gallocr *",
    "ggml_backend_buffer_t = struct ggml_backend_buffer *    ggml_backend_buffer_type_t = struct ggml_backend_buffer_type *",
    "ggml_backend_dev_t = struct ggml_backend_device *    ggml_backend_reg_t = struct ggml_backend_reg *    ggml_backend_event_t = struct ggml_backend_event *",
    "ggml_bitset_t = uint32_t    ggml_guid = uint8_t[16]    ggml_guid_t = ggml_guid * = uint8_t (*)[16]    ggml_backend_graph_plan_t = void *",
    "GGML_MAX_DIMS=4    GGML_MAX_SRC=10    GGML_MAX_OP_PARAMS=64    GGML_MAX_NAME=64（默认）    GGML_SCHED_MAX_BACKENDS=16（默认）",
    "GGML_SCHED_MAX_COPIES=4（默认）    GGML_VBUFFER_MAX_CHUNKS=16    MAX_FREE_BLOCKS=256    callback_eval: bool (*)(struct ggml_tensor *, bool, void *)",
    "void * 无静态结构体类型：具体 context 随后端实现而变，不推断为 CPU 或 GPU 私有对象。数组保留源码长度宏；枚举作为值类型终止展开。",
]
for i, line in enumerate(footer):
    c.text(155, footer_y + 77 + i * 33, line, 24, "#354861", mono=i < 4)

c.parts.append('</svg>')
svg = "\n".join(c.parts)
ET.fromstring(svg)  # Detect malformed SVG before saving any final image.
(OUT / f"{STEM}.svg").write_text(svg, encoding="utf-8")
c.im.save(OUT / f"{STEM}.png", optimize=True)

# A searchable companion preserves exact definitions, macro conditions, all
# cross-references, and source locations independently of the graphical layout.
doc = [
    "# simple_model 成员类型递归展开图", "",
    "依据当前工作区源码生成，日期：2026-10-17。", "",
    f"![simple_model 成员类型递归展开图]({STEM}.png)", "",
    f"[打开可无损放大的 SVG]({STEM}.svg) · [打开 PNG]({STEM}.png)", "",
    "从 `simple_model` 的每个数据成员开始，穿透结构体指针 typedef、指针、指针数组和内嵌结构体，直到基础类型、枚举、无类型指针或函数指针。"
    "同一结构体只展开一次，自引用、循环与共享关系指向已有节点。图中的编号是类型编号，不是运行时对象编号。", "",
    "图中实线和菱形表示按值内嵌，虚线表示结构体指针或指针数组；每个结构体成员右侧的 `→ 编号` 都对应目标卡片。"
    "同栏首次展开绘制连线，跨栏和重复引用使用编号，SVG 中可点击编号跳转。", "",
    f"共 **{len(NODES) - 1} 个具名结构体、1 个条件编译匿名结构体、{len(edges)} 条结构体成员关系**。", "",
    "## 展开边界", "",
    "- `std::vector<uint8_t> buf`：追踪到元素类型 `uint8_t`，不展开依赖编译器和标准库实现的容器内部结构。",
    "- `void *` / `const void *`：没有可静态追踪的结构体类型。`backend->context` 的具体类型由选中的后端决定；本例通过 `ggml_backend_init_best()` 选择后端，不能统一视为 CPU 私有结构体。",
    "- 函数指针也是成员，所有接口结构体均列出其完整签名；参数和返回值中的结构体不是该成员内嵌的子对象，不额外展开。",
    "- 枚举是值类型终点，完整枚举定义附于文末。`int32_t flags` 仍按其声明类型展示，不改写为枚举。",
    "- `allocated_tensors[1024]` 是 `GGML_ALLOCATOR_DEBUG` 条件编译成员，其匿名结构体也已展开。",
    "- 指针关系不表示所有权。`backend` 与 `cpu_backend` 在选择 CPU 时可以相同；共享引用不重复生成卡片。", "",
    "## 结构体索引", "", "| 编号 | 类型 | 源码 |", "|---|---|---|",
]
for n in NODES.values():
    display = "匿名结构体 allocated_tensors[]" if n.name == "allocated_tensor_debug" else n.name
    doc.append(f"| [{n.ident}](#{n.ident.lower()}) | `{display}` | [{n.path}:{n.line}](../{n.path}#L{n.line}) |")
doc += ["", "## 指针别名与基础别名", "", "```c"]
used_aliases = sorted(a for a in ALIASES if any(re.search(r"\b" + a + r"\b", m.code) for n in NODES.values() for m in n.members))
for alias in used_aliases:
    doc.append(f"typedef struct {ALIASES[alias]} * {alias};")
doc += ["typedef uint32_t ggml_bitset_t;", "typedef uint8_t ggml_guid[16];", "typedef ggml_guid * ggml_guid_t;  // uint8_t (*)[16]，指向整个数组", "typedef void * ggml_backend_graph_plan_t;", "typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);", "```", "", "## 完整结构体与每条成员关系", ""]
for n in NODES.values():
    doc += [f'<a id="{n.ident.lower()}"></a>', f"### {n.ident} · {n.name}", "", n.note + "。", "", f"源码：[{n.path}:{n.line}](../{n.path}#L{n.line})", "", "```cpp", n.source, "```", ""]
    structured = [m for m in n.members if m.target]
    if structured:
        doc += ["| 成员 | 声明 | 继续展开 | 关系 |", "|---|---|---|---|"]
        for m in structured:
            t = NODES[m.target]
            relation = "指针引用" if m.pointer else "按值内嵌"
            if m.conditional:
                relation += "；仅 GGML_ALLOCATOR_DEBUG"
            doc.append(f"| `{m.name}` | `{m.code}` | [{t.ident} · {t.name}](#{t.ident.lower()}) | {relation} |")
        doc.append("")
    else:
        doc += ["该结构体没有需要继续展开的数据成员。", ""]
doc += ["## 直接成员使用的枚举（值类型终点）", ""]
enum_names = sorted({m[1] for n in NODES.values() for member in n.members if not re.search(r"\(\s*\*", member.code) for m in re.finditer(r"\benum\s+(\w+)", member.code)})
for name in enum_names:
    path, line, source, _ = definition(name, "enum")
    doc += [f"### {name}", "", f"源码：[{path}:{line}](../{path}#L{line})", "", "```c", source, "```", ""]
doc += ["## 重新生成", "", "需要 Python 3 和 Pillow；在仓库根目录执行：", "", "```powershell", "python MyDoc/generate_simple_model_map.py", "```", "", "脚本从源码重新读取成员声明和源码行号，并校验每个结构体成员都有目标节点、所有节点均可从 `simple_model` 到达、SVG 是合法 XML。"
        "若源码新增可达结构体但布局尚未登记，脚本会报错，避免静默遗漏。", ""]
(OUT / f"{STEM}.md").write_text("\n".join(doc), encoding="utf-8")

# Small temporary previews make visual validation possible without opening a
# multi-megapixel image. These are deliberately placed in the ignored tmp dir.
preview_dir = ROOT / "tmp" / "simple-model-map"
preview_dir.mkdir(parents=True, exist_ok=True)
preview = c.im.copy()
preview.thumbnail((2100, 2100))
preview.save(preview_dir / "overview.png")
c.im.crop((root.x - 20, root.y - 20, root.x + root.w + 30, root.y + root.h + 20)).save(preview_dir / "root.png")
c.im.crop((NODES["ggml_gallocr"].x - 20, START_Y - 115, WIDTH - 100, START_Y + 1000)).save(preview_dir / "allocator.png")
print(f"Rendered {len(NODES)} structures, {len(edges)} member relationships; {WIDTH} x {HEIGHT} px.")
for ext in ["svg", "png", "md"]:
    p = OUT / f"{STEM}.{ext}"
    print(f"{p.relative_to(ROOT)} ({p.stat().st_size:,} bytes)")
