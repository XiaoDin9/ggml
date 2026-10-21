"""Build an offline, source-linked learning map of simple-backend.cpp.

Run: python MyDoc/simple-backend-callgraph/generate.py
No packages or network access required. The map is curated; definitions, direct
call sites, callback bindings and line numbers are extracted from this checkout.
It describes possible source paths, not a recorded execution or a complete AST.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
FILES = [
    "examples/simple/simple-backend.cpp", "src/ggml-backend.cpp",
    "src/ggml-backend-reg.cpp", "src/ggml-cpu/ggml-cpu.cpp",
    "src/ggml-cpu/ggml-cpu.c", "src/ggml-alloc.c", "src/ggml.c",
    "src/ggml-impl.h", "src/ggml-backend-impl.h",
]
TEXT = {p: (ROOT / p).read_text(encoding="utf-8") for p in FILES}


def mask(text):
    """Blank comments and literals without changing offsets or line numbers."""
    pattern = r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern, lambda m: re.sub(r"[^\n]", " ", m[0]), text, flags=re.S)


MASKED = {p: mask(s) for p, s in TEXT.items()}


def closing(text, start, left, right):
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == left) - (text[i] == right)
        if depth == 0:
            return i
    raise ValueError(f"Unbalanced {left} at {start}")


def editor_uri(path, line):
    absolute = (ROOT / path).as_posix()
    return "vscode://file/" + quote(absolute, safe="/:") + f":{line}:1"


def extract():
    indexed = {}
    excluded = {"if", "for", "while", "switch", "catch", "sizeof", "alignof", "return"}
    for path, clean in MASKED.items():
        for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", clean):
            name = match[1]
            if name in excluded:
                continue
            paren = clean.index("(", match.start())
            end_params = closing(clean, paren, "(", ")")
            tail = re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?\{", clean[end_params + 1:])
            if not tail:
                continue
            brace = end_params + 1 + tail.end() - 1
            end = closing(clean, brace, "{", "}")
            start = clean.rfind("\n", 0, match.start()) + 1
            line = clean.count("\n", 0, start) + 1
            last = clean.count("\n", 0, end) + 1
            # Keep alternative preprocessor definitions in the index. On this
            # Windows checkout the first ggml_time_init is the Windows variant.
            ident = name if name not in indexed else f"{name}@{path}:{line}"
            source = TEXT[path][start:end + 1]
            indexed[ident] = dict(
                id=ident, name=name, path=path, line=line, endLine=last,
                source=source, code=editor_uri(path, line), group="aux",
                note="辅助实现；点击源码中的函数名可继续阅读。", kind="function",
            )
    return indexed


FUNCTIONS = extract()
SPECS = {
    "entry": ("main", "main 总览", "#eab77b", [
        ("main", "入口：初始化 → 建图 → 分配与计算 → 读取结果 → 释放"),
        ("ggml_time_init", "初始化计时设施；Windows 分支"),
        ("ggml_nelements", "计算结果的元素个数"),
        ("ggml_nbytes", "计算张量占用的字节数，用于数据读写"),
        ("ggml_graph_node", "取得最后一个图节点，作为输出"),
    ]),
    "init": ("init_model", "后端初始化", "#74c8ee", [
        ("init_model", "选择执行后端、准备 CPU 兜底并创建调度器"),
        ("ggml_log_set", "设置日志回调；这里只保存回调"),
        ("ggml_backend_load_all", "仅 GGML_BACKEND_DL 编译分支显式加载插件"),
        ("ggml_backend_init_best", "优先寻找 GPU，其次 IGPU，最后 CPU"),
        ("ggml_backend_init_by_type", "主后端不是 CPU 时，另外创建 CPU 执行实例"),
        ("ggml_backend_dev_by_type", "枚举已注册设备，寻找匹配类型"),
        ("ggml_backend_dev_count", "读取全局设备列表大小"),
        ("ggml_backend_dev_get", "按下标取得设备对象"),
        ("ggml_backend_dev_type", "通过设备接口查询类型"),
        ("ggml_backend_get_device", "从执行实例取得设备描述"),
        ("ggml_backend_dev_init", "通过所选 device 的 init_backend 接口创建实例"),
        ("ggml_backend_cpu_device_init_backend", "所选设备为 CPU 时，进入此实现"),
        ("ggml_backend_cpu_init", "创建 CPU 上下文与 backend，绑定 CPU 执行接口表"),
        ("ggml_backend_sched_new", "建立调度器、后端列表和图分配器；parallel=false"),
        ("ggml_backend_get_default_buffer_type", "取得执行后端的默认存储类型"),
        ("ggml_backend_dev_buffer_type", "通过设备接口查询 buffer type"),
        ("ggml_backend_cpu_device_get_buffer_type", "CPU 设备返回 CPU buffer type"),
        ("ggml_backend_cpu_buffer_type", "提供 CPU 内存的分配、对齐与访问接口"),
        ("ggml_gallocr_new_n", "创建支持多个 buffer type 的图分配器"),
    ]),
    "registry": ("get_reg", "注册与设备", "#b9a1ef", [
        ("get_reg", "首次执行局部 static 声明时构造全局注册表；随后复用"),
        ("ggml_backend_registry", "构造注册表；按编译选项注册已链接后端"),
        ("register_backend", "保存注册对象，并枚举设备加入 devices"),
        ("register_device", "保存设备指针；同一设备去重"),
        ("ggml_backend_cpu_reg", "取得 CPU 静态注册对象与注册接口表"),
        ("ggml_cpu_init", "初始化 CPU 特性、查表等基础设施"),
        ("ggml_backend_reg_dev_count", "调用注册对象的 get_device_count 接口"),
        ("ggml_backend_reg_dev_get", "调用注册对象的 get_device 接口"),
        ("ggml_backend_cpu_reg_get_device_count", "CPU 注册对象报告一个设备"),
        ("ggml_backend_cpu_reg_get_device", "取得 CPU 静态设备，绑定 device 接口表"),
        ("ggml_backend_cpu_device_get_type", "CPU 设备的 get_type 接口实现"),
    ]),
    "build": ("build_graph", "构建计算图", "#83d5b5", [
        ("build_graph", "只建立 a、b、MUL_MAT 和依赖关系；不执行矩阵乘法"),
        ("ggml_init", "no_alloc=true：上下文保存元数据，数值存储稍后分配"),
        ("ggml_new_graph", "创建默认容量的图"),
        ("ggml_new_graph_custom", "在上下文中建立图对象、节点数组和访问集合"),
        ("ggml_new_tensor_2d", "创建二维输入张量描述；本例调用两次"),
        ("ggml_new_tensor", "创建指定维数的张量"),
        ("ggml_new_tensor_impl", "建立 tensor 元数据与视图关系"),
        ("ggml_new_object", "从 context 元数据池中取得对象空间"),
        ("ggml_mul_mat", "创建 GGML_OP_MUL_MAT 节点，并设置两个 src"),
        ("ggml_build_forward_expand", "从结果张量扩展前向图"),
        ("ggml_build_forward_impl", "调用依赖遍历，把结果相关节点加入图"),
        ("ggml_visit_parents_graph", "递归访问 src，区分 leaf 与计算节点"),
        ("ggml_free", "释放 context 管理对象；本例外部 model.buf 继续存在"),
    ]),
    "alloc": ("compute", "调度与分配", "#e9bd73", [
        ("compute", "reset → alloc_graph → 写 a、b → graph_compute → 取输出"),
        ("ggml_backend_sched_reset", "清空后端分配与副本记录，准备下一张图"),
        ("ggml_backend_sched_alloc_graph", "先分配后端与切图，再给数值数据分配内存"),
        ("ggml_backend_sched_split_graph", "Pass 1～4 分配后端，Pass 5 切图并建立副本描述"),
        ("ggml_backend_sched_backend_id_from_cur", "根据现有存储、输入标记和权重位置推断后端"),
        ("ggml_backend_sched_backend_from_buffer", "寻找同时支持 buffer type 与操作的后端"),
        ("ggml_backend_sched_set_if_supported", "Pass 2/4：后端支持操作时写入分配"),
        ("ggml_backend_sched_buffer_supported", "判断目标后端能否直接使用源张量的存储类型"),
        ("ggml_backend_supports_op", "通过设备能力接口判断是否支持操作"),
        ("ggml_backend_dev_supports_op", "设备 supports_op 接口入口"),
        ("ggml_backend_cpu_device_supports_op", "CPU 支持操作的判定，可能使用额外 buffer 实现"),
        ("ggml_backend_supports_buft", "通过设备接口判断能否使用此 buffer type"),
        ("ggml_backend_dev_supports_buft", "设备 supports_buft 接口入口"),
        ("ggml_backend_cpu_device_supports_buft", "CPU 对 buffer type 的兼容判定"),
        ("ggml_dup_tensor_layout", "只复制张量布局描述，数值数据稍后分配与搬运"),
        ("ggml_graph_view", "用原图的一段节点构造子图视图"),
        ("ggml_backend_graph_optimize", "调用可选的后端图优化接口；CPU 的该接口为空"),
        ("ggml_backend_sched_alloc_splits", "检查后端分配变化，申请或复用图存储"),
        ("ggml_gallocr_alloc_graph", "按分配记录设置各张量的 buffer 与 data"),
        ("ggml_gallocr_reserve", "单 buffer 的 reserve 包装入口"),
        ("ggml_gallocr_reserve_n", "按每个节点的 buffer ID 预留图存储"),
        ("ggml_gallocr_reserve_n_impl", "规划生命周期与复用，按需创建底层 buffer"),
        ("ggml_gallocr_alloc_graph_impl", "模拟图执行顺序，规划各张量所占空间"),
        ("ggml_gallocr_init_tensor", "把分配记录落实为 tensor->buffer / tensor->data"),
        ("ggml_vbuffer_alloc", "按 chunk 分配实际后端存储"),
        ("ggml_backend_buft_alloc_buffer", "通过 buffer type 的 alloc_buffer 分配存储"),
        ("ggml_backend_cpu_buffer_type_alloc_buffer", "CPU 分配内存，再包装为 backend buffer"),
        ("ggml_backend_buffer_init", "创建 buffer 描述，绑定 buffer 操作接口"),
    ]),
    "execute": ("ggml_backend_sched_graph_compute", "执行与 CPU 实现", "#e998ad", [
        ("ggml_backend_sched_graph_compute", "同步执行接口：提交计算，再同步"),
        ("ggml_backend_sched_graph_compute_async", "已分配则直接执行；否则先分配图"),
        ("ggml_backend_sched_compute_splits", "按 split 顺序复制必要输入并提交子图"),
        ("ggml_backend_graph_compute_async", "调用所选 backend 的 graph_compute 函数指针"),
        ("ggml_backend_cpu_graph_compute", "CPU 实现：准备计划与工作区，进入图计算"),
        ("ggml_graph_plan", "规划 CPU 线程数和临时工作区大小"),
        ("ggml_graph_compute", "启动/复用线程池，组织 CPU 图计算"),
        ("ggml_graph_compute_thread", "线程循环遍历图节点并分派操作"),
        ("ggml_compute_forward", "按 tensor->op 选择算子实现"),
        ("ggml_compute_forward_mul_mat", "MUL_MAT 的 CPU 实现；具体优化路径取决于条件"),
        ("ggml_compute_forward_mul_mat_one_chunk", "通用矩阵乘法的分块计算路径"),
        ("ggml_backend_sched_synchronize", "同步各后端，必要时更新副本槽位状态"),
        ("ggml_backend_synchronize", "可选同步接口；CPU 接口为空，调用后直接返回"),
        ("ggml_backend_tensor_copy", "必要时跨后端复制张量；当前简单同后端图通常无需"),
        ("ggml_backend_tensor_copy_async", "尝试异步复制，必要时回退到同步复制"),
    ]),
    "data": ("main", "数据读写与释放", "#a0c9e9", [
        ("ggml_backend_tensor_set", "把 matrix_A / matrix_B 写入已分配的张量"),
        ("ggml_backend_tensor_get", "把输出张量读回 out_data"),
        ("ggml_backend_cpu_buffer_set_tensor", "CPU buffer 的 set_tensor：memcpy 到 tensor->data"),
        ("ggml_backend_cpu_buffer_get_tensor", "CPU buffer 的 get_tensor：memcpy 到用户缓冲区"),
        ("ggml_backend_sched_free", "释放调度器元数据、图分配器等资源"),
        ("ggml_gallocr_free", "释放图分配器与其持有的 buffer"),
        ("ggml_vbuffer_free", "释放各 chunk 的后端 buffer"),
        ("ggml_backend_buffer_free", "调用 buffer 的 free_buffer 接口并释放描述"),
        ("ggml_backend_cpu_buffer_free_buffer", "释放 CPU buffer 持有的内存"),
        ("ggml_backend_free", "调用执行实例的 free 接口；共用 CPU 实例只释放一次"),
        ("ggml_backend_cpu_free", "释放 CPU 工作区、上下文和执行实例"),
    ]),
}

NODES = {}
for group, (_, _, _, specs) in SPECS.items():
    for name, note in specs:
        if name not in FUNCTIONS:
            raise ValueError(f"Definition missing: {name}")
        node = FUNCTIONS[name]
        node.update(group=group, note=note)
        NODES[name] = node


def location(path, line):
    return dict(path=path, line=line, code=editor_uri(path, line))


def direct_calls(source, name):
    clean = mask(source["source"])
    opening = clean.index("{")
    return [source["line"] + clean.count("\n", 0, m.start())
            for m in re.finditer(r"\b" + re.escape(name) + r"\s*\(", clean)
            if m.start() > opening]


EDGES = []
CONDITIONS = {
    ("init_model", "ggml_backend_load_all"): "仅 GGML_BACKEND_DL",
    ("init_model", "ggml_backend_init_by_type"): "主后端不是 CPU 时",
    ("main", "ggml_backend_free"): "CPU 兜底与主后端共用时只释放一次",
    ("ggml_backend_registry", "ggml_backend_cpu_reg"): "仅 GGML_USE_CPU 编译分支",
    ("ggml_backend_sched_graph_compute_async", "ggml_backend_sched_alloc_graph"): "仅 !sched->is_alloc；本例已显式分配",
    ("ggml_backend_sched_graph_compute_async", "ggml_backend_sched_reset"): "仅 !is_reset 且 !is_alloc",
    ("ggml_backend_sched_compute_splits", "ggml_backend_tensor_copy"): "存在需要复制的 split 输入时",
    ("ggml_backend_sched_compute_splits", "ggml_backend_tensor_copy_async"): "存在需复制的非用户输入时",
    ("ggml_gallocr_alloc_graph", "ggml_gallocr_reserve"): "单 buffer 且需要重新预留时",
    ("ggml_backend_sched_split_graph", "ggml_dup_tensor_layout"): "需要输入副本时",
    ("ggml_compute_forward", "ggml_compute_forward_mul_mat"): "GGML_OP_MUL_MAT 分支",
    ("ggml_compute_forward_mul_mat", "ggml_compute_forward_mul_mat_one_chunk"): "通用分块路径；优化实现可提前返回",
}
for caller, src in NODES.items():
    for callee in NODES:
        sites = direct_calls(src, callee)
        if sites:
            note = CONDITIONS.get((caller, callee), "")
            EDGES.append(dict(source=caller, target=callee, kind="direct",
                              note=note, label=note or f"L{sites[0]}",
                              site=location(src["path"], sites[0]), sites=sites))


def callback(caller, callee, field, kind="callback", note="CPU 实现；仅运行时选择该接口时进入"):
    src, dst = NODES[caller], NODES[callee]
    call = re.search(r"(?:->|\.)\s*" + field + r"\s*\(", mask(src["source"]))
    if not call:
        raise ValueError(f"Callback call missing: {caller}.{field}")
    site = src["line"] + src["source"].count("\n", 0, call.start())
    binding = re.search(r"/\*\s*\." + field + r"\s*=\s*\*/\s*" + callee + r"\b", TEXT[dst["path"]])
    if not binding:
        raise ValueError(f"Callback binding missing: {field} = {callee}")
    bind_line = TEXT[dst["path"]].count("\n", 0, binding.start()) + 1
    EDGES.append(dict(source=caller, target=callee, kind=kind, label=f"iface.{field}", note=note,
                      site=location(src["path"], site), sites=[site],
                      binding=location(dst["path"], bind_line)))


for args in [
    ("ggml_backend_dev_init", "ggml_backend_cpu_device_init_backend", "init_backend"),
    ("ggml_backend_dev_type", "ggml_backend_cpu_device_get_type", "get_type"),
    ("ggml_backend_reg_dev_count", "ggml_backend_cpu_reg_get_device_count", "get_device_count"),
    ("ggml_backend_reg_dev_get", "ggml_backend_cpu_reg_get_device", "get_device"),
    ("ggml_backend_dev_buffer_type", "ggml_backend_cpu_device_get_buffer_type", "get_buffer_type"),
    ("ggml_backend_dev_supports_op", "ggml_backend_cpu_device_supports_op", "supports_op"),
    ("ggml_backend_dev_supports_buft", "ggml_backend_cpu_device_supports_buft", "supports_buft"),
    ("ggml_backend_graph_compute_async", "ggml_backend_cpu_graph_compute", "graph_compute"),
    ("ggml_backend_tensor_set", "ggml_backend_cpu_buffer_set_tensor", "set_tensor"),
    ("ggml_backend_tensor_get", "ggml_backend_cpu_buffer_get_tensor", "get_tensor"),
    ("ggml_backend_buft_alloc_buffer", "ggml_backend_cpu_buffer_type_alloc_buffer", "alloc_buffer"),
    ("ggml_backend_free", "ggml_backend_cpu_free", "free"),
    ("ggml_backend_buffer_free", "ggml_backend_cpu_buffer_free_buffer", "free_buffer"),
]:
    callback(*args)

reg = NODES["get_reg"]
static_line = reg["line"] + next(i for i, s in enumerate(reg["source"].splitlines()) if "static ggml_backend_registry reg" in s)
EDGES.append(dict(source="get_reg", target="ggml_backend_registry", kind="lifecycle",
                  label="首次访问：局部 static 初始化", note="后续 get_reg() 直接复用；不是每次都构造",
                  site=location(reg["path"], static_line), sites=[static_line]))

# Named source regions keep the previously discussed passes inspectable without
# pretending that these blocks are separate C/C++ functions.
split = NODES["ggml_backend_sched_split_graph"]
pass_notes = [
    ("根据位置分配", "利用现有 buffer、输入标记和权重位置，形成初始后端分配"),
    ("扩展后端分配", "先非 CPU 后 CPU，正反向扫描，给未分配节点扩展兼容后端"),
    ("选择与升级", "补分配；在存储兼容时尝试更高优先级后端"),
    ("补齐输入与视图", "传播 view_src 和使用者的后端归属，并完成兜底分配"),
    ("切图与副本", "划分 split，建立必要的副本描述，改写节点 src"),
]
matches = list(re.finditer(r"// pass ([1-5]):", split["source"]))
assert len(matches) == 5
for i, match in enumerate(matches):
    end = matches[i + 1].start() if i < 4 else split["source"].index("    if (sched->debug)")
    source = split["source"][match.start():end].rstrip()
    start_line = split["line"] + split["source"].count("\n", 0, match.start())
    ident = f"pass{i + 1}"
    node = dict(id=ident, name=f"Pass {i + 1} · {pass_notes[i][0]}", path=split["path"],
                line=start_line, endLine=start_line + source.count("\n"), source=source,
                code=editor_uri(split["path"], start_line), group="passes", kind="phase", note=pass_notes[i][1])
    NODES[ident] = node
    EDGES.append(dict(source="ggml_backend_sched_split_graph", target=ident,
                      kind="phase", label="函数内部阶段", note="函数内的源码块；各 Pass 按编号顺序执行",
                      site=location(split["path"], start_line), sites=[start_line]))

for ident in [f"pass{i}" for i in range(1, 6)]:
    for callee, target in list(NODES.items()):
        if target["kind"] != "function":
            continue
        sites = direct_calls(NODES[ident], callee)
        if sites:
            EDGES.append(dict(source=ident, target=callee, kind="direct", label=f"L{sites[0]}",
                              note="此源码阶段内的调用", site=location(split["path"], sites[0]), sites=sites))

for i, edge in enumerate(EDGES):
    edge["id"] = f"edge{i}"


def view(ident, title, root, names):
    names = list(dict.fromkeys(names))
    missing = set(names) - set(NODES)
    assert not missing, missing
    return dict(id=ident, title=title, root=root, nodes=names)


VIEWS = [view("overview", "main 调用树", "main", [
    "main", "init_model", "build_graph", "compute",
    "ggml_backend_tensor_get", "ggml_backend_sched_free", "ggml_backend_free",
])]
EXTRAS = {
    "init": ["ggml_backend_cpu_reg", "ggml_backend_reg_dev_get", "ggml_backend_sched_reset"],
    "registry": ["ggml_backend_dev_by_type", "ggml_backend_dev_count", "ggml_backend_dev_get", "ggml_backend_dev_type"],
    "build": [],
    "alloc": ["ggml_backend_tensor_set", "ggml_backend_sched_graph_compute", "ggml_graph_node"],
    "execute": ["compute", "ggml_backend_sched_alloc_graph"],
    "data": ["main", "compute", "ggml_backend_tensor_copy", "ggml_backend_tensor_copy_async"],
}
for group, (root, title, _, specs) in SPECS.items():
    if group == "entry":
        continue
    names = [n for n, _ in specs] + EXTRAS[group]
    # Registry discovery starts with device search, which leads into get_reg.
    actual_root = "ggml_backend_dev_by_type" if group == "registry" else root
    VIEWS.append(view(group, title, actual_root, names))
VIEWS.append(view("passes", "切图 Pass 1～5", split["id"], [split["id"]] + [f"pass{i}" for i in range(1, 6)]))
VIEWS.append(view("all", "全树逐层浏览", "main", list(NODES)))

GROUPS = {g: dict(title=s[1], color=s[2]) for g, s in SPECS.items()}
GROUPS["passes"] = dict(title="函数内部阶段", color="#c5adf0")
GROUPS["aux"] = dict(title="辅助实现", color="#a9b6ca")
DATA = dict(root=ROOT.as_posix(), entry="examples/simple/simple-backend.cpp", nodes=NODES,
            edges=EDGES, views=VIEWS, groups=GROUPS, functions=FUNCTIONS,
            files={p: dict(sha256=hashlib.sha256(t.encode()).hexdigest(), lines=len(t.splitlines()),
                           text=t) for p, t in TEXT.items()},
            scope="主要 ggml 调用与 CPU 实现；标准库和平台细节省略。条件分支展示可能路径，不是运行记录。")

DATA["chains"] = [
    dict(title="CPU 矩阵乘法完整链", target="ggml_compute_forward_mul_mat"),
    dict(title="CPU 矩阵乘法 → 分块实现", target="ggml_compute_forward_mul_mat_one_chunk"),
    dict(title="调度切图 → Pass 2", target="pass2"),
    dict(title="设备注册 → CPU 设备", target="ggml_backend_cpu_reg_get_device"),
    dict(title="图分配 → CPU 内存", target="ggml_backend_cpu_buffer_type_alloc_buffer"),
    dict(title="读取结果 → CPU buffer", target="ggml_backend_cpu_buffer_get_tensor"),
]


def svg_export():
    compact = dict(nodes={k: {p: n[p] for p in ("id", "name", "note", "code", "path", "line", "group")}
                         for k, n in NODES.items()}, edges=EDGES, groups=GROUPS)
    payload = json.dumps(compact, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    result = (OUT / "svg.template.svg").read_text(encoding="utf-8")
    result = result.replace("__TREE_ENGINE__", (OUT / "tree.js").read_text(encoding="utf-8"))
    result = result.replace("__SVG_DATA__", payload)
    result = result.replace("__SVG_MAIN_URI__", NODES["main"]["code"])
    result = result.replace("__SVG_MAIN_LINE__", str(NODES["main"]["line"]))
    ET.fromstring(result)
    return result


def validate():
    assert NODES["main"]["path"] == "examples/simple/simple-backend.cpp"
    assert len({e["id"] for e in EDGES}) == len(EDGES)
    for node in list(FUNCTIONS.values()) + list(NODES.values()):
        lines = TEXT[node["path"]].splitlines()
        assert "\n".join(lines[node["line"] - 1:node["endLine"]]).strip() == node["source"].strip(), node["id"]
    for edge in EDGES:
        src, dst = NODES[edge["source"]], NODES[edge["target"]]
        assert src and dst
        if edge["kind"] == "direct":
            assert direct_calls(src, dst["name"]) == edge["sites"], edge
        assert 1 <= edge["site"]["line"] <= len(TEXT[edge["site"]["path"]].splitlines())
        assert dst["code"].startswith("vscode://file/")
    # Every project-defined direct call from main appears in the map.
    for name in FUNCTIONS:
        if "@" not in name and direct_calls(NODES["main"], name):
            assert name in NODES, f"Unmapped main call: {name}"


def main():
    validate()
    payload = json.dumps(DATA, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = (OUT / "viewer.template.html").read_text(encoding="utf-8")
    assert template.count("__CALLGRAPH_DATA__") == 1
    assert template.count("__TREE_ENGINE__") == 1
    assert template.count("__VIEWER_SCRIPT__") == 1
    result = template.replace("__CALLGRAPH_DATA__", payload)
    result = result.replace("__TREE_ENGINE__", (OUT / "tree.js").read_text(encoding="utf-8"))
    result = result.replace("__VIEWER_SCRIPT__", (OUT / "viewer.js").read_text(encoding="utf-8"))
    (OUT / "index.html").write_text(result, encoding="utf-8")
    (OUT / "callgraph.svg").write_text(svg_export(), encoding="utf-8")
    report = dict(nodes=len(NODES), edges=len(EDGES), indexedDefinitions=len(FUNCTIONS),
                  callbacks=sum(e["kind"] == "callback" for e in EDGES),
                  views={v["id"]: len(v["nodes"]) for v in VIEWS})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Generated: {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
