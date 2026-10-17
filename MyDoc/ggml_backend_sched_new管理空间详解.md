# ggml_backend_sched_new 管理空间详解

本文把两段初始化代码分别整理为两个详细例子：第一个说明容量参数、内部图与四张后端编号表；第二个说明临时上下文、执行片段、流水线资源和图内存分配器如何协作。内容依据当前仓库源码整理，核对日期为 2026-10-07。

阅读入口：

- [示例一：后端编号数组与内部图](#example-1)：从 3 个原始计算节点展开到 7 个内部图节点，逐项填写编号，并比较不同轮次。
- [示例二：管理资源与执行流程](#example-2)：沿用同一个图，说明每块空间保存什么、何时使用，以及开启流水线后的变化。

主要源码入口：

- [`src/ggml-backend.cpp`](../src/ggml-backend.cpp)：调度器创建、后端分配、切图、复制、执行和同步。
- [`src/ggml-alloc.c`](../src/ggml-alloc.c)：图内存分配器 `ggml_gallocr`。
- [`src/ggml.c`](../src/ggml.c)：上下文、张量描述对象和图管理空间。
- [`examples/simple/simple-backend.cpp`](../examples/simple/simple-backend.cpp)：调度器的实际创建与使用示例。
- [ggml_tensor 与 ggml_cgraph 详解](ggml_tensor与ggml_cgraph详解.md)：张量与计算图的基础概念。

<a id="example-1"></a>

## 示例一：后端编号数组与内部图

### 1.1 场景与原始计算图

假设调用者已经创建 GPU 和 CPU 后端，创建调度器时使用以下配置：

```cpp
// 配置示意；GPU、CPU 表示已经创建的后端句柄。
ggml_backend_t backends[2] = { GPU, CPU };
size_t graph_size = 8;
bool parallel = false;
bool op_offload = true;
// bufts 传 nullptr，使用各后端的默认缓冲区类型。
```

后端编号就是 `backends[]` 的下标：

| 编号 | 后端 |
|---|---|
| 0 | GPU |
| 1 | CPU |

实际计算图如下，节点的后端归属已经确定：

```mermaid
flowchart LR
    X["X：CPU 上的输入张量"] --> A["A：GPU 计算"]
    A --> B["B：GPU 计算"]
    B --> C["C：CPU 计算"]
```

为便于观察，假设原图只有叶子 X 和计算节点 A、B、C，后端优化没有改变节点顺序，也没有额外添加分配依赖。CPU 和 GPU 不能直接使用对方在本例中的张量缓冲区，因此需要复制 X 和 B。

原图的数组是：

```text
graph->nodes = [A, B, C]
graph->leafs = [X]
graph->n_nodes = 3
graph->n_leafs = 1
```

`graph_size = 8` 是创建调度器时提供的容量参数；本次图的实际节点数是 3。后面讨论流水线时，再把 `parallel` 改为 `true`。

### 1.2 后端编号数组的预留容量

对应代码：

```cpp
const size_t ggml_sched_max_splits = graph_size;
const size_t nodes_size =
    graph_size + ggml_sched_max_splits * GGML_SCHED_MAX_SPLIT_INPUTS * 2;

sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));
```

#### ggml_sched_max_splits：估算最大片段数

`split` 是交给同一个后端执行的连续图片段。初始化代码按每个节点都可能独立形成一个片段来估算，所以使用 `graph_size` 作为估算的最大片段数。

本例实际切分结果：

| 片段 | 原图节点范围 | 节点 | 后端 | 需要复制的输入 |
|---|---|---|---|---|
| split 0 | `[0, 2)` | A、B | GPU，编号 0 | X |
| split 1 | `[2, 3)` | C | CPU，编号 1 | B |

因此：

```text
ggml_sched_max_splits = 8    // 初始化时的容量估算
sched->n_splits = 2         // 本次实际切出的片段数量
```

#### nodes_size：每张编号表的元素容量

调度器需要构建一份供图内存分配器使用的内部图。在每个片段开始处，每个需要跨后端复制的输入会加入两个条目：

| 条目 | 用途 |
|---|---|
| `input_dep` | 引用源张量，延长它在分配规划中的生命周期，保证复制时源内存仍然有效 |
| `input_cpy` | 目标后端上的张量副本，让分配器在片段开始时为它准备内存 |

本例为 X 加入 `X_dep`、`X_gpu`，为 B 加入 `B_dep`、`B_cpu`。

按源码构建内部图的顺序，过程如下：

| 步骤 | 新加入的条目 | 加入后的 `sched->graph.nodes` | `n_nodes` |
|---|---|---|---|
| 开始 | 无 | `[]` | 0 |
| 处理 GPU 片段的输入 X | X_dep、X_gpu | `[X_dep, X_gpu]` | 2 |
| 加入 GPU 片段的计算节点 | A、B | `[X_dep, X_gpu, A, B]` | 4 |
| 处理 CPU 片段的输入 B | B_dep、B_cpu | `[X_dep, X_gpu, A, B, B_dep, B_cpu]` | 6 |
| 加入 CPU 片段的计算节点 | C | `[X_dep, X_gpu, A, B, B_dep, B_cpu, C]` | 7 |

最终内部图的节点数组顺序为：

```text
sched->graph.nodes = [X_dep, X_gpu, A, B, B_dep, B_cpu, C]
```

这个列表表示内部图的数组顺序。`X_dep`、`B_dep` 是 view 描述对象，共享源张量的数据；副本的数据传输由后续调度执行代码安排。

计算节点的输入指针也会调整：

```text
原图建好时： A->src[0] = X       C->src[0] = B
调度切图后： A->src[0] = X_gpu   C->src[0] = B_cpu
```

因此 GPU 片段实际读取 X 的 GPU 副本，CPU 片段实际读取 B 的 CPU 副本。供分配器使用的内部图加入了辅助条目；提交给后端计算的两个片段仍分别包含 A、B 和 C。

实际节点数变成：

```text
3 个原始节点 + 2 个需要复制的输入 × 2 = 7 个内部图节点
```

当前宏默认值为：

```cpp
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
```

初始化时的容量计算为：

```text
nodes_size = 原图容量 + 估算片段数 × 估算的每片段输入容量 × 2
           = 8 + 8 × 30 × 2
           = 488
```

四个数组各自分配 488 个 `int` 元素。`nodes_size` 的单位是元素个数，`calloc` 再通过第二个参数转换为字节数。

以第一行分配为例：

```cpp
sched->node_backend_ids = (int *) calloc(
    nodes_size, sizeof(sched->node_backend_ids[0]));

// 在本例中，相当于：
sched->node_backend_ids = (int *) calloc(488, sizeof(int));
```

`sizeof(...[0])` 取得数组元素类型的大小，在这里就是 `sizeof(int)`。若目标平台的 `int` 为 4 字节，每张表占 1952 字节，四张表合计 7808 字节。实际有效元素数仍由内部图的 `n_nodes`、`n_leafs` 决定。

本版本片段输入容器支持动态扩容，`30` 同时用于输入容器的初始容量。因此这里是初始化容量估算，不应把它理解成每个片段始终只能有 30 个输入，或者把 488 当作本次实际节点数。

### 1.3 四个后端编号数组如何对应内部图

#### node_backend_ids：当前节点编号

`node_backend_ids[i]` 对应 `sched->graph.nodes[i]`，记录该条目的后端编号。编号还用于选择图内存分配器中的缓冲区类型。

| 下标 i | 内部图节点 | 后端 | `node_backend_ids[i]` |
|---|---|---|---|
| 0 | X_dep | CPU | 1 |
| 1 | X_gpu | GPU | 0 |
| 2 | A | GPU | 0 |
| 3 | B | GPU | 0 |
| 4 | B_dep | GPU | 0 |
| 5 | B_cpu | CPU | 1 |
| 6 | C | CPU | 1 |

有效部分为：

```text
node_backend_ids = [1, 0, 0, 0, 0, 1, 1]
sched->graph.n_nodes = 7
```

下标按当前内部图的顺序解释。例如 A 在原图的下标是 0，加入辅助条目后，在内部图中的下标是 2。

#### leaf_backend_ids：当前叶子编号

`leaf_backend_ids[i]` 对应 `sched->graph.leafs[i]`。

当 `parallel = false`，本例内部图叶子仍然只有 X：

```text
sched->graph.leafs = [X]
leaf_backend_ids = [1]
sched->graph.n_leafs = 1
```

节点表使用前 7 个元素，叶子表使用前 1 个元素。两张表拥有各自独立的 488 个元素容量。

开启流水线后，输入副本还会加入内部图的叶子表，让分配器提前分配并保留其内存。

#### prev_node_backend_ids 与 prev_leaf_backend_ids：上一轮编号

这两张表保存上一轮构建内部图时的后端编号，供分配逻辑比较。

每次切图时，调度器交换当前表与上一轮表的指针，然后填写新的当前表：

```cpp
int * tmp = sched->node_backend_ids;
sched->node_backend_ids = sched->prev_node_backend_ids;
sched->prev_node_backend_ids = tmp;

tmp = sched->leaf_backend_ids;
sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
sched->prev_leaf_backend_ids = tmp;
```

假设节点表底层两块内存分别叫 P、Q：

| 时刻 | 当前表指向 | 上一轮表指向 |
|---|---|---|
| 初始化后 | P，全部为 0 | Q，全部为 0 |
| 第一轮交换并填写后 | Q，保存第一轮编号 | P，仍为初始值 |
| 第二轮交换并填写后 | P，保存第二轮编号 | Q，保留第一轮编号 |

如果第二轮图结构和后端归属相同：

```text
当前节点编号： [1, 0, 0, 0, 0, 1, 1]
上一轮编号：   [1, 0, 0, 0, 0, 1, 1]
```

分配代码检查同一位置的编号。若编号改变，且对应的 `bufts` 也不同，就重新预留图的内存。即使编号没有改变，旧分配方案无法用于当前图时，分配器也会要求重新预留。

`calloc` 将数组初始化为 0。这个初始值只是清零结果；真正有效的后端编号在构建内部图时填写。

### 1.4 再看一次后端归属发生变化的比较

前两轮都使用 GPU 执行 A、B，CPU 执行 C。第三轮重新创建一份相同依赖关系的原图，并假设 A、B、C 都被指定给 CPU。

此时 X 和所有计算节点都在 CPU 上，本例需要的片段与内部图为：

```text
原图节点：        [A, B, C]
实际片段：        1 个，全部交给 CPU
跨后端复制输入：  0 个
内部图节点：      [A, B, C]
当前节点编号：    [1, 1, 1]
当前叶子编号：    [1]
```

指针交换后，上一轮节点表仍保存：

```text
prev_node_backend_ids = [1, 0, 0, 0, 0, 1, 1]
```

源码按当前内部图的节点数量 3，比较前 3 个位置：

| 位置 i | 上一轮内部图条目 / 编号 | 当前内部图条目 / 编号 | 编号是否变化 |
|---|---|---|---|
| 0 | X_dep / 1 | A / 1 | 否 |
| 1 | X_gpu / 0 | B / 1 | 是 |
| 2 | A / 0 | C / 1 | 是 |

在位置 1，编号从 0 变为 1。本例 GPU 和 CPU 使用不同的缓冲区类型，因此满足：

```cpp
node_backend_ids[1] != prev_node_backend_ids[1] &&
bufts[node_backend_ids[1]] != bufts[prev_node_backend_ids[1]]
```

于是 `backend_ids_changed` 设置为 `true`，分配代码重新预留图的数据内存。

这个例子也说明，编号表的下标对应各轮内部图中的位置。图结构变化时，同一个位置可能对应不同的张量。比较结果用于判断旧内存分配方案能否继续使用。

### 1.5 示例一的变量对照

| 变量 | 本例取值 | 含义与单位 |
|---|---|---|
| `graph_size`，创建函数参数 | 8 | 创建时提供的图容量参数，单位是条目数 |
| `ggml_sched_max_splits` | 8 | 用于初始化空间估算的片段数 |
| `GGML_SCHED_MAX_SPLIT_INPUTS` | 默认 30 | 此版本输入容器的初始容量，也参与初始化估算 |
| `nodes_size` | 488 | 四张编号表各自分配的 `int` 元素数量 |
| `sched->n_splits`，原场景 | 2 | 本次实际执行片段数 |
| `sched->graph.n_nodes`，原场景 | 7 | 本次内部图实际节点数 |
| `sched->graph.n_leafs`，非流水线场景 | 1 | 本次内部图实际叶子数 |
| `node_backend_ids` 有效部分 | `[1, 0, 0, 0, 0, 1, 1]` | 当前内部图节点的后端编号 |
| `leaf_backend_ids` 有效部分 | `[1]` | 当前内部图叶子的后端编号 |
| 两张 `prev_*` 表 | 上一轮有效编号 | 用来判断分配归属是否变化 |

<a id="example-2"></a>

## 示例二：管理资源与执行流程

沿用示例一的后端与原始计算图：X 在 CPU 上，A、B 由 GPU 执行，C 由 CPU 执行。假设 X 已通过 `ggml_set_input(X)` 标记为用户输入。

先按 `parallel = false` 查看初始化与执行，再单独将其改为 `true`，观察输入副本槽位和事件的变化。

这一段代码为后续执行准备的资源分别是：

| 资源 | 创建时准备什么 | 后续何时使用 |
|---|---|---|
| `context_buffer` | 临时对象使用的连续主机空间 | 切图时初始化临时上下文、创建副本和 view 描述对象 |
| `splits` | 16 个执行片段描述位置 | 切图时填写，计算时依次读取 |
| `graph_inputs` | 默认 30 个输入张量指针位置 | 开启流水线且切图时收集需要管理的用户输入 |
| `backends`、`bufts` | 后端句柄和所选缓冲区类型 | 选择执行后端、分配数据存储 |
| `events` | 开启流水线时尝试创建各后端、各槽位事件 | 异步计算提交后记录，复用槽位前等待 |
| `galloc` | 图数据内存的分配管理结构 | 根据内部图预留、绑定和复用数据空间 |
| `op_offload` | 一个布尔配置 | 后端选择时判断是否启用特定 offload 分支 |

### 2.1 context_buffer：临时张量描述对象的空间

对应代码：

```cpp
sched->context_buffer_size =
    ggml_sched_max_splits * GGML_SCHED_MAX_SPLIT_INPUTS * 2 * sizeof(struct ggml_tensor)
    + ggml_graph_overhead_custom(graph_size, false);
sched->context_buffer = (char *) malloc(sched->context_buffer_size);
```

`context_buffer_size` 的单位是字节。`context_buffer` 指向主机内存中的一块连续空间，后续交给临时 `ggml_context` 管理。

沿用本例，设：

```text
T = sizeof(struct ggml_tensor)
H8 = ggml_graph_overhead_custom(8, false)
```

则创建时计算：

```text
context_buffer_size = 8 × 30 × 2 × T + H8
                    = 480 × T + H8 字节
```

第一项按预计需要创建的辅助张量描述对象估算空间。`sizeof(struct ggml_tensor)` 描述类型、形状、步长、输入指针等元数据的大小。实际在上下文中创建对象还涉及对象头和对齐，因此这个表达式应理解为初始化预留公式。

第二项是一个容量为 8、无需梯度数组的计算图所需的管理空间估算，包含对象头、图结构体、节点与叶子指针数组、哈希表等，以及对齐填充。`false` 表示不包含梯度数组。

当前切图实现将 `sched->graph.nodes` 和 `sched->graph.leafs` 通过 `realloc` 单独管理；公式中包含图管理空间的预留项，并不意味着内部图的所有配套数组实际都放在 `context_buffer` 内。

后续切图时，源码使用这块空间初始化临时上下文：

```cpp
struct ggml_init_params params = {
    /* .mem_size =   */ sched->context_buffer_size,
    /* .mem_buffer = */ sched->context_buffer,
    /* .no_alloc =   */ true
};

ggml_free(sched->ctx);
sched->ctx = ggml_init(params);
```

本例中的 `X_gpu`、`B_cpu` 副本描述对象，以及 `X_dep`、`B_dep` view 描述对象，都会通过 `sched->ctx` 创建。每轮切图释放旧临时上下文，再用同一块预留空间初始化新的上下文。

例如一个 F32 张量包含 1024 个元素：

```text
张量描述对象：保存 shape、stride、buffer、data、src 等信息
数值数据空间：1024 × sizeof(float) = 4096 字节
```

`no_alloc = true` 使这个临时上下文负责描述对象的创建。4096 字节的实际数据存储由后端缓冲区和图内存分配器安排；view 的数据则共享源张量。

### 2.2 splits：记录每个执行片段

对应代码：

```cpp
const int initial_splits_capacity = 16;
sched->splits = (ggml_backend_sched_split *) calloc(
    initial_splits_capacity, sizeof(sched->splits[0]));
sched->splits_capacity = initial_splits_capacity;
```

初始化时分配 16 个片段描述结构。`splits_capacity` 是可容纳的片段数量，`n_splits` 是本次实际片段数量。

一个片段保存：

| 成员 | 用途 |
|---|---|
| `backend_id` | 执行此片段的后端编号 |
| `i_start`、`i_end` | 原图节点数组中的范围 `[i_start, i_end)` |
| `inputs`、`n_inputs` | 需要复制到此片段后端的源张量及其数量 |
| `inputs_capacity` | 当前片段输入指针数组的容量 |
| `graph` | 此片段的图视图，交给后端执行 |

本例实际填入的内容：

```text
splits[0]:
    backend_id = 0             // GPU
    i_start = 0
    i_end = 2                  // 原图节点 A、B
    inputs = [X]
    n_inputs = 1

splits[1]:
    backend_id = 1             // CPU
    i_start = 2
    i_end = 3                  // 原图节点 C
    inputs = [B]
    n_inputs = 1

sched->n_splits = 2
sched->splits_capacity = 16
```

调度执行时，先读取 `splits[0]`，准备 X 的 GPU 副本，再在 GPU 上执行 A、B；随后读取 `splits[1]`，准备 B 的 CPU 副本，再在 CPU 上执行 C。

创建 `splits` 时，各片段自己的 `inputs` 指针仍为空，后续需要时才分配。片段超过 16 个时，容器容量翻倍，例如 16 → 32 → 64。

### 2.3 graph_inputs：流水线需要管理的用户输入

对应代码：

```cpp
sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
sched->graph_inputs = (struct ggml_tensor **) calloc(
    sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));
```

当前默认预留 30 个张量指针的位置。这个数组保存张量地址；每个张量的数值数据存储在其关联缓冲区中。

在切图过程中，源码针对带 `GGML_TENSOR_FLAG_INPUT` 标记、且 `n_copies > 1` 的输入准备副本，并把需要管理的源输入加入 `graph_inputs`。

假设本例通过 `ggml_set_input(X)` 标记了 X，B 没有输入标记：

| 配置 | `n_copies` | 本例的 `graph_inputs` 有效内容 |
|---|---|---|
| `parallel = false` | 1 | 空，`n_graph_inputs = 0` |
| `parallel = true`，默认配置 | 4 | `[X]`，`n_graph_inputs = 1` |

B 是图内计算产生的中间结果。它需要跨后端复制，所以登记在 `splits[1].inputs` 中。

不同数组各自回答不同的问题：

| 数组 | 本例内容 | 回答的问题 |
|---|---|---|
| 原图 `leafs` | X | 哪些张量是原图的叶子？ |
| `splits[0].inputs` | X | GPU 片段开始前要复制哪个源张量？ |
| `splits[1].inputs` | B | CPU 片段开始前要复制哪个源张量？ |
| `graph_inputs`，开启流水线时 | X | 哪些用户输入需要参与流水线副本管理？ |

本例开启流水线后，调度器为 X 准备源后端上的副本槽位。当前槽位使用原始 X，其余槽位创建新的描述对象；需要送往 GPU 的 X 还会在 GPU 后端准备对应副本槽位。实际副本指针存放在 `hv_tensor_copies` 中，`graph_inputs` 保存的是源输入列表。

以第一次使用槽位 0 为例，副本登记关系可简化表示为：

| 源张量 / 副本所在后端 | 槽位 0 | 槽位 1 | 槽位 2 | 槽位 3 |
|---|---|---|---|---|
| X / CPU | 原始 X | X_cpu_1 | X_cpu_2 | X_cpu_3 |
| X / GPU | X_gpu_0 | X_gpu_1 | X_gpu_2 | X_gpu_3 |
| B / CPU | B_cpu_0 | B_cpu_1 | B_cpu_2 | B_cpu_3 |

表中的副本名字用于说明。源码通过 `hv_tensor_copies[哈希槽位][后端编号][副本槽位]` 的展平布局记录它们的指针。原始计算结果 B 位于 GPU，表中列出的 B_cpu_* 是 CPU 片段需要的跨后端副本。

这些张量的副本槽位在流水线模式下设置输入和输出标记，防止分配器按普通临时张量处理它们的存储。

超过初始容量时，`graph_inputs` 也会翻倍扩容，例如 30 → 60。

### 2.4 backends 与 bufts：计算后端和数据存储类型

对应代码：

```cpp
for (int b = 0; b < n_backends; b++) {
    sched->backends[b] = backends[b];
    sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
    GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));
    // 后面创建流水线事件。
}
```

`backends[b]` 是已经创建的后端句柄，负责执行计算和相关操作；`bufts[b]` 是缓冲区类型，规定如何分配、对齐和使用数据存储。

本例将整个 `bufts` 参数传为 `nullptr`，初始化后的对应关系为：

| b | `sched->backends[b]` | `sched->bufts[b]` |
|---|---|---|
| 0 | GPU 后端 | 此 GPU 后端的默认缓冲区类型，本例用来分配设备数据 |
| 1 | CPU 后端 | 此 CPU 后端的默认缓冲区类型，本例用来分配主机数据 |

如果调用者传入非空的 `bufts` 数组，代码直接使用 `bufts[b]`。默认类型回退的条件是整个数组参数为空。

缓冲区类型决定分配方式。实际分配出来的 `ggml_backend_buffer` 对象才承载一块具体的数据空间。

`GGML_ASSERT` 检查指定后端是否支持使用所选缓冲区类型，保证随后根据后端编号选择存储类型时，两者能够配合工作。

### 2.5 events：防止副本槽位被过早覆盖

对应代码：

```cpp
if (sched->n_copies > 1) {
    for (int c = 0; c < sched->n_copies; c++) {
        sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
    }
}
```

`events[b][c]` 是后端 b 上、副本槽位 c 对应的同步事件，用来追踪该槽位相关异步工作的完成情况。

当前默认宏配置下：

```cpp
sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;
// GGML_SCHED_MAX_COPIES 默认为 4。
```

两个后端、四个槽位时，循环会尝试创建以下事件：

| 后端 | 槽位 0 | 槽位 1 | 槽位 2 | 槽位 3 |
|---|---|---|---|---|
| GPU，b = 0 | `events[0][0]` | `events[0][1]` | `events[0][2]` | `events[0][3]` |
| CPU，b = 1 | `events[1][0]` | `events[1][1]` | `events[1][2]` | `events[1][3]` |

是否真正创建出事件取决于后端设备的支持情况。`ggml_backend_event_new()` 在设备没有事件接口时返回空指针；后续调度代码会使用后端同步来保证顺序。

考虑一个需要复用 GPU 输入副本槽位的场景：

```text
先前的一次运行：GPU 正在读取槽位 0 的输入副本
新的一次运行：准备再次向槽位 0 写入新的输入数据
```

如果先前的计算尚未完成，直接覆盖会影响正在进行的计算。调度器在片段提交计算后记录对应事件；再次使用该槽位时，先等待相关工作完成，再安排覆盖。

| 操作 | 作用 |
|---|---|
| `ggml_backend_event_record` | 在后端上记录当前工作的完成位置 |
| `ggml_backend_event_synchronize` | 在主机侧等待事件完成 |
| `ggml_backend_event_wait` | 让后端的后续工作等待事件 |
| `ggml_backend_synchronize` | 没有对应事件时使用的后端同步方式 |

用户输入跨后端复制时，源码还会立即完成复制，以避免用户随后修改输入数据时影响尚未完成的数据传输。

槽位选择在 `ggml_backend_sched_alloc_graph()` 中推进：

```cpp
sched->cur_copy = sched->next_copy;
sched->next_copy = (sched->next_copy + 1) % sched->n_copies;
```

在槽位选择没有被同步逻辑重置的情况下，多次重新分配图可以依次使用 0、1、2、3、0 等槽位。已分配图的重复计算调用不一定推进槽位；同步逻辑也可能将 `next_copy` 重置为 0。

四个槽位提供流水线资源。实际重叠执行还取决于异步调用方式、图分配状态，以及后端的计算和复制支持。

在允许槽位连续轮换、且后端支持对应事件的异步运行场景下，可以用下表理解槽位复用。每一轮都假设进入图分配流程，期间没有将 `next_copy` 重置为 0：

| 轮次 | 本轮槽位 | GPU 片段读取的 X 副本 | 片段提交后的事件记录 |
|---|---|---|---|
| 第 1 轮 | 0 | X_gpu_0 | 记录 `events[0][0]` |
| 第 2 轮 | 1 | X_gpu_1 | 记录 `events[0][1]` |
| 第 3 轮 | 2 | X_gpu_2 | 记录 `events[0][2]` |
| 第 4 轮 | 3 | X_gpu_3 | 记录 `events[0][3]` |
| 第 5 轮 | 0 | X_gpu_0 | 复用前等待该槽位的先前工作，提交后再次记录事件 |

如果第 1 轮对 X_gpu_0 的使用仍未完成，第 5 轮会先等待。不同槽位提供独立的输入存储，使调度器能够在满足依赖与同步条件的前提下安排后续工作。

本例中，把 B 从 GPU 复制回 CPU 的步骤可能需要等待 GPU 完成，因此实际执行是否出现不同轮次的重叠，仍取决于后端与调用流程。上表用来说明槽位轮换和事件复用的对应关系。

### 2.6 galloc：为图中的张量安排数据内存

对应代码：

```cpp
sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
```

`galloc` 是图内存分配器。创建时，它保存各缓冲区类型，并建立分配管理结构。此时还没有依据具体计算图分配完整的计算数据缓冲区。

后续切图完成后，调度器把内部图与后端编号表交给分配器：

```cpp
ggml_gallocr_reserve_n(
    sched->galloc,
    &sched->graph,
    sched->node_backend_ids,
    sched->leaf_backend_ids);

ggml_gallocr_alloc_graph(sched->galloc, &sched->graph);
```

本例中的对应关系是：

```text
X_gpu、A、B  → 后端编号 0 → bufts[0] → GPU 数据缓冲区
B_cpu、C    → 后端编号 1 → bufts[1] → CPU 数据缓冲区
X_dep、B_dep → 共享各自源张量的存储
```

已提前分配的张量保留其原有数据空间。分配器为需要分配的张量安排存储，按生命周期规划复用，并避免源张量在复制前被过早回收。

例如两个张量都需要同一后端上的 4096 字节，且生命周期不重叠，那么分配器可能让它们复用同一段空间。对于还要保留的输入或输出，则必须遵守相应的生命周期约束。

`ggml_gallocr_new_n()` 还会检查多个编号是否使用同一个缓冲区类型；类型相同的条目可以共享分配器。

与前面的空间分工如下：

| 资源 | 本例用途 |
|---|---|
| `context_buffer` / `sched->ctx` | 创建 X_gpu、B_cpu、X_dep、B_dep 等临时张量描述对象 |
| 后端编号数组 | 记录内部图各条目的后端编号，为分配器选择缓冲区类型 |
| `splits` | 记录 A、B 交给 GPU，C 交给 CPU，以及各片段的复制输入 |
| `galloc` 与后端缓冲区 | 规划、分配和绑定实际张量数据的存储 |

### 2.7 op_offload：允许特定算子优先交给高优先级后端

对应代码：

```cpp
sched->op_offload = op_offload;
```

创建时保存这个布尔开关，实际选择后端时再使用它。

换一个权重相关的例子：

```text
W：存放在 CPU 主机缓冲区中的模型权重，标记为 WEIGHTS
Y = W × X：待执行的矩阵乘法
后端列表：[GPU, CPU]
```

选择后端时，调度器通常优先考虑权重所在的缓冲区。如果此时选择的是最后一个 CPU 后端，权重缓冲区位于主机，并且 `op_offload = true`，它还会检查前面的高优先级后端。

GPU 既支持这个算子，又通过自己的 `offload_op` 判断愿意接收该算子时，调度器可以选择 GPU 执行矩阵乘法，并在需要时准备权重副本。

这个开关启用的是特定选择规则。它不会强制所有算子交给 GPU。设为 `false` 会关闭这条权重相关的 offload 分支；其他后端选择规则仍然可能选择 GPU。

### 2.8 初始化后的状态与后续流程

这两行后面，创建函数还调用：

```cpp
ggml_backend_sched_reset(sched);
return sched;
```

初次重置时，后端哈希记录中的编号初始化为 `-1`，表示尚未分配后端；张量副本指针清空；`is_alloc` 设置为 `false`。调度器已准备好资源，等待实际图进入切图、分配和执行阶段。

非流水线配置下，不同阶段的状态可以对照如下：

| 项目 | 创建函数刚返回 | 切图和分配完成后 | 执行时的作用 |
|---|---|---|---|
| `context_buffer` | 已申请，容量为 `480 × T + H8` 字节 | 临时上下文使用其中空间创建描述对象 | 描述副本、view 和输入关系 |
| `sched->ctx` | 空，尚未创建临时上下文 | 已使用预留空间初始化 | 提供本轮临时对象 |
| `splits_capacity` / `n_splits` | 16 / 0 | 16 / 2 | 按顺序执行两段 |
| `graph_inputs_capacity` / `n_graph_inputs` | 30 / 0 | 30 / 0 | 非流水线场景没有登记源输入列表 |
| `n_copies` | 1 | 1 | 使用唯一的副本槽位 |
| `events` | 没有通过事件创建接口创建对象 | 仍为空 | 必要时通过后端同步保证顺序 |
| `galloc` | 分配管理结构已创建 | 图的数据存储已经安排 | 使用已绑定的张量数据空间 |
| `is_alloc` | false | true | 表示图已有可用分配 |

这里的 T、H8 沿用 2.1 节中的定义。开启流水线时，`n_copies` 默认变为 4，切图后本例 `n_graph_inputs` 变为 1，并尝试创建同步事件。

对应整个例子的流程为：

```text
创建调度器
    预留编号数组、临时对象空间、片段表、流水线输入表
    保存后端与缓冲区类型
    按需尝试创建事件
    创建图内存分配器
    重置初始状态

收到实际计算图 [A, B, C]
    确定后端归属
    切成 GPU 片段 [A, B] 和 CPU 片段 [C]
    创建 X_gpu、B_cpu 和生命周期辅助条目
    构建内部图与后端编号表
    通过 galloc 安排所需的数据存储

执行
    将 X 复制到 GPU 副本
    GPU 执行 A、B
    将 B 复制到 CPU 副本
    CPU 执行 C
    按需记录事件与同步
```

### 2.9 把执行动作与管理资源对应起来

| 执行步骤 | 使用的片段记录 | 使用的数据对象 | 相关管理资源 |
|---|---|---|---|
| 准备 GPU 输入 | `splits[0].inputs = [X]` | 从 X 复制到当前槽位的 X_gpu | 临时描述对象、图分配器；流水线时还使用对应事件 |
| GPU 计算 A、B | `splits[0].graph` | 读取 X_gpu，生成 A、B | `backends[0]` 及 GPU 数据缓冲区 |
| 准备 CPU 输入 | `splits[1].inputs = [B]` | 从 B 复制到当前槽位的 B_cpu | 源张量依赖、目标副本空间及复制同步 |
| CPU 计算 C | `splits[1].graph` | 读取 B_cpu，生成 C | `backends[1]` 及 CPU 数据缓冲区 |
| 后续复用空间或槽位 | 根据本次图与分配状态判断 | 复用不再被使用的存储 | `galloc` 的生命周期规划；流水线时使用事件保护槽位 |

源码阅读顺序可以从 `ggml_backend_sched_new()` 开始，再依次查看 `ggml_backend_sched_split_graph()`、`ggml_backend_sched_alloc_splits()`、`ggml_backend_sched_compute_splits()`，把创建资源、填写记录、分配数据和执行计算连起来理解。
