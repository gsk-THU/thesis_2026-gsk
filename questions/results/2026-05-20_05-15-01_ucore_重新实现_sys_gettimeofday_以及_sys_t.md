# OS 实验面试问题集

**生成时间**: 2026-05-20T05:15:01.919873

**OS 类型**: ucore

**实验要求**: 重新实现 sys_gettimeofday 以及 sys_trace
引入虚存机制后，原来内核的 sys_gettimeofday 和 sys_trace 函数实现就无效了。 请你重写这两个系统调用的代码，恢复其正常功能。

此外，由于本章我们有了地址空间作为隔离机制，所以 sys_trace 需要考虑一些额外的情况：

在读取（trace_request 为 0）时，如果对应地址用户不可见或不可读，则返回值应为 -1（int 的 -1，而非 uint8_t）。

在写入（trace_request 为 1）时，如果对应地址用户不可见或不可写，则返回值应为 -1（int 的 -1，而非 uint8_t）。

mmap 匿名映射
mmap 在 Linux 中主要用于在内存中映射文件，本次实验简化它的功能，仅使用匿名映射来申请内存。

请实现 mmap 和 munmap 系统调用，mmap 定义如下：

int mmap(void* start, unsigned long long len, int prot, int flags)
syscall ID：222

功能：申请长度为 len 字节的匿名物理内存（不要求实际物理内存位置，可以随便找一块），并映射到 addr 开始的虚拟地址，内存页属性为 prot。

参数：
start：需要映射的虚存起始地址。

len：映射字节长度，可以为 0。

prot：第 0 位表示是否可读，第 1 位表示是否可写，第 2 位表示是否可执行。其他位无效且必须为0。

flags：默认为 MAP_ANONYMOUS，可忽略该参数。

返回值:
成功返回 0，错误返回 -1。

说明：
为了简单， addr 要求按页对齐(否则报错)， len 可直接按页上取整。

为了简单，不考虑分配失败时的页回收。

flags 参数留待后续实验拓展。

错误：
[addr, addr + len) 存在已经被映射的页。

物理内存不足。

prot & ~0x7 == 0， prot 其他位必须为 0

prot & 0x7 != 0，不可读不可写不可执行的内存无意义

munmap 系统调用定义如下：

int munmap(void* start, unsigned long long len)
syscall ID：215

功能：取消到虚拟地址区间 [start, start + len) 的映射。

参数和返回值：参考 mmap

说明：
为了简单，参数错误时不考虑内存的恢复和回收。

错误：
[start, start + len) 中存在未被映射的虚存。

正确实现后，你的 os 应该能够正确运行 ch4_* 对应的一些测试用例，make test BASE=0 来执行测试。

**摘要**: 基于0个修改文件和36个新增文件生成4个问题（OS类型: ucore, RAG检索: 4次）

**RAG 检索**: 执行了 4 次工具调用
  - `search_by_lab`: {'lab_name': 'ch4'}
  - `search_os_concept`: {'concept': '页表', 'detail_level': 'detailed'}
  - `search_os_concept`: {'concept': 'mmap', 'detail_level': 'standard'}
  - `search_by_code_symbol`: {'symbol_name': 'kalloc', 'context': '物理内存分配'}

---

## 问题 1 [IMPLEMENTATION]

在 `sys_trace` 的实现中，你需要检查用户提供的地址是否可读或可写。请看这段典型的页表遍历权限检查代码：



这里的 `PTE_U`（User位）在权限检查中起什么作用？如果省略对 `PTE_U` 的检查，仅仅检查 `PTE_R` 或 `PTE_W`，会导致什么安全问题？请结合 RISC-V SV39 页表项标志位的定义解释。

**参考代码**:

```
int check_user_permission(uint64_t va, int write) {
    pte_t *pte = walk(current->pagetable, va, 0);  // 查找页表项
    if (pte == 0) return -1;  // 页表项不存在
    if (!(*pte & PTE_V)) return -1;  // 无效页
    
    uint64_t flags = PTE_FLAGS(*pte);
    if (write) {
        if (!(flags & PTE_W) || !(flags & PTE_U)) return -1;
    } else {
        if (!(flags & PTE_R) || !(flags & PTE_U)) return -1;  // 读取检查
    }
    return 0;
}
```

**出题理由**: 课程资料指出"页表项的标志位来源于当前逻辑段的类型为MapPermission的统一配置"，且强调地址空间作为隔离机制。`PTE_U` 是区分用户态和内核态访问权限的关键位，忽略此检查会导致用户态可能访问内核页表映射的物理页，破坏隔离性。此问题考察学生对权限检查完整性的理解。

---

## 问题 2 [IMPLEMENTATION]

`mmap` 系统调用需要将 `prot` 参数（如 `PROT_READ`、`PROT_WRITE`）转换为页表项标志位。请看这段权限映射代码：



代码中先检查 `(prot & 0x7) == 0` 返回错误，即拒绝"不可读不可写不可执行"的映射请求。但后续却允许单独设置 `PTE_R`、`PTE_W`、`PTE_X` 的任意组合。这种处理与 Linux 实际的 `mmap` 行为有何差异？在你的实现中，如果用户请求 `prot = 0x3`（可读可写，但包含保留位），为什么 `prot & ~0x7` 的检查是必要的？

**参考代码**:

```
int sys_mmap(uint64_t start, uint64_t len, int prot, int flags) {
    // 权限检查：prot & ~0x7 == 0 且 prot & 0x7 != 0
    if ((prot & ~0x7) != 0 || (prot & 0x7) == 0)
        return -1;
    
    int pte_flags = PTE_U | PTE_V;  // 用户态可访问且有效
    if (prot & 0x1) pte_flags |= PTE_R;  // 可读
    if (prot & 0x2) pte_flags |= PTE_W;  // 可写
    if (prot & 0x4) pte_flags |= PTE_X;  // 可执行
    
    // 后续映射逻辑...
    for (uint64_t va = start; va < start + len; va += PGSIZE) {
        uint64_t pa = (uint64_t)kalloc();  // 分配物理页
        if (pa == 0) return -1;  // 物理内存不足
        mappages(current->pagetable, va, PGSIZE, pa, pte_flags);
    }
    return 0;
}
```

**出题理由**: 课程资料提到页表项标志位来源于 MapPermission 的转换。此问题考察学生对权限位掩码检查的理解，以及对"无意义映射"（零权限）和"非法标志位"（保留位被设置）的区分处理，确保与实验要求中的错误检查条件一致。

---

## 问题 3 [DEBUGGING]

在 `munmap` 的实现中，需要检查 `[start, start + len)` 区间内是否存在未被映射的虚存页。请看这段解除映射的代码：



这段代码采用了"先遍历检查，再遍历解除"的两遍扫描策略。如果改为"边检查边解除"（即在第一个循环中直接 `kfree` 并清除页表项），在并发场景下（假设未来支持多线程或多核）可能会引入什么风险？此外，实验要求提到"参数错误时不考虑内存的恢复和回收"，如果第一个循环中发现中间某页未映射而返回 -1，前面已经释放的物理页应该如何处理？你的代码是否遵循了这一点？

**参考代码**:

```
int sys_munmap(uint64_t start, uint64_t len) {
    if (start % PGSIZE != 0) return -1;  // 页对齐检查
    
    for (uint64_t va = start; va < start + len; va += PGSIZE) {
        pte_t *pte = walk(current->pagetable, va, 0);
        // 检查该虚拟页是否已被映射
        if (pte == 0 || !(*pte & PTE_V)) {
            return -1;  // 存在未映射的页，报错
        }
    }
    
    // 确认全部已映射后，执行解除映射
    for (uint64_t va = start; va < start + len; va += PGSIZE) {
        pte_t *pte = walk(current->pagetable, va, 0);
        uint64_t pa = PTE2PA(*pte);
        kfree((void*)pa);  // 释放物理页
        *pte = 0;  // 清除页表项
    }
    return 0;
}
```

**出题理由**: 考察学生对原子性操作和资源管理的理解。课程资料中 `kalloc` 采用链表管理物理页，强调页粒度管理。此问题区分"参数验证"和"实际操作"两个阶段，避免在验证失败时留下不一致状态（部分页已释放），同时考察对实验简化要求的遵循。

---

## 问题 4 [CONCEPT]

`mmap` 要求 `addr` 必须按页对齐（`PGSIZE` 对齐），且 `len` 按页上取整。请看这段地址对齐处理代码：



如果用户传入 `len = 0`，按照实验说明"len 可为 0"且"len 可直接按页上取整"，你的代码会如何处理？`PGROUNDUP(0)` 的结果是 0，此时如果直接返回成功而不分配任何物理页，与分配一页（如注释所示）相比，哪种更符合实验要求？另外，如果用户传入 `start = 0x1000`（已对齐）和 `len = 0x1`（未对齐），映射后的实际覆盖范围是 `[0x1000, 0x2000)`，这种"超额映射"（over-allocation）在页表管理中是否安全？请解释。

**参考代码**:

```
#define PGROUNDUP(sz) (((sz) + PGSIZE - 1) & ~(PGSIZE - 1))
#define PGROUNDDOWN(a) (((a)) & ~(PGSIZE - 1))

int sys_mmap(uint64_t start, uint64_t len, int prot, int flags) {
    if (start != PGROUNDDOWN(start))  // 检查是否页对齐
        return -1;
    
    uint64_t map_len = PGROUNDUP(len);  // 长度向上取整
    if (map_len == 0) map_len = PGSIZE;  // len为0时至少映射一页？
    
    // 检查重叠...
    for (uint64_t va = start; va < start + map_len; va += PGSIZE) {
        pte_t *pte = walk(current->pagetable, va, 0);
        if (pte && (*pte & PTE_V))  // 已存在映射
            return -1;
    }
    // ...
}
```

**出题理由**: 考察对页粒度管理的理解。课程资料指出"页表只能以页为单位"维护映射关系。此问题针对边界情况（零长度、非对齐长度）的处理，以及超额映射对地址空间隔离的影响，确保学生理解页对齐的硬件约束。

---


## 代码变更摘要

```
新增文件: 36个
删除文件: 32个
修改文件: 0个
未变文件: 0个
```
