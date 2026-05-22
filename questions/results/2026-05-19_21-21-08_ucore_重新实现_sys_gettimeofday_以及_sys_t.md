# OS 实验面试问题集

**生成时间**: 2026-05-19T21:21:08.008923

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

**摘要**: 基于0个修改文件和36个新增文件生成4个问题（OS类型: ucore, RAG检索: 5次）

**RAG 检索**: 执行了 5 次工具调用
  - `search_os_concept`: {'concept': '虚拟内存', 'detail_level': 'detailed'}
  - `search_os_concept`: {'concept': '页表', 'detail_level': 'detailed'}
  - `search_by_code_symbol`: {'symbol_name': 'copyin', 'context': '系统调用访问用户地址'}
  - `search_by_code_symbol`: {'symbol_name': 'mmap', 'context': '内存映射系统调用'}
  - `search_by_lab`: {'lab_name': 'lab4'}

---

## 问题 1 [IMPLEMENTATION]

在重新实现 `sys_trace` 时，需要检查用户提供的地址是否具有相应的访问权限。请看这段可能的实现代码：



这段代码在权限检查逻辑上存在什么问题？特别是当 `addr` 位于虚拟页的末尾附近时（例如 `addr = 0x1000 - 1`），如果请求是读取/写入一个多字节数据（如 `int`），可能会发生什么？请结合页粒度权限管理说明原因，并给出正确的处理方法。

**参考代码**:

```
int sys_trace(void) {
    int pid, request, addr, data;
    argint(0, &pid);
    argint(1, &request);
    argint(2, &addr);
    argint(3, &data);
    
    struct proc *p = find_proc(pid);
    if (p == 0) return -1;
    
    pte_t *pte = walk(p->pagetable, addr, 0);
    if (pte == 0) return -1;
    
    if (request == 0) {  // 读取
        if ((*pte & PTE_V) == 0 || (*pte & PTE_R) == 0) 
            return -1;
        // 尝试读取...
    } else {  // 写入
        if ((*pte & PTE_V) == 0 || (*pte & PTE_W) == 0)
            return -1;
        // 尝试写入...
    }
    return 0;
}
```

**出题理由**: 考察学生是否真正理解页表权限检查的范围是页粒度而非字节粒度，以及是否考虑到数据可能跨越页边界的情况。这能区分"照搬代码"和"理解虚存本质"的学生。

---

## 问题 2 [CONCEPT]

课程资料中明确指出："用户在启用了虚拟内存之后，用户 syscall 给出的指针是不能直接用的，因为与内核的映射不一样... 更加推荐的是 copyin/out 接口"。请看 `sys_gettimeofday` 的两种实现方式：



请解释为什么方式A在引入虚存机制后必然失败，并详细说明 `copyout` 内部是如何完成地址转换的（从用户虚拟地址到物理地址的映射过程）。如果 `tv` 指针跨越了两个虚拟页（如 `tv = 0x1FFE`，`sizeof(t) = 8`），`copyout` 相比直接解引用有什么优势？

**参考代码**:

```
// 方式A：直接解引用（虚存引入前）
int sys_gettimeofday_old(struct timeval *tv) {
    tv->tv_sec = ticks / 100;
    tv->tv_usec = (ticks % 100) * 10000;
    return 0;
}

// 方式B：使用 copyout（虚存引入后）
int sys_gettimeofday(struct timeval *tv) {
    struct timeval t;
    t.tv_sec = ticks / 100;
    t.tv_usec = (ticks % 100) * 10000;
    
    if (copyout(myproc()->pagetable, (uint64)tv, (char*)&t, sizeof(t)) < 0)
        return -1;
    return 0;
}
```

**出题理由**: 结合课程资料中关于 copyin/out 的说明，考察学生对用户/内核地址空间隔离的理解，以及对跨页访问问题的认识。

---

## 问题 3 [DEBUGGING]

在 `mmap` 实现中，需要检查 `[start, start+len)` 区间是否存在已被映射的页。请看这段重叠检测和映射代码：



这段代码在错误处理方面存在明显的资源泄漏问题。请描述一种具体的执行场景（如特定参数序列），导致物理页帧泄漏。虽然实验要求"不考虑分配失败时的页回收"，请从操作系统原理角度分析，如果在实际生产系统中采用这种简化，会有什么严重后果？

**参考代码**:

```
int mmap(uint64 start, uint64 len, int prot, int flags) {
    if (start % PGSIZE != 0) return -1;  // 页对齐检查
    if (prot & ~0x7) return -1;          // 非法位检查
    
    uint64 end = PGROUNDUP(start + len);
    struct proc *p = myproc();
    
    // 第一阶段：检查是否已映射
    for (uint64 va = start; va < end; va += PGSIZE) {
        pte_t *pte = walk(p->pagetable, va, 0);
        if (pte && (*pte & PTE_V)) {
            return -1;  // 已映射
        }
    }
    
    // 第二阶段：分配并映射
    for (uint64 va = start; va < end; va += PGSIZE) {
        uint64 pa = (uint64)kalloc();
        if (pa == 0) {
            // 内存不足！但之前分配的页未回收
            return -1;
        }
        int perm = PTE_U | PTE_V;
        if (prot & 1) perm |= PTE_R;
        if (prot & 2) perm |= PTE_W;
        if (prot & 4) perm |= PTE_X;
        mappages(p->pagetable, va, PGSIZE, pa, perm);
    }
    return 0;
}
```

**出题理由**: 考察学生对资源管理完整性的思考，以及理解"实验简化"与"工程实现"的区别。这需要真正理解物理页帧的生命周期管理。

---

## 问题 4 [IMPLEMENTATION]

`mmap` 的 `prot` 参数与 RISC-V 页表项标志位存在映射关系。实验要求规定：`prot` 第0位表示可读，第1位表示可写，第2位表示可执行。请看这段权限转换代码：



根据实验要求："prot & 0x7 != 0，不可读不可写不可执行的内存无意义"，以及 "prot & ~0x7 == 0，prot 其他位必须为0"。这段代码缺少哪些必要的参数校验？特别是当 `prot` 为0或包含非法位时，应该如何处理？另外，在RISC-V架构中，如果设置了写权限（PTE_W），通常必须同时设置读权限（PTE_R），这段代码是否满足这一架构约束？

**参考代码**:

```
int mmap(uint64 start, uint64 len, int prot, int flags) {
    // 前置检查...
    
    int perm = PTE_U | PTE_V;  // 用户可访问且有效
    if (prot & 0x1) perm |= PTE_R;  // 读权限
    if (prot & 0x2) perm |= PTE_W;  // 写权限  
    if (prot & 4)   perm |= PTE_X;  // 执行权限
    
    // 当 prot == 0 时（无读/写/执行权限），perm 只有 PTE_U | PTE_V
    
    uint64 end = PGROUNDUP(start + len);
    for (uint64 va = start; va < end; va += PGSIZE) {
        uint64 pa = (uint64)kalloc();
        mappages(p->pagetable, va, PGSIZE, pa, perm);
    }
    return 0;
}
```

**出题理由**: 深入考察学生对权限位操作的理解，以及对RISC-V页表项规范的掌握（特别是W位必须配合R位的架构要求），区分"功能实现"和"规范符合性"。

---


## 代码变更摘要

```
新增文件: 36个
删除文件: 32个
修改文件: 0个
未变文件: 0个
```
