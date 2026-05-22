# OS 实验面试问题集

**生成时间**: 2026-05-20T08:26:19.695845

**OS 类型**: ucore

**实验要求**: 关于之前的 syscall
你仍需要迁移上一章的 sys_gettimeofday sys_mmap sys_munmap 以适应新的进程结构。不过， 从本章节开始，不再要求维护 sys_trace 这一系统调用。

进程创建
大家一定好奇过为啥进程创建要用 fork + execve 这么一个奇怪的系统调用，就不能直接搞一个新进程吗？思而不学则殆，我们就来试一试！这章的编程练习请大家实现一个完全 DIY 的系统调用 spawn，用以创建一个新进程。

spawn 系统调用定义( 标准spawn看这里 )：

int sys_spawn(char *filename)
syscall ID: 400

功能：相当于 fork + exec，新建子进程并执行目标程序。

说明：成功返回子进程id，否则返回 -1。

可能的错误：
无效的文件名。

进程池满/内存不足等资源错误。

实现完成之后，你应该能通过 ch5_spawn* 对应的所有测例，在 shell 中执行 ch5_usertest 来执行所有测试，应当发现除了setprio相关的测例均正确。

tips:

注意 fork 的执行流，新进程 context 的 ra 和 sp 与父进程不同。所以你不能在内核中通过 fork 和 exec 的简单组合实现 spawn。

在 spawn 中不应该有任何形式的内存拷贝。

stride 调度算法
lab3中我们引入了任务调度的概念，可以在不同任务之间切换，目前我们实现的调度算法十分简单，存在一些问题且不存在优先级。现在我们要为我们的 os 实现一种带优先级的调度算法：stide 调度算法。

算法描述如下:

为每个进程设置一个当前 stride，表示该进程当前已经运行的“长度”。另外设置其对应的 pass 值（只与进程的优先权有关系），表示对应进程在调度后，stride 需要进行的累加值。

每次需要调度时，从当前 runnable 态的进程中选择 stride 最小的进程调度。对于获得调度的进程 P，将对应的 stride 加上其对应的步长 pass。

一个时间片后，回到上一步骤，重新调度当前 stride 最小的进程。

可以证明，如果令 P.pass = BigStride / P.priority 其中 P.pass 为进程的 pass 值，P.priority 表示进程的优先权（大于 1），而 BigStride 表示一个预先定义的大常数，则该调度方案为每个进程分配的时间将与其优先级成正比。证明过程我们在这里略去，有兴趣的同学可以在网上查找相关资料。

其他实验细节：

stride 调度要求进程优先级 
，所以设定进程优先级 
 会导致错误。

进程初始 stride 设置为 0 即可。

进程初始优先级设置为 16。

为了实现该调度算法，内核还要增加 sys_set_priority 系统调用:

int sys_set_priority(long long prio);
功能描述：设定进程优先级 - syscall ID: 140 - 功能：设定进程优先级。 - 说明：设定自身进程优先级，只要 prio 在 [2, isize_max] 就成功，返回 prio，否则返回 -1。

针对测例 - ch5_setprio

完成之后你需要调整框架的代码调度机制，是的可以设置不同进程优先级之后可以按照 stride 算法进行调度。实现正确后，代码应该能够通过用户测例 ch5t_stride*。最终输出的 priority 和 exitcode 应该大致成正比，由于我们的时间片比较粗糙，qemu 的模拟也不是十分准确，我们最终的 CI 测试会允许最大 30% 的误差。

实现 tips:

你应该给 proc 结构体加入新的字段来支持优先级。

我们的测例运行时间不很长，不要求处理 stride 的溢出（详见问答作业，当然处理了更好）。

为了减少整数除的误差，BIG_STRIDE 一般需要很大，但测例中的优先级都是 2 的整数次幂，结合第二点，BIG_STRIDE不需要太大，65536 是一个不错的数字。

用户态的 printf 支持了行缓冲，所以如果你想要增加用户程序的输出，记得换行。

stride 算法要找到　stride 最小的进程，使用优先级队列是效率不错的办法，但是我们的实验测例很简单，所以效率完全不是问题。事实上，我很推荐使用暴力扫一遍的办法找最小值。

注意设置进程的初始优先级。

**摘要**: 基于0个修改文件和38个新增文件生成4个问题（OS类型: ucore, RAG检索: 4次）

**RAG 检索**: 执行了 4 次工具调用
  - `search_os_concept`: {'concept': 'stride 调度算法', 'detail_level': 'detailed'}
  - `search_os_concept`: {'concept': '进程创建', 'detail_level': 'detailed'}
  - `search_by_lab`: {'lab_name': 'ch5'}
  - `search_by_code_symbol`: {'symbol_name': 'proc', 'context': '进程控制块和调度'}

---

## 问题 1 [UNDERSTANDING]

在实现 `sys_spawn` 时，实验提示明确指出："注意 fork 的执行流，新进程 context 的 ra 和 sp 与父进程不同。所以你不能在内核中通过 fork 和 exec 的简单组合实现 spawn"。请看你实现的 spawn 相关代码（或参考以下典型实现）：



请解释：为什么在内核中不能直接先调用 `fork()` 再调用 `exec()` 来实现 spawn？特别是从 **trapframe 的 ra/sp 寄存器设置** 和 **内存拷贝开销** 两个角度分析。另外，`allocproc` + `loader` 的组合相比 `fork` + `exec` 避免了哪些不必要的操作？

**参考代码**:

```
int sys_spawn(char *filename)
{
    struct proc *p = allocproc();
    if (p == 0) return -1;
    
    // 加载目标程序到进程地址空间
    if (loader(filename, p) < 0) {
        freeproc(p);
        return -1;
    }
    
    // 设置新进程的初始执行上下文
    p->trapframe->epc = p->entry;  // 程序入口
    p->trapframe->sp = p->sz;      // 用户栈顶
    
    p->state = RUNNABLE;
    return p->id;
}
```

**出题理由**: 考察学生对进程创建机制的理解，特别是 spawn 与 fork+exec 的本质区别。课程资料明确指出 spawn 不应有内存拷贝，且执行流不同。此问题要求学生理解进程控制块初始化、trapframe 设置以及地址空间生命周期的差异，区分"真正理解"和"简单拼接代码"的学生。

---

## 问题 2 [IMPLEMENTATION]

Stride 调度算法的核心是在调度时选择 stride 值最小的进程。请看你的 scheduler 实现（或以下典型代码）：



这段代码在**选择进程后立即更新 stride**（`min_p->stride += min_p->pass`），而不是在进程用完时间片被换下时才更新。请解释：
1. 如果改为在进程被换下时才更新 stride，会导致什么调度问题？
2. 结合 stride 算法的公平性定义（`pass = BigStride / priority`），说明立即更新对保证"时间片与优先级成正比"的重要性。

**参考代码**:

```
void scheduler(void)
{
    struct proc *p;
    struct proc *min_p;
    
    for (;;) {
        min_p = 0;
        // 遍历进程池寻找 stride 最小的 RUNNABLE 进程
        for (p = proc; p < &proc[NPROC]; p++) {
            if (p->state == RUNNABLE) {
                if (min_p == 0 || p->stride < min_p->stride)
                    min_p = p;
            }
        }
        
        if (min_p) {
            min_p->state = RUNNING;
            // 关键点：立即增加 stride
            min_p->stride += min_p->pass;
            swtch(&c->context, &min_p->context);
        }
    }
}
```

**出题理由**: 考察 stride 调度算法的实现细节和正确性理解。课程资料提到 stride 表示进程已运行的"长度"。此问题针对关键修改点（stride 更新时机），区分学生是否真正理解算法的数学保证，还是仅仅照搬了"选最小值"的代码。

---

## 问题 3 [CONCEPT]

课程知识库中的问答作业提到了 stride 算法的溢出问题：

> "可以证明，在不考虑溢出的情况下，在进程优先级全部 >= 2 的情况下，如果严格按照算法执行，那么 STRIDE_MAX – STRIDE_MIN <= BigStride / 2。"

假设使用 32 位无符号整数存储 stride，设 `BigStride = 65536`，有两个进程 P1 和 P2：
P1.priority = 2，当前 P1.stride = 0xFFFFFFFF
P2.priority = 2，当前 P2.stride = 0xFFFFFFFE

*问题**：
1. 如果 P2 获得调度执行一个时间片，其 stride 更新为 `0xFFFFFFFE + 32768 = 0x00007FFE`（发生回绕）。此时比较两者的 stride 值，调度器会认为 P1.stride (0xFFFFFFFF) > P2.stride (0x00007FFE)，从而继续选择 P2 执行。这与理论上的公平调度是否一致？实际应该轮到哪个进程执行？

2. 请解释：为什么要求 `priority >= 2`（即 `pass <= BigStride/2`）可以避免这种因整数溢出导致的调度错误？（提示：考虑 STRIDE_MAX - STRIDE_MIN 的界限）

**出题理由**: 直接对应课程知识库中的"stride 算法深入"问答作业。这是 stride 算法的经典陷阱，考察学生对算法数学原理的理解深度，特别是对溢出处理和优先级约束原因的掌握，能有效区分"调试通过"和"理论清晰"的学生。

---

## 问题 4 [IMPLEMENTATION]

在实现 `sys_set_priority` 时，需要动态修改进程的 pass 值。请看以下典型实现：



假设进程 P 当前正在 RUNNING 状态（正在 CPU 上执行），在其用户态执行过程中调用了 `sys_set_priority` 将优先级从 16 改为 2（pass 从 4096 变为 32768）。

*问题**：
1. 上述代码中 `p->pass` 的更新会立即生效吗？这对**当前正在执行的时间片**的 stride 计算有何影响？（提示：考虑 scheduler 中 stride 更新的时机）

2. 如果要求"新优先级从下一个时间片开始生效"，你的代码需要如何修改？这种设计的优缺点是什么？

**参考代码**:

```
int sys_set_priority(long long prio)
{
    struct proc *p = myproc();
    
    if (prio < 2)  // 课程要求优先级 >= 2
        return -1;
    
    p->priority = prio;
    p->pass = BIG_STRIDE / prio;  // 立即重新计算 pass
    
    return prio;
}
```

**出题理由**: 考察动态优先级调整的实现细节和边界情况。这是 stride 调度在实际系统中的常见问题，涉及内核态/用户态切换、系统调用执行时机与调度器状态的交互。能检验学生对并发和状态一致性的理解。

---


## 代码变更摘要

```
新增文件: 38个
删除文件: 36个
修改文件: 0个
未变文件: 0个
```
