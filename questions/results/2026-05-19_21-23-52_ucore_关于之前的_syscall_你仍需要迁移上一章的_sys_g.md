# OS 实验面试问题集

**生成时间**: 2026-05-19T21:23:52.575617

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
  - `search_os_concept`: {'concept': 'stride调度算法', 'detail_level': 'detailed'}
  - `search_os_concept`: {'concept': '进程创建', 'detail_level': 'standard'}
  - `search_by_code_symbol`: {'symbol_name': 'spawn', 'context': '系统调用实现'}
  - `search_by_lab`: {'lab_name': 'lab5'}

---

## 问题 1 [IMPLEMENTATION]

在实现 `sys_spawn` 时，你需要为新进程设置 trapframe。请看下面这段代码：



对比 `fork()` 的实现中 `*p->trapframe = *p->parent->trapframe` 这种复制父进程上下文的方式，**spawn 为什么要采用这种"清零+重新设置"的方式？** 如果简单地在 `fork()` 后立即调用 `exec()` 来实现 spawn，会出现什么问题？请结合课程资料中提到的"新进程 context 的 ra 和 sp 与父进程不同"进行分析。

**参考代码**:

```
int sys_spawn(char *filename) {
    struct proc *p = allocproc();
    if(p == 0) return -1;
    
    // 设置新进程的 trapframe
    memset(p->trapframe, 0, sizeof(struct trapframe));
    p->trapframe->epc = loader_get_entry(filename);  // 程序入口
    p->trapframe->sp = PGROUNDUP(p->sz);             // 栈顶
    
    loader(filename, p);  // 加载程序到地址空间
    return p->pid;
}
```

**出题理由**: 根据课程资料提示，spawn 不能通过 fork+exec 简单组合实现，因为新进程的返回地址（ra）和栈指针（sp）与父进程不同。此问题考察学生是否真正理解 spawn 与 fork 在进程创建语义上的本质区别，以及是否正确设置了新进程的执行上下文。

---

## 问题 2 [IMPLEMENTATION]

你在实现 stride 调度算法时，需要在调度器中选择 stride 最小的进程。请看这段调度逻辑：



*这段代码的时间复杂度是多少？** 如果系统中有大量进程（如 NPROC=1000），这种实现是否会成为性能瓶颈？课程资料中提到"推荐使用暴力扫一遍的办法"，但为什么在生产级 OS 中通常使用优先级队列（如最小堆）？在什么情况下当前的暴力扫描方法会出现调度延迟问题？

**参考代码**:

```
void scheduler(void) {
    struct proc *p, *selected = 0;
    uint64 min_stride = UINT64_MAX;
    
    for(p = proc; p < &proc[NPROC]; p++) {
        if(p->state == RUNNABLE) {
            if(p->stride < min_stride) {
                min_stride = p->stride;
                selected = p;
            }
        }
    }
    
    if(selected) {
        selected->stride += BIG_STRIDE / selected->priority;
        // 切换到 selected 执行...
    }
}
```

**出题理由**: 考察学生对调度算法实现细节的掌握，以及基础的数据结构与算法分析能力。暴力扫描虽然能通过实验测例，但学生应该理解其复杂度局限性，体现对代码实现的深度思考而非简单照搬。

---

## 问题 3 [DEBUGGING]

课程资料中的问答作业提到了 stride 溢出的经典问题。假设你使用 8 位无符号整数存储 stride，BIG_STRIDE=256，两个进程的 priority 都为 2（pass=128）。当前状态如下：



*此时调度器再次选择最小 stride 的进程，会选择 p1 还是 p2？** 这种结果是否符合 stride 算法的预期（理论上应该让 p1 执行，因为 p1 的 stride 更大）？课程资料中提到"要求进程优先级 >= 2"可以解决此类问题，请解释为什么当 priority >= 2 时，能保证 `STRIDE_MAX - STRIDE_MIN <= BigStride / 2`，以及这个性质如何防止上述错误调度？

**参考代码**:

```
struct proc p1 = { .stride = 255, .state = RUNNABLE, .priority = 2 };
struct proc p2 = { .stride = 250, .state = RUNNABLE, .priority = 2 };

// p2 被调度执行一个时间片后：
p2.stride += BIG_STRIDE / p2.priority;  // 250 + 128 = 378
// 8位无符号整数溢出：378 % 256 = 122
```

**出题理由**: 直接引用课程资料中的 stride 算法深入问答，考察学生是否理解 stride 溢出的本质问题以及优先级限制的数学原理。这是区分"真正理解算法"和"简单实现功能"的关键问题。

---

## 问题 4 [OPTIMIZATION]

课程资料明确要求"在 spawn 中不应该有任何形式的内存拷贝"。请看下面两段代码的对比：



*为什么 spawn 不需要像 fork 那样拷贝内存？** 从进程创建语义和地址空间生命周期的角度，解释 spawn 与 fork 在内存管理上的根本差异。如果 spawn 像 fork 那样先拷贝父进程内存再加载新程序，会造成什么资源浪费？

**参考代码**:

```
// fork 的实现：需要拷贝父进程地址空间
int fork(void) {
    struct proc *np = allocproc();
    // ...
    if(uvmcopy(p->pagetable, np->pagetable, p->sz) < 0) {
        freeproc(np);
        return -1;
    }
    // 复制了父进程的内存页
    return np->pid;
}

// spawn 的实现：直接加载，无拷贝
int sys_spawn(char *filename) {
    struct proc *p = allocproc();
    // ...
    // 清空可能存在的旧映射（如果是替换当前进程）
    uvmunmap(p->pagetable, 0, p->max_page, 1);
    
    // 直接建立文件到地址空间的映射，不经过"拷贝"阶段
    int id = get_id_by_name(filename);
    loader(id, p);  // 直接设置页表映射，指向程序镜像
    return p->pid;
}
```

**出题理由**: 考察学生对 spawn 语义的理解。spawn 是"创建并执行新程序"，与 fork 的"复制当前进程"语义不同，因此不需要继承父进程的地址空间内容。此问题检验学生是否理解"无内存拷贝"的设计初衷。

---


## 代码变更摘要

```
新增文件: 38个
删除文件: 36个
修改文件: 0个
未变文件: 0个
```
