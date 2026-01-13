## 核心模块功能分析

### 1. __lib.rs - 主入口模块__

- __Rknpu结构体__：驱动的主接口，管理所有硬件核心

  - `base`: 存储多个RknpuCore实例（支持最多3个核心）
  - `data`: 包含硬件特定的数据配置
  - `gem`: 内存池管理器
  - `iommu_enabled`: IOMMU启用状态标志

```rust
pub struct Rknpu {
    base: Vec<RknpuCore>,        // 存储3个硬件核心的寄存器访问接口
    config: RknpuConfig,          // NPU配置
    data: RknpuData,              // 硬件特定数据（DMA掩码、任务参数等）
    iommu_enabled: bool,          // IOMMU启用标志
    pub(crate) gem: GemPool,      // 内存池管理器
}
```

- __RknpuAction枚举__：定义了硬件操作

  - 硬件版本获取 (`GetHwVersion`, `GetDrvVersion`)
  - 频率/电压管理 (`GetFreq`, `SetFreq`, `GetVolt`, `SetVolt`)
  - 带宽控制 (`GetBwPriority`, `SetBwExpect`, `GetBwTw`)
  - 电源管理 (`PowerOn`, `PowerOff`)
  - 内存统计 (`GetTotalSramSize`, `GetFreeSramSize`)
  - IOMMU管理 (`GetIommuEn`, `SetIommuDomainId`)

- __主要功能__：

  - `action()`: 执行硬件操作动作
  - `submit()`: 提交计算任务
  - `handle_interrupt0()`: 处理硬件中断
  - 通过`Deref` trait代理GemPool的内存管理功能

### 2. __registers/ - 硬件寄存器抽象层__

使用`tock_registers`库实现类型安全的MMIO访问，包含以下子模块：

- __core.rs__：核心控制寄存器
- __pc.rs__：程序计数器/命令解析器寄存器
- __int.rs__：中断控制寄存器
- __cna.rs__：CNA（Convolutional Neural Accelerator）寄存器
- __dpu.rs/dpu_rdma.rs__：数据处理单元及其DMA寄存器
- __ppu.rs/ppu_rdma.rs__：后处理单元及其DMA寄存器
- __sdma.rs/ddma.rs__：系统DMA和设备DMA寄存器
- __global.rs__：全局控制寄存器
- __consts.rs__：寄存器偏移量常量定义

__核心操作__：

- `submit_pc()`: PC模式任务提交，配置命令解析器
- `handle_interrupt()`: 中断状态处理和清除
- `rknpu_fuzz_status()`: 中断状态模糊化处理

- __submit_pc()方法__:
```rust
pub(crate) fn submit_pc(
    &mut self,
    config: &RknpuData,
    args: &SubmitRef,
) -> Result<(), RknpuError> {
    // 1. 设置PC基地址
    self.pc().base_address.set(1);
    
    // 2. 配置ping-pong模式（双缓冲）
    let task_pp_en = if args.base.flags.contains(JobMode::PINGPONG) { 1 } else { 0 };
    
    // 3. 设置中断指针（CNA和Core寄存器）
    if config.irqs.get(args.base.core_idx).is_some() {
        let val = 0xe + 0x10000000 * args.base.core_idx as u32;
        self.cna().s_pointer.set(val);
        self.core().s_pointer.set(val);
    }
    
    // 4. 设置PC寄存器基地址（命令地址）
    let pc_base_addr = args.regcmd_base_addr;
    self.pc().base_address.set(pc_base_addr);
    
    // 5. 计算并设置寄存器数量
    // 公式：(regcfg_amount + 4) / 2 - 1
    let amount = (args.base.regcfg_amount + RKNPU_PC_DATA_EXTRA_AMOUNT)
        .div_ceil(config.pc_data_amount_scale) - 1;
    self.pc().register_amounts.set(amount);
    
    // 6. 配置中断
    self.pc().interrupt_mask.set(args.base.int_mask);
    self.pc().interrupt_clear.set(args.base.int_clear);
    
    // 7. 设置任务控制寄存器
    let task_number = args.task_number as u32;
    let task_control = ((0x6 | task_pp_en) << config.pc_task_number_bits) | task_number;
    self.pc().task_control.set(task_control);
    
    // 8. 设置任务DMA基地址
    self.pc().task_dma_base_addr.set(args.base.task_base_addr);
    
    // 9. 内存屏障，确保寄存器写入顺序
    mb();
    
    // 10. 启用操作（写1启用，立即写0）
    self.pc().operation_enable.set(1);
    mb();
    self.pc().operation_enable.set(0);
    
    Ok(())
}
```

__handle_interrupt()方法__ - 中断处理：

```rust
pub fn handle_interrupt(&self) -> u32 {
    // 读取中断状态
    let int_status = self.pc().interrupt_status.get();
    mb();  // 内存屏障
    
    // 清除所有中断
    self.pc().interrupt_clear.set(INT_CLEAR_ALL);
    
    // 模糊化中断状态（合并相关位）
    rknpu_fuzz_status(int_status)
}
```

### 3. __gem.rs - 内存管理模块__

基于DMA一致性内存的内存池管理：

- __GemPool结构体__：

  - `pool`: BTreeMap存储句柄到DVec的映射
  - `handle_counter`: 生成唯一内存句柄

```rust
pub struct GemPool {
    pool: BTreeMap<u32, DVec<u8>>,  // 句柄到DMA向量的映射
    handle_counter: u32,             // 句柄计数器
}
```

- __主要功能__：

  - `create()`: 创建DMA内存缓冲区
  - `get_phys_addr_and_size()`: 获取物理地址和大小
  - `sync()`: 内存同步（cache一致性）
  - `destroy()`: 释放内存
  - `comfirm_write_all()`: 确认所有写入完成
  - `prepare_read_all()`: 准备读取操作

- __create()__:
```rust
pub fn create(&mut self, args: &mut RknpuMemCreate) -> Result<(), RknpuError> {
    // 分配零初始化的DMA一致性内存
    let data = DVec::zeros(
        u32::MAX as _,           // 设备ID（未使用）
        args.size as _,          // 分配大小
        0x1000,                  // 对齐到4KB
        Direction::Bidirectional // 双向DMA
    ).unwrap();
    
    // 生成唯一句柄
    let handle = self.handle_counter;
    self.handle_counter = self.handle_counter.wrapping_add(1);
    
    // 返回句柄和地址信息
    args.handle = handle;
    args.sram_size = data.len() as _;
    args.dma_addr = data.bus_addr();     // DMA总线地址
    args.obj_addr = data.as_ptr() as _;  // 虚拟地址
    self.pool.insert(args.handle, data);
    Ok(())
}
```
- __内存同步__:
```rust
pub fn comfirm_write_all(&mut self) -> Result<(), RknpuError> {
    // 确保所有CPU写入对DMA可见（刷cache）
    for data in self.pool.values_mut() {
        data.confirm_write_all();
    }
    Ok(())
}

pub fn prepare_read_all(&mut self) -> Result<(), RknpuError> {
    // 准备CPU读取（失效cache）
    for data in self.pool.values_mut() {
        data.prepare_read_all();
    }
    Ok(())
}
```

### 4. __job.rs - 任务描述符模块__

定义任务提交的数据结构：

- __常量定义__：

  - 支持最多3个核心 (`RKNPU_MAX_CORES`)
  - 每次提交最多5个子核心任务 (`RKNPU_MAX_SUBCORE_TASKS`)
  - 核心掩码：`RKNPU_CORE0_MASK`, `RKNPU_CORE1_MASK`, `RKNPU_CORE2_MASK`

- __数据结构__：

  - `RknpuTask`: 硬件任务描述符（C布局，包含flags、op_idx、int_mask等）
  - `RknpuSubcoreTask`: 子核心任务请求
  - `JobMode`: 位标志定义（PC模式、非阻塞、ping-pong执行等）

- __工具函数__：

  - `core_mask_from_index()`: 核心索引到掩码转换
  - `core_count_from_mask()`: 掩码到核心数量转换

### 5. __task/ - 任务构建和提交模块__

- __Submit结构体__：封装完整的任务提交信息

  - `base`: 基本提交参数（flags、中断掩码等）
  - `regcmd_all`: 寄存器命令DMA缓冲区
  - `tasks`: 操作列表


- __Submit结构体实现__:

```rust
pub struct Submit {
    pub base: SubmitBase,           // 基本提交参数
    pub regcmd_all: DVec<u64>,      // 寄存器命令DMA缓冲区
    pub tasks: Vec<Operation>,      // 操作列表
}

impl Submit {
    pub fn new(tasks: Vec<Operation>) -> Self {
        let base = SubmitBase {
            flags: JobMode::PC | JobMode::BLOCK | JobMode::PINGPONG,
            task_base_addr: 0,
            core_idx: 0,
            int_mask: 0x300,  // 等待DPU完成
            int_clear: 0x1ffff,  // 清除所有中断
            regcfg_amount: tasks[0].reg_amount(),
        };
        
        // 分配寄存器命令缓冲区
        let regcmd_all: DVec<u64> = DVec::zeros(
            u32::MAX as _,
            base.regcfg_amount as usize * tasks.len(),
            0x1000,
            Direction::Bidirectional,
        ).unwrap();
        
        // 填充每个操作的寄存器命令
        let amount = base.regcfg_amount as usize;
        for (i, task) in tasks.iter().enumerate() {
            let regcmd = unsafe {
                core::slice::from_raw_parts_mut(
                    regcmd_all.as_ptr().add(i * amount),
                    amount
                )
            };
            task.fill_regcmd(regcmd);  // 调用操作trait填充命令
        }
        regcmd_all.confirm_write_all();  // 确保写入对DMA可见
        
        Self { base, regcmd_all, tasks }
    }
}
```

- __SubmitRef结构体__：Submit的只读引用，用于硬件提交

- __主要功能__：

  - `new()`: 创建任务提交，分配DMA内存并填充寄存器命令
  - `as_ref()`: 转换为SubmitRef

- __子模块__：

  - `op/`: 操作定义
  - `dpu.rs`: DPU操作
  - `cna.rs`: CNA操作

### 6. __osal.rs - 操作系统抽象层__

定义平台无关的类型和错误：

- __类型定义__：

  - `PhysAddr`: 物理地址（u64）
  - `DmaAddr`: DMA地址（u64）
  - `TimeStamp`: 时间戳（u64）

- __OsalError枚举__：

  - `OutOfMemory`: 内存不足
  - `InvalidParameter`: 无效参数
  - `TimeoutError`: 超时错误
  - `DeviceError`: 设备错误
  - `NotSupported`: 不支持的操作

### 7. __config/ - 配置管理模块__

- __RknpuType枚举__：定义支持的NPU类型
- __RknpuConfig结构体__：NPU配置参数

### 8. __err.rs - 错误处理模块__

定义RknpuError错误类型

### 9. __ioctrl.rs - IOCTL接口模块__

提供设备控制接口，兼容Linux驱动IOCTL语义

- __RknpuSubmit结构体__:
```rust
#[repr(C)]
pub struct RknpuSubmit {
    pub flags: u32,                      // 作业标志（PC模式、非阻塞等）
    pub timeout: u32,                    // 超时时间
    pub task_start: u32,                 // 任务起始索引
    pub task_number: u32,                // 任务数量
    pub task_counter: u32,               // 任务计数器（输出）
    pub priority: i32,                   // 优先级
    pub task_obj_addr: u64,              // 任务对象地址（RknpuTask数组）
    pub iommu_domain_id: u32,            // IOMMU域ID
    pub task_base_addr: u64,             // 任务基地址
    pub hw_elapse_time: i64,             // 硬件运行时间（输出）
    pub core_mask: u32,                  // 核心掩码
    pub fence_fd: i32,                   // DMA信号量文件描述符
    pub subcore_task: [RknpuSubcoreTask; 5],  // 子核心任务数组
}
```

- __submit_ioctrl()实现__:

```rust
pub fn submit_ioctrl(&mut self, args: &mut RknpuSubmit) -> Result<(), RknpuError> {
    // 1. 确认所有CPU写入对DMA可见
    self.gem.comfirm_write_all()?;
    
    // 2. 检查非阻塞标志
    if args.flags & 1 << 1 > 0 {
        debug!("Nonblock task");
    }
    
    // 3. 遍历5个子核心任务
    for idx in 0..5 {
        if args.subcore_task[idx].task_number == 0 {
            continue;  // 跳过空任务
        }
        debug!("Submitting subcore task index: {}", idx);
        let submitted_tasks = self.submit_one(idx, args)?;  // 提交单个子核心任务
    }
    
    // 4. 准备CPU读取
    self.gem.prepare_read_all()?;
    
    // 5. 更新输出参数
    args.task_counter = args.task_number as _;
    args.hw_elapse_time = (args.timeout / 2) as _;
    
    Ok(())
}
```

- __submit_one()方法__:

```rust
fn submit_one(&mut self, idx: usize, args: &mut RknpuSubmit) -> Result<usize, RknpuError> {
    let task_ptr = args.task_obj_addr as *mut RknpuTask;
    let subcore = &args.subcore_task[idx];
    
    let mut task_iter = subcore.task_start as usize;
    let task_iter_end = task_iter + subcore.task_number as usize;
    let max_submit_number = self.data.max_submit_number as usize;  // 4095
    
    // 批量提交任务（每次最多4095个）
    while task_iter < task_iter_end {
        let task_number = (task_iter_end - task_iter).min(max_submit_number);
        let submit_tasks = unsafe {
            core::slice::from_raw_parts_mut(task_ptr.add(task_iter), task_number)
        };
        
        // 构建提交引用
        let job = SubmitRef {
            base: SubmitBase {
                flags: JobMode::from_bits_retain(args.flags),
                task_base_addr: args.task_base_addr as _,
                core_idx: idx,  // 使用子核心索引
                int_mask: submit_tasks.last().unwrap().int_mask,
                int_clear: submit_tasks[0].int_mask,
                regcfg_amount: submit_tasks[0].regcfg_amount,
            },
            task_number,
            regcmd_base_addr: submit_tasks[0].regcmd_addr as _,
        };
        
        // 等待之前的中断处理完成
        while self.base[idx].handle_interrupt() != 0 {
            spin_loop();  // 自旋等待
        }
        
        // 提交PC任务
        self.base[idx].submit_pc(&self.data, &job).unwrap();
        
        // 轮询等待完成
        let int_status;
        loop {
            let status = self.base[idx].pc().interrupt_status.get();
            let status = rknpu_fuzz_status(status);
            
            if status & job.base.int_mask > 0 {
                int_status = job.base.int_mask & status;
                break;  // 任务完成
            }
            if status != 0 {
                debug!("Interrupt status changed: {:#x}", status);
                return Err(RknpuError::TaskError);  // 错误状态
            }
        }
        
        // 清除中断并更新任务状态
        self.base[idx].pc().clean_interrupts();
        submit_tasks.last_mut().unwrap().int_status = int_status;
        
        task_iter += task_number;
    }
    
    Ok(subcore.task_number as usize)
}
```

### 10. __data/ - 硬件数据模块__

RknpuData结构体，包含硬件特定的配置参数如dma_mask、amount_top、amount_core等

## 工作流程

1. __初始化__：

   ```javascript
   创建Rknpu实例 → 提供MMIO基地址 → 配置参数 → 初始化
   ```

2. __内存分配__：

   ```javascript
   GemPool创建DMA内存 → 返回句柄 → 获取物理地址
   ```

3. __任务执行__：

   ```javascript
   构建操作（Operation）→ 创建Submit → 分配regcmd缓冲区 → 
   填充寄存器命令 → submit() → 硬件执行 → 中断处理
   ```

4. __中断处理__：

   ```javascript
   硬件触发中断 → handle_interrupt() → 清除中断状态 → 
   rknpu_fuzz_status() → 返回状态
   ``` 

## TODO:实现多进程共享硬件资源

### 1. 竞态条件问题

__Rknpu结构体没有同步保护__：

```rust
pub struct Rknpu {
    base: Vec<RknpuCore>,      // 多个核心共享，无锁保护
    gem: GemPool,              // 全局内存池，无锁保护
    // ... 其他字段也无保护
}
```

__任务提交直接操作硬件寄存器__：

```rust
// src/ioctrl.rs
pub fn submit_ioctrl(&mut self, args: &mut RknpuSubmit) -> Result<(), RknpuError> {
    // 多个进程同时调用会导致寄存器配置冲突
    self.gem.comfirm_write_all()?;
    for idx in 0..5 {
        self.submit_one(idx, args)?;  // 无同步保护
    }
}
```

__内存句柄计数器不是原子的__：

```rust
// src/gem.rs
pub struct GemPool {
    pool: BTreeMap<u32, DVec<u8>>,
    handle_counter: u32,  // 非原子操作，多进程下可能重复
}
```

### 2. 内存隔离问题

- 所有进程共享同一个`GemPool`
- 没有进程ID或上下文标识
- 一个进程可能误访问另一个进程的内存

### 3. 中断处理问题

- 中断处理函数`handle_interrupt()`不知道中断来源
- 没有进程特定的中断队列
- 无法将中断通知到正确的进程

### 4. 无调度机制

- 先到先服务（FCFS）
- 没有优先级调度
- 没有核心分配策略

---

## 多进程支持改进方案

### 方案一：添加互斥锁保护（简单方案）

__修改Rknpu结构体，添加互斥锁__：

```rust
use core::sync::atomic::{AtomicU32, Ordering};
use alloc::sync::{Arc, Mutex};

// 进程上下文标识
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessId(u32);

impl ProcessId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

// 进程上下文
pub struct ProcessContext {
    pid: ProcessId,
    gem_handle_offset: u32,  // 句柄偏移，避免冲突
    priority: i32,           // 进程优先级
    task_queue: Vec<RknpuTask>,  // 进程任务队列
}

// 修改后的Rknpu结构体
pub struct Rknpu {
    base: Vec<RknpuCore>,
    config: RknpuConfig,
    data: RknpuData,
    iommu_enabled: bool,
    gem: Mutex<GemPool>,  // 添加互斥锁保护
    
    // 进程管理
    process_table: Mutex<BTreeMap<ProcessId, ProcessContext>>,
    global_handle_counter: AtomicU32,
    
    // 任务调度
    pending_tasks: Mutex<VecDeque<TaskSubmission>>,
    active_tasks: Mutex<[Option<TaskSubmission>; RKNPU_MAX_CORES]>,
    
    // 中断通知
    task_completion: Mutex<BTreeMap<u32, CompletionEvent>>,
}

// 任务提交信息
pub struct TaskSubmission {
    pid: ProcessId,
    submit: RknpuSubmit,
    submit_time: TimeStamp,
}

// 完成事件
pub struct CompletionEvent {
    int_status: u32,
    hw_elapse_time: i64,
    completed: bool,
}
```

__修改GemPool，支持进程隔离__：

```rust
pub struct GemPool {
    pool: BTreeMap<u32, DVec<u8>>,
    handle_counter: AtomicU32,  // 原子计数器
    process_mapping: BTreeMap<u32, ProcessId>,  // 句柄到进程的映射
}

impl GemPool {
    pub fn create(&mut self, pid: ProcessId, args: &mut RknpuMemCreate) 
        -> Result<(), RknpuError> {
        let data = DVec::zeros(u32::MAX as _, args.size as _, 0x1000, 
                              Direction::Bidirectional).unwrap();
        
        // 原子分配句柄
        let handle = self.handle_counter.fetch_add(1, Ordering::SeqCst);
        
        args.handle = handle;
        args.sram_size = data.len() as _;
        args.dma_addr = data.bus_addr();
        args.obj_addr = data.as_ptr() as _;
        
        self.pool.insert(args.handle, data);
        self.process_mapping.insert(args.handle, pid);  // 记录进程归属
        
        Ok(())
    }
    
    pub fn verify_ownership(&self, handle: u32, pid: ProcessId) -> bool {
        self.process_mapping.get(&handle)
            .map(|&owner| owner == pid)
            .unwrap_or(false)
    }
    
    pub fn destroy(&mut self, pid: ProcessId, handle: u32) {
        // 验证所有权
        if self.verify_ownership(handle, pid) {
            self.pool.remove(&handle);
            self.process_mapping.remove(&handle);
        }
    }
}
```

__修改任务提交，添加进程参数__：

```rust
impl Rknpu {
    pub fn submit_ioctrl(&mut self, pid: ProcessId, 
                        args: &mut RknpuSubmit) -> Result<(), RknpuError> {
        // 1. 验证进程是否存在
        {
            let table = self.process_table.lock().unwrap();
            if !table.contains_key(&pid) {
                return Err(RknpuError::InvalidProcess);
            }
        }
        
        // 2. 确认内存写入（锁保护）
        {
            let mut gem = self.gem.lock().unwrap();
            gem.comfirm_write_all()?;
        }
        
        // 3. 检查核心可用性
        let core_mask = self.allocate_core_mask(pid, args.core_mask)?;
        
        // 4. 提交任务到队列
        let submission = TaskSubmission {
            pid,
            submit: args.clone(),
            submit_time: self.get_current_time(),
        };
        
        {
            let mut pending = self.pending_tasks.lock().unwrap();
            pending.push_back(submission);
        }
        
        // 5. 尝试执行任务
        self.try_execute_next_task()?;
        
        Ok(())
    }
    
    fn allocate_core_mask(&self, pid: ProcessId, requested: u32) 
        -> Result<u32, RknpuError> {
        let active = self.active_tasks.lock().unwrap();
        
        // 检查哪些核心已被占用
        let mut available = RKNPU_CORE0_MASK | RKNPU_CORE1_MASK | RKNPU_CORE2_MASK;
        for idx in 0..RKNPU_MAX_CORES {
            if active[idx].is_some() {
                available &= !(1 << idx);
            }
        }
        
        // 分配可用核心
        let allocated = available & requested;
        if allocated == 0 {
            return Err(RknpuError::NoAvailableCore);
        }
        
        Ok(allocated)
    }
    
    fn try_execute_next_task(&mut self) -> Result<(), RknpuError> {
        let task = {
            let mut pending = self.pending_tasks.lock().unwrap();
            pending.pop_front()
        };
        
        if let Some(mut submission) = task {
            // 检查核心可用性
            let core_mask = self.allocate_core_mask(submission.pid, 
                                                    submission.submit.core_mask)?;
            
            // 标记核心为占用
            {
                let mut active = self.active_tasks.lock().unwrap();
                for idx in 0..RKNPU_MAX_CORES {
                    if (core_mask & (1 << idx)) != 0 {
                        active[idx] = Some(submission.clone());
                    }
                }
            }
            
            // 执行任务
            self.execute_task(&mut submission)?;
        }
        
        Ok(())
    }
}
```

__修改中断处理，支持多进程__：

```rust
impl Rknpu {
    pub fn handle_interrupt(&mut self, core_idx: usize) -> Result<(), RknpuError> {
        let int_status = self.base[core_idx].handle_interrupt();
        
        // 获取该核心上运行的任务
        let task = {
            let mut active = self.active_tasks.lock().unwrap();
            active[core_idx].take()
        };
        
        if let Some(submission) = task {
            // 更新任务完成状态
            {
                let mut completion = self.task_completion.lock().unwrap();
                completion.insert(
                    submission.submit.task_counter,
                    CompletionEvent {
                        int_status,
                        hw_elapse_time: self.get_hw_elapse_time(),
                        completed: true,
                    }
                );
            }
            
            // 准备内存读取
            {
                let mut gem = self.gem.lock().unwrap();
                gem.prepare_read_all()?;
            }
            
            // 释放核心
            self.release_core(core_idx);
            
            // 尝试执行下一个任务
            self.try_execute_next_task()?;
        }
        
        Ok(())
    }
    
    pub fn wait_completion(&self, pid: ProcessId, task_counter: u32) 
        -> Result<CompletionEvent, RknpuError> {
        loop {
            {
                let completion = self.task_completion.lock().unwrap();
                if let Some(event) = completion.get(&task_counter) {
                    if event.completed {
                        return Ok(event.clone());
                    }
                }
            }
            
            // 短暂休眠后重试
            self.osal_msleep(1);
        }
    }
}
```

---

### 方案二：实现调度器（高级方案）

__添加任务调度器__：

```rust
use alloc::collections::BinaryHeap;

// 优先级队列项
#[derive(Debug, Clone)]
struct QueueItem {
    priority: i32,
    submission_time: TimeStamp,
    submission: TaskSubmission,
}

// 实现优先级队列
impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // 优先级高的先执行，同优先级按时间排序
        other.priority.cmp(&self.priority)
            .then_with(|| self.submission_time.cmp(&other.submission_time))
    }
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for QueueItem {}
impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && 
        self.submission_time == other.submission_time
    }
}

// 调度器
pub struct NpuScheduler {
    queue: BinaryHeap<QueueItem>,
    running_tasks: [Option<TaskSubmission>; RKNPU_MAX_CORES],
}

impl NpuScheduler {
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            running_tasks: Default::default(),
        }
    }
    
    pub fn enqueue(&mut self, submission: TaskSubmission, priority: i32) {
        let item = QueueItem {
            priority,
            submission_time: self.get_time(),
            submission,
        };
        self.queue.push(item);
    }
    
    pub fn schedule(&mut self, available_cores: u32) -> Vec<TaskSubmission> {
        let mut scheduled = Vec::new();
        let mut cores = available_cores;
        
        while let Some(item) = self.queue.pop() {
            if cores == 0 {
                break;
            }
            
            // 分配核心
            let core_mask = self.find_available_cores(&mut self.running_tasks, cores);
            if core_mask == 0 {
                self.queue.push(item);  // 放回队列
                break;
            }
            
            cores &= !core_mask;
            scheduled.push(item.submission);
        }
        
        scheduled
    }
    
    fn find_available_cores(&self, running: &[Option<TaskSubmission>; 3], 
                           requested: u32) -> u32 {
        let mut available = 0;
        for idx in 0..3 {
            if running[idx].is_none() && (requested & (1 << idx)) != 0 {
                available |= 1 << idx;
            }
        }
        available
    }
}
```

---

## 实施步骤建议

### 第一阶段：基础同步保护

1. __添加互斥锁__

   - `Rknpu.gem: Mutex<GemPool>`
   - `Rknpu.process_table: Mutex<BTreeMap<ProcessId, ProcessContext>>`

2. __进程管理__

   - 添加`register_process()`方法
   - 添加`unregister_process()`方法
   - 添加进程上下文验证

3. __内存隔离__

   - 修改`GemPool`支持进程ID
   - 添加所有权验证
   - 添加`verify_ownership()`方法

### 第二阶段：任务调度

4. __任务队列__

   - 实现`pending_tasks: Mutex<VecDeque<TaskSubmission>>`
   - 实现`active_tasks: Mutex<[Option<TaskSubmission>; 3]>`

5. __核心分配__

   - 实现`allocate_core_mask()`
   - 实现`release_core()`
   - 添加核心可用性检查

6. __调度逻辑__

   - 实现`try_execute_next_task()`
   - 实现先到先服务调度

### 第三阶段：中断处理

7. __中断通知__

   - 实现`task_completion: Mutex<BTreeMap<u32, CompletionEvent>>`
   - 修改`handle_interrupt()`记录完成事件

8. __等待机制__

   - 实现`wait_completion()`
   - 实现轮询或事件通知

### 第四阶段：高级调度（可选）

9. __优先级调度__

   - 实现`NpuScheduler`
   - 支持`RknpuSubmit.priority`

10. __公平调度__

    - 实现时间片轮转
    - 实现饥饿避免

---

## 使用示例

__多进程使用示例__：

```rust
// 进程A
let pid_a = ProcessId::new(1);
rknpu.register_process(pid_a, 0)?;

// 分配内存
let mut mem_create = RknpuMemCreate {
    size: 1024 * 1024,
    ..Default::default()
};
rknpu.create(pid_a, &mut mem_create)?;

// 提交任务
let mut submit = RknpuSubmit {
    core_mask: RKNPU_CORE0_MASK,
    task_number: 1,
    ..Default::default()
};
rknpu.submit_ioctrl(pid_a, &mut submit)?;

// 等待完成
let event = rknpu.wait_completion(pid_a, submit.task_counter)?;

// 进程B（同时运行）
let pid_b = ProcessId::new(2);
rknpu.register_process(pid_b, 0)?;
let mut submit_b = RknpuSubmit {
    core_mask: RKNPU_CORE1_MASK,  // 使用不同的核心
    task_number: 1,
    ..Default::default()
};
rknpu.submit_ioctrl(pid_b, &mut submit_b)?;
```

---

## 注意事项

1. __性能权衡__：互斥锁会增加开销，需要根据实际需求选择粒度
2. __死锁预防__：避免嵌套锁，使用一致的锁获取顺序
3. __优先级反转__：考虑使用优先级继承协议
4. __内存限制__：为每个进程设置内存配额
5. __超时处理__：实现任务超时和取消机制
6. __错误恢复__：实现进程崩溃后的资源清理
