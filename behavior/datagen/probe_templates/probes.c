#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/fs.h>

// Whether to block accumulate (keyed on bpf_get_current_pid_tgid() >> 32).
BPF_HASH(block, u32, u64, 512);
// Whether we have started executing query.
BPF_HASH(execstart, u32, u64, 512);
// Whether we have started tracking the off-time.
BPF_HASH(start, u32, u64, 512);
// Where we accumulate offtimes and whether we should.
BPF_HASH(offtime, u32, u64, 512);

BPF_HISTOGRAM(dist0);
BPF_HISTOGRAM(dist6);
BPF_HISTOGRAM(dist10);

void qss_ExecutorStart(struct pt_regs* ctx)
{
    u64 offtime_start = 0;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;
    offtime.update(&tgid, &offtime_start);

    u64 ts = bpf_ktime_get_ns();
    execstart.update(&tgid, &ts);
    block.delete(&tgid);
}

static void Block(struct pt_regs* ctx)
{
    u64 block_ind = 1;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;

    u64 *blk = block.lookup(&tgid);
    if (blk == 0)
        block.update(&tgid, &block_ind);
    else
        block_ind += *blk;
        block.update(&tgid, &block_ind);
}

static void Unblock(struct pt_regs* ctx)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;
    u64 *blk = block.lookup(&tgid);
    if (blk != 0)
    {
        u64 newblk = *blk - 1;
        if (newblk == 0)
        {
            block.delete(&tgid);
        }
        else
        {
            block.update(&tgid, &newblk);
        }
    }
}

void qss_Block(struct pt_regs* ctx)
{
    Block(ctx);
}

void qss_Unblock(struct pt_regs* ctx)
{
    Unblock(ctx);
}

void qss_ExecutorEnd(struct pt_regs* ctx)
{
    s64 elapsed = 0;
    bpf_usdt_readarg(1, ctx, &elapsed);

    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;

    u64 *startts = execstart.lookup(&tgid);
    if (startts == 0)
        return;

    u64 *tsp = offtime.lookup(&tgid);
    if (tsp == 0)
        return;

    u64 slp = (*tsp) == 0 ? 1 : (*tsp);
    if (elapsed > 0 && slp < elapsed) {
        // This is the time that OU models try to predict I think...
        // Under the assumption that we don't capture the I/O in offcpu.
        u32 oncpu = bpf_log2l(elapsed - slp);
        if (oncpu < 7)
            dist0.increment(bpf_log2l(slp));
        else if (oncpu < 11)
            dist6.increment(bpf_log2l(slp));
        else
            dist10.increment(bpf_log2l(slp));
    }
    block.delete(&tgid);
}

int trace_read_entry(struct pt_regs *ctx, struct file *file, char __user *buf, size_t count)
{
    Block(ctx);
    return 0;
}

int trace_write_entry(struct pt_regs *ctx, struct file *file, char __user *buf, size_t count)
{
    Block(ctx);
    return 0;
}

int trace_read_return(struct pt_regs *ctx)
{
    Unblock(ctx);
    return 0;
}

int trace_write_return(struct pt_regs *ctx)
{
    Unblock(ctx);
    return 0;
}

static inline void store_start(u32 tgid, u64 ts)
{
    if (tgid != TARGET_PID)
        return;

    // Check that we should trace.
    u64 *tsp = offtime.lookup(&tgid);
    if (tsp == 0)
        return;

    // Check that the block guard isn't on.
    u64 *b = block.lookup(&tgid);
    if (b != 0)
        return;

    start.update(&tgid, &ts);
}

static inline void update_hist(u32 tgid, u64 ts)
{
    if (tgid != TARGET_PID)
        return;

    u64 *tsp = start.lookup(&tgid);
    if (tsp == 0)
        return;

    u64 *off = offtime.lookup(&tgid);
    if (off == 0)
        return;

    if (ts < *tsp) {
        // Probably a clock issue where the recorded on-CPU event had a
        // timestamp later than the recorded off-CPU event, or vice versa.
        return;
    }

    u64 delta = (ts - *tsp) / 1000;
    u64 offtotal = (*off) + delta;
    offtime.update(&tgid, &offtotal);
    start.delete(&tgid);
}

int sched_switch(struct pt_regs *ctx, struct task_struct *prev)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;

    u32 prev_tgid = prev->tgid;
    u64 ts = bpf_ktime_get_ns();

    // Update runtime for old.
    update_hist(tgid, ts);

    // Store that we are now tracking the new one.
    store_start(prev_tgid, ts);
    return 0;
}
