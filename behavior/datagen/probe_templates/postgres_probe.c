BPF_PERF_OUTPUT(postmaster_events);

struct postmaster_event_t {
  int type_;
  int pid_;
  int socket_fd_;
};

void fork_backend(struct pt_regs *ctx) {
  struct postmaster_event_t event = {.type_ = 0};
  bpf_usdt_readarg(1, ctx, &(event.pid_));
  bpf_usdt_readarg(2, ctx, &(event.socket_fd_));
  postmaster_events.perf_submit(ctx, &event, sizeof(event));
}

void reap_backend(struct pt_regs *ctx) {
  struct postmaster_event_t event = {.type_ = 1};
  bpf_usdt_readarg(1, ctx, &(event.pid_));
  postmaster_events.perf_submit(ctx, &event, sizeof(event));
}
