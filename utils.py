import mteb


def load_tasks() -> list[mteb.AbsTask]:
    all_tasks = mteb.get_tasks()
    all_task_names = [task.metadata.name for task in all_tasks]
    benchmarks = mteb.get_benchmarks()
    benchmark_tasks = []
    for benchmark in benchmarks:
        benchmark_tasks.extend([task.metadata.name for task in benchmark.tasks])
    tasks_not_in_benchmarks = set(all_task_names) - set(benchmark_tasks)
    # This one's biiiig
    tasks_not_in_benchmarks -= set(["MSMARCOv2"])
    tasks = mteb.get_tasks(tasks=tasks_not_in_benchmarks)
    return tasks
