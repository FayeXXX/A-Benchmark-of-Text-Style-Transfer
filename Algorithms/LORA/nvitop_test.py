import time

from nvitop import Device, GpuProcess, NA, colored

print(colored(time.strftime('%a %b %d %H:%M:%S %Y'), color='red', attrs=('bold',)))

devices = Device.cuda.all()  # or `Device.all()` to use NVML ordinal instead
separator = False
for device in devices:
    processes = device.processes()  # type: Dict[int, GpuProcess]

    print(colored(str(device), color='green', attrs=('bold',)))
    print(colored('  - Fan speed:       ', color='blue', attrs=('bold',)) + f'{device.fan_speed()}%')
    print(colored('  - Temperature:     ', color='blue', attrs=('bold',)) + f'{device.temperature()}C')
    print(colored('  - GPU utilization: ', color='blue', attrs=('bold',)) + f'{device.gpu_utilization()}%')
    print(colored('  - Total memory:    ', color='blue', attrs=('bold',)) + f'{device.memory_total_human()}')
    print(colored('  - Used memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_used_human()}')
    print(colored('  - Free memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_free_human()}')
    if len(processes) > 0:
        processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
        processes.sort(key=lambda process: (process.username, process.pid))

        print(colored(f'  - Processes ({len(processes)}):', color='blue', attrs=('bold',)))
        fmt = '    {pid:<5}  {username:<8} {cpu:>5}  {host_memory:>8} {time:>8}  {gpu_memory:>8}  {sm:>3}  {command:<}'.format
        print(colored(fmt(pid='PID', username='USERNAME',
                          cpu='CPU%', host_memory='HOST-MEM', time='TIME',
                          gpu_memory='GPU-MEM', sm='SM%',
                          command='COMMAND'),
                      attrs=('bold',)))
        for snapshot in processes:
            print(fmt(pid=snapshot.pid,
                      username=snapshot.username[:7] + ('+' if len(snapshot.username) > 8 else snapshot.username[7:8]),
                      cpu=snapshot.cpu_percent, host_memory=snapshot.host_memory_human,
                      time=snapshot.running_time_human,
                      gpu_memory=(snapshot.gpu_memory_human if snapshot.gpu_memory_human is not NA else 'WDDM:N/A'),
                      sm=snapshot.gpu_sm_utilization,
                      command=snapshot.command))
    else:
        print(colored('  - No Running Processes', attrs=('bold',)))

    if separator:
        print('-' * 120)
    separator = True