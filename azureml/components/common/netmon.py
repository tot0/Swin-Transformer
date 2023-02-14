from datetime import datetime
import psutil
import time
from threading import Thread
import os
import pandas as pd
from pathlib import Path
import mlflow

UPDATE_DELAY = 10 # in seconds

def get_size(bytes):
    """
    Returns size of bytes in a nice format
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024


def monitor_eth0(self):
    # get the network I/O stats from psutil on each network interface
    # by setting `pernic` to `True`
    io = psutil.net_io_counters(pernic=True)

    first = True
    step = 0
    # open file to log network metrics to
    output_dir = Path('./user_logs/netmon')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(f'user_logs/netmon/node_{os.environ.get("AZUREML_CR_NODE_RANK", "0")}.log', 'w') as f:
        while not self.should_stop:
            # sleep for `UPDATE_DELAY` seconds
            time.sleep(UPDATE_DELAY)
            # get the network I/O stats again per interface 
            io_2 = psutil.net_io_counters(pernic=True)
            # initialize the data to gather (a list of dicts)
            data = []
            #for iface, iface_io in io.items():
            iface = 'eth0'
            iface_io = io[iface]
            # new - old stats gets us the speed
            upload_speed, download_speed = io_2[iface].bytes_sent - iface_io.bytes_sent, io_2[iface].bytes_recv - iface_io.bytes_recv
            data.append({
                "iface": iface,
                "Download": get_size(io_2[iface].bytes_recv),
                "Upload": get_size(io_2[iface].bytes_sent),
                "Upload Speed": f"{get_size(upload_speed / UPDATE_DELAY)}/s",
                "Download Speed": f"{get_size(download_speed / UPDATE_DELAY)}/s",
            })
            try:
                mlflow.log_metrics({
                    f'network.{iface}.upload.MBs': (upload_speed / UPDATE_DELAY) / 1024 / 1024,
                    f'network.{iface}.download.MBs': (download_speed / UPDATE_DELAY) / 1024 / 1024,
                }, step=step*UPDATE_DELAY)
            except Exception as e:
                f.write(f'Failed to log_metrics: {e}')
            # update the I/O stats for the next iteration
            io = io_2
            # construct a Pandas DataFrame to print stats in a cool tabular style
            df = pd.DataFrame(data)
            # sort values per column, feel free to change the column
            df.sort_values("Download", inplace=True, ascending=False)
            # clear the screen based on your OS
            # os.system("cls") if "nt" in os.name else os.system("clear")
            # print the stats
            f.write(df.to_string(header=first))
            if first:
                first = False

            f.write('\n')
            f.flush()
            step = step + 1


class NetmonThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.should_stop = False

    def run(self):
        monitor_eth0(self)

    def stop(self):
        self.should_stop = True


class NetmonProc:
    def __init__(self):
        self.should_stop = False
        self.process = None
    
    def start(self):
        class ShouldStop:
            def __init__(self):
                self.should_stop = False

        # Start a new process
        from multiprocessing import Process
        self.process = Process(target=monitor_eth0, args=(ShouldStop(),))
        self.process.daemon = True
        self.process.start()

    def stop(self):
        self.process.kill()
