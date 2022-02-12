import datetime
import os
import subprocess
import time
import random

run_py_iterval = 4 * 60 * 60  # 4 hours
commod_max_time = 50 * 60  # 20 minutes
# run_py_iterval = 4
# commod_max_time = 2


def execute_command(cmdstring, timeout=2):


    end_time = datetime.datetime.now() + datetime.timedelta(seconds=timeout)

    sub = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, bufsize=4096, shell=True)

    while True:
        if sub.poll() is not None:
            break
        # time.sleep(0.1)
        if timeout:
            if end_time <= datetime.datetime.now():
                # sub.kill()
                os.system('kill {}'.format(sub.pid))
                print("i am kill the pid {}".format(sub.pid))
                return "TIME_OUT"

    return "is done"


def main(sleep_time=60):
    while True:
        # os.system("python /home/ubuntu/TechXueXi/SourcePackages/pandalearning.py &")
        res_command = execute_command(
            "python /home/ubuntu/TechXueXi/SourcePackages/pandalearning.py",
            commod_max_time,
        )
        print(res_command + " ------> " + time.asctime(time.localtime(time.time())))
        if res_command == "TIME_OUT":
            # 失败重新执行一次
            print("-" * 300)
            print("run again")
            print("-" * 300)
            continue

        delay_time = random.random() * 1000
        time.sleep(sleep_time + delay_time)


main(run_py_iterval)