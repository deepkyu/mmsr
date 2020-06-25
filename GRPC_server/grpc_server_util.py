import csv
import os
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from codes.EDVR_inference import makeoutput


def read_csv():
    ret_list = list()
    f = open("server_tmp_files/filequeue.csv", 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        ret_list.append(line)
    f.close()
    return ret_list


def read_findkey_csv(key_to_find):
    ret_line = ["None", "None", "None", "None"]
    queue = 0
    f = open("server_tmp_files/filequeue.csv", 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        if key_to_find == line[0]:
            ret_line = line
            break
        if ("None" in line[2]) or ("Proc" in line[2]):
            queue += 1
    return ret_line, queue


def write_csv_new_line(newline):
    ret_list = list()
    f = open("server_tmp_files/filequeue.csv", 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        ret_list.append(line)
    f.close()

    f = open('server_tmp_files/filequeue.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for T in ret_list:
        wr.writerow(T)
    wr.writerow(newline)
    f.close()


def find_proc_file():
    csv_list = read_csv()
    now_csv_proc = ""
    for T in csv_list:
        if T[0] == "KEY": continue
        if T[2] == 'None':
            now_csv_proc = T[0]
            T[2] = "Proc"
            break

    f = open('server_tmp_files/filequeue.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for T in csv_list:
        wr.writerow(T)
    f.close()

    if now_csv_proc == "":
        print("No Process")
        return
    print("Processing in " + now_csv_proc)


    # Processing Block
    try:
        makeoutput('server_tmp_files/input/' + now_csv_proc,'server_tmp_files/output/' + now_csv_proc)
    except:
        csv_list = read_csv()
        for T in csv_list:
            if T[0] == "KEY": continue
            if T[2] == 'Proc':
                T[2] = "Error"
                break

        f = open('server_tmp_files/filequeue.csv', 'w', encoding='utf-8', newline='')
        wr = csv.writer(f)
        for T in csv_list:
            wr.writerow(T)
        f.close()
        return

    csv_list = read_csv()
    for T in csv_list:
        if T[0] == "KEY": continue
        if T[2] == 'Proc':
            T[2] = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            break

    f = open('server_tmp_files/filequeue.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for T in csv_list:
        wr.writerow(T)
    f.close()


def initial_csv():
    if not (os.path.isfile("server_tmp_files/filequeue.csv")):
        f = open('server_tmp_files/filequeue.csv', 'w', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow(["KEY", "upload_at", "act_at", "deleted_at"])
        f.close()
    else:
        csv_list = read_csv()
        for T in csv_list:
            if(T[0]=="KEY") : continue
            else:
                if(T[2]=='Proc'):
                    T[2]=="Error"
        f = open('server_tmp_files/filequeue.csv', 'w', encoding='utf-8', newline='')
        wr = csv.writer(f)
        for T in csv_list:
            wr.writerow(T)
        f.close()


def initial_dir():
    if not (os.path.isdir("server_tmp_files")):
        os.makedirs(os.path.join("server_tmp_files"))
    if not (os.path.isdir("server_tmp_files/input")):
        os.makedirs(os.path.join("server_tmp_files/input"))
    if not (os.path.isdir("server_tmp_files/output")):
        os.makedirs(os.path.join("server_tmp_files/output"))


def tmpfile_del():
    now_list = read_csv()
    for T in now_list:
        if(T[0]=="KEY"):
            continue
        else:
            if(T[3]=="None"):
                try:
                    os.remove("server_tmp_files/input/" + T[0])
                except:
                    {}
                try:
                    os.remove("server_tmp_files/output/" + T[0])
                except:
                    {}
                T[3] = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    f = open('server_tmp_files/filequeue.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for T in now_list:
        wr.writerow(T)
    f.close()