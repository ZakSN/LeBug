#!/usr/bin/env python

import os, re, datetime, time, signal, shutil, random, sys, fileinput, subprocess, datetime
from subprocess import call, Popen, STDOUT
import numpy as np

####################################
########## Configurations ##########
####################################
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
quartus_path,tests,top,seeds = np.load('config.npy')
results_file = 'results/summary.rpt'

####################################
############# Functions ############
####################################

#Check if this test was previously made
def testPreviouslyMade(t):
    test_id=getID(t)
    if os.path.isfile(results_file):
        with open(results_file) as file:
            for line in file:
                line=line.split(test_id,1)
                if len(line)>1:
                    return True
    return False

# Convert test configurations into a unique ID
def getID(t):
    return '_'.join(str(a) for a in t)

# function for getting first word after keyword in file
def getKeywordValue(key,filePath):
    if os.path.isfile(filePath):
        with open(filePath) as file:
            for line in file:
                line=line.split(key,1)
                if len(line)>1:
                    return line[1].strip().partition(' ')[0].replace(',','')
        return "SEARCH_FAIL" #keyword not found in file
    else: 
        return "FAIL" #File does not exist ()

def getKeywordValueWithTwoKeys(key1,key2,filePath):
    if os.path.isfile(filePath):
        firstKeyWasFound = False
        key=key1
        with open(filePath) as file:
            for line in file:
                line=line.split(key,1)
                if len(line)>1:
                    if not firstKeyWasFound:
                        firstKeyWasFound = True
                        key=key2
                    else:
                        return line[1].strip().partition(' ')[0].replace(',','')
        return "SEARCH_FAIL" #keyword not found in file
    else: 
        return "FAIL" #File does not exist ()

# Parse results and save to file
def saveResults(t,programWasKilled):

    if programWasKilled:
        fMax="TIMEOUT"
        LE="TIMEOUT"
        memBits="TIMEOUT"
        multipliers="TIMEOUT"
        RAM="TIMEOUT"
    else:
        #Getting results
        topFitFile=f'quartus_project/output_files/{top}.fit.rpt'
        topStaFile=f'quartus_project/output_files/{top}.sta.rpt'
        board="Stratix 10"
        if board=="Stratix 10":
            fMax = getKeywordValueWithTwoKeys("Restricted Fmax ;","MHz ;",topStaFile)
            LE = getKeywordValue("Logic utilization (in ALMs)     ;",topFitFile)
            memBits = getKeywordValue("Total block memory bits                                     ;",topFitFile)
            multipliers = getKeywordValue("DSP Blocks Needed [=A+B-C]                                  ;",topFitFile)
            RAM = getKeywordValue("Total MLAB memory bits                                      ;",topFitFile)
        else:
            print("board not found!")
            exit()

        #Saving unfiltered results in backup folder
        test_id=getID(t)
        backupFolder='results/'+test_id
        call(["mkdir", "-p",backupFolder])
        call(["cp", topFitFile, backupFolder])
        call(["cp", topStaFile, backupFolder])

    #Printing results
    print ("\t\tfMax: "+fMax +" MHz")
    print ("\t\tLE: "+LE)
    print ("\t\tmemBits: "+memBits)
    print ("\t\tmultipliers: "+multipliers)
    print ("\t\tRAM: "+RAM)

    #Saving filtered results
    f = open(results_file,'a+')
    f.write(test_id+' ')
    f.write(fMax+' ')
    f.write(LE+' ')
    f.write(memBits+' ')
    f.write(multipliers+' ')
    f.write(RAM+'\n')
    f.close

# Finds a line in a file and replaces it
def replaceAfter(file,searchExp,replaceExp):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = replaceExp
        sys.stdout.write(line)

def randomizeQuartusSeed(seed=None):
    assignments_file=f"quartus_project/{top}.qsf"
    f = open(assignments_file,'r')

    # We will cache all file in an list to make it simpler to move information around
    f_content=[]
    while True:
        # Read line by line and exit when done
        line = f.readline()
        if not line:
            break
        f_content.append(line)
    f.close()
    if seed is None:
        seed = random.randint(0, 999)
    for idx, line in enumerate(f_content):
        if "set_global_assignment -name SEED " in line:
            f_content[idx]="set_global_assignment -name SEED "+str(seed)+"\n"
            break
    f = open(assignments_file,'w')
    for line in f_content:
        # Process line
        f.write(line)
    f.close()

#get time stamp
def getTimeStamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
     
def waitFinishOrTimeout (p,timeOut):
    count = 0
    while True:
        if isinstance(p,int) or count>=timeOut:
            break
        else:
            time.sleep(1)
            count +=1
    if count>=timeOut:
        p.terminate()
        print ("Killed!")
        return True
    return False

####################################
############ Start Tests ###########
####################################

#Used for suppressing calls outputs on STDOUT
FNULL = open(os.devnull, 'w')

#Go over all tests
for idx, t in enumerate(tests):
    test_id=getID(t)
    print(f'Test {idx+1}/{len(tests)}: {test_id}')
    print("\tTest previously made? "+str(testPreviouslyMade(t)))

    if not testPreviouslyMade(t):
        
        print("\tUpdate M and N")
        N,M,k=t
        top_file="quartus_project/top.sv"
        replaceAfter(top_file,"N=",f'    parameter N={N};\n')
        replaceAfter(top_file,"M=",f'    parameter M={M};\n')

        print("\tUpdate seed")
        randomizeQuartusSeed(seeds[N][M])

        # Run quartus synthesis
        print(f"\tRunning Quartus (Time: {getTimeStamp()})")
        f = open("quartus_stdio.txt", "w")
        p = subprocess.Popen([f"{quartus_path}/bin/quartus_sh","--flow","compile","quartus_project/mlDebug.qpf"],stdout=f,stderr=f).wait()
        programWasKilled = waitFinishOrTimeout(p,60*60*3)#Wait 3h 
        print(f"\tDone running Quartus (Time: {getTimeStamp()})")

        print("\tSaving Results")
        saveResults(t,programWasKilled)


