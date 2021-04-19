import os
import zipfile
import os
import json
import numpy as np
import quaternion

def downloadScanNetScene(sceneID, scanDir='outputs'):
    filesToDload = ['_vh_clean_2.ply', '.txt', '_vh_clean_2.labels.ply']

    print('Downloading ScanNet V2 Scan %s...'%sceneID)
    scenesList = [sceneID]

    for scene in scenesList:
        for file in filesToDload:
            if not os.path.exists(os.path.join(scanDir, 'scans', scene, scene+file)):
                os.system('python2 download-scannet.py -o %s --id %s --type %s'%(scanDir, scene, file))
            if 'zip' in file:
                if (not os.path.exists(os.path.join(scanDir, 'scans', scene, scene+file[:-4]))) \
                    and (os.path.exists(os.path.join(scanDir, 'scans', scene, scene+file))):
                    with zipfile.ZipFile(os.path.join(scanDir, 'scans', scene, scene+file), 'r') as zip_ref:
                        zip_ref.extractall(os.path.join(scanDir, 'scans', scene, scene+file[:-4]))
                        print('Unizipping %s'%(scene))


def downloadValScenes(scanDir='outputs'):
    with open('scannetv2_val.txt','r') as f:
        sceneIDs = f.readlines()
    sceneIDs = [s.strip() for s in sceneIDs if '_00' in s]
    for i,scene in enumerate(sceneIDs):
        print('Downloading Scene %d of %d'%(i, len(sceneIDs)))
        if os.path.exists(os.path.join(scanDir, 'scans', scene, 'monte_carlo', 'FinalCandidates.pickle')):
            downloadScanNetScene(scene, scanDir)

def jsonWrite(filename, data):
	with open(filename, 'w') as outfile:
		json.dump(data, outfile)

def jsonRead(filename):
	with open(filename, 'r') as infile:
		return json.load(infile)


