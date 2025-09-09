'''
First run the whole pipeline up this point for static pose:
python src/pose/rtmw3d_pose_estimation.py \
  -m manifests/OpenCapDataset/subject2.yaml -p config/paths.yaml \
  --trials static1 --video-field video_sync --stride 1\
  --metainfo-from-file external/datasets_config/h3wb.py --refine-pass

python src/pose/rtmw3d_scale_from_height.py \
  -m manifests/OpenCapDataset/subject2.yaml \
  -p config/paths.yaml \
  --trial static1 \
  --trc --trc-rate 100 --trc-lpf-hz 6.0

python src/marker_enhancer/marker_enhancer2.py \
    -m manifests/OpenCapDataset/subject2.yaml \
    -p config/paths.yaml \
    --trial static1 \
    --models-path \
    ./models/marker_enhancer/ \
    --version 0.3 \
    --upsampled

And then:
python src/OpenSim/open_sim_scaling.py \
    -m manifests/OpenCapDataset/subject2.yaml \
    -p config/paths.yaml \
    --trial static1 \
    --scaling-xml ./models/OpenSim/Setup_scaling_LaiUhlrich2022.xml \
    --base-model ./models/OpenSim/LaiUhlrich2022.osim \
    --upsampled
'''
import argparse
import json
from pathlib import Path
import numpy as np
from IO.load_manifest import load_manifest
import os
import utils.utilsDataman as utilsDataman
import opensim

def getScaleTimeRange(pathTRCFile, thresholdPosition=0.005, thresholdTime=0.3,
                      withArms=True, withOpenPoseMarkers=False, isMocap=False,
                      removeRoot=False):
    
    c_trc_file = utilsDataman.TRCFile(pathTRCFile)
    c_trc_time = c_trc_file.time    
    if withOpenPoseMarkers:
        markers = ["neck", "right_shoulder", "left_shoulder", "right_hip", "left_hip", "right_knee", 
                   "left_knee", "right_ankle", "left_ankle", "right_heel", "left_heel", "right_small_toe", 
                   "left_small_toe", "right_elbow", "left_elbow", "right_wrist", "left_wrist"]        
    else:
        markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                   "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                   "L.PSIS_study", "r_knee_study", "L_knee_study",
                   "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                   "L_ankle_study", "r_mankle_study", "L_mankle_study",
                   "r_calc_study", "L_calc_study", "r_toe_study", 
                   "L_toe_study", "r_5meta_study", "L_5meta_study",
                   "RHJC_study", "LHJC_study"]
        if withArms:
            markers.append("r_lelbow_study")
            markers.append("L_lelbow_study")
            markers.append("r_melbow_study")
            markers.append("L_melbow_study")
            markers.append("r_lwrist_study")
            markers.append("L_lwrist_study")
            markers.append("r_mwrist_study")
            markers.append("L_mwrist_study")
            
        if isMocap:
            markers = [marker.replace('_study','') for marker in markers]
            markers = [marker.replace('r_shoulder','R_Shoulder') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_shoulder','L_Shoulder') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('RHJC','R_HJC') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('LHJC','L_HJC') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('r_lelbow','R_elbow_lat') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_lelbow','L_elbow_lat') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('r_melbow','R_elbow_med') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_melbow','L_elbow_med') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('r_lwrist','R_wrist_radius') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_lwrist','L_wrist_radius') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('r_mwrist','R_wrist_ulna') for marker in markers] # should just change the mocap marker set
            markers = [marker.replace('L_mwrist','L_wrist_ulna') for marker in markers] # should just change the mocap marker set
            

    trc_data = np.zeros((c_trc_time.shape[0], 3*len(markers)))
    for count, marker in enumerate(markers):
        trc_data[:, count*3:count*3+3] = c_trc_file.marker(marker)
    
    if removeRoot:
        try:
            root_data = c_trc_file.marker('midHip')
            trc_data -= np.tile(root_data,len(markers))
        except:
            pass
   
    if np.max(trc_data)>10: # in mm, turn to m
        trc_data/=1000
        
    # Sampling frequency.
    sf = np.round(1/np.mean(np.diff(c_trc_time)),4)
    # Minimum duration for time range in seconds.
    timeRange_min = 1
    # Corresponding number of frames.
    nf = int(timeRange_min*sf + 1)
    
    detectedWindow = False
    i = 0
    while not detectedWindow:
        c_window = trc_data[i:i+nf,:]
        c_window_max = np.max(c_window, axis=0)
        c_window_min = np.min(c_window, axis=0)
        c_window_diff = np.abs(c_window_max - c_window_min)
        detectedWindow = np.alltrue(c_window_diff<thresholdPosition)
        if not detectedWindow:
            i += 1
            if i > c_trc_time.shape[0]-nf:
                i = 0
                nf -= int(0.1*sf) 
            if np.round((nf-1)/sf,2) < thresholdTime: # number of frames got too small without detecting a window
                log_warn(f"Musculoskeletal model scaling failed; could not detect a static phase of at least {thresholdTime}fs.")
    
    timeRange = [c_trc_time[i], c_trc_time[i+nf-1]]
    timeRangeSpan = np.round(timeRange[1] - timeRange[0], 2)
    
    log_info("Static phase of %.2fs detected in staticPose between [%.2f, %.2f]."
          % (timeRangeSpan,np.round(timeRange[0],2),np.round(timeRange[1],2)))
 
    return timeRange

def runScaleTool(pathGenericSetupFile, pathGenericModel, subjectMass,
                 pathTRCFile, timeRange, pathOutputFolder, 
                 scaledModelName='not_specified', subjectHeight=0,
                 createModelWithContacts=False, fixed_markers=False,
                 suffix_model=''):
    
    dirGenericModel, scaledModelNameA = os.path.split(pathGenericModel)
    
    # Paths.
    if scaledModelName == 'not_specified':
        scaledModelName = scaledModelNameA[:-5] + "_scaled"
    pathOutputModel = os.path.join(
        pathOutputFolder, scaledModelName + '.osim')
    pathOutputMotion = os.path.join(
        pathOutputFolder, scaledModelName + '.mot')
    pathOutputSetup =  os.path.join(
        pathOutputFolder, 'Setup_Scale_' + scaledModelName + '.xml')
    pathUpdGenericModel = os.path.join(
        pathOutputFolder, scaledModelNameA[:-5] + "_generic.osim")
    
    # Marker set.
    _, setupFileName = os.path.split(pathGenericSetupFile)
    if 'Lai' in scaledModelName or 'Rajagopal' in scaledModelName:
        if 'Mocap' in setupFileName:
            markerSetFileName = 'LaiUhlrich2022_markers_mocap{}.xml'.format(suffix_model)
        elif 'openpose' in setupFileName:
            markerSetFileName = 'LaiUhlrich2022_markers_openpose.xml'
        elif 'mmpose' in setupFileName:
            markerSetFileName = 'LaiUhlrich2022_markers_mmpose.xml'
        elif 'rtmw3d' in setupFileName:
            markerSetFileName = 'LaiUhlrich2022_markers_rtmw3d.xml'
        else:
            if fixed_markers:
                markerSetFileName = 'LaiUhlrich2022_markers_augmenter_fixed.xml'
            else:
                markerSetFileName = 'LaiUhlrich2022_markers_augmenter{}.xml'.format(suffix_model)
    else:
        raise log_warn("Unknown model type: scaling")
    pathMarkerSet = os.path.join(dirGenericModel, markerSetFileName)
    log_info(f"Marker se: {pathMarkerSet}")

    # Add the marker set to the generic model and save that updated model.
    log_info(f"Loading generic model: {pathGenericModel}")
    opensim.Logger.setLevelString('error')
    genericModel = opensim.Model(os.fspath(pathGenericModel))
    log_info(f"Loading marker set: {pathMarkerSet}")
    markerSet = opensim.MarkerSet(os.fspath(pathMarkerSet))
    genericModel.set_MarkerSet(markerSet)
    genericModel.printToXML(pathUpdGenericModel)    

    # Time range.
    timeRange_os = opensim.ArrayDouble(timeRange[0], 0)
    timeRange_os.insert(1, timeRange[-1])
                
    # Setup scale tool.
    log_info(f"Loading scaling file: {pathGenericSetupFile}")
    scaleTool = opensim.ScaleTool(os.fspath(pathGenericSetupFile))
    scaleTool.setName(scaledModelName)
    scaleTool.setSubjectMass(subjectMass)
    scaleTool.setSubjectHeight(subjectHeight)
    genericModelMaker = scaleTool.getGenericModelMaker()     
    genericModelMaker.setModelFileName(pathUpdGenericModel)
    modelScaler = scaleTool.getModelScaler() 
    modelScaler.setMarkerFileName(pathTRCFile)
    modelScaler.setOutputModelFileName("")       
    modelScaler.setOutputScaleFileName("")
    modelScaler.setTimeRange(timeRange_os) 
    markerPlacer = scaleTool.getMarkerPlacer() 
    markerPlacer.setMarkerFileName(pathTRCFile)                
    markerPlacer.setOutputModelFileName(pathOutputModel)
    markerPlacer.setOutputMotionFileName(pathOutputMotion) 
    markerPlacer.setOutputMarkerFileName("")
    markerPlacer.setTimeRange(timeRange_os)
    
    # Disable tasks of dofs that are locked and markers that are not present.
    model = opensim.Model(pathUpdGenericModel)
    coordNames = []
    for coord in model.getCoordinateSet():
        if not coord.getDefaultLocked():
            coordNames.append(coord.getName())            
    modelMarkerNames = [marker.getName() for marker in model.getMarkerSet()]          
              
    for task in markerPlacer.getIKTaskSet():
        # Remove IK tasks for dofs that are locked or don't exist.
        if (task.getName() not in coordNames and 
            task.getConcreteClassName() == 'IKCoordinateTask'):
            task.setApply(False)
            log_info('{} is a locked coordinate - ignoring IK task'.format(
                task.getName()))
        # Remove Marker tracking tasks for markers not in model.
        if (task.getName() not in modelMarkerNames and 
            task.getConcreteClassName() == 'IKMarkerTask'):
            task.setApply(False)
            log_info('{} is not in model - ignoring IK task'.format(
                task.getName()))
            
    # Remove measurements from measurement set when markers don't exist.
    # Disable entire measurement if no complete marker pairs exist.
    measurementSet = modelScaler.getMeasurementSet()
    for meas in measurementSet:
        mkrPairSet = meas.getMarkerPairSet()
        iMkrPair = 0
        while iMkrPair < meas.getNumMarkerPairs():
            mkrPairNames = [
                mkrPairSet.get(iMkrPair).getMarkerName(i) for i in range(2)]
            if any([mkr not in modelMarkerNames for mkr in mkrPairNames]):
                mkrPairSet.remove(iMkrPair)
                log_info('{} or {} not in model. Removing associated \
                      MarkerPairSet from {}.'.format(mkrPairNames[0], 
                      mkrPairNames[1], meas.getName()))
            else:
                iMkrPair += 1
            if meas.getNumMarkerPairs() == 0:
                meas.setApply(False)
                log_info('There were no marker pairs in {}, so this measurement \
                      is not applied.'.format(meas.getName()))
    # Run scale tool.                      
    scaleTool.printToXML(pathOutputSetup)            
    command = 'opensim-cmd -o error' + ' run-tool ' + pathOutputSetup
    log_info(f"Running scale command: {command}")
    os.system(command)
    log_info(f"Scale saved: {pathOutputSetup}")

    
    # Sanity check
    scaled_model = opensim.Model(pathOutputModel)
    bodySet = scaled_model.getBodySet()
    nBodies = bodySet.getSize()
    scale_factors = np.zeros((nBodies, 3))
    for i in range(nBodies):
        bodyName = bodySet.get(i).getName()
        body = bodySet.get(bodyName)
        attached_geometry = body.get_attached_geometry(0)
        scale_factors[i, :] = attached_geometry.get_scale_factors().to_numpy()
    diff_scale = np.max(np.max(scale_factors, axis=0)-
                        np.min(scale_factors, axis=0))
    # A difference in scaling factor larger than 1 would indicate that a 
    # segment (e.g., humerus) would be more than twice as large as its generic
    # counterpart, whereas another segment (e.g., pelvis) would have the same
    # size as the generic segment. This is very unlikely, but might occur when
    # the camera calibration went wrong (i.e., bad extrinsics).
    if diff_scale > 1:
        log_warn("Musculoskeletal model scaling failed; the segment sizes are not anthropometrically realistic")   

    return pathOutputModel

# ---------- Logging ----------
def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_err (msg): print(f"[ERROR] {msg}")
def log_done(msg): print(f"[DONE] {msg}")

# ---------- IO helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Run scaling of model for OpenSim with Marker-Enahncer output")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--scaling-xml", default=None, help="XML for scaling")
    ap.add_argument("--base-model", default=None, help="Base OSIM model")
    ap.add_argument("--trc-type", required=True, choices=["metric_upsampled", "cannonical", "abs_cannonical", "world", "cam", "metric"])

    args = ap.parse_args()

    log_step("Loading and resolving manifest")
    manifest = load_manifest(args.manifest, args.paths)

    # Find trial
    log_step(f"Finding trial '{args.trial}'")
    trial = None
    for subset, trials in manifest.get("trials", {}).items():
        for t in trials:
            if t.get("id") == args.trial:
                trial = t
                break
        if trial: break
    if trial is None:
        raise SystemExit(f"[ERROR] Trial '{args.trial}' not found in manifest.")

    # Paths
    base = manifest.get('output_dir')
    if not base:
        subj = manifest.get('subject_id', 'subject')
        sess = manifest.get('session', 'Session')
        cam  = manifest.get('camera', 'Cam')
        base = Path(manifest.get('outputs_root', Path.cwd() / "outputs")) / subj / sess / cam
    trial_root = Path(base) / trial['id']
    rtmw3d_dir = trial_root / "rtmw3d"
    eval_dir = trial_root / "rtmw3d_eval"
    enh_dir  = trial_root / "enhancer"
    osim_dir  = trial_root / "OpenSim"
    enh_output = os.path.join(enh_dir, f"enhancer_{args.trial}_{args.trc_type}.trc")
    ensure_dir(eval_dir); 
    ensure_dir(enh_dir)
    ensure_dir(osim_dir)

    #osim_scaling_output = osim_dir / "model_scaled.osim"
    meta_path  = trial_root / "meta.json"
    scaling_xml_path  = Path(args.scaling_xml)
    base_model_path = Path(args.base_model) 

    log_info(f"Trial root : {trial_root}")
    log_info(f"Enhancer output preds path : {enh_output}")
    log_info(f"meta.json  : {meta_path}")
    log_info(f"Scaling xml  : {scaling_xml_path}")
    log_info(f"Base model osim  : {base_model_path}")

    # Subject info
    log_step("Reading meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    subj = meta.get("subject") or {}
    height_m = subj.get("height_m", None)
    mass_kg  = subj.get("mass_kg", None)
    if not (isinstance(height_m, (int, float)) and height_m > 0):
        raise log_err("subject.height_m missing/invalid in meta.json")
    if not (isinstance(mass_kg, (int, float)) and mass_kg > 0):
        raise log_err("subject.mass_kg missing/invalid in meta.json")
    log_info(f"Subject info: height={height_m} m ({height_m*1000:.1f} mm), mass={mass_kg} kg")

    height_m = height_m * 1000.0
    mass_kg = float(mass_kg)

    # Get time range.
    try:
        thresholdPosition = 0.5
        maxThreshold = 100
        increment = 0.001
        success = False
        while thresholdPosition <= maxThreshold and not success:
            try:
                timeRange4Scaling = getScaleTimeRange(
                    enh_output,
                    thresholdPosition=thresholdPosition,
                    thresholdTime=0.1, removeRoot=True)
                success = True
            except Exception as e:
                log_warn(f"Attempt identifying scaling time range with thresholdPosition {thresholdPosition} failed: {e}")
                thresholdPosition += increment

        # Run scale tool.
        if success:
            log_info('Running Scaling')
            pathScaledModel = runScaleTool(
                scaling_xml_path, base_model_path,
                mass_kg, enh_output, 
                timeRange4Scaling, osim_dir,
                subjectHeight=height_m, 
                suffix_model='',
                scaledModelName=f"LaiUhlrich2022_scaled_{args.trc_type}")
        else:
            log_warn('Did not start Scaling')
        
    except Exception as e:
        if len(e.args) == 2: # specific exception
            log_warn(f"Error: {e.args[0]}, \n{e.args[1]}")
        elif len(e.args) == 1: # generic exception
            log_warn(f"Musculoskeletal model scaling failed. {e}")

if __name__ == "__main__":
    main()
