'''
python src/marker_enhancer/marker_enhancer2.py   -m manifests/OpenCapDataset/subject2.yaml   -p config/paths.yaml   --trial walking1   --body-model ./models/marker_enhancer/body   --arms-model ./models/marker_enhancer/arm
'''
import argparse
import json
from pathlib import Path
import numpy as np
from IO.load_manifest import load_manifest
import utilsDataman
import os
import copy
import tensorflow as tf

# ---------- Logging ----------
def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_err (msg): print(f"[ERROR] {msg}")
def log_done(msg): print(f"[DONE] {msg}")

# ---------- IO helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def TRC2numpy(pathFile, markers,rotation=None):
    # rotation is a dict, eg. {'y':90} with axis, angle for rotation
    
    trc_file = utilsDataman.TRCFile(pathFile)
    time = trc_file.time
    num_frames = time.shape[0]
    data = np.zeros((num_frames, len(markers)*3))
    
    if rotation != None:
        for axis,angle in rotation.items():
            trc_file.rotate(axis,angle)
    for count, marker in enumerate(markers):
        data[:,3*count:3*count+3] = trc_file.marker(marker)    
    this_dat = np.empty((num_frames, 1))
    this_dat[:, 0] = time
    data_out = np.concatenate((this_dat, data), axis=1)
    
    return data_out

# ---------- Markers (OpenCap code) ----------
def getOpenPoseMarkers_lowerExtremity2():

    feature_markers = [
        "neck", "right_shoulder", "left_shoulder", "right_hip", "left_hip", "right_knee", "left_knee",
        "right_ankle", "left_ankle", "right_heel", "left_heel", "right_small_toe", "left_small_toe",
        "right_big_toe", "left_big_toe"]

    response_markers = [
        'r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study',
        'L.PSIS_study', 'r_knee_study', 'r_mknee_study', 
        'r_ankle_study', 'r_mankle_study', 'r_toe_study', 
        'r_5meta_study', 'r_calc_study', 'L_knee_study', 
        'L_mknee_study', 'L_ankle_study', 'L_mankle_study',
        'L_toe_study', 'L_calc_study', 'L_5meta_study', 
        'r_shoulder_study', 'L_shoulder_study', 'C7_study', 
        'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study',
        'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study',
        'r_sh1_study', 'r_sh2_study', 'r_sh3_study', 'L_sh1_study',
        'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']

    return feature_markers, response_markers

def getMarkers_upperExtremity_noPelvis2():

    feature_markers = [
        "neck", "right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_wrist",
        "left_wrist"]

    response_markers = ["r_lelbow_study", "r_melbow_study", "r_lwrist_study",
                        "r_mwrist_study", "L_lelbow_study", "L_melbow_study",
                        "L_lwrist_study", "L_mwrist_study"]

    return feature_markers, response_markers

def main():
    ap = argparse.ArgumentParser(description="Prepare OCAP20 from RTMW3D (metric) and run Marker-Enhancer.")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--body-model", default=None, help="Directory containing model.json, weights.h5, mean.npy, std.npy, metadata.json")
    ap.add_argument("--arms-model", default=None, help="Directory containing model.json, weights.h5, mean.npy, std.npy, metadata.json")
    ap.add_argument("--upsampled", action="store_true", help="Use upsampled trc as input")

    args = ap.parse_args()

    featureHeight = True
    featureWeight = True
    offset = True

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
    enh_output = os.path.join(enh_dir, f"enhancer_{args.trial}{'_upsampled' if args.upsampled else ''}.trc")
    ensure_dir(eval_dir); ensure_dir(enh_dir)

    rtmw3d_trc = rtmw3d_dir / f"rtmw3d{'_upsampled' if args.upsampled else ''}.trc"
    meta_path  = trial_root / "meta.json"

    log_info(f"Trial root : {trial_root}")
    log_info(f"Trc preds path : {rtmw3d_trc}")
    log_info(f"meta.json  : {meta_path}")

    # Subject info
    log_step("Reading meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    subj = meta.get("subject") or {}
    height_m = subj.get("height_m", None)
    mass_kg  = subj.get("mass_kg", None)
    if not (isinstance(height_m, (int, float)) and height_m > 0):
        raise SystemExit("[ERROR] subject.height_m missing/invalid in meta.json")
    if not (isinstance(mass_kg, (int, float)) and mass_kg > 0):
        raise SystemExit("[ERROR] subject.mass_kg missing/invalid in meta.json")
    log_info(f"Subject info: height={height_m} m ({height_m*1000:.1f} mm), mass={mass_kg} kg")

    # Get markers
    feature_markers_lower, response_markers_lower = getOpenPoseMarkers_lowerExtremity2()
    feature_markers_upper, response_markers_upper = getMarkers_upperExtremity_noPelvis2() 
    feature_markers_all = [feature_markers_lower, feature_markers_upper]
    response_markers_all = [response_markers_lower, response_markers_upper]
    augmenterModelType_all = ["body", "arms"]

    # Load trc
    trc_file = utilsDataman.TRCFile(rtmw3d_trc)

    # Loop over augmenter types to handle separate augmenters for lower and
    # upper bodies.
    outputs_all = {}
    n_response_markers_all = 0
    for idx_augm, augmenterModelType in enumerate(augmenterModelType_all):
        outputs_all[idx_augm] = {}
        feature_markers = feature_markers_all[idx_augm]
        response_markers = response_markers_all[idx_augm]
        
        model_dir = (Path(args.body_model) if augmenterModelType == "body" else Path(args.arms_model))

        # Step 1: import .trc file with OpenPose marker trajectories.
        log_info(f"Step 1: import .trc") 
        trc_data = TRC2numpy(rtmw3d_trc, feature_markers)
        trc_data_data = trc_data[:,1:]
        
        # Step 2: Normalize with reference marker position.
        log_info(f"Step 2: Normalize with reference marker position.") 
        with open(os.path.join(model_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        referenceMarker = metadata['reference_marker']
        referenceMarker_data = trc_file.marker(referenceMarker)
        norm_trc_data_data = np.zeros((trc_data_data.shape[0],
                                       trc_data_data.shape[1]))
        for i in range(0,trc_data_data.shape[1],3):
            norm_trc_data_data[:,i:i+3] = (trc_data_data[:,i:i+3] - 
                                           referenceMarker_data)
            
        # Step 3: Normalize with subject's height.
        log_info(f"Step 3: Normalize with subject's height.") 
        norm2_trc_data_data = copy.deepcopy(norm_trc_data_data)
        norm2_trc_data_data = norm2_trc_data_data / height_m

        # Step 4: Add remaining features.
        log_info(f"Step 4: Add remaining features.") 
        inputs = copy.deepcopy(norm2_trc_data_data)
        if featureHeight:    
            inputs = np.concatenate(
                (inputs, height_m*np.ones((inputs.shape[0],1))), axis=1)
        if featureWeight:    
            inputs = np.concatenate(
                (inputs, mass_kg*np.ones((inputs.shape[0],1))), axis=1)

        # Step 5: Pre-process data
        log_info(f"Step 5: Pre-process data") 
        pathMean = os.path.join(model_dir, "mean.npy")
        pathSTD = os.path.join(model_dir, "std.npy")
        if os.path.isfile(pathMean):
            trainFeatures_mean = np.load(pathMean, allow_pickle=True)
            inputs -= trainFeatures_mean
        if os.path.isfile(pathSTD):
            trainFeatures_std = np.load(pathSTD, allow_pickle=True)
            inputs /= trainFeatures_std 

        # Step 6: Reshape inputs if necessary (eg, LSTM)
        log_info(f"Step 6: Reshape inputs") 
        inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))
            
        # Load model and weights, and predict outputs.
        log_info(f"Load model and weights, and predict outputs.") 
        json_file = open(os.path.join(model_dir, "model.json"), 'r')
        pretrainedModel_json = json_file.read()
        json_file.close()
        model = tf.keras.models.model_from_json(pretrainedModel_json)
        model.load_weights(os.path.join(model_dir, "weights.h5"))  
        outputs = model.predict(inputs, verbose=2)

        # Post-process outputs.
        # Step 1: Reshape if necessary (eg, LSTM)
        log_info(f"Post-process outputs.")
        log_info(f"Step 1: Reshape") 
        outputs = np.reshape(outputs, (outputs.shape[1], outputs.shape[2]))

        # Step 2a: Un-normalize with subject's height
        log_info(f"Step 2a: Un-normalize with subject's height") 
        unnorm_outputs = outputs * height_m
        
        # Step 2b: Un-normalize with reference marker position.
        log_info(f"Step 2b: Un-normalize with reference marker position.") 
        unnorm2_outputs = np.zeros((unnorm_outputs.shape[0],
                                    unnorm_outputs.shape[1]))
        for i in range(0,unnorm_outputs.shape[1],3):
            unnorm2_outputs[:,i:i+3] = (unnorm_outputs[:,i:i+3] + 
                                        referenceMarker_data)

        # Add markers to .trc file.
        log_info(f"Add markers to .trc file.") 
        for c, marker in enumerate(response_markers):
            x = unnorm2_outputs[:,c*3]
            y = unnorm2_outputs[:,c*3+1]
            z = unnorm2_outputs[:,c*3+2]
            trc_file.add_marker(marker, x, y, z)
            
        # Gather data for computing minimum y-position.
        log_info(f"Gather data for computing minimum y-position.") 
        outputs_all[idx_augm]['response_markers'] = response_markers   
        outputs_all[idx_augm]['response_data'] = unnorm2_outputs
        n_response_markers_all += len(response_markers)

    # Extract minimum y-position across response markers. This is used
    # to align feet and floor when visualizing.
    log_info(f"Extract minimum y-position across response markers. This is used to align feet and floor when visualizing.") 
    responses_all_conc = np.zeros((unnorm2_outputs.shape[0],
                                   n_response_markers_all*3))
    idx_acc_res = 0
    for idx_augm in outputs_all:
        idx_acc_res_end = (idx_acc_res + 
                           (len(outputs_all[idx_augm]['response_markers']))*3)
        responses_all_conc[:,idx_acc_res:idx_acc_res_end] = (
            outputs_all[idx_augm]['response_data'])
        idx_acc_res = idx_acc_res_end
    # Minimum y-position across response markers.
    log_info(f"Minimum y-position across response markers.") 
    min_y_pos = np.min(responses_all_conc[:,1::3])

    # If offset
    #if offset:
    #    trc_file.offset('y', -(min_y_pos-0.01))
        
    # Return augmented .trc file   
    log_info(f"Save augmented .trc file ") 
    trc_file.write(enh_output)
    
    #return min_y_pos

if __name__ == "__main__":
    main()
