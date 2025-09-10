'''
python src/OpenSim/open_sim_ik.py \
    -m manifests/OpenCapDataset/subject2.yaml \
    -p config/paths.yaml \
    --trial walking1 \
    --ik-xml ./models/OpenSim/Setup_IK.xml \
    --upsampled
'''
from IO.load_manifest import load_manifest
import opensim
import os
from pathlib import Path
import json
import argparse

# ---------- Logging ----------
def log_step(msg): print(f"[STEP] {msg}")
def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")
def log_err (msg): print(f"[ERROR] {msg}")
def log_done(msg): print(f"[DONE] {msg}")

# ---------- IO helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# Inverse kinematics.
def runIKTool(pathGenericSetupFile, pathScaledModel, pathTRCFile,
              pathOutputFolder, timeRange=[], IKFileName='not_specified'):
    
    # Paths
    if IKFileName == 'not_specified':
        _, IKFileName = os.path.split(pathTRCFile)
        IKFileName = IKFileName[:-4]
    pathOutputMotion = os.path.join(pathOutputFolder, IKFileName + '.mot')
    pathOutputSetup =  os.path.join(pathOutputFolder, 'Setup_IK_' + IKFileName + '.xml')
    
    # To make IK faster, we remove the patellas and their constraints from the
    # model. Constraints make the IK problem more difficult, and the patellas
    # are not used in the IK solution for this particular model. Since muscles
    # are attached to the patellas, we also remove all muscles.
    opensim.Logger.setLevelString('error')
    model = opensim.Model(pathScaledModel)
    # Remove all actuators.                                         
    forceSet = model.getForceSet()
    forceSet.setSize(0)
    # Remove patellofemoral constraints.
    constraintSet = model.getConstraintSet()
    patellofemoral_constraints = [
        'patellofemoral_knee_angle_r_con', 'patellofemoral_knee_angle_l_con']
    for patellofemoral_constraint in patellofemoral_constraints:
        i = constraintSet.getIndex(patellofemoral_constraint, 0)
        constraintSet.remove(i)       
    # Remove patella bodies.
    bodySet = model.getBodySet()
    patella_bodies = ['patella_r', 'patella_l']
    for patella in patella_bodies:
        i = bodySet.getIndex(patella, 0)
        bodySet.remove(i)
    # Remove patellofemoral joints.
    jointSet = model.getJointSet()
    patellofemoral_joints = ['patellofemoral_r', 'patellofemoral_l']
    for patellofemoral in patellofemoral_joints:
        i = jointSet.getIndex(patellofemoral, 0)
        jointSet.remove(i)
    # Print the model to a new file.
    model.finalizeConnections
    model.initSystem()
    pathScaledModelWithoutPatella = pathScaledModel.replace('.osim', '_no_patella.osim')
    model.printToXML(pathScaledModelWithoutPatella)   

    # Setup IK tool.
    log_info(f"Loading IK setup file: {pathGenericSetupFile}")
    IKTool = opensim.InverseKinematicsTool(os.fspath(pathGenericSetupFile))            
    IKTool.setName(IKFileName)
    log_info(f"Setting scaled model: {pathGenericSetupFile}")
    IKTool.set_model_file(pathScaledModelWithoutPatella)          
    IKTool.set_marker_file(pathTRCFile)
    if timeRange:
        IKTool.set_time_range(0, timeRange[0])
        IKTool.set_time_range(1, timeRange[-1])
    IKTool.setResultsDir(os.fspath(pathOutputFolder))                     
    IKTool.set_report_errors(True)
    IKTool.set_report_marker_locations(False)
    IKTool.set_output_motion_file(pathOutputMotion)
    IKTool.printToXML(pathOutputSetup)
    command = 'opensim-cmd -o error' + ' run-tool ' + pathOutputSetup
    log_info(f"Running scale command: {command}")
    os.system(command)
    log_info(f"IK saved: {pathOutputSetup}")
    
    return pathOutputMotion, pathScaledModelWithoutPatella

def main():
    ap = argparse.ArgumentParser(description="Run scaling of model for OpenSim with Marker-Enahncer output")
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--static-trial",  default='')
    ap.add_argument("--ik-xml", default=None, help="XML for IK")
    #ap.add_argument("--base-model", default=None, help="Base OSIM model")
    ap.add_argument("--trc-type", required=True, choices=["metric_upsampled", "cannonical", "abs_cannonical", "world", "cam", "metric"])
    ap.add_argument("--mocap", action="store_true",default=False)


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
    ensure_dir(eval_dir); 
    ensure_dir(enh_dir)
    ensure_dir(osim_dir)

    #osim_scaling_output = osim_dir / "model_scaled.osim"
    if args.mocap:
        enh_output = trial["open_pose"]
    else:
        enh_output = os.path.join(enh_dir, f"enhancer_{args.trial}_{args.trc_type}.trc")

    meta_path  = trial_root / "meta.json"
    ik_xml_path  = Path(args.ik_xml)
    scaled_model_path = os.path.join(base, f"{args.static_trail if args.static_trial != '' else args.trial}", "OpenSim", f"{'usporedba' if args.mocap else ''}", f"LaiUhlrich2022_scaled_{args.trc_type}.osim")

    log_info(f"Trial root : {trial_root}")
    log_info(f"Enhancer output preds path : {enh_output}")
    log_info(f"meta.json  : {meta_path}")
    log_info(f"IK xml  : {ik_xml_path}")
    log_info(f"Scaled model osim  : {scaled_model_path}")

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

    log_info('Running Inverse Kinematics')
    try:
        pathOutputIK, pathModelIK = runIKTool(
            ik_xml_path, scaled_model_path, 
            enh_output, osim_dir, IKFileName=f"{args.trial}__{args.trc_type}{'_mocap' if args.mocap else ''}_IK")
    except Exception as e:
        if len(e.args) == 2:
            log_err(f"Error: {e.args[0]}, \n{e.args[1]}")
        elif len(e.args) == 1:
            log_err(f"Inverse kinematics failed. {e}")
            
if __name__ == "__main__":
    main()