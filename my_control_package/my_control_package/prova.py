"""
Questo script calcola la trasformazione del camera_joint
espressa rispetto a link8 invece che link6.

COME USARLO:
1. Lascia l'URDF COM'È ORA (camera attaccata a link6)
2. Esegui questo script standalone (non come nodo ROS2)
3. Copia i valori xyz e rpy stampati nell'URDF, cambiando
   il parent da link6 a link8
"""

import numpy as np
import pinocchio as pin
import xacro
import os

# ── Parametri da adattare ────────────────────────────────────────────────────
ARM_PREFIX   = "left"
ROBOT_TYPE   = "fr3"

# Questi devono corrispondere esattamente a quelli nel tuo xacro
XYZ_BASE     = "-0.338 -0.8 0"
RPY_BASE     = "0 0 -1.570796"

# Trasformazione camera→link6 attuale (dall'URDF)
T_CAM_FROM_LINK6 = {
    "xyz": np.array([0.25, -0.1, 0.0]),
    "rpy": np.array([-np.pi/2, 0.0, -np.pi/2 - np.pi/4])
}

from ament_index_python.packages import get_package_share_directory
XACRO_FILE = os.path.join(
    get_package_share_directory('franka_gazebo_bringup'),
    'urdf', 'franka_arm.gazebo.xacro'
)
# ────────────────────────────────────────────────────────────────────────────


def load_model(xacro_file, arm_prefix, xyz, rpy):
    doc = xacro.process_file(xacro_file, mappings={
        'arm_prefix':    arm_prefix,
        'robot_type':    'fr3',
        'hand':          'false',
        'ros2_control':  'true',
        'gazebo':        'true',
        'ee_id':         'franka_hand',
        'gazebo_effort': 'true',
        'xyz':           xyz,
        'rpy':           rpy,
        'robot_namespace': arm_prefix
    })
    urdf_string = doc.toxml()
    model = pin.buildModelFromXML(urdf_string)
    return model


def main():
    print("=" * 60)
    print("Carico il modello Pinocchio dall'URDF attuale...")
    print("(camera attaccata a link6)")
    print("=" * 60)

    model = load_model(XACRO_FILE, ARM_PREFIX, XYZ_BASE, RPY_BASE)
    data  = model.createData()

    # Configurazione neutra (tutti i giunti a zero)
    q0 = pin.neutral(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    prefix = f"{ARM_PREFIX}_{ROBOT_TYPE}_"

    # Recupera i frame id
    link6_name = f"{prefix}link6"
    link8_name = f"{prefix}link8"

    link6_id = model.getFrameId(link6_name)
    link8_id = model.getFrameId(link8_name)

    if link6_id == model.nframes:
        raise ValueError(f"Frame '{link6_name}' non trovato. "
                         f"Controlla il prefix. Frame disponibili:\n"
                         + "\n".join([model.frames[i].name 
                                      for i in range(model.nframes)]))
    if link8_id == model.nframes:
        raise ValueError(f"Frame '{link8_name}' non trovato.")

    oMlink6 = data.oMf[link6_id]
    oMlink8 = data.oMf[link8_id]

    print(f"\n[INFO] Posa di {link6_name} a q=0:")
    print(f"  translation : {oMlink6.translation}")
    print(f"  RPY         : {pin.rpy.matrixToRpy(oMlink6.rotation)}")

    print(f"\n[INFO] Posa di {link8_name} a q=0:")
    print(f"  translation : {oMlink8.translation}")
    print(f"  RPY         : {pin.rpy.matrixToRpy(oMlink8.rotation)}")

    # Trasformazione relativa link8 → link6
    # (come vede link6 chi sta seduto in link8)
    link8Mlink6 = oMlink8.inverse() * oMlink6

    print(f"\n[INFO] Trasformazione relativa link8 → link6:")
    print(f"  translation : {link8Mlink6.translation}")
    print(f"  RPY         : {pin.rpy.matrixToRpy(link8Mlink6.rotation)}")

    # Costruisci SE3 della camera rispetto a link6 (configurazione attuale)
    R_cam_in_link6 = pin.rpy.rpyToMatrix(*T_CAM_FROM_LINK6["rpy"])
    t_cam_in_link6 = T_CAM_FROM_LINK6["xyz"]
    link6Mcamera   = pin.SE3(R_cam_in_link6, t_cam_in_link6)

    print(f"\n[INFO] Camera rispetto a link6 (URDF attuale):")
    print(f"  xyz : {link6Mcamera.translation}")
    print(f"  RPY : {pin.rpy.matrixToRpy(link6Mcamera.rotation)}")

    # Trasformazione camera rispetto a link8
    # link8Mcamera = (link8Mlink6) * (link6Mcamera)
    link8Mcamera = link8Mlink6 * link6Mcamera

    xyz_result = link8Mcamera.translation
    rpy_result = pin.rpy.matrixToRpy(link8Mcamera.rotation)

    print("\n" + "=" * 60)
    print("RISULTATO — copia questi valori nell'URDF:")
    print("=" * 60)
    print(f"\n  <joint name=\"${{prefix}}camera_joint\" type=\"fixed\">")
    print(f"      <parent link=\"${{prefix}}link8\"/>")
    print(f"      <child link=\"${{prefix}}camera_link\"/>")
    print(f"      <origin xyz=\"{xyz_result[0]:.6f} {xyz_result[1]:.6f} {xyz_result[2]:.6f}\"")
    print(f"              rpy=\"{rpy_result[0]:.6f} {rpy_result[1]:.6f} {rpy_result[2]:.6f}\"/>")
    print(f"  </joint>")
    print()

    # Verifica: ricostruisci la posa della camera nel mondo da entrambe le strade
    oMcamera_via_link6 = oMlink6 * link6Mcamera
    oMcamera_via_link8 = oMlink8 * link8Mcamera

    print("[VERIFICA] Posa camera nel mondo via link6:")
    print(f"  translation : {oMcamera_via_link6.translation}")
    print(f"  RPY         : {pin.rpy.matrixToRpy(oMcamera_via_link6.rotation)}")

    print("[VERIFICA] Posa camera nel mondo via link8:")
    print(f"  translation : {oMcamera_via_link8.translation}")
    print(f"  RPY         : {pin.rpy.matrixToRpy(oMcamera_via_link8.rotation)}")

    diff_t = np.linalg.norm(
        oMcamera_via_link6.translation - oMcamera_via_link8.translation)
    diff_R = np.linalg.norm(
        oMcamera_via_link6.rotation    - oMcamera_via_link8.rotation)

    print(f"\n[VERIFICA] Differenza traslazione : {diff_t:.2e}  (deve essere ~0)")
    print(f"[VERIFICA] Differenza rotazione   : {diff_R:.2e}  (deve essere ~0)")

    if diff_t < 1e-6 and diff_R < 1e-6:
        print("\n✓ Trasformazione corretta. Puoi aggiornare l'URDF.")
    else:
        print("\n✗ Qualcosa non torna — controlla i parametri in cima allo script.")


if __name__ == "__main__":
    main()