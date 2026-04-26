"""
SynthPose marker definitions.

52-keypoint model extending COCO 17 with anatomical landmarks.
"""

SYNTHPOSE_MARKERS = {
    0: "Nose", 1: "L_Eye", 2: "R_Eye", 3: "L_Ear", 4: "R_Ear",
    5: "L_Shoulder", 6: "R_Shoulder", 7: "L_Elbow", 8: "R_Elbow",
    9: "L_Wrist", 10: "R_Wrist", 11: "L_Hip", 12: "R_Hip",
    13: "L_Knee", 14: "R_Knee", 15: "L_Ankle", 16: "R_Ankle",
    17: "sternum", 18: "rshoulder", 19: "lshoulder", 20: "r_lelbow",
    21: "l_lelbow", 22: "r_melbow", 23: "l_melbow", 24: "r_lwrist",
    25: "l_lwrist", 26: "r_mwrist", 27: "l_mwrist", 28: "r_ASIS",
    29: "l_ASIS", 30: "r_PSIS", 31: "l_PSIS", 32: "r_knee",
    33: "l_knee", 34: "r_mknee", 35: "l_mknee", 36: "r_ankle",
    37: "l_ankle", 38: "r_mankle", 39: "l_mankle", 40: "r_5meta",
    41: "l_5meta", 42: "r_toe", 43: "l_toe", 44: "r_big_toe",
    45: "l_big_toe", 46: "l_calc", 47: "r_calc", 48: "C7",
    49: "L2", 50: "T11", 51: "T6",
}

NUM_MARKERS = len(SYNTHPOSE_MARKERS)  # 52


def find_marker_index(markers: dict[int, str], name: str) -> int:
    for idx, marker_name in markers.items():
        if marker_name == name:
            return idx
    raise KeyError(f"marker {name!r} not found in schema (have {sorted(markers.values())})")


HIP_INDICES = (
    find_marker_index(SYNTHPOSE_MARKERS, "L_Hip"),
    find_marker_index(SYNTHPOSE_MARKERS, "R_Hip"),
)
