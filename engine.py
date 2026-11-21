"""Run real Kimodo inference and export a single SOMA-77 take."""

MODEL = "Kimodo-SOMA-RP-v1.2"


def export_take(output, skeleton, fps, metadata, directory):
    import json

    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from kimodo.exports.bvh import save_motion_bvh
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.skeleton import global_rots_to_local_rots

    names = list(skeleton.bone_order_names)
    parents = skeleton.joint_parents.detach().cpu().numpy()
    root_index = int(skeleton.root_idx)
    neutral = skeleton.neutral_joints.detach().cpu().numpy()
    offsets = neutral.copy()
    offsets[parents >= 0] -= neutral[parents[parents >= 0]]
    floor = min(neutral[names.index(name), 1] for name in (
        "LeftFoot", "LeftToeEnd", "RightFoot", "RightToeEnd"
    ))
    offsets[root_index] = neutral[root_index] - [0, floor, 0]

    roots = output["posed_joints"][:, root_index, :]
    device = skeleton.joint_parents.device
    global_rotations = torch.as_tensor(output["global_rot_mats"], device=device)
    rotations = global_rots_to_local_rots(global_rotations, skeleton)
    frame_count = len(roots)
    quaternions = Rotation.from_matrix(rotations.detach().cpu().numpy().reshape(-1, 3, 3))
    motion = {
        "schema": "kimodo-maker.soma-motion.v1",
        "skeleton": "SOMA-77",
        "units": "m",
        "up": "+Y",
        "forward": "+Z",
        "quaternionOrder": "xyzw",
        "fps": float(fps),
        "frameCount": frame_count,
        "generation": metadata,
        "joints": [
            {"name": name, "parent": int(parent), "restOffset": offset.tolist()}
            for name, parent, offset in zip(names, parents, offsets)
        ],
        "rootPositions": roots.round(6).tolist(),
        "localRotations": quaternions.as_quat().reshape(frame_count, len(names) * 4).round(7).tolist(),
    }

    save_kimodo_npz(str(directory / "motion.npz"), output)
    save_motion_bvh(
        directory / "motion.bvh",
        rotations,
        torch.as_tensor(roots, device=device),
        skeleton=skeleton,
        fps=fps,
        standard_tpose=True,
    )
    with (directory / "motion.json").open("w", encoding="utf-8") as handle:
        json.dump(motion, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return {"fps": float(fps), "frame_count": frame_count}


class KimodoEngine:
    def __init__(self):
        self.model = None

    def generate(self, options, directory, report):
        import torch
        from kimodo import load_model
        from kimodo.constraints import load_constraints_lst
        from kimodo.skeleton import SOMASkeleton30
        from kimodo.tools import seed_everything

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch에서 CUDA를 찾지 못했습니다. CUDA 버전의 PyTorch와 GPU 연결을 확인하세요.")
        if self.model is None:
            report("loading")
            self.model = load_model(MODEL, device="cuda:0")

        model = self.model
        skeleton = model.skeleton
        frame_count = round(options["duration"] * model.fps)
        constraints = load_constraints_lst(options["constraints"], skeleton, device=model.device)
        seed_everything(options["seed"])
        prompt = options["prompt"]
        if prompt[-1] not in ".!?":
            prompt += "."
        report("generating")
        with torch.no_grad():
            batch = model(
                [prompt],
                [frame_count],
                num_denoising_steps=options["steps"],
                constraint_lst=constraints,
                num_samples=1,
                multi_prompt=True,
                post_processing=True,
                return_numpy=True,
            )
        output = {name: value[0] for name, value in batch.items()}
        export_skeleton = skeleton.somaskel77 if isinstance(skeleton, SOMASkeleton30) else skeleton
        report("exporting")
        metadata = {key: value for key, value in options.items() if key != "constraints"}
        metadata.update(model=MODEL, constraintCount=len(constraints))
        return export_take(output, export_skeleton, model.fps, metadata, directory)
