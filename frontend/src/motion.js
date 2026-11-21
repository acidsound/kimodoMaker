import { Quaternion, Vector3 } from 'three';

const SCHEMA = 'kimodo-maker.soma-motion.v1';

export class Motion {
  constructor(data) {
    const { joints, fps, frameCount, rootPositions, localRotations } = data;
    if (data.schema !== SCHEMA || data.skeleton !== 'SOMA-77' || data.units !== 'm'
        || data.up !== '+Y' || data.forward !== '+Z' || data.quaternionOrder !== 'xyzw'
        || !Number.isFinite(fps) || fps <= 0 || !Number.isInteger(frameCount) || frameCount < 1
        || !Array.isArray(joints) || joints.length !== 77 || joints[0]?.name !== 'Hips'
        || rootPositions?.length !== frameCount || localRotations?.length !== frameCount) {
      throw new Error('SOMA-77 모션 JSON 형식이 올바르지 않습니다.');
    }

    this.fps = fps;
    this.frameCount = frameCount;
    this.joints = joints;
    this.rootPositions = rootPositions;
    this.rotations = localRotations;
    this.indices = new Map(joints.map(({ name }, index) => [name, index]));
    this.restPosition = joints.map(({ restOffset, parent }, index) => {
      if (index === 0 ? parent !== -1 : !Number.isInteger(parent) || parent < 0 || parent >= index) {
        throw new Error('SOMA-77 관절 계층이 올바르지 않습니다.');
      }
      return new Vector3().fromArray(restOffset);
    });
    for (let i = 1; i < joints.length; i++) {
      this.restPosition[i].add(this.restPosition[joints[i].parent]);
    }
    this.worldPosition = joints.map(() => new Vector3());
    this.worldRotation = joints.map(() => new Quaternion());
    this.nextRotation = new Quaternion();
    this.origin = new Vector3(rootPositions[0][0], 0, rootPositions[0][2]);
    this.root = new Vector3();
    this.nextRoot = new Vector3();
  }

  sample(frame, rootMotion) {
    const bounded = Math.max(0, Math.min(this.frameCount - 1, frame));
    const a = Math.floor(bounded);
    const b = Math.min(a + 1, this.frameCount - 1);
    const fraction = bounded - a;

    this.root.fromArray(this.rootPositions[a])
      .lerp(this.nextRoot.fromArray(this.rootPositions[b]), fraction)
      .sub(this.origin);
    if (!rootMotion) this.root.set(0, this.root.y, 0);

    for (let i = 0; i < this.joints.length; i++) {
      const { parent, restOffset } = this.joints[i];
      const position = this.worldPosition[i];
      const rotation = this.worldRotation[i];
      rotation.fromArray(this.rotations[a], i * 4);
      this.nextRotation.fromArray(this.rotations[b], i * 4);
      rotation.slerp(this.nextRotation, fraction).normalize();
      if (parent === -1) {
        position.copy(this.root);
      } else {
        position.fromArray(restOffset).applyQuaternion(this.worldRotation[parent]).add(this.worldPosition[parent]);
        rotation.premultiply(this.worldRotation[parent]);
      }
    }
  }
}
