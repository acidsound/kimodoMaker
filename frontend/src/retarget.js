import { Box3, Quaternion, Vector3 } from 'three';

const bindings = [
  ['hips', 'Hips'],
  ['spine', 'Spine1'],
  ['chest', 'Spine2'],
  ['upperChest', 'Chest'],
  ['neck', 'Neck2'],
  ['head', 'Head'],
];

for (const side of ['left', 'right']) {
  const somaSide = side === 'left' ? 'Left' : 'Right';
  for (const [vrm, source, nextVrm, nextSource] of [
    ['Shoulder', 'Shoulder', 'UpperArm', 'Arm'],
    ['UpperArm', 'Arm', 'LowerArm', 'ForeArm'],
    ['LowerArm', 'ForeArm', 'Hand', 'Hand'],
    ['Hand', 'Hand', 'MiddleProximal', 'HandMiddle1'],
    ['UpperLeg', 'Leg', 'LowerLeg', 'Shin'],
    ['LowerLeg', 'Shin', 'Foot', 'Foot'],
    ['Foot', 'Foot', 'Toes', 'ToeBase'],
    ['Toes', 'ToeBase'],
  ]) {
    bindings.push([side + vrm, somaSide + source,
      nextVrm && side + nextVrm, nextSource && somaSide + nextSource]);
  }
  for (const [vrm, source, segments] of [
    ['Thumb', 'Thumb', ['Metacarpal', 'Proximal', 'Distal']],
    ['Index', 'Index', ['Proximal', 'Intermediate', 'Distal']],
    ['Middle', 'Middle', ['Proximal', 'Intermediate', 'Distal']],
    ['Ring', 'Ring', ['Proximal', 'Intermediate', 'Distal']],
    ['Little', 'Pinky', ['Proximal', 'Intermediate', 'Distal']],
  ]) {
    for (const [i, segment] of segments.entries()) {
      bindings.push([side + vrm + segment, `${somaSide}Hand${source}${i + 1}`,
        i < 2 ? side + vrm + segments[i + 1] : null, `${somaSide}Hand${source}${i + 2}`]);
    }
  }
}

export class Retarget {
  constructor(vrm, motion) {
    const humanoid = vrm.humanoid;
    humanoid.resetNormalizedPose();
    vrm.update(0);
    vrm.scene.updateMatrixWorld(true);
    const hips = humanoid.getNormalizedBoneNode('hips');
    const world = new Vector3();
    const targetDirection = new Vector3();
    const sourceDirection = new Vector3();
    const floor = new Box3().setFromObject(vrm.scene).min.y;
    this.hips = hips;
    this.hipsRest = hips.getWorldPosition(new Vector3());
    this.hipsHeight = this.hipsRest.y - floor;
    const sourceHeight = motion.restPosition[0].y;
    this.scale = this.hipsHeight > 0 && Number.isFinite(this.hipsHeight) && sourceHeight > 0
      ? this.hipsHeight / sourceHeight : 1;
    this.hipsHeight = Number.isFinite(this.hipsHeight) ? this.hipsHeight : sourceHeight;
    this.parentRotation = new Quaternion();
    this.position = new Vector3();
    this.links = [];

    for (const [vrmName, sourceName, nextVrm, nextSource] of bindings) {
      const node = humanoid.getNormalizedBoneNode(vrmName);
      const source = motion.indices.get(sourceName);
      if (!node || source === undefined) continue;
      const targetChild = nextVrm && humanoid.getNormalizedBoneNode(nextVrm);
      const sourceChild = motion.indices.get(nextSource);
      const correction = new Quaternion();
      if (targetChild && sourceChild !== undefined) {
        targetDirection.copy(targetChild.getWorldPosition(world)).sub(node.getWorldPosition(world));
        sourceDirection.copy(motion.restPosition[sourceChild]).sub(motion.restPosition[source]);
        if (targetDirection.lengthSq() > 1e-8 && sourceDirection.lengthSq() > 1e-8) {
          correction.setFromUnitVectors(targetDirection.normalize(), sourceDirection.normalize());
        }
      }
      const rest = node.getWorldQuaternion(new Quaternion());
      this.links.push({ node, source, rest: correction.multiply(rest) });
    }
  }

  apply(motion) {
    this.position.copy(motion.root).multiplyScalar(this.scale);
    this.position.x += this.hipsRest.x;
    this.position.z += this.hipsRest.z;
    this.hips.position.copy(this.hips.parent.worldToLocal(this.position));
    for (const { node, source, rest } of this.links) {
      node.parent.getWorldQuaternion(this.parentRotation).invert();
      node.quaternion.copy(this.parentRotation).multiply(motion.worldRotation[source]).multiply(rest);
      node.updateMatrixWorld();
    }
  }
}
