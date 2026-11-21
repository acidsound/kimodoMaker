import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Quaternion, Vector3 } from 'three';
import { Motion } from '../src/motion.js';

function motionData() {
  const joints = Array.from({ length: 77 }, (_, i) => ({
    name: i === 0 ? 'Hips' : `Joint${i}`,
    parent: i === 0 ? -1 : 0,
    restOffset: [0, 1, 0],
  }));
  const neutral = Array.from({ length: 77 }, () => [0, 0, 0, 1]).flat();
  const rotated = neutral.slice();
  new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), Math.PI).toArray(rotated);
  return {
    schema: 'kimodo-maker.soma-motion.v1', skeleton: 'SOMA-77', units: 'm',
    up: '+Y', forward: '+Z', quaternionOrder: 'xyzw',
    joints, fps: 30, frameCount: 2,
    rootPositions: [[3, 1, 4], [5, 3, 6]],
    localRotations: [neutral, rotated],
  };
}

test('samples intermediate root positions and composes child rotations', () => {
  const motion = new Motion(motionData());
  motion.sample(0.5, true);
  assert.deepEqual(motion.root.toArray(), [1, 2, 1]);
  assert.ok(motion.worldPosition[1].distanceTo(new Vector3(0, 2, 1)) < 1e-8);
  assert.ok(motion.worldRotation[1].angleTo(motion.worldRotation[0]) < 1e-7);
});

test('root motion toggle only removes horizontal movement', () => {
  const motion = new Motion(motionData());
  motion.sample(1, false);
  assert.deepEqual(motion.root.toArray(), [0, 3, 0]);
});

test('rejects out-of-order joints', () => {
  const data = motionData();
  data.joints[2].parent = 3;
  assert.throws(() => new Motion(data), /계층/);
});
