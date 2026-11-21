import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { Retarget } from './retarget.js';

export class Viewer {
  constructor(viewport, onFrame) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#10181e');
    this.scene.fog = new THREE.FogExp2('#10181e', 0.065);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.025, 120);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.domElement.setAttribute('aria-label', 'SOMA와 VRM 모션 미리보기');
    viewport.prepend(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 0.15;
    this.controls.maxDistance = 30;
    const grid = new THREE.GridHelper(50, 100, '#3d575d', '#22333a');
    this.scene.add(grid, new THREE.HemisphereLight(0xffffff, 0x67677d, 2.2));
    const light = new THREE.DirectionalLight(0xffeddc, 2.0);
    light.position.set(3, 5, 3);
    this.scene.add(light);

    const avatar = new THREE.Group();
    this.scene.add(avatar);
    this.avatar = avatar;
    this.skeleton = new THREE.Group();
    this.scene.add(this.skeleton);
    const boneGeometry = new THREE.BufferGeometry();
    boneGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(76 * 6), 3));
    this.bones = new THREE.LineSegments(boneGeometry, new THREE.LineBasicMaterial({ color: '#b6f2d6' }));
    const jointGeometry = new THREE.BufferGeometry();
    jointGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(77 * 3), 3));
    this.joints = new THREE.Points(jointGeometry, new THREE.PointsMaterial({ color: '#e8ffe8', size: 0.022 }));
    this.path = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#64a994' }));
    this.path.position.y = 0.006;
    this.skeleton.add(this.bones, this.joints);
    this.scene.add(this.path);
    this.path.visible = this.skeleton.visible = false;
    this.bones.frustumCulled = this.joints.frustumCulled = false;

    this.frame = 0;
    this.playing = false;
    this.speed = 1;
    this.loop = true;
    this.rootMotion = true;
    this.showSkeleton = true;
    this.onFrame = onFrame;
    this.lastTime = null;
    this.focus = new THREE.Vector3();
    this.nextFocus = new THREE.Vector3();
    this.focusDelta = new THREE.Vector3();
    this.resetView();
    new ResizeObserver(() => {
      this.renderer.setSize(viewport.clientWidth, viewport.clientHeight, false);
      this.camera.aspect = viewport.clientWidth / Math.max(1, viewport.clientHeight);
      this.camera.updateProjectionMatrix();
    }).observe(viewport);
    this.renderer.setAnimationLoop((time) => this.tick(time));
  }

  measureFocus(target) {
    const motion = this.motion;
    if (!motion) return target.set(0, 0, 0);
    const { x, z } = motion.root;
    if (!this.retarget) return target.set(x, 0, z);
    const avatarX = x * this.retarget.scale + this.retarget.hipsRest.x;
    const avatarZ = z * this.retarget.scale + this.retarget.hipsRest.z;
    if (!this.showSkeleton) return target.set(avatarX, 0, avatarZ);
    return target.set((avatarX + x + this.skeleton.position.x) / 2, 0, (avatarZ + z) / 2);
  }

  resetView() {
    this.measureFocus(this.focus);
    const sourceHeight = this.motion && this.showSkeleton ? this.motion.restPosition[0].y : 0;
    const height = Math.max(0.6, sourceHeight, this.avatarHeight ?? 0, !this.motion && !this.vrm ? 0.9 : 0);
    const comparison = this.motion && this.vrm && this.showSkeleton;
    const distance = Math.max(height * 4.9, comparison ? 3.7 / Math.min(1, this.camera.aspect) : 0);
    const { x, z } = this.focus;
    this.controls.target.set(x, height, z);
    this.camera.position.set(x + distance * 0.23, height + distance * 0.21, z + distance);
    this.controls.update();
  }

  async loadVrm(file) {
    const manager = new THREE.LoadingManager();
    manager.setURLModifier((url) => {
      if (/^(blob:|data:)/i.test(url)) return url;
      throw new Error('외부 리소스에 의존하는 VRM은 지원하지 않습니다.');
    });
    const loader = new GLTFLoader(manager).register((parser) => new VRMLoaderPlugin(parser));
    const gltf = await loader.parseAsync(await file.arrayBuffer(), '');
    const vrm = gltf.userData.vrm;
    if (!vrm?.humanoid?.getNormalizedBoneNode('hips')) {
      VRMUtils.deepDispose(gltf.scene);
      throw new Error('휴머노이드 VRM 0.x 또는 1.0 파일이 필요합니다.');
    }
    VRMUtils.rotateVRM0(vrm);
    VRMUtils.removeUnnecessaryVertices(vrm.scene);
    this.clearVrm();
    this.vrm = vrm;
    this.avatar.add(vrm.scene);
    const floor = new THREE.Box3().setFromObject(vrm.scene).min.y;
    if (Number.isFinite(floor)) this.avatar.position.y = -floor;
    const hipsHeight = vrm.humanoid.getNormalizedBoneNode('hips').getWorldPosition(new THREE.Vector3()).y;
    this.avatarHeight = hipsHeight > 0 ? hipsHeight : 0.9;
    this.bindVrm();
    this.drawPose();
    this.resetView();
  }

  clearVrm() {
    if (this.vrm) {
      this.avatar.remove(this.vrm.scene);
      VRMUtils.deepDispose(this.vrm.scene);
    }
    this.avatar.position.set(0, 0, 0);
    this.vrm = null;
    this.retarget = null;
    this.avatarHeight = null;
    this.updateVisibility();
  }

  bindVrm() {
    this.retarget = this.vrm && this.motion ? new Retarget(this.vrm, this.motion) : null;
    this.updateVisibility();
  }

  setMotion(motion) {
    this.motion = motion;
    this.frame = 0;
    this.playing = false;
    this.path.geometry.dispose();
    const points = motion.rootPositions.map(([x, , z]) => new THREE.Vector3(x - motion.origin.x, 0, z - motion.origin.z));
    this.path.geometry = new THREE.BufferGeometry().setFromPoints(points);
    this.bindVrm();
    this.drawPose();
    this.resetView();
    this.onFrame(this.frame, this.playing);
  }

  updateVisibility() {
    this.skeleton.visible = Boolean(this.motion && this.showSkeleton);
    this.skeleton.position.x = this.vrm ? -1.65 : 0;
    this.path.position.x = this.skeleton.visible ? this.skeleton.position.x : 0;
    const scale = this.retarget && !this.skeleton.visible ? this.retarget.scale : 1;
    this.path.scale.set(scale, 1, scale);
    this.path.visible = Boolean(this.motion && this.rootMotion);
  }

  seek(frame) {
    this.frame = Math.max(0, Math.min(this.motion.frameCount - 1, frame));
    this.drawPose();
    this.onFrame(this.frame, this.playing);
  }

  togglePlay() {
    if (!this.motion) return;
    if (!this.playing && this.frame >= this.motion.frameCount - 1) this.frame = 0;
    this.playing = !this.playing;
    this.onFrame(this.frame, this.playing);
  }

  drawPose() {
    if (!this.motion) return;
    const motion = this.motion;
    motion.sample(this.frame, this.rootMotion);
    const joints = this.joints.geometry.attributes.position;
    const bones = this.bones.geometry.attributes.position;
    for (let index = 0; index < motion.joints.length; index++) {
      const p = motion.worldPosition[index];
      joints.setXYZ(index, p.x, p.y, p.z);
      if (index === 0) continue;
      const parent = motion.worldPosition[motion.joints[index].parent];
      const i = (index - 1) * 2;
      bones.setXYZ(i, parent.x, parent.y, parent.z);
      bones.setXYZ(i + 1, p.x, p.y, p.z);
    }
    joints.needsUpdate = bones.needsUpdate = true;
    this.retarget?.apply(motion);
    this.measureFocus(this.nextFocus);
    this.focusDelta.copy(this.nextFocus).sub(this.focus);
    this.camera.position.add(this.focusDelta);
    this.controls.target.add(this.focusDelta);
    this.focus.copy(this.nextFocus);
  }

  tick(time) {
    const delta = this.lastTime === null ? 0 : Math.min((time - this.lastTime) / 1000, 0.1);
    this.lastTime = time;
    if (this.playing) {
      this.frame += delta * this.motion.fps * this.speed;
      if (this.loop) {
        this.frame %= this.motion.frameCount;
      } else if (this.frame >= this.motion.frameCount - 1) {
        this.frame = this.motion.frameCount - 1;
        this.playing = false;
      }
      this.drawPose();
      this.onFrame(this.frame, this.playing);
    }
    this.vrm?.update(delta);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
