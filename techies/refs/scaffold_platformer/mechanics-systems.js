// mechanics-systems.js
export default class MechanicsSystem {
  constructor(scene) {
    this.scene = scene;
  }

  applyGravity(entity) {
    // If you ever needed custom gravity outside of arcade physics
  }

  handlePlatformCollision(entity, platform) {
    // Example: do something special on collision
  }

  // etc.
}
