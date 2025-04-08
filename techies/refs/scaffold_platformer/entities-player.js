// entities-player.js
export default class Player extends Phaser.Physics.Arcade.Sprite {
  constructor(scene, x, y) {
    // Use null for texture; the player will be a tinted rectangle.
    super(scene, x, y, null);
    scene.add.existing(this);
    scene.physics.add.existing(this);

    // Tint so the player is visible.
    this.setTintFill(0x00ccff);

    // Basic movement settings.
    this.speed = 160;
    this.jumpPower = 300;
  }

  handleMovement({ left, right, jump }) {
    // Horizontal movement.
    if (left) {
      this.setVelocityX(-this.speed);
    } else if (right) {
      this.setVelocityX(this.speed);
    } else {
      this.setVelocityX(0);
    }

    // Jump only if on the floor.
    if (jump && this.body.onFloor()) {
      this.setVelocityY(-this.jumpPower);
    }
  }
}
