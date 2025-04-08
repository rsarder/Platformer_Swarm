// entities-enemy.js
export default class Enemy extends Phaser.Physics.Arcade.Sprite {
  constructor(scene, x, y) {
    super(scene, x, y, null);
    scene.add.existing(this);
    scene.physics.add.existing(this);

    this.setTintFill(0xff4444);
    this.setCollideWorldBounds(true);
  }

  update() {
    // Optional: Move left/right or chase the player
  }
}
