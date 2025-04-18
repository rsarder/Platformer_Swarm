// entities‑player.js
export default class Player extends Phaser.Physics.Arcade.Sprite {
  /**
   * @param {Phaser.Scene} scene              Owning scene.
   * @param {number}       x                  Spawn X.
   * @param {number}       y                  Spawn Y.
   * @param {object}       [cfg]              Behaviour & visual options.
   *        {string} cfg.texture              Frame key (omit ➜ plain rect).
   *        {number} cfg.tint                 Tint colour (e.g. 0x00ccff).
   *        {number} cfg.speed                Horizontal speed (px / s).
   *        {number} cfg.jumpPower            Jump impulse (px / s).
   *        {number} cfg.gravity              Gravity override (px / s²). 0 → use scene default.
   */
  constructor(scene, x, y, cfg = {}) {
    const { texture = null } = cfg;     // Allows optional texture
    super(scene, x, y, texture);

    // ---------------- Scene & Physics
    scene.add.existing(this);
    scene.physics.add.existing(this);
    this.setCollideWorldBounds(true);

    // ---------------- Visuals
    if (cfg.tint !== undefined) this.setTintFill(cfg.tint);

    // ---------------- Movement parameters (with safe defaults)
    this.speed     = cfg.speed     ?? 200;
    this.jumpPower = cfg.jumpPower ?? 350;
    this.gravity   = cfg.gravity   ?? 0;

    if (this.gravity !== 0) this.body.setGravityY(this.gravity);
  }

  /* ----------------------------------------------------------- */
  /*               Movement helpers (left / right / jump)        */
  /* ----------------------------------------------------------- */

  /**
   * Apply movement based on a control state object.
   * Expected shape: { left:Boolean, right:Boolean, jump:Boolean }
   */
  handleMovement(ctrl = {}) {
    const { left, right, jump } = ctrl;

    // Horizontal
    if (left) {
      this.setVelocityX(-this.speed);
    } else if (right) {
      this.setVelocityX(this.speed);
    } else {
      this.setVelocityX(0);
    }

    // Vertical (single‑jump)
    if (jump && this.body.onFloor()) {
      this.setVelocityY(-this.jumpPower);
    }
  }

  /**
   * Public update — call from your scene’s `update`.
   * @param {object} [ctrl] Control state object passed to `handleMovement`.
   */
  update(ctrl) {
    this.handleMovement(ctrl);
  }
}
