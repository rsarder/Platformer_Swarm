// entities‑enemy.js
export default class Enemy extends Phaser.Physics.Arcade.Sprite {
  /**
   * Generic enemy entity.
   *
   * @param {Phaser.Scene}  scene            – Owning scene
   * @param {number}        x                – Spawn X
   * @param {number}        y                – Spawn Y
   * @param {object}        [cfg]            – Behaviour & visual options
   *        {string}   cfg.behaviour         – 'static' | 'patrol' | 'chase' | 'custom'
   *        {number}   cfg.speed             – Pixels/second for moving behaviours
   *        {number}   cfg.patrolDistance    – Max offset before turning (patrol AI)
   *        {number}   cfg.tint              – Tint colour (e.g. 0xff4444)
   *        {string}   cfg.texture           – Frame key (omit ⟹ invisible rect)
   *        {function} cfg.aiFn(enemy, player, dt) – Custom AI callback
   */
  constructor(scene, x, y, cfg = {}) {
    const { texture = null } = cfg;            // Allows optional texture
    super(scene, x, y, texture);

    // --------------------------------- Essentials
    scene.add.existing(this);
    scene.physics.add.existing(this);
    this.setCollideWorldBounds(true);

    // --------------------------------- Visuals
    if (cfg.tint !== undefined) this.setTintFill(cfg.tint);

    // --------------------------------- Behaviour
    this.behaviour      = cfg.behaviour      ?? 'patrol';
    this.speed          = cfg.speed          ?? 100;
    this.patrolDistance = cfg.patrolDistance ?? 120;
    this._aiFn          = typeof cfg.aiFn === 'function' ? cfg.aiFn : null;

    // Internal state
    this._dir     = 1;      // 1 = right, -1 = left
    this._originX = x;      // For patrol distance check
  }

  /* ---------- Built‑in AI helpers ---------- */

  _patrol() {
    const walkedTooFar = Math.abs(this.x - this._originX) > this.patrolDistance;
    if (this.body.blocked.left || this.body.blocked.right || walkedTooFar) {
      this._dir *= -1;
    }
    this.setVelocityX(this.speed * this._dir);
  }

  _chase(target) {
    if (!target) return this.setVelocityX(0);
    const dir = target.x < this.x ? -1 : 1;
    this.setVelocityX(this.speed * dir);
  }

  /* ---------- Public update ---------- */

  /**
   * Call from the owning scene’s update loop.
   * @param {Phaser.GameObjects.Sprite} [player] – Optional chase target.
   * @param {number}                    [dt]     – Delta time (ms)
   */
  update(player, dt) {
    if (this._aiFn) {
      // Full user‑supplied control
      return this._aiFn(this, player, dt);
    }

    switch (this.behaviour) {
      case 'static':
        this.setVelocityX(0);
        break;

      case 'chase':
        this._chase(player);
        break;

      case 'patrol':
      default:
        this._patrol();
        break;
    }
  }
}
