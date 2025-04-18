// systems.js
export default class GameSystems {
  /**
   * @param {Phaser.Scene} scene         – The owning scene.
   * @param {object}       [cfg]         – Optional customisation.
   *        {number}  cfg.startLives     – Initial lives   (default 3)
   *        {number}  cfg.startLevel     – Initial level   (default 1)
   *        {object}  cfg.keyMap         – Key ↔ action map (see initializeInput)
   *        {object}  cfg.callbacks      – Hook functions for engine‑specific actions:
   *                                      { onMove(dir), onJump(), onShoot(),
   *                                        onPauseToggle(isPaused) }
   */
  constructor(scene, cfg = {}) {
    this.scene = scene;

    /* ---------- Runtime state ---------- */
    this.state = {
      score: 0,
      lives: cfg.startLives ?? 3,
      level: cfg.startLevel ?? 1
    };

    /* ---------- Input & callbacks ---------- */
    this.keyMap    = cfg.keyMap    ?? null;          // set later if null
    this.callbacks = {
      onMove:       cfg.callbacks?.onMove       ?? (() => {}),
      onJump:       cfg.callbacks?.onJump       ?? (() => {}),
      onShoot:      cfg.callbacks?.onShoot      ?? (() => {}),
      onPauseToggle:cfg.callbacks?.onPauseToggle?? (() => {})
    };

    this.keys = {};
  }

  /* ============================================================
     INPUT
     ============================================================ */
  initializeInput() {
    /* Default WASD + extras – override via cfg.keyMap */
    const defaults = {
      up:    "W",
      left:  "A",
      down:  "S",
      right: "D",
      jump:  "SPACE",
      pause: "ESC",
      shoot: "SHIFT"
    };
    this.keys = this.scene.input.keyboard.addKeys(this.keyMap ?? defaults);
  }

  #handleInput() {
    const k = this.keys;
    if (!k) return;

    /* Movement (continuous) */
    if (k.left?.isDown)  this.callbacks.onMove("left");
    if (k.right?.isDown) this.callbacks.onMove("right");

    /* Jump / Shoot (edge‑trigger) */
    if (Phaser.Input.Keyboard.JustDown(k.jump))  this.callbacks.onJump();
    if (k.shoot?.isDown)                         this.callbacks.onShoot();

    /* Pause toggle (edge‑trigger) */
    if (Phaser.Input.Keyboard.JustDown(k.pause)) this.#togglePause();
  }

  /* ============================================================
     GAME STATE HELPERS
     ============================================================ */
  addScore(amt = 1) {
    this.state.score += amt;
    this.scene.events.emit("score:changed", this.state.score);
  }

  loseLife() {
    this.state.lives -= 1;
    this.scene.events.emit("lives:changed", this.state.lives);
    if (this.state.lives < 0) this.transitionToScene("GameOverScene");
  }

  nextLevel() {
    this.state.level += 1;
    this.scene.events.emit("level:changed", this.state.level);
    this.transitionToScene(`Level${this.state.level}Scene`);
  }

  /* ============================================================
     PAUSE / AUDIO
     ============================================================ */
  #togglePause() {
    const wasPaused = this.scene.scene.isPaused();
    if (wasPaused) {
      this.scene.scene.resume();
      this.#manageAudio("resume");
    } else {
      this.scene.scene.pause();
      this.#manageAudio("pause");
    }
    this.callbacks.onPauseToggle(!wasPaused);
  }

  #manageAudio(mode) {
    if (!this.scene.sound) return;
    if (mode === "pause")  this.scene.sound.pauseAll();
    if (mode === "resume") this.scene.sound.resumeAll();
  }

  /* ============================================================
     SCENE FLOW
     ============================================================ */
  transitionToScene(key, data = {}) {
    this.scene.scene.start(key, data);
  }

  /* ============================================================
     PER‑FRAME UPDATE  – call from your Scene.update()
     ============================================================ */
  update() {
    this.#handleInput();            // input ➜ callbacks
    // other per‑tick systems can be added here
  }
}
