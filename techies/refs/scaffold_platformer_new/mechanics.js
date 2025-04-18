// mechanics.js
export default class GameMechanics {
  /**
   * @param {object}   cfg
   * @param {object}   cfg.levelData      – Parsed JSON describing platforms, enemies, etc.
   * @param {object}   cfg.adapters       – Optional callbacks to bridge into a render/physics engine.
   *        {function(ent):void}  spawnPlatform
   *        {function(ent):void}  spawnEnemy
   *        {function(ent):void}  spawnEntity
   *        {function(score):void}updateScoreUI
   *        {function(lives):void}updateLivesUI
   *        {function():void}     showGameOver
   *        {function():void}     showLevelComplete
   */
  constructor({ levelData, adapters = {} } = {}) {
    /* ---------- Bind or fallback ---------- */
    Object.assign(this, {
      spawnPlatform:      adapters.spawnPlatform      ?? (() => {}),
      spawnEnemy:         adapters.spawnEnemy         ?? (() => {}),
      spawnEntity:        adapters.spawnEntity        ?? (() => {}),
      updateScoreUI:      adapters.updateScoreUI      ?? (() => {}),
      updateLivesUI:      adapters.updateLivesUI      ?? (() => {}),
      showGameOver:       adapters.showGameOver       ?? (() => {}),
      showLevelComplete:  adapters.showLevelComplete  ?? (() => {})
    });

    /* ---------- Static level data ---------- */
    this.levelData = levelData;
    this.gravity   = levelData?.gravity ?? 500;

    /* ---------- Dynamic state ---------- */
    this.player = {
      x: 0, y: 0, w: 32, h: 48,
      speed: 200,
      jumpPower: 350,
      velocityX: 0,
      velocityY: 0,
      lives: 3,
      score: 0,
      isJumping: false
    };

    this.enemies   = [];   // Populated in initialise()
    this.bullets   = [];
    this.entities  = [];
    this.isPaused  = false;
    this.highScore = 0;
  }

  /* ============================================================
     INITIALISATION
     ============================================================ */
  initialise() {
    /* Spawn geometry & objects through the chosen rendering engine */
    this.levelData?.platforms?.forEach(p => this.spawnPlatform(p));
    this.levelData?.enemies?.forEach(e   => {
      const enemy = { ...e, direction: 1, health: e.health ?? 1 };
      this.enemies.push(enemy);
      this.spawnEnemy(enemy);
    });
    this.levelData?.entities?.forEach(o  => {
      this.entities.push(o);
      this.spawnEntity(o);
    });

    /* Player start position */
    if (this.levelData?.playerStart) {
      this.player.x = this.levelData.playerStart.x;
      this.player.y = this.levelData.playerStart.y;
    }
  }

  /* ============================================================
     FRAME‑BY‑FRAME UPDATE (call from your main loop)
     ============================================================ */
  update(dt, controlState) {
    if (this.isPaused) return;

    this.#handlePlayerMovement(controlState, dt);
    this.#applyGravity(dt);
    this.#updateEnemies(dt);
    this.#updateBullets(dt);
    this.#detectCollisions();
    this.#checkWinLoss();
  }

  /* ============================================================
     PLAYER
     ============================================================ */
  #handlePlayerMovement(ctrl = {}, dt) {
    const { left, right, jump } = ctrl;
    const accel = this.player.speed;

    if (left)  this.player.velocityX = -accel;
    if (right) this.player.velocityX =  accel;
    if (!left && !right) this.player.velocityX = 0;

    if (jump && !this.player.isJumping) {
      this.player.velocityY = -this.player.jumpPower;
      this.player.isJumping = true;
    }

    this.player.x += this.player.velocityX * dt;
    this.player.y += this.player.velocityY * dt;
  }

  #applyGravity(dt) {
    this.player.velocityY += this.gravity * dt;

    // Simple floor clamp; replace with engine collision in production
    if (this.player.y > 9999) { /* fell off map */ this.#loseLife(); }
  }

  /* ============================================================
     ENEMIES
     ============================================================ */
  #updateEnemies(dt) {
    this.enemies.forEach(e => {
      if (e.movementPattern === 'chase') {
        const dir = this.player.x < e.x ? -1 : 1;
        e.x += dir * (e.speed ?? 100) * dt;
      } else if (e.movementPattern === 'patrol') {
        e.x += e.direction * (e.speed ?? 100) * dt;
        if (e.x < (e.minX ?? 0) || e.x > (e.maxX ?? 9999)) {
          e.direction *= -1;
        }
      }
    });
  }

  /* ============================================================
     PROJECTILES
     ============================================================ */
  fireBullet(angleRad = 0, speed = 600) {
    this.bullets.push({
      x: this.player.x,
      y: this.player.y,
      vx: Math.cos(angleRad) * speed,
      vy: Math.sin(angleRad) * speed,
      damage: 1
    });
  }

  #updateBullets(dt) {
    this.bullets = this.bullets.filter(b => {
      b.x += b.vx * dt;
      b.y += b.vy * dt;
      return b.x > -100 && b.x < 10000; // simple bounds check
    });
  }

  /* ============================================================
     COLLISIONS
     ============================================================ */
  #detectCollisions() {
    // Bullets → Enemies
    this.bullets.forEach(b => {
      this.enemies.forEach(e => {
        if (this.#overlap(b, e)) {
          e.health -= b.damage;
          b.hit = true;
        }
      });
    });
    this.bullets = this.bullets.filter(b => !b.hit);
    this.enemies = this.enemies.filter(e => e.health > 0);

    // Player → Entities (coins, hazards, etc.)
    this.entities = this.entities.filter(ent => {
      if (!this.#overlap(this.player, ent)) return true;

      switch (ent.type) {
        case 'collectible':
          this.player.score += 1;
          this.updateScoreUI(this.player.score);
          return false; // remove collected
        case 'obstacle':
          this.#loseLife();
          return true;
        default:
          return true;
      }
    });
  }

  #overlap(a, b) {
    return (
      a.x < b.x + (b.w ?? b.width ?? 16) &&
      a.x + (a.w ?? a.width ?? 16) > b.x &&
      a.y < b.y + (b.h ?? b.height ?? 16) &&
      a.y + (a.h ?? a.height ?? 16) > b.y
    );
  }

  /* ============================================================
     LIFE / SCORE / WIN / LOSS
     ============================================================ */
  #loseLife() {
    if (--this.player.lives < 0) {
      this.showGameOver();
      this.isPaused = true;
    } else {
      this.updateLivesUI(this.player.lives);
      // optional: respawn player at checkpoint
    }
  }

  #checkWinLoss() {
    const winPos = this.levelData?.winCondition?.position;
    if (winPos && this.player.x >= winPos.x && this.player.y >= winPos.y) {
      this.showLevelComplete();
      this.isPaused = true;
    }
    if (this.player.lives < 0) this.showGameOver();
  }

  /* ============================================================
     PAUSE
     ============================================================ */
  togglePause() { this.isPaused = !this.isPaused; }

  /* ============================================================
     HIGH SCORE
     ============================================================ */
  saveHighScore() {
    if (this.player.score > this.highScore) {
      this.highScore = this.player.score;
      try { localStorage.setItem('highScore', String(this.highScore)); }
      catch { /* storage may be unavailable */ }
    }
  }
}

/* ----------------------------------------------------------------
   USAGE EXAMPLE (pseudo‑engine loop)

import levelJSON from './level‑1.json' with { type: 'json' };
import GameMechanics from './mechanics.js';

const gm = new GameMechanics({
  levelData: levelJSON,
  adapters:  { /* supply Phaser bridge functions here  }
});
*/
gm.initialise();

let lastT = performance.now();
function loop(t = 0) {
  const dt = (t - lastT) / 1000; // seconds
  lastT = t;

  const controls = {
    left:  keyLeft.isDown,
    right: keyRight.isDown,
    jump:  keyUp.isDown
  };
  gm.update(dt, controls);
  requestAnimationFrame(loop);
}
loop();
