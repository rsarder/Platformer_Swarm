// scenes‑game.js
import Mechanics from "./mechanics.js";
import Systems   from "./systems.js";
import Player    from "./entities-player.js";
import Enemy     from "./entities-enemy.js";

export default class GameScene extends Phaser.Scene {
  constructor() { super({ key: "GameScene" }); }

  /* ------------------------------------------------------------
     1.  Receive levelData (and anything else you like) from the
         scene that launched us.
     ------------------------------------------------------------ */
  init(data = {}) {
    this.levelData = data.levelData ?? null;
  }

  preload() {
    //  Place to load level‑specific assets if required.
    //  (Generic GameScene normally has nothing to preload.)
  }

  /* ------------------------------------------------------------
     2.  Scene creation — build world from levelData
     ------------------------------------------------------------ */
  create() {
    /* -------------------------------------------------- */
    /* 2a. House‑keeping & engine helpers                 */
    /* -------------------------------------------------- */
    const { width, height } = this.scale;
    this.cameras.main.setBackgroundColor(
      this.levelData?.background?.color ?? "#87CEEB"
    );
    this.physics.world.setBounds(0, 0, width, height);

    this.mechanics = new Mechanics(this);
    this.systems   = new Systems(this);

    /* -------------------------------------------------- */
    /* 2b. Platforms                                      */
    /* -------------------------------------------------- */
    this.platforms = this.physics.add.staticGroup();
    const firstLevel = this.levelData?.levels?.[0];

    if (firstLevel?.platforms?.length) {
      firstLevel.platforms.forEach(p => {
        const rect = this.add.rectangle(p.x, p.y, p.width, p.height, 0x666666);
        this.physics.add.existing(rect, true);
        rect.body.setSize(p.width, p.height);
        rect.body.updateFromGameObject();
        this.platforms.add(rect);
      });
    } else {
      this.add.text(16, 16, "⚠ No platforms in level", { color: "#ff0000" });
      console.warn("[GameScene] No platforms found");
    }

    /* -------------------------------------------------- */
    /* 2c. Player                                         */
    /* -------------------------------------------------- */
    const spawn = firstLevel?.playerStart ?? { x: width / 2, y: height / 2 };
    this.player = new Player(this, spawn.x, spawn.y);
    this.player.setCollideWorldBounds(true);
    this.physics.add.collider(this.player, this.platforms);

    /* -------------------------------------------------- */
    /* 2d. Enemies                                        */
    /* -------------------------------------------------- */
    this.enemies = this.physics.add.group();
    (firstLevel?.enemies ?? []).forEach(e => {
      const enemy = new Enemy(this, e.x, e.y, e.type);
      this.enemies.add(enemy);
      this.physics.add.collider(enemy, this.platforms);
    });
    this.physics.add.overlap(
      this.player,
      this.enemies,
      this.onPlayerEnemyHit,
      null,
      this
    );

    /* -------------------------------------------------- */
    /* 2e. HUD (score & lives)                            */
    /* -------------------------------------------------- */
    this.state = { score: 0, lives: 3 };
    this.scoreTxt = this.add.text(16, 16, "Score: 0",  { fontSize: 18 });
    this.livesTxt = this.add.text(16, 40, "Lives: 3",  { fontSize: 18 });

    /* -------------------------------------------------- */
    /* 2f. Input                                          */
    /* -------------------------------------------------- */
    this.cursors   = this.input.keyboard.createCursorKeys();
    this.keyW      = this.input.keyboard.addKey("W");
    this.keyA      = this.input.keyboard.addKey("A");
    this.keyD      = this.input.keyboard.addKey("D");
    this.keySpace  = this.input.keyboard.addKey("SPACE");
    this.pauseKey  = this.input.keyboard.addKey("P");
  }

  /* ------------------------------------------------------------
     3.  Frame update
     ------------------------------------------------------------ */
  update() {
    if (Phaser.Input.Keyboard.JustDown(this.pauseKey)) {
      this.scene.launch("PauseScene");
      this.scene.pause();
    }

    const move = {
      left:  this.cursors.left.isDown  || this.keyA.isDown,
      right: this.cursors.right.isDown || this.keyD.isDown,
      jump:  this.cursors.up.isDown    || this.keyW.isDown || this.keySpace.isDown,
    };
    this.player.handleMovement(move);

    /* Example generics — replace with your own win/loss logic */
    if (this.checkWinCondition())  this.completeLevel();
    if (this.state.lives <= 0)     this.gameOver();
  }

  /* ------------------------------------------------------------
     4.  Generic collision & HUD helpers
     ------------------------------------------------------------ */
  onPlayerEnemyHit(player, enemy) {
    enemy.destroy();
    this.state.lives--;
    this.livesTxt.setText(`Lives: ${this.state.lives}`);
  }

  addScore(amount = 1) {
    this.state.score += amount;
    this.scoreTxt.setText(`Score: ${this.state.score}`);
  }

  checkWinCondition() {
    //  Example placeholder: player reaches right edge
    return this.player.x >= this.physics.world.bounds.right - 32;
  }

  completeLevel() {
    this.scene.restart({ levelData: this.levelData }); // or next level
    console.info("[GameScene] Level complete");
  }

  gameOver() {
    this.scene.start("GameOverScene", { score: this.state.score });
    console.info("[GameScene] Game over");
  }
}
