// scenes-game.js
import Mechanics from "./mechanics.js";
import System from "./systems.js";
import Player from "./entities-player.js";
import Enemy from "./entities-enemy.js";

export default class GameScene extends Phaser.Scene {
  constructor() {
    super({ key: "GameScene" });
  }

  init(data) {
    this.levelData = data.levelData || null;
  }

  preload() {
    // No texture generation; we’re using built-in rectangle shapes.
  }

  create() {
    this.cameras.main.setBackgroundColor("#87CEEB");

    // Use the first level in the config
    const firstLevel = this.levelData?.levels?.[0];
    if (!firstLevel) {
      this.add.text(20, 20, "No level data found!", { font: "18px Arial", fill: "#000" });
      return;
    }

    // Set world bounds so the player can’t leave the screen.
    this.physics.world.setBounds(0, 0, 800, 600);

    // Initialize the mechanics (placeholder)
    this.mechanics = new Mechanics(this);

    // Initialize the systems (placeholder)
    this.systems = new Systems(this);

    // Create platforms as static rectangle shapes
    this.platforms = this.physics.add.staticGroup();

    firstLevel.platforms.forEach((plat) => {
      // Create a rectangle shape at the given x,y.
      // Note: Phaser positions rectangle centers by default.
      const platform = this.add.rectangle(plat.x, plat.y, plat.width, plat.height, 0x666666);
      // Add a static physics body to the rectangle.
      this.physics.add.existing(platform, true);
      // Ensure the body's size is updated
      platform.body.setSize(plat.width, plat.height);
      platform.body.updateFromGameObject();
      // Add to the static group for easier collision management.
      this.platforms.add(platform);
    });

    // Create the player at the start position.
    const px = firstLevel.playerStart?.x || 100;
    const py = firstLevel.playerStart?.y || 100;
    this.player = new Player(this, px, py);
    this.physics.add.collider(this.player, this.platforms);
    this.player.setCollideWorldBounds(true);

    // Create enemies.
    this.enemies = this.physics.add.group();
    firstLevel.enemies.forEach((enemyData) => {
      const enemy = new Enemy(this, enemyData.x, enemyData.y);
      this.enemies.add(enemy);
      this.physics.add.collider(enemy, this.platforms);
    });
    this.physics.add.overlap(this.player, this.enemies, () => {
      console.log("Player overlapped enemy!");
    });

    // Setup input keys.
    this.cursors = this.input.keyboard.createCursorKeys();
    this.keyW = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.W);
    this.keyA = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.A);
    this.keyD = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.D);
    this.keySpace = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.SPACE);

    this.pauseKey = this.input.keyboard.addKey("P");
  }

  update() {
    if (Phaser.Input.Keyboard.JustDown(this.pauseKey)) {
      this.scene.launch("PauseScene");
      this.scene.pause();
    }

    const movement = {
      left: this.cursors.left.isDown || this.keyA.isDown,
      right: this.cursors.right.isDown || this.keyD.isDown,
      jump: this.cursors.up.isDown || this.keyW.isDown || this.keySpace.isDown,
    };

    this.player.handleMovement(movement);
  }
}
