// scenes‑pause.js
export default class PauseScene extends Phaser.Scene {
  constructor() {
    super({ key: "PauseScene" });
  }

  /**
   * Optional data on launch:
   * {
   *   overlayColor : "rgba(0,0,0,0.5)",
   *   title        : "Game Paused",
   *   menuItems    : [
   *     { label: "Resume", scene: "GameScene", baseColor: "#00ff00", resume: true },
   *     { label: "Main Menu", scene: "MenuScene" }
   *   ]
   * }
   */
  init(data = {}) {
    this.cfg = {
      overlayColor: data.overlayColor ?? "rgba(0, 0, 0, 0.5)",
      title:        data.title        ?? "Game Paused",
      menuItems:    data.menuItems    ?? [
        { label: "Resume",    scene: "GameScene", resume: true, baseColor: "#00ff00" },
        { label: "Main Menu", scene: "MenuScene" }
      ]
    };
  }

  create() {
    const { width, height } = this.scale;
    this.cameras.main.setBackgroundColor(this.cfg.overlayColor);

    /* --------------- Title --------------- */
    this.add.text(width / 2, height * 0.3, this.cfg.title, {
      font: "30px Arial",
      color: "#ffffff"
    }).setOrigin(0.5);

    /* --------------- Menu items --------------- */
    const startY = height * 0.5;
    const gap    = 50;

    this.cfg.menuItems.forEach((item, idx) => {
      const baseColor = item.baseColor ?? "#ffffff";
      const txt = this.add.text(width / 2, startY + idx * gap, item.label, {
        font: "20px Arial",
        color: baseColor
      })
      .setOrigin(0.5)
      .setInteractive({ useHandCursor: true });

      txt.on("pointerover", () => txt.setColor("#ffff00"));
      txt.on("pointerout",  () => txt.setColor(baseColor));
      txt.on("pointerdown", () => {
        this.scene.stop("PauseScene");
        if (item.resume) {
          /* Resume the paused gameplay scene */
          this.scene.resume(item.scene);
        } else {
          /* Stop gameplay and go elsewhere (e.g., main menu) */
          this.scene.stop("GameScene");
          this.scene.start(item.scene);
        }
      });
    });
  }
}
