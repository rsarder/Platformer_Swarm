// scenes‑menu.js
export default class MenuScene extends Phaser.Scene {
  constructor() {
    super({ key: "MenuScene" });
  }

  /**
   * Optional data you can pass when starting this scene:
   * {
   *   title:  "YOUR GAME TITLE",
   *   bgColor: "#24252A",
   *   hoverColor: "#ffff00",
   *   items: [
   *     { label: "Play",        scene: "GameScene",      baseColor: "#00ff00" },
   *     { label: "Settings",    scene: "SettingsScene" },
   *     { label: "Instructions",scene: "InstructionsScene" }
   *   ],
   *   levelData: { ... }   // forwarded to target scenes
   * }
   */
  init(data = {}) {
    this.menuCfg = {
      title:      data.title      ?? "GAME TITLE",
      bgColor:    data.bgColor    ?? "#24252A",
      hoverColor: data.hoverColor ?? "#ffff00",
      items:      data.items      ?? [
        { label: "Start",       scene: "GameScene",       baseColor: "#00ff00" },
        { label: "Settings",    scene: "SettingsScene" },
        { label: "Instructions",scene: "InstructionsScene" }
      ],
      levelData: data.levelData ?? null
    };
  }

  create() {
    const { width } = this.scale;
    let y = 100;

    /* -------------- Background & Title -------------- */
    this.cameras.main.setBackgroundColor(this.menuCfg.bgColor);
    this.add
      .text(width / 2, y, this.menuCfg.title, {
        font: "32px Arial",
        color: "#ffffff"
      })
      .setOrigin(0.5);

    /* -------------- Menu Items -------------- */
    y += 100;
    this.menuCfg.items.forEach((item, idx) => {
      const baseColor = item.baseColor ?? "#ffffff";
      const txt = this.add
        .text(width / 2, y + idx * 50, item.label, {
          font: "20px Arial",
          color: baseColor
        })
        .setOrigin(0.5)
        .setInteractive({ useHandCursor: true });

      txt.on("pointerover", () => txt.setColor(this.menuCfg.hoverColor));
      txt.on("pointerout",  () => txt.setColor(baseColor));
      txt.on("pointerdown", () => {
        if (item.scene)
          this.scene.start(item.scene, { levelData: this.menuCfg.levelData });
      });
    });
  }
}
