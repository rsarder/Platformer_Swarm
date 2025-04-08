// scenes-menu.js
export default class MenuScene extends Phaser.Scene {
  constructor() {
    super({ key: "MenuScene" });
  }

  create() {
    this.cameras.main.setBackgroundColor("#24252A");

    const titleText = this.add
      .text(400, 100, "My Minimal Platformer", {
        font: "32px Arial",
        fill: "#ffffff",
      })
      .setOrigin(0.5);

    const startText = this.add
      .text(400, 200, "Start Game", { font: "20px Arial", fill: "#00ff00" })
      .setOrigin(0.5);

    const settingsText = this.add
      .text(400, 250, "Settings", { font: "20px Arial", fill: "#ffffff" })
      .setOrigin(0.5);

    const instrText = this.add
      .text(400, 300, "Instructions", { font: "20px Arial", fill: "#ffffff" })
      .setOrigin(0.5);

    [startText, settingsText, instrText].forEach((item) => {
      item.setInteractive({ useHandCursor: true });
      item.on("pointerover", () => item.setStyle({ fill: "#ffff00" }));
      item.on("pointerout", () => {
        if (item === startText) item.setStyle({ fill: "#00ff00" });
        else item.setStyle({ fill: "#ffffff" });
      });
    });

    startText.on("pointerdown", () => this.scene.start("GameScene"));
    settingsText.on("pointerdown", () => this.scene.start("SettingsScene"));
    instrText.on("pointerdown", () => this.scene.start("InstructionsScene"));
  }
}
