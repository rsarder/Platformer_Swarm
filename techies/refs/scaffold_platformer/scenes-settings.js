// scenes-settings.js
export default class SettingsScene extends Phaser.Scene {
  constructor() {
    super({ key: "SettingsScene" });
  }

  create() {
    this.cameras.main.setBackgroundColor("#333333");

    this.add
      .text(400, 100, "Settings", {
        font: "28px Arial",
        fill: "#ffffff",
      })
      .setOrigin(0.5);

    this.add
      .text(400, 200, "No real settings implemented", {
        font: "20px Arial",
        fill: "#aaaaaa",
      })
      .setOrigin(0.5);

    const backText = this.add
      .text(400, 400, "Back to Menu", {
        font: "20px Arial",
        fill: "#00ff00",
      })
      .setOrigin(0.5);

    backText.setInteractive({ useHandCursor: true });
    backText.on("pointerdown", () => this.scene.start("MenuScene"));

    backText.on("pointerover", () => backText.setStyle({ fill: "#ffff00" }));
    backText.on("pointerout", () => backText.setStyle({ fill: "#00ff00" }));
  }
}
