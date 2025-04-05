// scenes-instructions.js
export default class InstructionsScene extends Phaser.Scene {
  constructor() {
    super({ key: "InstructionsScene" });
  }

  create() {
    this.cameras.main.setBackgroundColor("#222222");

    this.add
      .text(400, 100, "How to Play (Placeholder)", {
        font: "28px Arial",
        fill: "#ffffff",
      })
      .setOrigin(0.5);

    this.add
      .text(400, 180, "Use arrow keys or WASD to move", {
        font: "20px Arial",
        fill: "#ffffff",
      })
      .setOrigin(0.5);

    this.add
      .text(400, 220, "Press Space or W/Up to jump", {
        font: "20px Arial",
        fill: "#ffffff",
      })
      .setOrigin(0.5);

    this.add
      .text(400, 260, "Press P to pause", {
        font: "20px Arial",
        fill: "#ffffff",
      })
      .setOrigin(0.5);

    const backText = this.add
      .text(400, 500, "Back to Menu", {
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
