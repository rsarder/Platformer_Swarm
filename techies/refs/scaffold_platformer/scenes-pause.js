// scenes-pause.js
export default class PauseScene extends Phaser.Scene {
  constructor() {
    super({ key: "PauseScene" });
  }

  create() {
    this.cameras.main.setBackgroundColor("rgba(0, 0, 0, 0.5)");

    this.add
      .text(400, 200, "Game Paused", {
        font: "30px Arial",
        fill: "#ffffff",
      })
      .setOrigin(0.5);

    const resumeText = this.add
      .text(400, 300, "Resume", { font: "20px Arial", fill: "#00ff00" })
      .setOrigin(0.5);
    const menuText = this.add
      .text(400, 350, "Main Menu", { font: "20px Arial", fill: "#ffffff" })
      .setOrigin(0.5);

    [resumeText, menuText].forEach((item) => {
      item.setInteractive({ useHandCursor: true });
      item.on("pointerover", () => item.setStyle({ fill: "#ffff00" }));
      item.on("pointerout", () => {
        if (item === resumeText) item.setStyle({ fill: "#00ff00" });
        else item.setStyle({ fill: "#ffffff" });
      });
    });

    resumeText.on("pointerdown", () => {
      this.scene.stop("PauseScene");
      this.scene.resume("GameScene");
    });
    menuText.on("pointerdown", () => {
      this.scene.stop("GameScene");
      this.scene.start("MenuScene");
    });
  }
}
