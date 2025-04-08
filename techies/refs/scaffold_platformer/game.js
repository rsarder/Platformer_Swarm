// game.js
import BootScene from "./scenes-boot.js";
import MenuScene from "./scenes-menu.js";
import GameScene from "./scenes-game.js";
import PauseScene from "./scenes-pause.js";
import SettingsScene from "./scenes-settings.js";
import InstructionsScene from "./scenes-instructions.js";

let levelData = null;

fetch("./level-config.json")
  .then((res) => res.json())
  .then((json) => {
    levelData = json;
    initGame(levelData);
  })
  .catch((err) => {
    console.error("Failed to load level-config.json:", err);
    initGame(null);
  });

function initGame(levelData) {
  const config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    parent: "game",
    backgroundColor: "#000000",
    physics: {
      default: "arcade",
      arcade: { 
        debug: false,
        gravity: { y: 500 } // set gravity
      },
    },
    scene: [
      BootScene,
      MenuScene,
      new GameScene(levelData),
      PauseScene,
      SettingsScene,
      InstructionsScene,
    ],
  };

  new Phaser.Game(config);
}
