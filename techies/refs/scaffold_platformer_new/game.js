// game.js ───────────────────────────────────────────────────────
import Phaser from "phaser";

/* 1 ▸ Register your scenes (barrel file keeps the list tidy) */
import {
  SceneBoot,     // loads common assets & level JSON
  SceneMenu,
  ScenePlay,
  ScenePause,
  SceneSettings,
  SceneHelp,
} from "./scenes/index.js";

/* 2 ▸ Tunables you’ll touch most often */
const GAME_CONTAINER_ID = "game";                       // <div id="game"></div>
const VIEWPORT          = { width: 800, height: 600 };
const GRAVITY_Y         = 500;                          // 0 → Scene default
const LEVEL_JSON_PATH   = "./level-config.json";        // null → skip fetch

/* 3 ▸ Safe JSON fetch that NEVER rejects (returns null on error) */
async function fetchJSON(path) {
  if (!path) return null;
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.warn(`[game.js] Couldn’t load "${path}":`, err.message);
    return null;
  }
}

/* 4 ▸ Bootstrap in an async IIFE so we can await the fetch */
(async () => {
  const levelData = await fetchJSON(LEVEL_JSON_PATH);

  /* 4a ▸ Build the standard Phaser config */
  const config = {
    type: Phaser.AUTO,
    ...VIEWPORT,
    parent: GAME_CONTAINER_ID,
    backgroundColor: "#000",
    physics: {
      default: "arcade",
      arcade: { gravity: { y: GRAVITY_Y }, debug: false },
    },
    scene: [
      /* Order doesn’t matter because we start manually */
      SceneBoot,
      SceneMenu,
      ScenePlay,
      ScenePause,
      SceneSettings,
      SceneHelp,
    ],
  };

  /* 4b ▸ Kick‑start Phaser */
  const game = new Phaser.Game(config);

  /* 4c ▸ Start the very first scene & hand it levelData */
  game.scene.start("SceneBoot", { levelData });
})();
