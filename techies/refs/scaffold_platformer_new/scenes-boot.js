// scenes‑boot.js
export default class BootScene extends Phaser.Scene {
  constructor() {
    super({ key: "BootScene" });
  }

  /**
   * Expect optional data when this scene is started:
   * {
   *   assets: [         // array of asset descriptors (see below)
   *     { type: 'image',       key: 'player',     url: 'assets/player.png' },
   *     { type: 'audio',       key: 'bgm',        url: 'assets/bgm.ogg' },
   *     { type: 'atlas',       key: 'ui',         textureURL: 'ui.png', atlasURL: 'ui.json' },
   *     { type: 'spritesheet', key: 'explosion',  url: 'explosion.png',
   *                            frameConfig: { frameWidth: 64, frameHeight: 64 } }
   *   ],
   *   levelData: {...}  // forwarded untouched to the next scene
   * }
   */
  init(data = {}) {
    this.assetManifest = data.assets ?? [];
    this.levelData     = data.levelData ?? null;
  }

  preload() {
    const { width, height } = this.cameras.main;

    /* --------- Simple progress text (replace with a bar if you like) --------- */
    const progressTxt = this.add
      .text(width / 2, height / 2, "Loading 0%", {
        fontFamily: "sans-serif",
        fontSize: "20px",
        color: "#ffffff",
      })
      .setOrigin(0.5);

    this.load.on("progress", p => progressTxt.setText(`Loading ${Math.floor(p * 100)}%`));
    this.load.on("complete", () => progressTxt.destroy());

    /* --------- Dynamically queue every asset from the manifest --------- */
    for (const asset of this.assetManifest) {
      switch (asset.type) {
        case "image":
          this.load.image(asset.key, asset.url);
          break;
        case "audio":
          this.load.audio(asset.key, asset.url);
          break;
        case "atlas":
          this.load.atlas(asset.key, asset.textureURL, asset.atlasURL);
          break;
        case "spritesheet":
          this.load.spritesheet(asset.key, asset.url, asset.frameConfig);
          break;
        case "tilemapTiledJSON":
          this.load.tilemapTiledJSON(asset.key, asset.url);
          break;
        /* Add cases for bitmap fonts, videos, etc. as your engine grows */
        default:
          console.warn(`[BootScene] Unknown asset type "${asset.type}"`, asset);
      }
    }

    /* If you pass an empty manifest the scene still works (no assets to load). */
  }

  create() {
    /* Kick off the next scene—rename key to match your flow. */
    this.scene.start("MenuScene", { levelData: this.levelData });
  }
}
