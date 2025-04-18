// scenes‑settings.js
export default class SettingsScene extends Phaser.Scene {
  constructor() {
    super({ key: "SettingsScene" });
  }

  /**
   * Accept optional config when launching:
   * {
   *   bgColor: "#333333",
   *   title:   "Settings",
   *   items: [
   *     { type: "audio",      labelOn: "Audio: ON",       labelOff: "Audio: OFF" },
   *     { type: "fullscreen", labelOn: "Fullscreen: ON",  labelOff: "Fullscreen: OFF" }
   *   ],
   *   backLabel: "Back to Menu",
   *   backScene: "MenuScene"
   * }
   */
  init(data = {}) {
    this.cfg = {
      bgColor:   data.bgColor   ?? "#333333",
      title:     data.title     ?? "Settings",
      items:     data.items     ?? [
        { type: "audio",      labelOn: "Audio: ON",      labelOff: "Audio: OFF" },
        { type: "fullscreen", labelOn: "Fullscreen: ON", labelOff: "Fullscreen: OFF" }
      ],
      backLabel: data.backLabel ?? "Back to Menu",
      backScene: data.backScene ?? "MenuScene"
    };
  }

  create() {
    const { width } = this.scale;
    let y = 100;

    /* ---------- Background & Title ---------- */
    this.cameras.main.setBackgroundColor(this.cfg.bgColor);
    this.add.text(width / 2, y, this.cfg.title, {
      font: "28px Arial",
      color: "#ffffff"
    }).setOrigin(0.5);

    /* ---------- Toggle items ---------- */
    y += 100;
    this.cfg.items.forEach((item, idx) => {
      const currentState = this.#getStoredState(item.type);
      const txt = this.add.text(width / 2, y + idx * 50,
        currentState ? item.labelOn : item.labelOff,
        { font: "20px Arial", color: "#ffffff" }
      ).setOrigin(0.5)
       .setInteractive({ useHandCursor: true });

      txt.on("pointerover", () => txt.setColor("#ffff00"));
      txt.on("pointerout",  () => txt.setColor("#ffffff"));
      txt.on("pointerdown", () => {
        const nextState = !this.#getStoredState(item.type);
        this.#applySetting(item.type, nextState);
        txt.setText(nextState ? item.labelOn : item.labelOff);
      });
    });

    /* ---------- Back button ---------- */
    this.#makeButton(width / 2, y + this.cfg.items.length * 50 + 100,
      this.cfg.backLabel, "#00ff00", "#ffff00",
      () => this.scene.start(this.cfg.backScene)
    );

    /* ---------- Apply stored prefs on load ---------- */
    this.cfg.items.forEach(i => this.#applySetting(i.type, this.#getStoredState(i.type)));
  }

  /* ===== Helpers ================================================= */

  #makeButton(x, y, label, color, hoverColor, cb) {
    const btn = this.add.text(x, y, label, {
      font: "20px Arial", color
    }).setOrigin(0.5).setInteractive({ useHandCursor: true });

    btn.on("pointerover", () => btn.setColor(hoverColor));
    btn.on("pointerout",  () => btn.setColor(color));
    btn.on("pointerdown", cb);
  }

  #getStoredState(type) {
    /* LocalStorage returns string; treat anything other than "off" as ON */
    return localStorage.getItem(type) !== "off";
  }

  #applySetting(type, stateOn) {
    localStorage.setItem(type, stateOn ? "on" : "off");

    switch (type) {
      case "audio":
        this.sound?.setMute && this.sound.setMute(!stateOn);
        break;
      case "fullscreen":
        if (stateOn && !this.scale.isFullscreen) {
          this.scale.startFullscreen();
        } else if (!stateOn && this.scale.isFullscreen) {
          this.scale.stopFullscreen();
        }
        break;
      /* Extend with more setting types here */
    }
  }
}
