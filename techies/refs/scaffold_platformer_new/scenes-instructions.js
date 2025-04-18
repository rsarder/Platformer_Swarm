// scenes‑instructions.js
export default class InstructionsScene extends Phaser.Scene {
  constructor() {
    super({ key: "InstructionsScene" });
  }

  /**
   * Accepts an optional data object:
   * {
   *   title: string,
   *   objectiveHeading: string,
   *   objective: string[],                      // list of lines
   *   controlsHeading: string,
   *   controls: { label: string, value: string }[],
   *   tipsHeading: string,
   *   tips: string[],                           // list of lines
   *   backText: string,
   *   backScene: string                         // key to return to (default: "MenuScene")
   * }
   */
  init(data = {}) {
    /* Merge caller‑supplied fields with sensible defaults */
    this.content = {
      /* Headings & titles */
      title:            data.title            ?? "How to Play",
      objectiveHeading: data.objectiveHeading ?? "Objective",
      controlsHeading:  data.controlsHeading  ?? "Controls",
      tipsHeading:      data.tipsHeading      ?? "Tips",

      /* Body text */
      objective: data.objective ?? [
        "Reach the level goal without losing all lives."
      ],
      controls:  data.controls  ?? [
        { label: "Move",  value: "Arrow Keys / W A S D" },
        { label: "Jump",  value: "Space / W / Up" },
        { label: "Pause", value: "P" }
      ],
      tips: data.tips ?? [
        "Collect items to boost your score.",
        "Avoid hazards to keep your lives."
      ],

      /* Navigation */
      backText:  data.backText  ?? "Back to Menu",
      backScene: data.backScene ?? "MenuScene"
    };
  }

  create() {
    const { width } = this.scale;
    let y = 60;
    const line = (txt, style = {}) =>
      this.add.text(width / 2, y, txt, { font: "20px Arial", color: "#ffffff", ...style }).setOrigin(0.5);

    this.cameras.main.setBackgroundColor("#222222");

    /* ---------- Title ---------- */
    line(this.content.title, { font: "32px Arial" });
    y += 60;

    /* ---------- Objective ---------- */
    line(this.content.objectiveHeading, { font: "24px Arial" });
    y += 28;
    this.content.objective.forEach(t => { line(t); y += 28; });
    y += 20;

    /* ---------- Controls ---------- */
    line(this.content.controlsHeading, { font: "24px Arial" });
    y += 28;
    this.content.controls.forEach(c => { line(`- ${c.label}: ${c.value}`); y += 28; });
    y += 20;

    /* ---------- Tips ---------- */
    line(this.content.tipsHeading, { font: "24px Arial" });
    y += 28;
    this.content.tips.forEach(t => { line(`- ${t}`); y += 28; });

    /* ---------- Back button ---------- */
    const back = line(this.content.backText, { color: "#00ff00" })
      .setInteractive({ useHandCursor: true });

    back.on("pointerover", () => back.setColor("#ffff00"));
    back.on("pointerout",  () => back.setColor("#00ff00"));
    back.on("pointerdown", () => this.scene.start(this.content.backScene));
  }
}
