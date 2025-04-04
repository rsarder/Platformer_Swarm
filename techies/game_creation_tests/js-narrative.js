"use strict";

class NarrativeManager {
    constructor() {
        this.overlay = document.getElementById("narrative-overlay");
        if (!this.overlay) {
            this.createOverlay();
        }
    }

    createOverlay() {
        this.overlay = document.createElement("div");
        this.overlay.id = "narrative-overlay";
        Object.assign(this.overlay.style, {
            position: "fixed",
            top: "0",
            left: "0",
            width: "100%",
            height: "100%",
            backgroundColor: "rgba(0, 0, 0, 0.7)",
            color: "#fff",
            display: "none",
            justifyContent: "center",
            alignItems: "center",
            fontSize: "24px",
            zIndex: "1000",
            textAlign: "center",
            padding: "20px"
        });
        document.body.appendChild(this.overlay);
    }

    showDialogue(text, duration = 3000, callback = null) {
        this.overlay.innerHTML = `<div>${text}</div>`;
        this.overlay.style.display = "flex";
        setTimeout(() => {
            this.hideDialogue();
            if (callback) callback();
        }, duration);
    }

    hideDialogue() {
        this.overlay.style.display = "none";
    }

    showCutscene(cutsceneId, duration = 5000, callback = null) {
        const cutsceneText = this.getCutsceneText(cutsceneId);
        this.showDialogue(cutsceneText, duration, callback);
    }

    getCutsceneText(cutsceneId) {
        const cutscenes = {
            "bossIntro": "You have entered the lair. Prepare for the ultimate battle!",
            "bossDefeat": "Victory! The boss has been vanquished. A new chapter begins..."
        };
        return cutscenes[cutsceneId] || "";
    }
}

window.NarrativeManager = NarrativeManager;
