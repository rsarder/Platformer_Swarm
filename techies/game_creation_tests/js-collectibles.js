"use strict";

// Define a Collectible class for coins and power-ups
class Collectible {
    constructor(x, y, type = "coin") {
        this.x = x;
        this.y = y;
        this.width = 20;
        this.height = 20;
        this.type = type; // 'coin' increases score, 'powerup' grants a temporary buff
        // Duration of powerup effect (milliseconds), applicable if type is 'powerup'
        this.duration = (type === "powerup") ? 5000 : 0;
    }

    // Render the collectible onto the canvas context
    render(ctx) {
        if (this.type === "coin") {
            ctx.fillStyle = "#FFD700"; // golden coin
            ctx.beginPath();
            ctx.arc(this.x + this.width/2, this.y + this.height/2, this.width/2, 0, 2*Math.PI);
            ctx.fill();
        } else if (this.type === "powerup") {
            ctx.fillStyle = "#00FF00"; // green square for powerups
            ctx.fillRect(this.x, this.y, this.width, this.height);
        }
    }
}

// Utility function for simple axis-aligned bounding box collision detection
function isColliding(a, b) {
    return !(
        a.x > b.x + b.width ||
        a.x + a.width < b.x ||
        a.y > b.y + b.height ||
        a.y + a.height < b.y
    );
}

// Expose to global scope
window.Collectible = Collectible;
window.isColliding = isColliding;
