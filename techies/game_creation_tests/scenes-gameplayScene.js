"use strict";

// GameplayScene handles the main game logic, including player control, scene updates, rendering, health management, and collectibles integration.

class GameplayScene {
    constructor() {
        console.log("Constructing GameplayScene...");
        // Initialize the player with a starting position
        this.player = new Player(50, 100);
        // Flag to indicate game over state
        this.isGameOver = false;

        // Collectibles system
        this.collectibles = [];
        this.score = 0;
        // Power-up status
        this.activePowerup = false;
        this.powerupEndTime = 0;
        // Store the player's original speed for power-up reversion
        this.baseSpeed = this.player.speed;
    }

    initScene() {
        console.log("Initializing gameplay scene...");
        // Bind input events for player control specific to this scene
        this.bindInput();

        // Spawn initial collectibles
        this.spawnInitialCollectibles();
    }

    bindInput() {
        // Handle key down events for movement, jumping, and simulating damage for demonstration
        window.addEventListener("keydown", (e) => {
            if (e.repeat) return; // Ignore keys that are held down
            switch(e.key) {
                case "ArrowLeft":
                case "a":
                case "A":
                    this.player.moveLeft();
                    break;
                case "ArrowRight":
                case "d":
                case "D":
                    this.player.moveRight();
                    break;
                case "ArrowUp":
                case "w":
                case "W":
                case " ":
                    this.player.jump();
                    break;
                // For demo purposes: press 'k' to simulate taking damage
                case "k":
                case "K":
                    // Simulate player taking 20 points of damage
                    this.player.takeDamage(20);
                    break;
                default:
                    break;
            }
        });

        // Handle key up events to stop horizontal movement when keys are released
        window.addEventListener("keyup", (e) => {
            switch(e.key) {
                case "ArrowLeft":
                case "a":
                case "A":
                case "ArrowRight":
                case "d":
                case "D":
                    this.player.stopHorizontal();
                    break;
                default:
                    break;
            }
        });
    }

    spawnInitialCollectibles() {
        // For demonstration, spawn a few coins and one power-up at fixed positions
        // Assumes canvas dimensions available via #game-canvas element
        const canvas = document.getElementById("game-canvas");
        if (!canvas) return;

        // Spawn coins at various positions
        this.collectibles.push(new Collectible(150, canvas.height - 100, "coin"));
        this.collectibles.push(new Collectible(300, canvas.height - 150, "coin"));
        this.collectibles.push(new Collectible(450, canvas.height - 100, "coin"));

        // Spawn a power-up
        this.collectibles.push(new Collectible(600, canvas.height - 120, "powerup"));
    }

    startScene() {
        console.log("Starting gameplay scene...");
        // Trigger game events and animations as necessary
    }

    pauseScene() {
        console.log("Pausing gameplay scene...");
        // Pause timers, animations, or other in-game actions
    }

    resumeScene() {
        console.log("Resuming gameplay scene...");
        // Resume any paused activity
    }

    updateScene() {
        // Skip updates if game is over
        if (this.isGameOver) return;

        const canvas = document.getElementById("game-canvas");

        // Update player physics and collision detection
        this.player.update(16, canvas.height);

        // Process collectibles: check collisions with the player and apply effects
        // We'll use the globally available isColliding() from collectibles.js
        let remainingCollectibles = [];
        this.collectibles.forEach(collectible => {
            if (isColliding(this.player, collectible)) {
                // Process pickup based on collectible type
                if (collectible.type === "coin") {
                    this.score += 10;
                    console.log(`Coin collected! Score: ${this.score}`);
                } else if (collectible.type === "powerup") {
                    // Activate powerup: e.g., double player speed for a duration
                    this.activePowerup = true;
                    // Set powerup duration (use collectible.duration property)
                    this.powerupEndTime = Date.now() + collectible.duration;
                    // Increase player's speed
                    this.player.speed = this.baseSpeed * 2;
                    console.log(`Power-up activated! Speed boosted for ${collectible.duration}ms.`);
                }
                // Do not push this collectible back into remaining array (i.e., remove it)
            } else {
                remainingCollectibles.push(collectible);
            }
        });
        this.collectibles = remainingCollectibles;

        // Check if active powerup has expired
        if (this.activePowerup && Date.now() > this.powerupEndTime) {
            this.activePowerup = false;
            this.player.speed = this.baseSpeed;
            console.log('Power-up expired, player speed normalized.');
        }

        // Check for game over condition based on player health
        if (this.player.health <= 0 && !this.isGameOver) {
            this.gameOver();
        }
    }

    renderScene(ctx) {
        console.log("Rendering gameplay scene...");
        // Clear previous frame if necessary (assumed handled by main game loop)

        // Render collectibles
        this.collectibles.forEach(collectible => {
            collectible.render(ctx);
        });

        // Render the player as a simple rectangle for visualization
        ctx.fillStyle = "#FFD700"; // Gold color for the player
        ctx.fillRect(this.player.x, this.player.y, this.player.width, this.player.height);

        // Render HUD: Health bar, Score, and Power-up indicator
        // Health Bar
        const barWidth = 100;
        const barHeight = 10;
        const healthPercent = this.player.health / this.player.maxHealth;

        ctx.fillStyle = "#555"; // Dark grey background for health bar
        ctx.fillRect(20, 20, barWidth, barHeight);

        ctx.fillStyle = "#0f0"; // Green health bar
        ctx.fillRect(20, 20, barWidth * healthPercent, barHeight);

        ctx.strokeStyle = "#000";
        ctx.strokeRect(20, 20, barWidth, barHeight);

        // Score Display
        ctx.fillStyle = "#fff";
        ctx.font = "16px Arial";
        ctx.fillText(`Score: ${this.score}`, 20, 50);

        // Power-up Indicator if active
        if (this.activePowerup) {
            // Calculate remaining time in seconds
            const remaining = Math.max(0, Math.floor((this.powerupEndTime - Date.now()) / 1000));
            ctx.fillStyle = "#00FF00";
            ctx.fillText(`Power-up active (${remaining}s)`, 20, 70);
        }
    }

    // Handle game over sequence
    gameOver() {
        console.log("Game Over! Player health reached 0.");
        this.isGameOver = true;
        // Trigger state transition to a "Game Over" screen or handle respawn logic
        // For now, we simply log the event and reset the player after a short delay
        setTimeout(() => {
            // Reset player health and position
            this.player.resetHealth();
            this.player.x = 50;
            this.player.y = 100;
            this.isGameOver = false;
            console.log("Player respawned and health reset.");
        }, 2000);
    }
}

// Expose GameplayScene to the global scope if necessary
window.GameplayScene = GameplayScene;
