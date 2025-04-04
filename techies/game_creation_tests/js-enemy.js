"use strict";

// Define Enemy AI states
const EnemyState = {
    PATROL: 'PATROL',
    CHASE: 'CHASE',
    ATTACK: 'ATTACK'
};

class Enemy {
    constructor(x, y, patrolBounds = { left: x - 50, right: x + 50 }) {
        // Position and dimensions
        this.x = x;
        this.y = y;
        this.width = 30;
        this.height = 40;
        this.vx = 0; // Horizontal velocity
        this.vy = 0; // Vertical velocity if needed

        // Patrol configuration
        this.patrolBounds = patrolBounds; // Patrol boundaries
        this.patrolSpeed = 1.5; // Speed during patrol
        this.chaseSpeed = 2.5;  // Speed when chasing the player

        // Detection and attack parameters
        this.detectionRadius = 150; // Pixels to detect the player
        this.attackRange = 30;      // Attack range in pixels
        this.attackDamage = 10;     // Damage inflicted per attack
        this.attackCooldown = 1000; // Time in ms between attacks
        this.lastAttackTime = 0;

        // Initial state
        this.state = EnemyState.PATROL;
        // Patrol direction: 1: right, -1: left
        this.direction = 1;
    }

    // Utility: Check for collision between two entities (Axis-Aligned Bounding Box)
    static isColliding(entityA, entityB) {
        return !(
            entityA.x > entityB.x + entityB.width ||
            entityA.x + entityA.width < entityB.x ||
            entityA.y > entityB.y + entityB.height ||
            entityA.y + entityA.height < entityB.y
        );
    }

    // Main update method. Parameters:
    //    player: instance of Player
    //    deltaTime: elapsed time in ms
    //    currentTime: current timestamp (ms)
    update(player, deltaTime, currentTime) {
        // Calculate horizontal distance to player
        let dx = (player.x + player.width / 2) - (this.x + this.width / 2);
        let distance = Math.abs(dx);

        // State transitions based on player distance
        if (distance <= this.attackRange) {
            this.state = EnemyState.ATTACK;
        } else if (distance <= this.detectionRadius) {
            this.state = EnemyState.CHASE;
        } else {
            this.state = EnemyState.PATROL;
        }

        // Execute behavior based on the state
        switch (this.state) {
            case EnemyState.PATROL:
                this.patrol(deltaTime);
                break;
            case EnemyState.CHASE:
                this.chase(player, deltaTime);
                break;
            case EnemyState.ATTACK:
                // Stop horizontal movement while attacking
                this.vx = 0;
                // Attack if cooldown period has passed
                if (currentTime - this.lastAttackTime >= this.attackCooldown) {
                    this.attack(player);
                    this.lastAttackTime = currentTime;
                }
                break;
        }

        // Update horizontal position; vertical movement can be added as needed
        this.x += this.vx * deltaTime * 0.06;  // Adjust for consistent movement timing
    }

    // Patrol behavior: move back and forth within patrol bounds
    patrol(deltaTime) {
        this.vx = this.patrolSpeed * this.direction;
        
        // Reverse direction upon reaching patrol bounds
        if (this.x <= this.patrolBounds.left) {
            this.x = this.patrolBounds.left;
            this.direction = 1;
        } else if (this.x + this.width >= this.patrolBounds.right) {
            this.x = this.patrolBounds.right - this.width;
            this.direction = -1;
        }
    }

    // Chase behavior: move towards the player's position
    chase(player, deltaTime) {
        if ((player.x + player.width / 2) < (this.x + this.width / 2)) {
            // Player is to the left
            this.vx = -this.chaseSpeed;
        } else {
            // Player is to the right
            this.vx = this.chaseSpeed;
        }
    }

    // Attack behavior: inflict damage to the player if colliding
    attack(player) {
        if (Enemy.isColliding(this, player)) {
            console.log('Enemy attacking player!');
            player.takeDamage(this.attackDamage);
        } else {
            // If not colliding, return to chase mode
            this.state = EnemyState.CHASE;
        }
    }

    // Render the enemy on a canvas 2D context
    render(ctx) {
        ctx.fillStyle = '#FF0000'; // Red color for the enemy
        ctx.fillRect(this.x, this.y, this.width, this.height);
        
        // Debug: Draw detection radius
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.3)';
        ctx.beginPath();
        ctx.arc(this.x + this.width / 2, this.y + this.height / 2, this.detectionRadius, 0, 2 * Math.PI);
        ctx.stroke();
    }
}

// Expose Enemy class to global scope
window.Enemy = Enemy;

// End of enemy AI module
