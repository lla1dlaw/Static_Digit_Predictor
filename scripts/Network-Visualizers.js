
class rgb{
    /**
     * Converts a hexedicimal string into rgb values
     * @param {string} hex 
     */
    constructor(hex) {
        this.hex = hex.replace("#", "");
        this.parsedHex = parseInt(hex, 16);
        this.r = (this.parsedHex >> 16) & 255;
        this.g = (this.parsedHex >> 8) & 255;
        this.b = this.parsedHex & 255;
    }

    /**
     * Converts rgb values to a hex string.
     * @returns {string} Hex string representation of rgb values. 
     */

    toHex() {
        rhex = this.r.toString(16);
        ghex = this.g.toString(16);
        bhex = this.b.toString(16);
        return "#" + rhex + ghex + bhex;
    }
}


class Node {
    /**
     * A visual respresentation of a node in a network.  
     * @param {rgb} color - The inital color of the node. 
     * @param {int} x - Initial X coordinate of the node.
     * @param {int} y - Inital Y coordinate of the node.
     * @param {int} radius - Radius of the node. 
     */
    constructor(color, x, y, radius) {
        this.color = color;
        this.x = x;
        this.y = y;
        this.radius = radius;
    }
}


class NetworkVisualizer {
    /**
     * A visual respresentation of a network based on its name (ex. '10-10-10'). More activated nodes are brighter shade of the node color. 
     * @param {string} name - The name of the network. Note: This parameter defines its dimensions.
     * @param {string} nodeColor - Hex color for the displayed nodes.
     * @param {HTMLCanvasElement} canvas - The canvas to draw the visualization to. 
     * @param {CanvasRenderingContext2D} canvasContext - The rendering context for the associated canvas. . 
     */
    constructor(name,  canvas, canvasContext, nodeColor="#FF5733") {
        this.name = name;
        this.canvas = canvas;
        this.canvasContext = canvasContext;
        this.layers = name.split("-").forEach(element => parseInt(element));// turn name into integer layer counts
        this.nodeColor = rgb(nodeColor); // neon green
        this.edgeColor = "#000000"; // black
        this.nodeGap = 20; // px gap between nodes (vertically)
        this.radius = this.calcNodeRadius();
        this.layerGap = width/numLayers - (this.radius * numLayers * 2); // Gap to leave between layers
        this.nodes = generateNodes();
    }

    calcNodeRadius() {
        const maxLayerWidth = Math.max(...this.layers);
        const width = this.canvas.width;
        const maxDiameter = width/maxLayerWidth;
        const radius = Math.floor((maxDiameter - this.nodeGap)/2);
        return radius
    }

    generateNodes() {
        const nodes = [];
        // init vars
        let x = 0;
        let y = 0;
        let layerWidth = 0;
        
        // calc coords and generate nodes
        for (let i=0; i<this.layers.length; i++) {
            x = this.layerGap * (i+1);
            layerWidth = this.layers[i];
            for (let j=0; j < layerWdith; j++) {
                y = (2 * this.radius + this.nodeGap) * (j+1);
                node = Node(this.color, x, y, this.radius)
                nodes.push(node);
            }
        }

        return nodes;
    }

    drawBlankNetwork() {
        return none;
    }

    applyActivations() {
        return none;
    }

}


