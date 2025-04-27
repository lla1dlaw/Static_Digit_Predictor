
import * as ort from 'onnxruntime-web';
import * as fs from 'fs';
import * as path from 'path';

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


class Network {
    /**
     * A visual respresentation of a network based on its name (ex. '10-10-10'). More activated nodes are brighter shade of the node color. 
     * @param {string} name - The name of the network. Note: This parameter defines its dimensions.
     * @param {string} path - The path to the network onnx file.
     * @param {string} nodeColor - Hex color for the displayed nodes.
     * @param {HTMLCanvasElement} canvas - The canvas to draw the visualization to. 
     * @param {CanvasRenderingContext2D} canvasContext - The rendering context for the associated canvas. . 
     */
    constructor(name, path,  canvas, canvasContext, nodeColor="#FF5733") {
        this.name = name;
        this.path = path;
        this.canvas = canvas;
        this.canvasContext = canvasContext;
        this.layers = name.split("-").forEach(element => parseInt(element));// turn name into integer layer counts
        this.nodeColor = rgb(nodeColor); // neon green
        this.edgeColor = "#000000"; // black
        this.nodeGap = 20; // px gap between nodes (vertically)
        this.radius = this.calcNodeRadius();
        this.layerGap = width/numLayers - (this.radius * numLayers * 2); // Gap to leave between layers
        this.nodes = generateNodes();
        this.model = loadModel(path);

        // input dims 
        if ("cnn" in this.name) {
            this.inputDims = [1, 1, 28, 28];
        } else {
            this.inputDims = [1, 784];
        }
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

    draw() {
        this.canvasContext.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.nodes.forEach(node => {
            this.canvasContext.beginPath();
            this.canvasContext.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            this.canvasContext.fillStyle = node.color.toHex();
            this.canvasContext.fill();
            this.canvasContext.closePath();
        });
    }
    drawEdges() {
        this.nodes.forEach(node => {
            this.canvasContext.beginPath();
            this.canvasContext.moveTo(node.x, node.y);
            this.canvasContext.lineTo(node.x + this.layerGap, node.y);
            this.canvasContext.strokeStyle = this.edgeColor;
            this.canvasContext.stroke();
            this.canvasContext.closePath();
        });
    }
    drawNetwork() {
        this.draw();
        this.drawEdges();
    }

    /**
     * Loads the network from a file. 
     * @param {string} filePath - The path to the file. 
     */
    async loadModel(filePath) {
        try {
            return await ort.InferenceSession.create(filepath);
        } catch (error) {
            console.error(`Error Loading model ${this.name} from ${filePath}:`, error);
            return null;
        }
    }


    /**
     * Runs inference on the network. 
     * @param {Array} data - Image data to run through the network
     * @returns {int} - The index of the most activated node (infered digit).
     */
    async getInference(data) {
        const inputTensor = new ort.Tensor('float32', data, this.inputDims);
        if (this.inputDims )
        await this.model.run({ input: inputTensor }).then(output => {
            const outputTensor = output.values().next().value;
            const outputData = outputTensor.data;
            const maxIndex = outputData.indexOf(Math.max(...outputData));
            return maxIndex;
        }).catch(error => {
            console.error(`Error running inference on model ${this.name}:`, error);
            return null;
        });
    }
}

class NetworkManager {

    constructor(models_dir) {
        
    }
}


