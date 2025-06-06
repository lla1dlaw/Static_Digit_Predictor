// Author: Liam Laidlaw
// Filename: index.js
// Date: 03/17/2024
// Purpose: Allows user to draw on included canvas element
// Note: This is an introductory javascript project; comments may be verbose/redundant and are intended to emphasize understanding of functionality
//       rather than to serve as pure documentation.
import NetworkManager from './Network-Classes.js';

// make sure window is loaded
window.addEventListener('load', () => {
    resize();
    window.addEventListener('resize', resize);
    addNetworkOptions()
    .then()
    .catch((error) => {
        console.error(error.message);
    });
});

// -------------------------vars & objects----------------------

let mouseCoordinates = {x:0, y:0};
let draw = false; // draw lines to canvas

const canvas = document.querySelector('#canvas');
const context = canvas.getContext('2d', { willReadFrequently: true }); // context for 2d canvas operations
const clearButton = document.querySelector('#clear-button');
const selectNetworkMenu = document.getElementById("select-network-type");
const networkCanvas = document.getElementById("network-canvas");
const predictionText = document.getElementById("prediction");
const networkContext = networkCanvas.getContext('2d', { willReadFrequently: true });
const netManager = new NetworkManager("../onnx_models/", networkCanvas);

let canvasData;
let grayScaleImage;
let imageObject;
let serverReconnectDelay = 1000;
let savedWidth;
let savedHeight;

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawingPredict);
canvas.addEventListener('mousemove', drawCanvas);
clearButton.addEventListener('click', clearCanvas);
selectNetworkMenu.addEventListener('change', displayNewNetwork)

// ------------- functions --------------
function displayNewNetwork() {
    const layers_string = selectNetworkMenu.value;
    if (layers_string == "") { return; } // ensure that a network has been selected

    const layers_as_strings = layers_string.split("-")
    const layers = layers_as_strings.map(num => parseInt(num)); // cast string layers to integers
    
    predictionText.textContent = "";
}

async function getInference() {
    const selected_net = selectNetworkMenu.value;
    if (selected_net == "") { // if the user hasn't selected a network...
        selectNetworkMenu.style.background = "#ff6961";
        return;
    }

    // send query
    canvasData = context.getImageData(0, 0, canvas.width, canvas.height).data;
    grayScaleImage = processImageVector(canvasData);

    // request prediction from backend
    pred_info = netManager.getInference(grayScaleImage); // will include prediction and layer activations after models are altered to include them
    const prediction = pred_info['prediction'];
    const activations = pred_info['activations']; // returns [0] for now

    // draw network activations

    // reset predict button styling and enable it
    predictButton.style.background = "white";
    predictButton.style.cursor = "pointer";
    predictButton.disabled = false;

    // display prediction
    predictionText.textContent = prediction;
}


function addNetworkOptions() {

    // clear network select menu elements and add placeholder option
    selectNetworkMenu.innerHTML = ''; // ensure select tag is empty
    const placeHolder = document.createElement('option');
    placeHolder.value = "";
    placeHolder.text = "Select Model";
    placeHolder.class = "roboto-regular";
    selectNetworkMenu.appendChild(placeHolder);
    
    // TODO: get available networks from network manager object
    const available_networks = netManager.getAvailableNetworks();
    // populate selectNetworkMenu with available options from microservice
    available_networks.forEach(option => {
        const e = document.createElement('option');
        e.value = option;
        e.text = option;
        e.class = "roboto-regular";
        e.id = option;
        selectNetworkMenu.appendChild(e);
    });
}


function clearCanvas() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    predictionText.textContent = "";
    initCanvas();
}

function initCanvas() {
    context.fillStyle = '#FFFFFF';
    context.fill();
}

function initNetworkCanvas() {
    const networkCanvasContainer = document.getElementById("network-container");
    networkCanvasContainer.height = 2 * canvas.height; // 2x the draw canvas height
    //networkCanvas.style.height = "100%";
}

// Ensure relative canvas size
function resize() {
    const savedCanvas = document.createElement('canvas');
    const savedContext = savedCanvas.getContext('2d');
    savedCanvas.width = canvas.width;
    savedCanvas.height = canvas.height;

    // Save the current canvas content
    savedContext.drawImage(canvas, 0, 0);

    // Resize the canvas
    canvas.width = window.innerHeight * 0.3;
    canvas.height = canvas.width;

    // Restore the saved content
    context.drawImage(savedCanvas, 0, 0, savedCanvas.width, savedCanvas.height, 0, 0, canvas.width, canvas.height);

    initNetworkCanvas();
}


function getMouseCoordinates(mouseEvent) {
    mouseCoordinates.x = mouseEvent.clientX - canvas.offsetLeft;
    mouseCoordinates.y = mouseEvent.clientY - canvas.offsetTop;
}


function startDrawing(mouseEvent) {
    draw = true;
    getMouseCoordinates(mouseEvent);
}


function stopDrawingPredict() {
    draw = false;
    getInference();

}


function drawCanvas(mouseEvent) {
    if (!draw) return;

    initCanvas();
    context.beginPath();
    // ensure that line parameters are correct
    context.lineWidth = context.canvas.width*.10; // line width relative to canvas size
    context.lineCap = 'round';
    context.lineStyle = 'black';

    // draw line
    context.moveTo(mouseCoordinates.x, mouseCoordinates.y);
    getMouseCoordinates(mouseEvent); // update position as cursor moves
    context.lineTo(mouseCoordinates.x, mouseCoordinates.y);
    context.stroke(); // draw the stroke
}



function processImageVector(imageVector) { // removes the rgb values from the image data and returns only the alpha values (0: white - 255: black)
    let processedImageVector = [];
    for (let i = 0; i < imageVector.length; i += 4) {
        processedImageVector.push(imageVector[i + 3]);
    }
    return processedImageVector;
}



