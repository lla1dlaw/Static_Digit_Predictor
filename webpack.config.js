const path = require('path');

module.exports = {
    entry: './scripts/main.js', // Your main JS file
    output: {
        path: path.resolve(__dirname, 'docs'), // Output to the 'docs' folder
        filename: 'bundle.js',
    },
    mode: 'production',
};