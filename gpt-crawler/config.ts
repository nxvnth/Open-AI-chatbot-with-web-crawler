import { Config } from "./src/config";

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';



// Assuming the crawler_url.txt is in the root of your Node.js project
const tempFilePath = 'crawler_url.txt'

let crawlerUrl = '';

try {
    // Synchronously read from the file
    crawlerUrl = fs.readFileSync(tempFilePath, { encoding: 'utf8' });
} catch (error) {
    console.error('Error reading the URL file:', error);
}


export const defaultConfig: Config = {
  url: crawlerUrl,
  match: crawlerUrl+"**",
  maxPagesToCrawl: 10,
  outputFileName: "output.json",
  maxTokens: 2000000,
};
