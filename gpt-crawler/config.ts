import { Config } from "./src/config";

export const defaultConfig: Config = {
  url: "https://www.concur.co.in/",
  match: "https://www.concur.co.in/**",
  maxPagesToCrawl: 1000,
  outputFileName: "output.json",
  maxTokens: 2000000,
};
