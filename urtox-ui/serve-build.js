const fs = require("fs");
const http = require("http");
const path = require("path");

const root = path.join(__dirname, "build");
const port = Number(process.env.PORT || 3000);

const contentTypes = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

http
  .createServer((req, res) => {
    let requestPath = decodeURIComponent(req.url.split("?")[0]);
    if (requestPath === "/") requestPath = "/index.html";

    let filePath = path.join(root, requestPath);
    if (!filePath.startsWith(root) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
      filePath = path.join(root, "index.html");
    }

    res.setHeader("Content-Type", contentTypes[path.extname(filePath)] || "application/octet-stream");
    fs.createReadStream(filePath).pipe(res);
  })
  .listen(port, "127.0.0.1", () => {
    console.log(`Serving build on http://localhost:${port}`);
  });
