/* Zero-dependency static dev server. Forwards --host/--port (and HOST/PORT env). */
"use strict";
const http = require("http");
const fs = require("fs");
const path = require("path");

const argv = process.argv.slice(2);
function arg(name, fallback) {
  const i = argv.findIndex((a) => a === `--${name}`);
  if (i !== -1 && argv[i + 1]) return argv[i + 1];
  const eq = argv.find((a) => a.startsWith(`--${name}=`));
  if (eq) return eq.split("=")[1];
  return fallback;
}
const host = arg("host", process.env.HOST || "127.0.0.1");
const port = parseInt(arg("port", process.env.PORT || "7100"), 10);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

http.createServer((req, res) => {
  let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (p === "/") p = "/index.html";
  const file = path.join(__dirname, path.normalize(p).replace(/^([/\\])+/, ""));
  if (!file.startsWith(__dirname)) { res.writeHead(403); res.end(); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end("not found"); return; }
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  });
}).listen(port, host, () => {
  console.log(`dev server: http://${host}:${port}/`);
});
